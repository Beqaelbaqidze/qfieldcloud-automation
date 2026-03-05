#!/usr/bin/env python3
"""
QFieldCloud GeoPackage Processor
=================================
Downloads a GeoPackage from a QFieldCloud project, converts polygon
boundaries to a lines layer, then uploads the updated file back.

Environment variables (set as GitHub Secrets):
  QFIELDCLOUD_USERNAME    – Your QFieldCloud login email / username
  QFIELDCLOUD_PASSWORD    – Your QFieldCloud password
  QFIELDCLOUD_PROJECT_ID  – Project UUID from the QFieldCloud project URL
  QFIELDCLOUD_URL         – (optional) Base URL, default: https://app.qfield.cloud
  GPKG_FILENAME           – (optional) Remote file path, default: project.gpkg
  POLYGON_LAYER_NAME      – (optional) Layer to read,  default: polygons
  LINES_LAYER_NAME        – (optional) Layer to write, default: polygon_boundaries
"""

import os
import sys
import logging
import sqlite3
from pathlib import Path

import fiona
import geopandas as gpd
import requests

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Configuration (from environment / GitHub Secrets) ─────────────────────────
QFIELDCLOUD_URL      = os.environ.get("QFIELDCLOUD_URL", "https://app.qfield.cloud")
QFIELDCLOUD_USERNAME = os.environ["QFIELDCLOUD_USERNAME"]   # required
QFIELDCLOUD_PASSWORD = os.environ["QFIELDCLOUD_PASSWORD"]   # required
PROJECT_ID           = os.environ["QFIELDCLOUD_PROJECT_ID"] # required
GPKG_FILENAME        = os.environ.get("GPKG_FILENAME", "project.gpkg")
POLYGON_LAYER        = os.environ.get("POLYGON_LAYER_NAME", "polygons")
LINES_LAYER          = os.environ.get("LINES_LAYER_NAME", "polygon_boundaries")

LOCAL_GPKG = Path("project.gpkg")
API_BASE   = f"{QFIELDCLOUD_URL.rstrip('/')}/api/v1"


# ── QFieldCloud API helpers ───────────────────────────────────────────────────

def get_token() -> str:
    """Authenticate with username/password and return a session token."""
    url = f"{API_BASE}/auth/token/"
    log.info("Authenticating as '%s' …", QFIELDCLOUD_USERNAME)
    resp = requests.post(
        url,
        json={"username": QFIELDCLOUD_USERNAME, "password": QFIELDCLOUD_PASSWORD},
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise ValueError(f"No token in auth response: {resp.text}")
    log.info("Authentication successful.")
    return token


def download_gpkg(headers: dict) -> None:
    """Stream the GeoPackage file from QFieldCloud to disk."""
    url = f"{API_BASE}/files/{PROJECT_ID}/{GPKG_FILENAME}/"
    log.info("Downloading '%s' from project %s …", GPKG_FILENAME, PROJECT_ID)

    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with LOCAL_GPKG.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=8_192):
                fh.write(chunk)

    size_kb = LOCAL_GPKG.stat().st_size / 1024
    log.info("Download complete — %.1f KB saved to %s", size_kb, LOCAL_GPKG)


def upload_gpkg(headers: dict) -> None:
    """Upload the processed GeoPackage back to QFieldCloud."""
    url = f"{API_BASE}/files/{PROJECT_ID}/{GPKG_FILENAME}/"
    log.info("Uploading '%s' to project %s …", GPKG_FILENAME, PROJECT_ID)

    with LOCAL_GPKG.open("rb") as fh:
        resp = requests.post(
            url,
            headers=headers,
            files={"file": (GPKG_FILENAME, fh, "application/geopackage+sqlite3")},
            timeout=180,
        )
    resp.raise_for_status()
    log.info("Upload complete — HTTP %d", resp.status_code)


# ── GeoPackage layer helpers ──────────────────────────────────────────────────

def list_layers(gpkg_path: Path) -> list[str]:
    """Return all layer names present in the GeoPackage."""
    return fiona.listlayers(str(gpkg_path))


def drop_layer(gpkg_path: Path, layer_name: str) -> None:
    """
    Remove a layer from a GeoPackage using direct SQLite operations.
    This is the safest way to delete a single layer without touching others.
    """
    with sqlite3.connect(gpkg_path) as conn:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (layer_name,),
        ).fetchone()

        if not table_exists:
            return  # nothing to do

        conn.execute(f'DROP TABLE IF EXISTS "{layer_name}"')
        conn.execute("DELETE FROM gpkg_contents WHERE table_name=?", (layer_name,))
        conn.execute(
            "DELETE FROM gpkg_geometry_columns WHERE table_name=?", (layer_name,)
        )
        conn.commit()
    log.info("Dropped existing layer '%s'.", layer_name)


# ── Geometry processing ───────────────────────────────────────────────────────

def build_lines_layer(polygons_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Derive a lines GeoDataFrame from polygon boundaries.

    Steps:
    1. Extract .boundary from each polygon (returns LinearRing / MultiLineString).
    2. Explode multi-part geometries so every row is a single LineString.
    3. Drop any degenerate (empty) geometries.
    4. Copy all non-geometry attribute columns from the source polygons.
    5. Carry over the original CRS unchanged.
    """
    lines_gdf = polygons_gdf.copy()
    lines_gdf.geometry = polygons_gdf.geometry.boundary  # Polygon → LinearRing
    lines_gdf = lines_gdf.explode(index_parts=False)     # flatten multi-part rings
    lines_gdf = lines_gdf[~lines_gdf.is_empty].reset_index(drop=True)
    return lines_gdf


def process_gpkg() -> None:
    """
    Core processing pipeline:
    1. Read the polygons layer (CRS preserved automatically by geopandas).
    2. Validate geometry types — warn and filter out non-polygon features.
    3. Build the lines layer from polygon boundaries.
    4. Rewrite the polygons layer (unchanged) and write the new lines layer.
       Using drop + append mode prevents duplicate features on repeated runs.
    """
    # ── Read polygons ─────────────────────────────────────────────────────────
    existing_layers = list_layers(LOCAL_GPKG)
    log.info("Layers found in GeoPackage: %s", existing_layers)

    if POLYGON_LAYER not in existing_layers:
        raise ValueError(
            f"Layer '{POLYGON_LAYER}' not found in {LOCAL_GPKG}. "
            f"Available layers: {existing_layers}"
        )

    polygons_gdf = gpd.read_file(LOCAL_GPKG, layer=POLYGON_LAYER, engine="pyogrio")

    if polygons_gdf.empty:
        raise ValueError(f"Layer '{POLYGON_LAYER}' is empty — nothing to process.")

    log.info(
        "Loaded %d features | CRS: %s | geometry types: %s",
        len(polygons_gdf),
        polygons_gdf.crs,
        polygons_gdf.geometry.geom_type.unique().tolist(),
    )

    # ── Filter to polygon geometries only ────────────────────────────────────
    valid_types = {"Polygon", "MultiPolygon"}
    non_poly = polygons_gdf[~polygons_gdf.geometry.geom_type.isin(valid_types)]
    if not non_poly.empty:
        log.warning(
            "Skipping %d non-polygon feature(s) with type(s): %s",
            len(non_poly),
            non_poly.geometry.geom_type.unique().tolist(),
        )
        polygons_gdf = polygons_gdf[
            polygons_gdf.geometry.geom_type.isin(valid_types)
        ].copy()

    # ── Build lines layer ────────────────────────────────────────────────────
    lines_gdf = build_lines_layer(polygons_gdf)
    log.info(
        "Generated %d boundary line feature(s) | CRS: %s",
        len(lines_gdf),
        lines_gdf.crs,
    )

    # ── Write layers back to the GeoPackage ─────────────────────────────────
    # Drop the lines layer if it already exists (prevents duplicates on re-runs).
    drop_layer(LOCAL_GPKG, LINES_LAYER)

    # Append the new lines layer. All other layers in the file are untouched.
    log.info("Writing lines layer '%s' to %s …", LINES_LAYER, LOCAL_GPKG)
    lines_gdf.to_file(
        LOCAL_GPKG,
        layer=LINES_LAYER,
        driver="GPKG",
        engine="pyogrio",
        mode="a",  # append: adds a new layer without affecting existing ones
    )

    log.info("GeoPackage layers after processing: %s", list_layers(LOCAL_GPKG))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    try:
        token   = get_token()
        headers = {"Authorization": f"Token {token}"}
        download_gpkg(headers)
        process_gpkg()
        upload_gpkg(headers)
        log.info("All done.")
    except requests.HTTPError as exc:
        log.error(
            "QFieldCloud API error %d: %s",
            exc.response.status_code,
            exc.response.text,
        )
        sys.exit(1)
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
