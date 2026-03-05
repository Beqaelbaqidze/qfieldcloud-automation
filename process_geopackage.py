#!/usr/bin/env python3
"""
QFieldCloud – Automated Boundary Generator
==========================================
1. Downloads Surveyor.gpkg from QFieldCloud.
2. Finds nakveti parcels that have no topo_point children yet.
3. Generates topo_point + topo_line features for each new parcel.
4. Uploads the updated GeoPackage back to QFieldCloud.

Requires only the Python standard library + requests.
No QGIS, geopandas, or GDAL needed.

GitHub Secrets required:
  QFIELDCLOUD_USERNAME  – your QFieldCloud login (e.g. BeqaElbaqidze)
  QFIELDCLOUD_PASSWORD  – your QFieldCloud password
"""

import os
import sys
import struct
import logging
import sqlite3
from pathlib import Path

import requests

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Constants (project-specific, not secrets) ─────────────────────────────────
QFIELDCLOUD_URL = "https://app.qfield.cloud"
PROJECT_ID      = "a9ee4eb1-1b33-40c4-b122-525eb80acdd6"   # BeqaElbaqidze/Surveyor_Cad
GPKG_FILENAME   = "Surveyor.gpkg"
LOCAL_GPKG      = Path(GPKG_FILENAME)
API_BASE        = f"{QFIELDCLOUD_URL}/api/v1"

# ── GeoPackage domain constants ───────────────────────────────────────────────
SRS_ID  = 32638                 # EPSG:32638 – WGS 84 / UTM zone 38N
STATUS  = "არაფიქსირებული"      # Georgian: "unregistered"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ── Credentials from GitHub Secrets ──────────────────────────────────────────
QFIELDCLOUD_USERNAME = os.environ["QFIELDCLOUD_USERNAME"]
QFIELDCLOUD_PASSWORD = os.environ["QFIELDCLOUD_PASSWORD"]


# ════════════════════════════════════════════════════════════════════════════════
# QFieldCloud API
# ════════════════════════════════════════════════════════════════════════════════

def get_token() -> str:
    """Exchange username/password for a session token."""
    log.info("Authenticating as '%s' …", QFIELDCLOUD_USERNAME)
    resp = requests.post(
        f"{API_BASE}/auth/token/",
        json={"username": QFIELDCLOUD_USERNAME, "password": QFIELDCLOUD_PASSWORD},
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise ValueError(f"No token returned: {resp.text}")
    log.info("Authentication OK.")
    return token


def get_last_updated(headers: dict) -> str:
    """
    Return the project's data_last_updated_at timestamp from QFieldCloud.
    Used to skip processing when nothing has changed since the last run.
    """
    resp = requests.get(
        f"{API_BASE}/projects/{PROJECT_ID}/",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("data_last_updated_at", "")


def download_gpkg(headers: dict) -> None:
    """Stream Surveyor.gpkg from QFieldCloud to disk."""
    url = f"{API_BASE}/files/{PROJECT_ID}/{GPKG_FILENAME}/"
    log.info("Downloading %s …", GPKG_FILENAME)
    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with LOCAL_GPKG.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=8_192):
                fh.write(chunk)
    log.info("Downloaded %.1f KB.", LOCAL_GPKG.stat().st_size / 1024)


def upload_gpkg(headers: dict) -> None:
    """Upload the updated Surveyor.gpkg back to QFieldCloud."""
    url = f"{API_BASE}/files/{PROJECT_ID}/{GPKG_FILENAME}/"
    log.info("Uploading %s …", GPKG_FILENAME)
    with LOCAL_GPKG.open("rb") as fh:
        resp = requests.post(
            url,
            headers=headers,
            files={"file": (GPKG_FILENAME, fh, "application/geopackage+sqlite3")},
            timeout=180,
        )
    resp.raise_for_status()
    log.info("Upload complete — HTTP %d.", resp.status_code)


# ════════════════════════════════════════════════════════════════════════════════
# SQLite UDFs required by GeoPackage spatial-index triggers
# ════════════════════════════════════════════════════════════════════════════════

def _st_is_empty(blob):
    if blob is None or len(blob) < 4:
        return 1
    return (blob[3] >> 4) & 1

def _envelope_double(blob, offset):
    if blob is None or len(blob) < offset + 8:
        return None
    if ((blob[3] >> 1) & 0x07) == 0:
        return None
    return struct.unpack_from("<d", blob, offset)[0]

def _st_minx(blob): return _envelope_double(blob,  8)
def _st_maxx(blob): return _envelope_double(blob, 16)
def _st_miny(blob): return _envelope_double(blob, 24)
def _st_maxy(blob): return _envelope_double(blob, 32)

def open_gpkg(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.create_function("ST_IsEmpty", 1, _st_is_empty)
    conn.create_function("ST_MinX",    1, _st_minx)
    conn.create_function("ST_MaxX",    1, _st_maxx)
    conn.create_function("ST_MinY",    1, _st_miny)
    conn.create_function("ST_MaxY",    1, _st_maxy)
    return conn


# ════════════════════════════════════════════════════════════════════════════════
# WKB / GeoPackage geometry helpers
# ════════════════════════════════════════════════════════════════════════════════

def parse_vertices(blob) -> list:
    """Extract exterior-ring vertex list from a GeoPackage geometry blob."""
    if not blob or len(blob) < 8 or blob[:2] != b"GP":
        return []
    flags    = blob[3]
    if (flags >> 4) & 1:
        return []
    env_type = (flags >> 1) & 0x07
    env_size = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(env_type, 0)
    wkb      = blob[8 + env_size:]
    if len(wkb) < 5:
        return []
    endian = "<" if wkb[0] == 1 else ">"
    gt     = struct.unpack_from(endian + "I", wkb, 1)[0] & 0xFFFF

    if gt == 3:                         # POLYGON
        return _exterior_ring(wkb, 5, endian)[0]
    if gt == 6:                         # MULTIPOLYGON
        n = struct.unpack_from(endian + "I", wkb, 5)[0]
        if n == 0:
            return []
        off = 9
        en2 = "<" if wkb[off] == 1 else ">"
        gt2 = struct.unpack_from(en2 + "I", wkb, off + 1)[0] & 0xFFFF
        off += 5
        if gt2 == 3:
            return _exterior_ring(wkb, off, en2)[0]
    return []


def _exterior_ring(wkb, off, endian):
    num_rings = struct.unpack_from(endian + "I", wkb, off)[0]; off += 4
    pts = []
    for ring_idx in range(num_rings):
        num_pts = struct.unpack_from(endian + "I", wkb, off)[0]; off += 4
        ring = []
        for _ in range(num_pts):
            x, y = struct.unpack_from(endian + "dd", wkb, off); off += 16
            ring.append((x, y))
        if ring_idx == 0:
            if ring and ring[-1] == ring[0]:
                ring = ring[:-1]
            pts = ring
    return pts, off


def gpkg_point(x: float, y: float) -> bytes:
    """Build a GeoPackage point blob with XY envelope (required by spatial index)."""
    hdr = b"GP" + bytes([0, 0x03]) + struct.pack("<i", SRS_ID)
    env = struct.pack("<dddd", x, x, y, y)
    wkb = struct.pack("<B", 1) + struct.pack("<I", 1) + struct.pack("<dd", x, y)
    return hdr + env + wkb


def gpkg_linestring(pts: list) -> bytes:
    """Build a GeoPackage linestring blob with XY envelope."""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    hdr = b"GP" + bytes([0, 0x03]) + struct.pack("<i", SRS_ID)
    env = struct.pack("<dddd", min(xs), max(xs), min(ys), max(ys))
    wkb = struct.pack("<B", 1) + struct.pack("<I", 2) + struct.pack("<I", len(pts))
    for x, y in pts:
        wkb += struct.pack("<dd", x, y)
    return hdr + env + wkb


# ════════════════════════════════════════════════════════════════════════════════
# Boundary generation
# ════════════════════════════════════════════════════════════════════════════════

def generate_boundaries():
    """
    Find nakveti parcels without topo_point children and generate boundary
    features. Returns (parcels_processed, points_created, lines_created).
    """
    conn = open_gpkg(LOCAL_GPKG)
    cur  = conn.cursor()

    # Only process nakveti that have no topo_point entries yet (idempotent)
    cur.execute("""
        SELECT n.fid, n.identifi, n.geom
        FROM   nakveti n
        WHERE  n.identifi IS NOT NULL
          AND  n.identifi != ''
          AND  NOT EXISTS (
                   SELECT 1 FROM topo_point t
                   WHERE  t.identifi = n.identifi)
    """)
    rows = cur.fetchall()

    if not rows:
        log.info("Nothing to do — all nakveti parcels already have boundary data.")
        conn.close()
        return 0, 0, 0

    log.info("Found %d nakveti parcel(s) to process.", len(rows))

    tp_insert = []
    tl_insert = []
    processed = 0

    for row in rows:
        identifi = row["identifi"]
        verts    = parse_vertices(row["geom"])

        if not verts:
            log.warning("  SKIP fid=%s (%s): could not parse geometry.", row["fid"], identifi)
            continue

        n = len(verts)
        log.info("  %s — %d vertices", identifi, n)

        for i, (x, y) in enumerate(verts):
            point_id = LETTERS[i] if i < 26 else f"Z{i - 25}"
            tp_insert.append((
                gpkg_point(x, y),
                None,               # OBJECTID
                round(x, 3),        # POINT_X
                round(y, 3),        # POINT_Y
                None,               # type
                point_id,           # POINT_ID
                identifi,           # identifi
            ))
            x2, y2 = verts[(i + 1) % n]
            tl_insert.append((
                gpkg_linestring([(x, y), (x2, y2)]),
                None,               # OBJECTID
                STATUS,             # type
                None,               # Shape_Leng
                identifi,           # identifi
            ))
        processed += 1

    cur.executemany(
        "INSERT INTO topo_point (geom,OBJECTID,POINT_X,POINT_Y,type,POINT_ID,identifi)"
        " VALUES (?,?,?,?,?,?,?)",
        tp_insert,
    )
    cur.executemany(
        "INSERT INTO topo_line (geom,OBJECTID,type,Shape_Leng,identifi)"
        " VALUES (?,?,?,?,?)",
        tl_insert,
    )
    conn.commit()
    conn.close()

    return processed, len(tp_insert), len(tl_insert)


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════

LAST_RUN_FILE = Path(".last_updated")


def main() -> None:
    try:
        token   = get_token()
        headers = {"Authorization": f"Token {token}"}

        # ── Quick check: has QFieldCloud data changed since our last run? ──
        last_updated = get_last_updated(headers)
        log.info("QFieldCloud last updated: %s", last_updated)

        prev = LAST_RUN_FILE.read_text().strip() if LAST_RUN_FILE.exists() else ""
        if last_updated and last_updated == prev:
            log.info("No new data since last run — nothing to do.")
            return

        download_gpkg(headers)

        parcels, points, lines = generate_boundaries()

        if parcels == 0:
            log.info("No new parcels — skipping upload.")
        else:
            log.info(
                "Results: %d parcel(s) | %d topo_point | %d topo_line",
                parcels, points, lines,
            )
            upload_gpkg(headers)

        # ── Save timestamp so next run can skip if nothing changed ──────────
        if last_updated:
            LAST_RUN_FILE.write_text(last_updated)

        log.info("Done.")

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
