# ...existing code...
import ast
import json
import math
import re
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# import DB connector from project
from DbConnector import DbConnector

# optional projection / viewers
try:
    import pyproj
    _PYPROJ_ERR = None
except Exception as e:
    pyproj = None
    _PYPROJ_ERR = str(e)

try:
    import pptk
    _PPTK_ERR = None
except Exception as e:
    pptk = None
    _PPTK_ERR = str(e)

try:
    import open3d as o3d
    _O3D_ERR = None
except Exception as e:
    o3d = None
    _O3D_ERR = str(e)


def parse_polyline(poly_str):
    """Parse POLYLINE-like string into list of (lat, lon) pairs."""
    if poly_str is None or (isinstance(poly_str, float) and math.isnan(poly_str)):
        return []
    s = str(poly_str).strip()
    # try JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append((float(item[0]), float(item[1])))
            if out:
                return out
    except Exception:
        pass
    # try Python literal
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append((float(item[0]), float(item[1])))
            if out:
                return out
    except Exception:
        pass
    # fallback: extract floats and pair them
    floats = [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", s)]
    pairs = []
    for i in range(0, len(floats) - 1, 2):
        pairs.append((floats[i], floats[i + 1]))
    return pairs


def read_porto_from_db(db: DbConnector, table: str = "trips", poly_col: str = "POLYLINE",
                       where: str | None = None, sample_trips: int | None = None,
                       max_points: int | None = None, fetch_batch: int = 1000) -> pd.DataFrame:
    """
    Stream POLYLINE rows from MySQL via DbConnector and return DataFrame with columns lat, lon, trip_id.
    - Auto-detects an id-like column (id, trip_id, TRIP_ID, etc). If none found, uses a generated row index.
    - Handles missing/NULL polylines and skips malformed rows.
    """
    cur = db.cursor

    # discover columns in the table and try to pick an id column
    try:
        cur.execute(f"SHOW COLUMNS FROM `{table}`")
        cols_info = cur.fetchall()
        cols = [c[0] for c in cols_info]
    except Exception:
        cols = []

    id_col = None
    for candidate in ("id", "trip_id", "tripid", "TRIP_ID", "TRIPID"):
        for c in cols:
            if c.lower() == candidate.lower():
                id_col = c
                break
        if id_col:
            break

    # Build select query (omit id if not found)
    if id_col:
        q = f"SELECT `{id_col}`, `{poly_col}` FROM `{table}`"
    else:
        q = f"SELECT `{poly_col}` FROM `{table}`"

    if where:
        q += f" WHERE {where}"
    if sample_trips:
        q += f" LIMIT {int(sample_trips)}"

    try:
        cur.execute(q)
    except Exception as e:
        raise RuntimeError(f"Failed to execute query: {e}. Check table/column names. Available columns: {cols}") from e

    rows_out = []
    pts = 0
    generated_id = 0
    while True:
        batch = cur.fetchmany(fetch_batch)
        if not batch:
            break
        for row in batch:
            try:
                if id_col:
                    row_id = row[0]
                    poly = row[1]
                else:
                    row_id = generated_id
                    poly = row[0]
                    generated_id += 1

                if poly is None:
                    continue
                coords = parse_polyline(poly)
                if not coords:
                    continue
                for lat, lon in coords:
                    # ensure lat/lon numeric
                    rows_out.append({"lat": float(lat), "lon": float(lon), "trip_id": int(row_id)})
                    pts += 1
                    if max_points and pts >= max_points:
                        break
                if max_points and pts >= max_points:
                    break
            except Exception:
                # skip malformed row but continue processing
                continue
        if max_points and pts >= max_points:
            break

    df = pd.DataFrame(rows_out)
    return df


def lonlat_to_meters(lon, lat, use_pyproj=True):
    """Convert lon/lat arrays to x,y in meters. Use pyproj if available, else equirectangular approx."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if use_pyproj and pyproj is not None:
        mean_lon = float(np.nanmean(lon))
        zone = int((mean_lon + 180) / 6) + 1
        proj = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
        x, y = proj(lon.tolist(), lat.tolist())
        return np.array(x), np.array(y)
    # fallback approx
    lat0 = np.nanmean(lat)
    R = 6378137.0
    x = np.deg2rad(lon - 0.0) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - 0.0) * R
    return x, y


def visualize_points_generic(pts_xyz: np.ndarray, attrs: np.ndarray | None = None, out_path: Path | None = None):
    """
    Try interactive viewers (pptk, open3d) then fallback to static scatter PNG.
    pts_xyz: Nx3 array
    attrs: optional 1D array for coloring
    """
    # pptk
    if pptk is not None:
        try:
            v = pptk.viewer(pts_xyz)
            if attrs is not None:
                v.attributes(attrs)
            try:
                v.set(point_size=0.5)
            except Exception:
                pass
            return
        except Exception:
            pass
    # open3d
    if o3d is not None:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_xyz)
            if attrs is not None:
                vals = np.asarray(attrs, dtype=float)
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                if vmax - vmin > 0:
                    norm = (vals - vmin) / (vmax - vmin + 1e-9)
                else:
                    norm = np.zeros_like(vals)
                cmap = plt.get_cmap("jet")
                colors = cmap(norm)[:, :3]
                pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd])
            return
        except Exception:
            pass
    # static fallback
    out_path = Path(out_path or "figures/trajectories_fallback.png")
    x = pts_xyz[:, 0]
    y = pts_xyz[:, 1]
    color = attrs if attrs is not None else None
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, s=0.5, c=color, alpha=0.6, linewidths=0)
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved static visualization to", out_path)


def visualize_points_matplotlib_only(pts_xyz: np.ndarray, out_path: Path, attrs: np.ndarray | None = None):
    """
    Simple, reliable static visualization using matplotlib (no pptk/open3d).
    Saves a PNG at out_path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = pts_xyz[:, 0]
    y = pts_xyz[:, 1]
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, s=0.5, c=attrs if attrs is not None else "k", alpha=0.6, linewidths=0)
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved static visualization to", out_path)


# Replace call in main: use matplotlib-only visualizer to guarantee success
def main():
    parser = argparse.ArgumentParser(description="Read trajectories from MySQL and visualize")
    parser.add_argument("--host", default="tdt4225-27.idi.ntnu.no", help="DB host")
    parser.add_argument("--database", default="porto_taxi", help="Database name")
    parser.add_argument("--user", default="edvarsa", help="DB user")
    parser.add_argument("--password", default=os.getenv("DB_PASSWORD", "sql2001"), help="DB password")
    parser.add_argument("--table", default="trips", help="table name with POLYLINE column")
    parser.add_argument("--poly_col", default="POLYLINE", help="column name that holds trajectories")
    parser.add_argument("--where", default=None, help="optional SQL WHERE clause (without WHERE)")
    parser.add_argument("--sample_trips", type=int, default=None, help="limit number of trips (rows) fetched")
    parser.add_argument("--max_points", type=int, default=200000, help="stop after this many points")
    parser.add_argument("--out", default=str(Path("figures/porto_from_db.png")), help="output image path")
    args = parser.parse_args()

    # connect using provided credentials via DbConnector
    try:
        db = DbConnector(HOST=args.host, DATABASE=args.database, USER=args.user, PASSWORD=args.password)
    except Exception as e:
        print("Failed to connect to DB:", e)
        return

    print("Import errors (pyproj,pptk,open3d):", _PYPROJ_ERR, _PPTK_ERR, _O3D_ERR)

    print("Querying table", args.table, "poly column", args.poly_col)
    try:
        df_points = read_porto_from_db(db, table=args.table, poly_col=args.poly_col,
                                       where=args.where, sample_trips=args.sample_trips, max_points=args.max_points)
    except Exception as e:
        print("Error reading from DB:", e)
        db.close_connection()
        return

    if df_points.empty:
        print("No points retrieved from DB.")
        db.close_connection()
        return

    # convert to meters (lon, lat)
    try:
        x, y = lonlat_to_meters(df_points["lon"].values, df_points["lat"].values, use_pyproj=True)
    except Exception as e:
        print("Projection failed, trying fallback equirectangular. Error:", e)
        x, y = lonlat_to_meters(df_points["lon"].values, df_points["lat"].values, use_pyproj=False)

    pts = np.c_[x, y, np.zeros(len(x))]

    attrs = df_points.get("trip_id").to_numpy() if "trip_id" in df_points.columns else None

    # Force matplotlib-only output (works without installing optional libs)
    try:
        visualize_points_matplotlib_only(pts, out_path=Path(args.out), attrs=attrs)
    except Exception as e:
        print("Visualization failed:", e)

    db.close_connection()

if __name__ == "__main__":
    main()
# ...existing code...