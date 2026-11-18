"""Build aligned Sentinel chips for the forestry LAI lab.

Run from the repo root with NumPy, pandas, requests, Rasterio, and Shapely.
Zip the resulting directory's files (without a containing directory) for the
course data server. Contains modified Copernicus Sentinel data (2022–2024).
"""

import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from time import sleep
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform
from shapely.geometry import box, shape

CATALOG = "https://planetarycomputer.microsoft.com/api"
LAI = "https://harvardforest1.fas.harvard.edu/data/p06/hf069/hf069-02-lai-site.csv"
OUTPUT = Path("Lessons/11-Remote-Sensing-in-Forestry-and-Vegetation/data/harvard-forest")
TOWER = (-72.1715, 42.5378)
BBOX = (-72.180, 42.532, -72.171, 42.544)
CRS = "EPSG:32618"
x, y = (coords[0] for coords in transform("EPSG:4326", CRS, [TOWER[0]], [TOWER[1]]))
SIZE = 60
GRID = from_origin(round(x / 20) * 20 - 600, round(y / 20) * 20 + 600, 20, 20)
rows, cols = np.indices((SIZE, SIZE))
px, py = GRID * (cols + 0.5, rows + 0.5)
FOOTPRINT = (px < x) & (np.hypot(px - x, py - y) <= 500)
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12", "SCL")
TOKENS = {}


def get_json(url, params):
    for _ in range(5):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code not in (429, 502, 503):
            break
        sleep(max(2, int(response.headers.get("Retry-After", "2"))))
    response.raise_for_status()
    return response.json()


def search(collection):
    results = get_json(
        f"{CATALOG}/stac/v1/search",
        {"collections": collection, "bbox": ",".join(map(str, BBOX)), "limit": 1000,
         "datetime": "2022-04-01T00:00:00Z/2024-11-15T00:00:00Z"},
    )
    if any(link["rel"] == "next" for link in results["links"]):
        raise ValueError(f"Unexpected pagination for {collection}")
    return [item for item in results["features"] if shape(item["geometry"]).covers(box(*BBOX))]


def read(item, band):
    href = item["assets"][band]["href"]
    resampling = Resampling.nearest if band == "SCL" else Resampling.average
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY=2,
                      GDAL_HTTP_TIMEOUT=30), rasterio.open(href + "?" + TOKENS[container(href)]) as source:
        with WarpedVRT(source, crs=CRS, transform=GRID, width=SIZE, height=SIZE,
                       resampling=resampling, dtype="float32", nodata=np.nan) as aligned:
            return aligned.read(1)


def date(item):
    return pd.Timestamp(item["properties"]["datetime"][:10])


def container(href):
    url = urlsplit(href)
    return url.hostname.split(".")[0] + "/" + url.path.split("/")[1]


def prepare(field_date, optical, radar):
    near_optical = [item for item in optical if abs((date(item) - field_date).days) <= 10]
    near_optical.sort(key=lambda item: (abs((date(item) - field_date).days),
                                       item["properties"]["eo:cloud_cover"], item["id"]))
    near_radar = [item for item in radar if abs((date(item) - field_date).days) <= 10]
    near_radar.sort(key=lambda item: (abs((date(item) - field_date).days), item["id"]))
    if not near_radar:
        return None
    s1 = near_radar[0]
    for s2 in near_optical:
        if abs((date(s2) - date(s1)).days) > 10:
            continue
        scl = read(s2, "SCL")
        # Restrict the comparison to clear vegetation pixels in the study footprint.
        if (scl[FOOTPRINT] == 4).mean() < 0.75:
            continue
        optical_values = {band: (read(s2, band) - 1000) / 10000 for band in BANDS[:-1]}
        arrays = dict(optical_values, SCL=scl, VV=read(s1, "vv"), VH=read(s1, "vh"))
        common = FOOTPRINT & (scl == 4)
        for values in arrays.values():
            common &= np.isfinite(values) & (values > 0)
        if common.sum() < 0.75 * FOOTPRINT.sum():
            continue
        stamp = str(field_date.date())
        np.savez_compressed(OUTPUT / f"{stamp}.npz", **arrays)
        return {"field_date": stamp, "optical_date": str(date(s2).date()),
                "radar_date": str(date(s1).date()), "optical_id": s2["id"],
                "radar_id": s1["id"], "common_pixels": int(common.sum())}
    return None


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    optical = [
        item for item in search("sentinel-2-l2a")
        if float(item["properties"]["s2:processing_baseline"]) >= 4
        and item["properties"]["eo:cloud_cover"] < 65
    ]
    radar = [
        item for item in search("sentinel-1-rtc")
        if item["properties"]["sat:relative_orbit"] == 135 and {"vv", "vh"} <= item["assets"].keys()
    ]
    containers = {container(item["assets"][band]["href"])
                  for items, band in ((optical, "SCL"), (radar, "vv")) for item in items}
    for key in sorted(containers):
        TOKENS[key] = get_json(f"{CATALOG}/sas/v1/token/{key}", {})["token"]
    response = requests.get(LAI, timeout=30)
    response.raise_for_status()
    lai = pd.read_csv(BytesIO(response.content), parse_dates=["date"]).dropna(subset=["all_mean"])
    dates = lai.loc[(lai.site == "ems") & lai.year.between(2022, 2024), "date"].unique()
    print(f"{len(dates)} field dates; {len(optical)} optical scenes; {len(radar)} radar scenes", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        matches = [row for row in pool.map(lambda day: prepare(pd.Timestamp(day), optical, radar), dates)
                   if row is not None]
    # Keep one field visit per independent optical/radar scene pair, prioritizing closest dates.
    manifest = pd.DataFrame(matches)
    if manifest.empty:
        raise ValueError("No matching dates after cloud and coverage screening")
    field = pd.to_datetime(manifest.field_date)
    distance = sum((pd.to_datetime(manifest[col]) - field).abs().dt.days
                   for col in ("optical_date", "radar_date"))
    manifest = (manifest.assign(distance=distance).sort_values(["distance", "field_date"])
                .drop_duplicates("optical_id").drop_duplicates("radar_id")
                .sort_values("field_date").drop(columns="distance"))
    for path in OUTPUT.glob("*.npz"):
        if path.stem not in set(manifest.field_date):
            path.unlink()
    manifest.to_csv(OUTPUT / "scenes.csv", index=False)
    np.save(OUTPUT / "footprint.npy", FOOTPRINT)
    (OUTPUT / "grid.json").write_text(json.dumps({"crs": CRS, "transform": tuple(GRID)[:6],
                                                  "shape": [SIZE, SIZE], "tower_lonlat": TOWER},
                                                 indent=2) + "\n")
    print(manifest.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
