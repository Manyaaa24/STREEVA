"""
STREEVA — Area Risk Scoring Engine
scripts/download_worldpop.py

WorldPop Population Density Raster Download Script.

What this downloads:
  WorldPop Global High Resolution Population Denominators (2020)
  - India unconstrained individual countries, 100m resolution
  - File: ind_ppp_2020_1km_Aggregated_UNadj.tif (or 100m variant)
  - Values: estimated people per 100m × 100m grid cell
  - License: CC BY 4.0 — free to use with attribution

Citation required in your report:
  WorldPop (www.worldpop.org) - School of Geography and Environmental Science,
  University of Southampton; Department of Geography and Geosciences,
  University of Louisville; Departement de Geographie, Universite de Namur)
  and Center for International Earth Science Information Network (CIESIN),
  Columbia University (2018). Global High Resolution Population Denominators Project
  - Funded by The Bill and Melinda Gates Foundation (OPP1134076).
  https://dx.doi.org/10.5258/SOTON/WP00649

File sizes:
  - 100m unconstrained: ~300 MB for India
  - 1km aggregated: ~3 MB for India (lower resolution, faster)
  Default: 1km aggregated (faster to start, still useful for Chennai-level analysis)

To switch to 100m:
  Change USE_1KM = True to False (requires ~300MB download)

Run from project root:
    python scripts/download_worldpop.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent

# ── Configuration ─────────────────────────────────────────────────────────────
# Use 1km aggregated raster for quicker setup.
# Set to False to download 100m (~300 MB) for full resolution.
USE_1KM = True

if USE_1KM:
    # 1km aggregated — suitable for district/city level (Chennai occupies ~426 km²)
    FILE_NAME = "ind_ppp_2020_1km_Aggregated_UNadj.tif"
    DOWNLOAD_URL = (
        "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/"
        "2020/IND/ind_ppp_2020_1km_Aggregated_UNadj.tif"
    )
    OUTPUT_PATH = ROOT / "data" / "raw" / "worldpop_india_2020_1km.tif"
else:
    # 100m unconstrained — high resolution (~300 MB)
    FILE_NAME = "ind_ppp_2020.tif"
    DOWNLOAD_URL = (
        "https://data.worldpop.org/GIS/Population/Global_2000_2020/"
        "2020/IND/ind_ppp_2020.tif"
    )
    OUTPUT_PATH = ROOT / "data" / "raw" / "worldpop_india_2020_100m.tif"


def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Progress callback for urllib.request.urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, int(downloaded * 100 / total_size))
        mb_done = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        print(f"\r  Downloading: {pct}% ({mb_done:.1f} / {mb_total:.1f} MB)", end="", flush=True)
    else:
        mb_done = downloaded / 1_048_576
        print(f"\r  Downloaded: {mb_done:.1f} MB", end="", flush=True)


def crop_to_tamilnadu(input_path: Path, output_path: Path) -> bool:
    """
    Optional: crop the India raster to Tamil Nadu bounding box.
    Reduces file size significantly. Requires rasterio.

    Tamil Nadu bbox: south=7.9, west=76.2, north=13.6, east=80.4

    Args:
        input_path: Path to full India raster.
        output_path: Path for cropped Tamil Nadu raster.

    Returns:
        True if crop succeeded, False otherwise.
    """
    try:
        import rasterio
        from rasterio.mask import mask
        from shapely.geometry import box
        import json

        TN_BBOX = {
            "south": 7.9, "west": 76.2, "north": 13.6, "east": 80.4
        }

        print(f"\nCropping to Tamil Nadu bounding box...")
        bbox_geom = box(TN_BBOX["west"], TN_BBOX["south"], TN_BBOX["east"], TN_BBOX["north"])

        with rasterio.open(str(input_path)) as src:
            geom = [bbox_geom.__geo_interface__]
            out_image, out_transform = mask(src, geom, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
            })

        with rasterio.open(str(output_path), "w", **out_meta) as dest:
            dest.write(out_image)

        size_mb = output_path.stat().st_size / 1_048_576
        print(f"Cropped raster saved: {output_path} ({size_mb:.1f} MB)")
        return True

    except Exception as exc:
        print(f"Crop failed (not critical): {exc}")
        return False


def download() -> None:
    """Download the WorldPop raster."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
        print(f"File already exists: {OUTPUT_PATH} ({size_mb:.1f} MB)")
        print("Delete the file and re-run to force re-download.")
        return

    print(f"Downloading WorldPop raster: {FILE_NAME}")
    print(f"URL: {DOWNLOAD_URL}")
    print(f"Destination: {OUTPUT_PATH}")
    print("This may take a few minutes depending on your connection...")

    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, str(OUTPUT_PATH), show_progress)
        print()  # newline after progress
        size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
        print(f"\nDownload complete: {OUTPUT_PATH} ({size_mb:.1f} MB)")
    except Exception as exc:
        print(f"\nERROR downloading: {exc}")
        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()
        sys.exit(1)

    # Optionally crop to Tamil Nadu
    tn_path = ROOT / "data" / "raw" / f"worldpop_tamilnadu_2020{'_1km' if USE_1KM else '_100m'}.tif"
    crop_success = crop_to_tamilnadu(OUTPUT_PATH, tn_path)

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    if crop_success:
        print(f"1. The Tamil Nadu-cropped raster is at:")
        print(f"   {tn_path}")
        print(f"2. Update WORLDPOP_RASTER_PATH in your .env file:")
        print(f"   WORLDPOP_RASTER_PATH={tn_path.relative_to(ROOT)}")
    else:
        print(f"1. The India raster is at: {OUTPUT_PATH}")
        print(f"2. Update WORLDPOP_RASTER_PATH in your .env file:")
        print(f"   WORLDPOP_RASTER_PATH={OUTPUT_PATH.relative_to(ROOT)}")
    print("3. Restart the service: uvicorn app.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("STREEVA — WorldPop Population Raster Download")
    print(f"Resolution: {'1 km (aggregated)' if USE_1KM else '100 m (full)'}")
    print("=" * 60)
    download()
