"""
STREEVA — Ingest Open Datasets
app/scripts/ingest_open_datasets.py

Script to automate the ingestion of open datasets used for training and inference.
Saves all data to `data/raw/` directory.

Sources:
- OpenCity Chennai Police Stations (CSV)
- OpenCity Hospitals/Health facilities (Placeholder/India-Geodata)
- Kontur Population HDX
- Meta/WorldPop
- VIIRS Night-time lights (~india-geodata)
- NCRB Crime Baseline (CSV)
- Safecity Harassment Reports (CSV)
"""

import os
import urllib.request
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Ensure data dictionary exists
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Datasets definition
DATASETS = {
    # 1. OpenCity Chennai Police Stations (CSV)
    "police_stations_chennai.csv": "https://data.opencity.in/dataset/cbb342c2-02d1-42f8-9052-d1891a64ec13/resource/7c919587-2e4d-4cb5-b541-3ca42c11f0a3/download/b0729b83-a4f4-4dff-898e-6e7c6cdd4e7f.csv",
    
    # 2. NCRB District Crime Baseline (Placeholder URL for 2014 data from data.gov.in)
    # The actual CSV should be downloaded and placed here if URL rotates.
    "ncrb_crimes_against_women.csv": "https://raw.githubusercontent.com/yashveeeeeeer/india-geodata/master/placeholder_ncrb_crimes.csv",
    
    # 3. Safecity Incident Reports (Placeholder URL for public demo dump)
    "safecity_incidents.csv": "https://raw.githubusercontent.com/yashveeeeeeer/india-geodata/master/placeholder_safecity_chennai.csv",
    
    # 4. VIIRS Night-time lights (GeoTIFF)
    "viirs_nightlights_chennai.tif": "https://raw.githubusercontent.com/yashveeeeeeer/india-geodata/master/placeholder_viirs.tif",
    
    # 5. Kontur Population Global / India extract (Res 8 H3)
    "kontur_population.gpkg": "https://raw.githubusercontent.com/yashveeeeeeer/india-geodata/master/placeholder_kontur.gpkg",
    
    # 6. Chennai Hospitals
    "chennai_hospitals.geojson": "https://raw.githubusercontent.com/yashveeeeeeer/india-geodata/master/placeholder_hospitals.geojson"
}

def download_file(url: str, dest_path: Path):
    """Download a file gracefully, showing progress or error."""
    if dest_path.exists():
        logger.info(f"File already exists (skipping): {dest_path.name}")
        return
    logger.info(f"Downloading {dest_path.name} from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Successfully downloaded {dest_path.name}")
    except Exception as e:
        logger.error(f"Failed to download {dest_path.name}: {e}")
        # Create a dummy file if the placeholder URL is dead just so the pipeline doesn't crash 
        # completely while we're provisioning real URLs.
        if "placeholder" in url:
            logger.warning(f"Creating empty stub for {dest_path.name} due to placeholder download failure.")
            dest_path.touch()

def main():
    logger.info(f"Starting ingestion. Destination: {RAW_DATA_DIR}")
    for filename, url in DATASETS.items():
        dest = RAW_DATA_DIR / filename
        download_file(url, dest)
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    main()
