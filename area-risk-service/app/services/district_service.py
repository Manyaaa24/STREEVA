import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
GEOJSON_PATH = BASE_DIR / "data" / "processed" / "tn_districts_bbox.geojson"

_districts_gdf = None

def get_district_gdf():
    global _districts_gdf
    if _districts_gdf is None:
        if GEOJSON_PATH.exists():
            try:
                _districts_gdf = gpd.read_file(GEOJSON_PATH)
            except Exception as e:
                logger.error(f"Failed to load district GeoJSON: {e}")
                _districts_gdf = gpd.GeoDataFrame()
        else:
            logger.warning(f"District GeoJSON missing at {GEOJSON_PATH}")
            _districts_gdf = gpd.GeoDataFrame()
    return _districts_gdf

def get_district_for_location(lat: float, lng: float) -> str | None:
    """
    Returns the mapped internal NCRB district name for a lat/lng using
    post-2019 OSM district polygons.
    """
    gdf = get_district_gdf()
    if gdf.empty:
        return None
        
    point = Point(lng, lat)

    def _extract_district_name(name_str: str) -> str | None:
        if 'Chengalpattu' in name_str:
            return 'Chengalpattu'
        elif 'Kanchipuram' in name_str:
            return 'Kanchipuram'
        elif 'Tiruvallur' in name_str or 'Thiruvallur' in name_str:
            return 'Thiruvallur'
        elif 'Chennai' in name_str:
            return 'Chennai'
        return None

    # 1. Exact polygon containment check
    for idx, row in gdf.iterrows():
        if row['geometry'].contains(point):
            dist_name = _extract_district_name(str(row.get('display_name', '')))
            if dist_name:
                return dist_name

    # 2. Nearest district boundary fallback for border / suburban points (e.g., Vandalur / VIT Chennai)
    nearest_dist_name = None
    min_dist = float('inf')
    for idx, row in gdf.iterrows():
        d = row['geometry'].distance(point)
        if d < min_dist:
            dist_name = _extract_district_name(str(row.get('display_name', '')))
            if dist_name:
                min_dist = d
                nearest_dist_name = dist_name

    return nearest_dist_name
