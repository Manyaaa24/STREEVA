"""
STREEVA — Area Risk Scoring Engine
app/core/h3_utils.py

H3 hexagonal grid utilities.

What is H3?
  Uber's open-source hierarchical hexagonal grid system for the globe.
  Reference: https://h3geo.org/

Why H3 instead of a lat/lng grid?
  1. Equal-area: every hexagon at the same resolution covers the same area.
     Square grids distort area at different latitudes.
  2. Equal distance: the center of a hexagon is equidistant from the center
     of ALL 6 neighbours — no diagonal penalty like in square grids.
  3. Hierarchical: a resolution-9 cell is contained by one resolution-8 cell,
     making multi-resolution analysis trivial.
  4. Compact representation: a single 64-bit integer (string-encoded) identifies
     any cell on Earth. Efficient cache key.

Resolution 9 chosen because:
  - Edge length: ~174 m
  - Cell area: ~0.105 km²
  - Matches approximately one city block in Chennai density
  - Granular enough for hyperlocal risk, large enough to have OSM/Places data
"""

from __future__ import annotations

import math
from typing import Sequence

import h3
from shapely.geometry import Polygon

from app.utils.logger import get_logger

logger = get_logger(__name__)


def latlng_to_h3(lat: float, lng: float, resolution: int) -> str:
    """
    Convert a WGS-84 latitude/longitude to an H3 cell index string.

    Args:
        lat: Latitude in decimal degrees (WGS-84).
        lng: Longitude in decimal degrees (WGS-84).
        resolution: H3 resolution (0–15). Use 9 for hyperlocal.

    Returns:
        H3 cell index string, e.g. '89283082837ffff'.

    Example:
        >>> latlng_to_h3(13.0827, 80.2707, 9)
        '8928308283fffff'
    """
    return h3.latlng_to_cell(lat, lng, resolution)


def h3_to_centroid(h3_index: str) -> tuple[float, float]:
    """
    Return the (lat, lng) centroid of an H3 cell.

    Args:
        h3_index: H3 cell index string.

    Returns:
        Tuple of (latitude, longitude) for the cell center.
    """
    lat, lng = h3.cell_to_latlng(h3_index)
    return lat, lng


def h3_to_polygon(h3_index: str) -> Polygon:
    """
    Return the Shapely Polygon boundary of an H3 cell.
    Useful for spatial queries and visualisation.

    Args:
        h3_index: H3 cell index string.

    Returns:
        Shapely Polygon in WGS-84 coordinates.
    """
    # h3.cell_to_boundary returns list of (lat, lng) tuples
    boundary_latlng = h3.cell_to_boundary(h3_index)
    # Shapely uses (lng, lat) = (x, y) convention
    boundary_lnglat = [(lng, lat) for lat, lng in boundary_latlng]
    return Polygon(boundary_lnglat)


def get_k_ring(h3_index: str, k: int = 1) -> set[str]:
    """
    Return the set of H3 cells within k rings of the given cell.
    Includes the center cell itself.

    Args:
        h3_index: Center H3 cell index string.
        k: Ring distance. k=1 returns 7 cells (center + 6 neighbours).

    Returns:
        Set of H3 cell index strings.
    """
    return set(h3.grid_disk(h3_index, k))


def h3_edge_length_m(resolution: int) -> float:
    """
    Return the approximate edge length of H3 cells at given resolution (meters).

    Uses the average edge lengths from the H3 spec.
    Reference: https://h3geo.org/docs/core-library/restable

    Args:
        resolution: H3 resolution (0–15).

    Returns:
        Average edge length in meters.
    """
    # Average edge lengths in km from H3 spec
    edge_lengths_km = {
        0: 1281.256011, 1: 483.0568391, 2: 182.5129565, 3: 68.97922179,
        4: 26.07175968, 5: 9.854090990, 6: 3.724532667, 7: 1.406475763,
        8: 0.531414010, 9: 0.200786148, 10: 0.075863783, 11: 0.028663897,
        12: 0.010830188, 13: 0.004092010, 14: 0.001546100, 15: 0.000584169,
    }
    return edge_lengths_km.get(resolution, 0.0) * 1000.0


def fill_bbox_with_h3(
    south: float, west: float, north: float, east: float, resolution: int
) -> set[str]:
    """
    Fill a bounding box with H3 cells at the given resolution.

    Args:
        south, west, north, east: WGS-84 bounding box coordinates.
        resolution: H3 resolution.

    Returns:
        Set of H3 cell index strings covering the bounding box.
    """
    # Create a GeoJSON-compatible polygon for the bounding box
    bbox_polygon = {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]]
    }
    cells = set(h3.geo_to_cells(bbox_polygon, resolution))
    logger.info(
        "h3_bbox_filled",
        resolution=resolution,
        cell_count=len(cells),
        bbox={"south": south, "west": west, "north": north, "east": east},
    )
    return cells


def haversine_distance_m(
    lat1: float, lng1: float, lat2: float, lng2: float
) -> float:
    """
    Calculate the great-circle distance between two points on Earth (meters).

    Uses the Haversine formula. Accurate for distances < ~1000 km.

    Args:
        lat1, lng1: First point coordinates.
        lat2, lng2: Second point coordinates.

    Returns:
        Distance in meters.
    """
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
