"""
STREEVA — Area Risk Scoring Engine
tests/unit/test_h3_utils.py

Unit tests for H3 grid utilities.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from app.core.h3_utils import (
    fill_bbox_with_h3,
    get_k_ring,
    h3_edge_length_m,
    h3_to_centroid,
    h3_to_polygon,
    haversine_distance_m,
    latlng_to_h3,
)


class TestLatlngToH3:

    def test_returns_string(self):
        result = latlng_to_h3(13.0827, 80.2707, 9)
        assert isinstance(result, str)

    def test_returns_valid_h3_index(self):
        """H3 cell index should be 15 hex chars for resolution 9."""
        import h3
        result = latlng_to_h3(13.0827, 80.2707, 9)
        assert h3.is_valid_cell(result)

    def test_same_point_same_cell(self):
        """Identical coordinates → identical H3 cell."""
        cell1 = latlng_to_h3(12.9141, 80.1408, 9)
        cell2 = latlng_to_h3(12.9141, 80.1408, 9)
        assert cell1 == cell2

    def test_different_resolution_different_cell(self):
        """Same point at different resolutions → different cells."""
        cell_9 = latlng_to_h3(13.0827, 80.2707, 9)
        cell_8 = latlng_to_h3(13.0827, 80.2707, 8)
        assert cell_9 != cell_8

    def test_nearby_points_can_be_same_cell(self):
        """Points very close together should map to the same H3 cell."""
        cell1 = latlng_to_h3(13.0827, 80.2707, 9)
        cell2 = latlng_to_h3(13.08271, 80.27071, 9)  # ~1m apart
        assert cell1 == cell2

    def test_known_chennai_cell(self):
        """Marina Beach (approximate) should return a valid cell."""
        cell = latlng_to_h3(13.0500, 80.2824, 9)
        assert len(cell) == 15


class TestH3ToCentroid:

    def test_round_trip(self):
        """lat/lng → H3 → centroid should be close to original."""
        lat, lng = 13.0827, 80.2707
        cell = latlng_to_h3(lat, lng, 9)
        clat, clng = h3_to_centroid(cell)
        # Centroid should be within ~200m of the original point at res 9
        dist = haversine_distance_m(lat, lng, clat, clng)
        assert dist < 300.0, f"Centroid {dist:.0f}m from original, expected <300m"

    def test_centroid_within_valid_range(self):
        """Centroid should be valid WGS-84 coordinates."""
        cell = latlng_to_h3(13.0827, 80.2707, 9)
        lat, lng = h3_to_centroid(cell)
        assert -90 <= lat <= 90
        assert -180 <= lng <= 180


class TestH3ToPolygon:

    def test_returns_polygon(self):
        cell = latlng_to_h3(13.0827, 80.2707, 9)
        poly = h3_to_polygon(cell)
        assert isinstance(poly, Polygon)

    def test_polygon_is_valid(self):
        cell = latlng_to_h3(13.0827, 80.2707, 9)
        poly = h3_to_polygon(cell)
        assert poly.is_valid

    def test_hexagon_has_6_vertices(self):
        """H3 cells are hexagons — should have 6 exterior vertices."""
        cell = latlng_to_h3(13.0827, 80.2707, 9)
        poly = h3_to_polygon(cell)
        # exterior.coords includes repeated closing vertex
        assert len(poly.exterior.coords) == 7


class TestGetKRing:

    def test_k0_returns_only_center(self):
        cell = latlng_to_h3(13.0827, 80.2707, 9)
        ring = get_k_ring(cell, k=0)
        assert ring == {cell}

    def test_k1_returns_7_cells(self):
        """k=1 ring includes center + 6 neighbours = 7 cells."""
        cell = latlng_to_h3(13.0827, 80.2707, 9)
        ring = get_k_ring(cell, k=1)
        assert len(ring) == 7
        assert cell in ring

    def test_k2_returns_19_cells(self):
        """k=2 ring = 1 + 6 + 12 = 19 cells."""
        cell = latlng_to_h3(13.0827, 80.2707, 9)
        ring = get_k_ring(cell, k=2)
        assert len(ring) == 19


class TestH3EdgeLength:

    def test_res9_edge_length(self):
        """Resolution 9 should have ~200.8m edge length."""
        length = h3_edge_length_m(9)
        assert 180.0 < length < 220.0, f"Expected ~200m, got {length:.1f}m"

    def test_lower_resolution_is_longer(self):
        """Lower resolution = larger hexagons = longer edges."""
        assert h3_edge_length_m(7) > h3_edge_length_m(9)
        assert h3_edge_length_m(9) > h3_edge_length_m(11)


class TestFillBbox:

    def test_small_bbox_has_cells(self):
        """A reasonable bounding box should yield multiple H3 cells."""
        cells = fill_bbox_with_h3(
            south=13.04, west=80.22, north=13.08, east=80.28, resolution=9
        )
        assert len(cells) > 5

    def test_all_cells_valid(self):
        """All returned cells should be valid H3 indices."""
        import h3
        cells = fill_bbox_with_h3(
            south=13.04, west=80.22, north=13.08, east=80.28, resolution=9
        )
        for cell in cells:
            assert h3.is_valid_cell(cell)


class TestHaversineDistance:

    def test_same_point_is_zero(self):
        d = haversine_distance_m(13.08, 80.27, 13.08, 80.27)
        assert d < 1.0

    def test_known_distance(self):
        """
        Distance between Chennai Central (13.0827, 80.2707) and
        approximately 1km north (13.0917, 80.2707) ≈ 1000m.
        """
        d = haversine_distance_m(13.0827, 80.2707, 13.0917, 80.2707)
        assert 900 < d < 1100, f"Expected ~1000m, got {d:.0f}m"

    def test_symmetry(self):
        """Distance A→B should equal distance B→A."""
        d1 = haversine_distance_m(13.0827, 80.2707, 13.0500, 80.2824)
        d2 = haversine_distance_m(13.0500, 80.2824, 13.0827, 80.2707)
        assert abs(d1 - d2) < 0.01
