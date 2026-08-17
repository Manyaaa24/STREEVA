"""
STREEVA — Area Risk Scoring Engine
tests/integration/test_cache.py

Integration tests for the disk-based cache.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.cache.disk_cache import DiskCache


@pytest.fixture
def tmp_cache(tmp_path) -> DiskCache:
    """Cache instance using a temporary directory."""
    return DiskCache(cache_dir=tmp_path)


class TestDiskCache:

    def test_set_and_get_basic(self, tmp_cache):
        """Basic write then read."""
        data = {"key": "value", "number": 42}
        tmp_cache.set("cell123", "osm", data)
        result, is_stale = tmp_cache.get("cell123", "osm", ttl_seconds=3600)
        assert result == data
        assert is_stale is False

    def test_cache_miss_returns_none(self, tmp_cache):
        result, is_stale = tmp_cache.get("nonexistent", "osm", ttl_seconds=3600)
        assert result is None
        assert is_stale is False

    def test_expired_entry_returns_stale(self, tmp_cache):
        """Expired cache entry → returns (data, True) for stale fallback."""
        data = {"key": "value"}
        tmp_cache.set("cell123", "osm", data)
        # Retrieve with TTL of 0 (already expired)
        result, is_stale = tmp_cache.get("cell123", "osm", ttl_seconds=0)
        assert result == data  # Data available as stale fallback
        assert is_stale is True

    def test_no_expiry_never_stale(self, tmp_cache):
        """NO_EXPIRY (-1) should never mark as stale."""
        from app.cache.disk_cache import NO_EXPIRY
        data = {"permanent": True}
        tmp_cache.set("cell_perm", "pop", data)
        result, is_stale = tmp_cache.get("cell_perm", "pop", ttl_seconds=NO_EXPIRY)
        assert result == data
        assert is_stale is False

    def test_different_data_types_separate(self, tmp_cache):
        """Same H3 cell, different data types → separate cache entries."""
        tmp_cache.set("cell_abc", "osm", {"road": "primary"})
        tmp_cache.set("cell_abc", "places_commercial", {"count": 5})

        osm, _ = tmp_cache.get("cell_abc", "osm", ttl_seconds=3600)
        places, _ = tmp_cache.get("cell_abc", "places_commercial", ttl_seconds=3600)

        assert osm == {"road": "primary"}
        assert places == {"count": 5}

    def test_invalidate_removes_entry(self, tmp_cache):
        tmp_cache.set("cell_x", "osm", {"data": 1})
        deleted = tmp_cache.invalidate("cell_x", "osm")
        assert deleted is True
        result, _ = tmp_cache.get("cell_x", "osm", ttl_seconds=3600)
        assert result is None

    def test_invalidate_nonexistent_returns_false(self, tmp_cache):
        result = tmp_cache.invalidate("ghost_cell", "osm")
        assert result is False

    def test_clear_all_removes_all_files(self, tmp_cache):
        for i in range(5):
            tmp_cache.set(f"cell_{i}", "osm", {"i": i})
        count = tmp_cache.clear_all()
        assert count == 5
        stats = tmp_cache.stats()
        assert stats["total_entries"] == 0

    def test_stats_correct(self, tmp_cache):
        tmp_cache.set("cella", "osm", {"x": 1})
        tmp_cache.set("cellb", "osm", {"x": 2})
        tmp_cache.set("cellc", "places_police", {"dist": 300})
        stats = tmp_cache.stats()
        assert stats["total_entries"] == 3
        assert stats["by_data_type"]["osm"] == 2
        assert stats["by_data_type"]["places_police"] == 1

    def test_corrupted_cache_file_returns_none(self, tmp_cache):
        """Gracefully handle a corrupted JSON cache file."""
        path = tmp_cache._cache_path("corrupt_cell", "osm")
        path.write_text("{ NOT VALID JSON !!!", encoding="utf-8")
        result, is_stale = tmp_cache.get("corrupt_cell", "osm", ttl_seconds=3600)
        assert result is None

    def test_data_with_unicode(self, tmp_cache):
        """Non-ASCII data should be preserved."""
        data = {"name": "சென்னை", "value": 42}
        tmp_cache.set("cell_uni", "osm", data)
        result, _ = tmp_cache.get("cell_uni", "osm", ttl_seconds=3600)
        assert result["name"] == "சென்னை"
