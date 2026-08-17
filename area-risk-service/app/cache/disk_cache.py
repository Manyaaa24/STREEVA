"""
STREEVA — Area Risk Scoring Engine
app/cache/disk_cache.py

Disk-based JSON cache keyed by H3 cell ID.

Design decisions:
  - Storage format: JSON files (one per H3 cell, per data type)
  - Cache key: f"{h3_cell_id}_{data_type}" → "{key}.json"
  - TTL: configurable per data type (OSM=7 days, Places=1 day, Population=30 days)
  - Thread safety: file writes are atomic (write to temp, rename)
  - No external dependencies: no Redis, no SQLite — just the filesystem

Why disk cache (not in-memory)?
  In-memory caches (e.g., functools.lru_cache) are lost on restart.
  During development, we query the same few dozen H3 cells repeatedly —
  a disk cache saves API quota across server restarts and re-runs.

Cache file naming convention:
  data/cache/{h3_cell_id}_{data_type}.json
  Example: data/cache/8928308283fffff_places.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# TTL sentinel value for "no expiry"
NO_EXPIRY = -1


class DiskCache:
    """
    Simple disk-based JSON cache with configurable TTL.

    All cache entries store a 'cached_at' Unix timestamp. The get()
    method checks this against the provided TTL before returning data.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        settings = get_settings()
        self._cache_dir = cache_dir or settings.cache_dir_abs
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("disk_cache_initialized", cache_dir=str(self._cache_dir))

    def _cache_path(self, h3_cell: str, data_type: str) -> Path:
        """Return the file path for a cache entry."""
        safe_key = f"{h3_cell}_{data_type}"
        return self._cache_dir / f"{safe_key}.json"

    def get(
        self,
        h3_cell: str,
        data_type: str,
        ttl_seconds: float,
    ) -> tuple[Any | None, bool]:
        """
        Retrieve a cached value if it exists and has not expired.

        Args:
            h3_cell: H3 cell index string (cache key component).
            data_type: Data type string, e.g. 'places', 'osm', 'population'.
            ttl_seconds: Time-to-live in seconds. Use NO_EXPIRY for permanent cache.

        Returns:
            Tuple of (data, is_stale):
              - data: The cached data dict, or None if not found.
              - is_stale: True if data exists but is expired (allows stale fallback).
        """
        path = self._cache_path(h3_cell, data_type)

        if not path.exists():
            logger.debug("cache_miss", h3_cell=h3_cell, data_type=data_type)
            return None, False

        try:
            with open(path, encoding="utf-8") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "cache_read_error",
                h3_cell=h3_cell,
                data_type=data_type,
                error=str(exc),
            )
            return None, False

        cached_at: float = entry.get("cached_at", 0.0)
        age_seconds = time.time() - cached_at

        if ttl_seconds != NO_EXPIRY and age_seconds > ttl_seconds:
            logger.debug(
                "cache_expired",
                h3_cell=h3_cell,
                data_type=data_type,
                age_seconds=round(age_seconds),
                ttl_seconds=ttl_seconds,
            )
            return entry.get("data"), True  # stale but available as fallback

        logger.debug(
            "cache_hit",
            h3_cell=h3_cell,
            data_type=data_type,
            age_seconds=round(age_seconds),
        )
        return entry.get("data"), False

    def set(self, h3_cell: str, data_type: str, data: Any) -> None:
        """
        Write a value to the cache.

        Uses atomic write (temp file → rename) to prevent partial reads
        if the process is interrupted during the write.

        Args:
            h3_cell: H3 cell index string.
            data_type: Data type identifier string.
            data: JSON-serialisable data to cache.
        """
        path = self._cache_path(h3_cell, data_type)
        entry = {
            "h3_cell": h3_cell,
            "data_type": data_type,
            "cached_at": time.time(),
            "data": data,
        }

        # Atomic write: write to temp file, then rename
        tmp_path = path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=None)
            tmp_path.rename(path)
            logger.debug("cache_written", h3_cell=h3_cell, data_type=data_type)
        except OSError as exc:
            logger.error(
                "cache_write_error",
                h3_cell=h3_cell,
                data_type=data_type,
                error=str(exc),
            )
            # Clean up temp file if it exists
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def invalidate(self, h3_cell: str, data_type: str) -> bool:
        """
        Delete a specific cache entry.

        Args:
            h3_cell: H3 cell index string.
            data_type: Data type identifier string.

        Returns:
            True if the entry existed and was deleted, False otherwise.
        """
        path = self._cache_path(h3_cell, data_type)
        if path.exists():
            path.unlink()
            logger.info("cache_invalidated", h3_cell=h3_cell, data_type=data_type)
            return True
        return False

    def clear_all(self) -> int:
        """
        Delete all cache files.

        Returns:
            Number of files deleted.
        """
        count = 0
        for f in self._cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        logger.info("cache_cleared", files_deleted=count)
        return count

    def stats(self) -> dict[str, Any]:
        """
        Return cache statistics.

        Returns:
            Dict with total_entries, total_size_bytes, and breakdown by data_type.
        """
        files = list(self._cache_dir.glob("*.json"))
        total_bytes = sum(f.stat().st_size for f in files)
        by_type: dict[str, int] = {}
        for f in files:
            parts = f.stem.split("_", 1)
            dtype = parts[1] if len(parts) > 1 else "unknown"
            by_type[dtype] = by_type.get(dtype, 0) + 1
        return {
            "total_entries": len(files),
            "total_size_bytes": total_bytes,
            "by_data_type": by_type,
            "cache_dir": str(self._cache_dir),
        }


# ── Module-level singleton ────────────────────────────────────────────────────
_cache_instance: DiskCache | None = None


def get_cache() -> DiskCache:
    """Return the module-level DiskCache singleton."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DiskCache()
    return _cache_instance
