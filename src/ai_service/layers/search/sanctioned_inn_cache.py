"""
Fast lookup cache for sanctioned INNs.

Provides O(1) lookup for INN -> sanctioned person/organization mapping.
Much faster than AC search or Elasticsearch for INN-specific queries.
"""

import json
import os
import re
from ...data.resources import PACKAGE_DATA_DIR
import time
from pathlib import Path
from typing import Dict, Optional, Any
from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class SanctionedINNCache:
    """Fast in-memory cache for sanctioned INNs."""

    def __init__(self, data_dir=None):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.loaded_at: Optional[float] = None
        self.data_dir = Path(data_dir or os.getenv("SANCTIONS_DATA_DIR") or PACKAGE_DATA_DIR).resolve()
        self.cache_file = self.data_dir / "sanctioned_inns_cache.json"
        self._source_state = None
        self._all_records = {}
        self.stats = {
            "total_inns": 0,
            "persons": 0,
            "organizations": 0,
            "lookups": 0,
            "hits": 0,
            "misses": 0
        }

    def _current_source_state(self):
        files = [self.data_dir / name for name in ("sanctioned_persons.json", "sanctioned_companies.json")]
        return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_ctime_ns, path.stat().st_size)
                     for path in files if path.exists())

    def load_cache(self) -> bool:
        """Build the derived lookup from current sources, never a stale export."""
        state = self._current_source_state()
        records = {}
        for name, entity_type, fields in (
            ("sanctioned_persons.json", "person", ("itn", "itn_import")),
            ("sanctioned_companies.json", "organization", ("tax_number",)),
        ):
            path = self.data_dir / name
            if not path.exists():
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))
            for row in rows:
                numbers = set()
                for field in fields:
                    value = row.get(field)
                    if value is not None:
                        numbers.update(re.findall(r"(?<!\d)\d{8,12}(?!\d)", str(value)))
                record = {**row, "type": entity_type, "source": "ukrainian_sanctions"}
                for number in numbers:
                    records.setdefault(number, []).append(record)
        self._all_records = records
        self.cache = {number: matches[0] for number, matches in records.items()}
        self._source_state = state
        self.loaded_at = time.time()
        self.stats.update({"total_inns": len(self.cache),
                           "persons": sum(row["type"] == "person" for row in self.cache.values()),
                           "organizations": sum(row["type"] == "organization" for row in self.cache.values())})
        return bool(state)

    def lookup_all(self, inn: str):
        if self._source_state != self._current_source_state():
            self.load_cache()
        return [dict(record) for record in self._all_records.get(str(inn).strip(), [])]

    def lookup(self, inn: str) -> Optional[Dict[str, Any]]:
        """
        Fast lookup for sanctioned INN.

        Args:
            inn: INN to check

        Returns:
            Dict with sanctioned person/organization data if found, None otherwise
        """
        self.stats["lookups"] += 1

        if self._source_state != self._current_source_state():
            self.load_cache()
        if self.loaded_at is None and not self.load_cache():
            self.stats["misses"] += 1
            return None

        inn_normalized = str(inn).strip()
        result = self.cache.get(inn_normalized)

        if result:
            self.stats["hits"] += 1
            logger.debug(f"🚨 SANCTIONED INN FOUND: {inn_normalized} -> {result.get('name', 'Unknown')}")
            return result.copy()  # Return copy to prevent modification
        else:
            self.stats["misses"] += 1
            return None

    def is_sanctioned(self, inn: str) -> bool:
        """Check if INN is sanctioned."""
        return self.lookup(inn) is not None

    def get_sanctioned_person(self, inn: str) -> Optional[Dict[str, Any]]:
        """Get sanctioned person data by INN."""
        result = self.lookup(inn)
        if result and result.get("type") == "person":
            return result
        return None

    def get_sanctioned_organization(self, inn: str) -> Optional[Dict[str, Any]]:
        """Get sanctioned organization data by INN."""
        result = self.lookup(inn)
        if result and result.get("type") == "organization":
            return result
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (self.stats["hits"] / max(1, self.stats["lookups"])) * 100
        return {
            **self.stats,
            "hit_rate_percent": round(hit_rate, 2),
            "loaded_at": self.loaded_at,
            "cache_age_seconds": time.time() - self.loaded_at if self.loaded_at else None
        }

    def reload_cache(self) -> bool:
        """Force reload cache from file."""
        self.cache.clear()
        self.loaded_at = None
        # Reset lookup stats (but keep total counts)
        self.stats["lookups"] = 0
        self.stats["hits"] = 0
        self.stats["misses"] = 0

        return self.load_cache()


# Global cache instance
_inn_cache_instance: Optional[SanctionedINNCache] = None


def get_inn_cache() -> SanctionedINNCache:
    """Get global INN cache instance."""
    global _inn_cache_instance
    expected_dir = Path(os.getenv("SANCTIONS_DATA_DIR") or PACKAGE_DATA_DIR).resolve()
    if _inn_cache_instance is None or _inn_cache_instance.data_dir != expected_dir:
        _inn_cache_instance = SanctionedINNCache()
        _inn_cache_instance.load_cache()
    return _inn_cache_instance


def lookup_sanctioned_inn(inn: str) -> Optional[Dict[str, Any]]:
    """Fast lookup for sanctioned INN (convenience function)."""
    return get_inn_cache().lookup(inn)


def is_inn_sanctioned(inn: str) -> bool:
    """Check if INN is sanctioned (convenience function)."""
    return get_inn_cache().is_sanctioned(inn)