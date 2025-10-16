"""Thread-safe in-memory TTL/LRU cache used by the processing pipeline."""

from __future__ import annotations

import hashlib
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional


class CacheService:
    """A small, dependency-free TTL cache with least-recently-used eviction."""

    def __init__(self, max_size: int = 1024, default_ttl: int = 3600):
        if max_size < 1:
            raise ValueError("max_size must be greater than zero")
        if default_ttl < 1:
            raise ValueError("default_ttl must be greater than zero")

        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Return a cached value and refresh its LRU position."""
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                self._misses += 1
                return None

            value, expires_at = item
            if time.time() >= expires_at:
                del self._cache[key]
                self._misses += 1
                return None

            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value, evicting the least recently used item if needed."""
        effective_ttl = self._default_ttl if ttl is None else ttl
        if effective_ttl < 1:
            raise ValueError("ttl must be greater than zero")

        with self._lock:
            if key in self._cache:
                del self._cache[key]
            self._cache[key] = (value, time.time() + effective_ttl)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1
        return True

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._cache.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def size(self) -> int:
        self.cleanup_expired()
        with self._lock:
            return len(self._cache)

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        value = self.get(key)
        if value is not None:
            return value
        value = factory()
        self.set(key, value, ttl=ttl)
        return value

    def touch(self, key: str, ttl: Optional[int] = None) -> bool:
        effective_ttl = self._default_ttl if ttl is None else ttl
        if effective_ttl < 1:
            raise ValueError("ttl must be greater than zero")
        with self._lock:
            item = self._cache.get(key)
            if item is None or time.time() >= item[1]:
                self._cache.pop(key, None)
                return False
            self._cache[key] = (item[0], time.time() + effective_ttl)
            self._cache.move_to_end(key)
            return True

    def cleanup_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [key for key, (_, deadline) in self._cache.items() if now >= deadline]
            for key in expired:
                del self._cache[key]
            return len(expired)

    def set_max_size(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("max_size must be greater than zero")
        with self._lock:
            self._max_size = max_size
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

    def get_keys(self) -> list[str]:
        self.cleanup_expired()
        with self._lock:
            return list(self._cache.keys())

    def get_stats(self) -> dict[str, Any]:
        self.cleanup_expired()
        with self._lock:
            total_requests = self._hits + self._misses
            memory_bytes = sys.getsizeof(self._cache) + sum(
                sys.getsizeof(key) + sys.getsizeof(value)
                for key, (value, _) in self._cache.items()
            )
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total_requests,
                "hit_rate": self._hits / total_requests if total_requests else 0.0,
                "current_size": len(self._cache),
                "max_size": self._max_size,
                "evictions": self._evictions,
                "memory_usage_mb": memory_bytes / (1024 * 1024),
            }

    @staticmethod
    def _generate_key(*args: Any, **kwargs: Any) -> str:
        material = repr((args, sorted(kwargs.items()))).encode("utf-8")
        return hashlib.md5(material).hexdigest()
