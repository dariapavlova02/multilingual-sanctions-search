"""
Elasticsearch adapters for AC and Vector search modes.

Provides adapters for:
- AC (exact/almost-exact) search using Elasticsearch text search
- Vector (kNN) search using Elasticsearch dense vector search
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import wraps

from elasticsearch import (
    AsyncElasticsearch, ApiError as ElasticsearchException,
    ConnectionError, ConnectionTimeout as TimeoutError,
)

ELASTICSEARCH_AVAILABLE = True

from ...utils.logging_config import get_logger

from .contracts import Candidate, SearchOpts, ElasticsearchAdapter, SearchMode
from .config import HybridSearchConfig
from .elasticsearch_client import ElasticsearchClientFactory
from .elasticsearch_index_manager import ElasticsearchIndexManager
from .search_integrity import require_complete_response, source_metadata


def retry_elasticsearch(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator for Elasticsearch operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (ConnectionError, TimeoutError) as exc:
                    last_exception = exc
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise exc
                except ElasticsearchException as exc:
                    # Don't retry on Elasticsearch-specific errors (like index not found)
                    raise exc
            
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


class ElasticsearchACAdapter(ElasticsearchAdapter):
    """Elasticsearch adapter for AC (exact/almost-exact) search."""

    AC_PATTERN_FIELD = "pattern.keyword"
    AC_PATTERNS_INDEX = "ac_patterns"

    def __init__(
        self,
        config: HybridSearchConfig,
        client_factory: Optional[ElasticsearchClientFactory] = None,
    ) -> None:
        self.config = HybridSearchConfig.validated_copy(config)
        self.logger = get_logger(__name__)
        self._client_factory = client_factory or ElasticsearchClientFactory(self.config)
        if isinstance(self._client_factory, ElasticsearchClientFactory):
            self._client_factory.validate_connection_configuration(self.config)
        self._client: Optional[AsyncElasticsearch] = None
        self._index_manager: Optional[ElasticsearchIndexManager] = None
        self._latency_stats = {
            "total_requests": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "request_times": [],
        }
        self._connected = False
        self._last_connection_check: Optional[datetime] = None
        self._connection_lock = asyncio.Lock()
        self._ac_patterns_initialized = False

    async def _ensure_connection(self) -> AsyncElasticsearch:
        """Ensure Elasticsearch connection is established and cached."""
        if (
            self._client
            and self._connected
            and self._last_connection_check
            and (datetime.now() - self._last_connection_check).seconds < 300
        ):
            return self._client

        async with self._connection_lock:
            if (
                self._client
                and self._connected
                and self._last_connection_check
                and (datetime.now() - self._last_connection_check).seconds < 300
            ):
                return self._client
            try:
                self._client = await self._client_factory.get_client()
                self._index_manager = ElasticsearchIndexManager(self.config, self._client)
                self._connected = True
                self._last_connection_check = datetime.now()
                self.logger.debug("Elasticsearch AC adapter connected")
                
                # Initialize AC patterns index if not already done
                if not self._ac_patterns_initialized:
                    await self._ensure_ac_patterns_index()
                
                return self._client
            except Exception as exc:
                self._connected = False
                self.logger.error(f"Failed to connect to Elasticsearch: {exc}")
                raise

    async def _ensure_ac_patterns_index(self) -> None:
        """Ensure AC patterns index exists with proper mapping."""
        if self._ac_patterns_initialized:
            return
            
        try:
            client = await self._ensure_connection()
            
            if self._index_manager:
                success = await self._index_manager.create_ac_patterns_index()
                if success:
                    self._ac_patterns_initialized = True
                else:
                    raise RuntimeError("Failed to create AC patterns index")
            else:
                raise RuntimeError("Index manager not initialized")
            
        except Exception as exc:
            self.logger.error(f"Failed to create AC patterns index: {exc}")
            raise

    def _build_should_queries(self, query: str, opts: SearchOpts) -> List[Dict[str, Any]]:
        weights = self.config.ac_search.field_weights
        fuzziness = opts.ac_fuzziness if opts.ac_fuzziness >= 0 else 0
        weighted_fields = [
            f"{field}^{weights.get(field, 1.0)}"
            for field in weights
        ]

        should_queries: List[Dict[str, Any]] = [
            {
                "term": {
                    "normalized_text.keyword": {
                        "value": query.lower(),
                        "boost": opts.ac_boost * weights.get("normalized_text", 1.0) * 2,
                    }
                }
            },
            {
                "term": {
                    self.AC_PATTERN_FIELD: {
                        "value": query,
                        "boost": opts.ac_boost * 3.0,
                    }
                }
            },
            {
                "multi_match": {
                    "query": query,
                    "fields": weighted_fields,
                    "type": "best_fields",
                    "fuzziness": fuzziness,
                    "boost": opts.ac_boost,
                    "tie_breaker": self.config.ac_search.tie_breaker,
                }
            },
        ]

        # Add AC patterns search if enabled
        if getattr(self.config, 'enable_ac_es', False):
            ac_pattern_queries = self._build_ac_pattern_queries(query, opts)
            should_queries.extend(ac_pattern_queries)

        if self.config.ac_search.enable_phrase_queries:
            should_queries.append(
                {
                    "match_phrase": {
                        "normalized_text": {
                            "query": query,
                            "boost": opts.ac_boost * 1.5,
                        }
                    }
                }
            )

        if self.config.ac_search.enable_wildcard_queries:
            should_queries.append(
                {
                    "wildcard": {
                        "normalized_text": {
                            "value": f"{query}*",
                            "boost": 0.5 * opts.ac_boost,
                        }
                    }
                }
            )

        return should_queries

    def _build_ac_pattern_queries(self, query: str, opts: SearchOpts) -> List[Dict[str, Any]]:
        """Build optimized AC pattern queries for T0/T1 patterns."""
        ac_queries = []
        
        # Exact pattern match (T0/T1) - highest priority
        ac_queries.append({
            "term": {
                "pattern.keyword": {
                    "value": query,
                    "boost": opts.ac_boost * 10.0 * self.config.ac_query_boost_factor,
                }
            }
        })
        
        # Optimized edge ngram match with better scoring
        ac_queries.append({
            "match": {
                "pattern": {
                    "query": query,
                    "boost": opts.ac_boost * 3.0 * self.config.ac_query_boost_factor,
                    "fuzziness": "0",  # Disable fuzziness for better performance
                    "operator": "and"  # All terms must match
                }
            }
        })
        
        # Optimized phrase match for multi-word patterns
        ac_queries.append({
            "match_phrase": {
                "pattern": {
                    "query": query,
                    "boost": opts.ac_boost * 3.0,
                }
            }
        })
        
        return ac_queries

    def _build_filters(self, opts: SearchOpts) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = []

        if opts.entity_types:
            filters.append({"terms": {"entity_type.keyword": opts.entity_types}})

        metadata_filters = opts.metadata_filters or {}
        for key, value in metadata_filters.items():
            if value is None:
                continue

            if key in {"id", "doc_id"}:
                values = value if isinstance(value, list) else [value]
                filters.append({"ids": {"values": values}})
                continue

            field_name = key
            if key == "entity_id":
                field_name = "entity_id"
            elif key in {"country", "country_code"}:
                field_name = "country.keyword"
            elif key in {"dob", "date_of_birth"}:
                field_name = "dob"
            else:
                field_name = f"metadata.{key}.keyword"

            if isinstance(value, list):
                filters.append({"terms": {field_name: value}})
            else:
                filters.append({"term": {field_name: value}})

        return filters

    def _build_ac_query(self, query: str, opts: SearchOpts) -> Dict[str, Any]:
        should_queries = self._build_should_queries(query, opts)

        # Защита от пустых should запросов
        if not should_queries:
            should_queries = [{"match_none": {}}]

        bool_query: Dict[str, Any] = {
            "should": should_queries,
            "minimum_should_match": 1,
        }

        filters = self._build_filters(opts)
        if filters:
            bool_query["filter"] = filters

        search_body: Dict[str, Any] = {
            "query": {"bool": bool_query},
            "size": opts.top_k,
            "min_score": opts.ac_min_score,
            "timeout": f"{opts.timeout_ms}ms",
        }

        if opts.enable_highlighting:
            search_body["highlight"] = {
                "fields": {field: {} for field in self.config.ac_search.field_weights}
            }

        return search_body

    def _parse_candidates(self, response: Dict[str, Any], query: Optional[str] = None) -> List[Candidate]:
        require_complete_response(response)
        hits_section = response.get("hits", {})
        hits = hits_section.get("hits", [])
        max_score = hits_section.get("max_score") or 0.0
        candidates: List[Candidate] = []

        for hit in hits:
            source = hit.get("_source", {})
            metadata = source_metadata(source)
            score = float(hit.get("_score") or 0.0)
            confidence = 0.0
            if query is not None:
                from rapidfuzz.fuzz import ratio
                name = source.get("normalized_text") or source.get("name") or ""
                confidence = ratio(query.casefold(), name.casefold()) / 100.0
            elif max_score:
                confidence = min(1.0, score / max_score)
            metadata = dict(metadata)
            metadata["raw_es_score"] = score

            match_fields: List[str] = []
            if "matched_queries" in hit and isinstance(hit["matched_queries"], list):
                match_fields.extend(hit["matched_queries"])
            if not match_fields and "highlight" in hit:
                match_fields.extend(list(hit["highlight"].keys()))

            candidates.append(
                Candidate(
                    doc_id=hit.get("_id", ""),
                    score=confidence if query is not None else score,
                    text=source.get("normalized_text")
                    or source.get("legal_names")
                    or source.get("name")
                    or "",
                    entity_type=source.get("entity_type", metadata.get("entity_type", "unknown")),
                    metadata=metadata,
                    search_mode=SearchMode.AC,
                    match_fields=match_fields or ["normalized_text"],
                    confidence=confidence,
                )
            )

        return candidates

    @retry_elasticsearch(max_retries=3, delay=0.5, backoff=2.0)
    async def search_ac_patterns(
        self,
        query: str,
        opts: SearchOpts,
    ) -> List[Dict[str, Any]]:
        """Search AC patterns index for T0/T1 matches."""
        if not getattr(self.config, 'enable_ac_es', False):
            return []
            
        client = await self._ensure_connection()
        
        try:
            # Search AC patterns index
            ac_pattern_queries = self._build_ac_pattern_queries(query, opts)

            # Защита от пустых should запросов
            if not ac_pattern_queries:
                return []

            pattern_query = {
                "query": {
                    "bool": {
                        "should": ac_pattern_queries,
                        "minimum_should_match": 1
                    }
                },
                "size": opts.top_k,
                "sort": [
                    {"tier": {"order": "asc"}},  # T0 before T1
                    {"_score": {"order": "desc"}}
                ]
            }
            
            response = await client.search(index=self.config.elasticsearch.ac_index, body=pattern_query)
            require_complete_response(response)
            payload = response
            
            hits = payload.get("hits", {}).get("hits", [])
            ac_patterns = []
            
            for hit in hits:
                source = hit.get("_source", {})
                ac_patterns.append({
                    "pattern": source.get("pattern", ""),
                    "tier": source.get("tier", 0),
                    "meta": source.get("meta", {}),
                    "score": hit.get("_score", 0.0)
                })
            
            return ac_patterns
            
        except Exception as exc:
            self.logger.error("AC pattern retrieval failed")
            raise

    async def iter_documents(self, opts: SearchOpts, *, batch_size: int = 1000):
        """Read the configured active index without a result-window coverage limit.

        A PIT and _shard_doc/search_after keep pages consistent during refreshes.
        The caller checks the ingestion generation again before publishing results.
        """
        if not 1 <= batch_size <= 5000:
            raise ValueError("Invalid fuzzy scan batch size")
        client = await self._ensure_connection()
        keep_alive = f"{opts.timeout_ms + 5000}ms"
        opened = await client.open_point_in_time(
            index=self.config.elasticsearch.ac_index,
            keep_alive=keep_alive,
            allow_partial_search_results=False,
        )
        pit_id = opened["id"]
        try:
            if opened.get("_shards", {}).get("failed", 0):
                raise RuntimeError("Cannot open a complete sanctions snapshot")
            search_after = None
            while True:
                body = {
                    "pit": {"id": pit_id, "keep_alive": keep_alive},
                    "query": {"bool": {"filter": self._build_filters(opts)}},
                    "size": batch_size,
                    "sort": ["_shard_doc"],
                    "track_total_hits": False,
                    "timeout": f"{opts.timeout_ms}ms",
                    "_source": ["pattern", "normalized_text", "name", "aliases",
                                "entity_id", "entity_type", "source_list", "category",
                                "country", "dob", "metadata"],
                }
                if search_after is not None:
                    body["search_after"] = search_after
                response = await client.search(body=body, allow_partial_search_results=False)
                pit_id = response.get("pit_id", pit_id)
                require_complete_response(response)
                hits = response["hits"]["hits"]
                if not hits:
                    break
                next_page = hits[-1].get("sort")
                if not next_page or next_page == search_after:
                    raise RuntimeError("Invalid sanctions snapshot pagination")
                search_after = next_page
                yield hits
        finally:
            await client.options(request_timeout=2, max_retries=0).close_point_in_time(id=pit_id)

    @retry_elasticsearch(max_retries=3, delay=0.5, backoff=2.0)
    async def search(
        self,
        query: Any,
        opts: SearchOpts,
        index_name: Optional[str] = None,
    ) -> List[Candidate]:
        if not isinstance(query, str):
            raise TypeError("AC adapter expects query as string")

        client = await self._ensure_connection()
        start_time = time.time()

        try:
            search_body = self._build_ac_query(query, opts)
            response = await client.search(index=index_name or self.config.elasticsearch.ac_index, body=search_body)
            candidates = self._parse_candidates(response, query=query)
            
        except ElasticsearchException as exc:
            self.logger.error(f"Elasticsearch error during AC search: {exc}")
            self._connected = False
            raise
        except (ConnectionError, TimeoutError) as exc:
            self.logger.error(f"Connection error during AC search: {exc}")
            self._connected = False
            raise
        except Exception as exc:
            self.logger.error(f"Unexpected error during AC search: {exc}")
            self._connected = False
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency_stats(latency_ms)

        return candidates

    def _update_latency_stats(self, latency_ms: float) -> None:
        self._latency_stats["total_requests"] += 1
        self._latency_stats["request_times"].append(latency_ms)

        if len(self._latency_stats["request_times"]) > 1000:
            self._latency_stats["request_times"].pop(0)

        if self._latency_stats["request_times"]:
            times = self._latency_stats["request_times"]
            self._latency_stats["avg_latency_ms"] = sum(times) / len(times)
            sorted_times = sorted(times)
            p95_index = int(len(sorted_times) * 0.95)
            if p95_index >= len(sorted_times):
                p95_index = len(sorted_times) - 1
            self._latency_stats["p95_latency_ms"] = sorted_times[p95_index]

    async def health_check(self) -> Dict[str, Any]:
        try:
            health = await self._client_factory.health_check()
            health["connected"] = health.get("status") == "healthy"
            
            # Add index health information
            if self._index_manager:
                index_health = await self._index_manager.get_index_health()
                health["indices"] = index_health["indices"]
            
            health.update(
                {
                    "adapter": "elasticsearch_ac",
                    "latency_stats": self.get_latency_stats(),
                }
            )
            return health
        except Exception as exc:
            return {
                "adapter": "elasticsearch_ac",
                "status": "unhealthy",
                "connected": False,
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            }
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get adapter-specific latency statistics."""
        return {
            "total_requests": self._latency_stats["total_requests"],
            "avg_latency_ms": self._latency_stats["avg_latency_ms"],
            "p95_latency_ms": self._latency_stats["p95_latency_ms"]
        }


class ElasticsearchVectorAdapter(ElasticsearchAdapter):
    """Elasticsearch adapter for Vector (kNN) search."""

    # VECTOR_INDEX = "vector_index"  # Removed hardcoded constant - use config.elasticsearch.vector_index

    def __init__(
        self,
        config: HybridSearchConfig,
        client_factory: Optional[ElasticsearchClientFactory] = None,
    ) -> None:
        self.config = HybridSearchConfig.validated_copy(config)
        self.logger = get_logger(__name__)
        self._client_factory = client_factory or ElasticsearchClientFactory(self.config)
        if isinstance(self._client_factory, ElasticsearchClientFactory):
            self._client_factory.validate_connection_configuration(self.config)
        self._client: Optional[AsyncElasticsearch] = None
        self._index_manager: Optional[ElasticsearchIndexManager] = None
        self._latency_stats = {
            "total_requests": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "request_times": [],
        }
        self._connected = False
        self._last_connection_check: Optional[datetime] = None
        self._connection_lock = asyncio.Lock()
        self._vector_index_initialized = False

    async def _ensure_connection(self) -> AsyncElasticsearch:
        if (
            self._client
            and self._connected
            and self._last_connection_check
            and (datetime.now() - self._last_connection_check).seconds < 300
        ):
            return self._client

        async with self._connection_lock:
            if (
                self._client
                and self._connected
                and self._last_connection_check
                and (datetime.now() - self._last_connection_check).seconds < 300
            ):
                return self._client
            try:
                self._client = await self._client_factory.get_client()
                self._index_manager = ElasticsearchIndexManager(self.config, self._client)
                self._connected = True
                self._last_connection_check = datetime.now()
                self.logger.debug("Elasticsearch Vector adapter connected")
                
                # Initialize vector index if not already done
                if not self._vector_index_initialized:
                    await self._ensure_vector_index()
                
                return self._client
            except Exception as exc:
                self._connected = False
                self.logger.error(f"Failed to connect to Elasticsearch: {exc}")
                raise

    async def _ensure_vector_index(self) -> None:
        """Ensure vector index exists with proper mapping for dense vectors and BM25."""
        if self._vector_index_initialized:
            return
            
        try:
            client = await self._ensure_connection()
            
            if self._index_manager:
                success = await self._index_manager.create_vector_index()
                if success:
                    self._vector_index_initialized = True
                else:
                    raise RuntimeError("Failed to create vector index")
            else:
                raise RuntimeError("Index manager not initialized")
            
        except Exception as exc:
            self.logger.error(f"Failed to create vector index: {exc}")
            raise

    def _validate_query_vector(self, query_vector: Any) -> List[float]:
        if not isinstance(query_vector, (list, tuple)):
            raise TypeError("Vector adapter expects query_vector as a list/tuple of floats")

        vector = [float(v) for v in query_vector]
        expected_dimension = self.config.vector_search.vector_dimension
        if expected_dimension and len(vector) != expected_dimension:
            raise ValueError(
                f"Query vector dimension {len(vector)} does not match configured "
                f"dimension {expected_dimension}"
            )
        return vector

    async def search_vector_fallback(self, query_vector, query_text, opts):
        """Compatibility entry point using the same validated cosine search contract."""
        if not self.config.enable_vector_fallback:
            return []
        return await self.search(query_vector, opts)

    def _build_filters(self, opts: SearchOpts) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = []

        if opts.entity_types:
            filters.append({"terms": {"entity_type.keyword": opts.entity_types}})

        metadata_filters = opts.metadata_filters or {}
        for key, value in metadata_filters.items():
            if value is None:
                continue

            if key in {"id", "doc_id"}:
                values = value if isinstance(value, list) else [value]
                filters.append({"ids": {"values": values}})
                continue

            field_name = key
            if key == "entity_id":
                field_name = "entity_id"
            elif key in {"country", "country_code"}:
                field_name = "country.keyword"
            elif key in {"dob", "date_of_birth"}:
                field_name = "dob"
            else:
                field_name = f"metadata.{key}.keyword"

            if isinstance(value, list):
                filters.append({"terms": {field_name: value}})
            else:
                filters.append({"term": {field_name: value}})

        return filters

    def _build_vector_query(self, query_vector: List[float], opts: SearchOpts) -> Dict[str, Any]:
        max_candidates = min(
            self.config.vector_search.max_candidates,
            max(opts.top_k * 5, opts.max_escalation_results),
        )

        request: Dict[str, Any] = {
            "knn": {
                "field": self.config.vector_search.vector_field,
                "query_vector": query_vector,
                "k": opts.top_k,
                "num_candidates": max_candidates,
                # Note: ef_search не поддерживается в Elasticsearch 8+ KNN запросах
                # Используется num_candidates для контроля точности
            },
            "size": opts.top_k,
            "min_score": opts.vector_min_score,
            "timeout": f"{opts.timeout_ms}ms",
        }

        filters = self._build_filters(opts)
        if filters:
            request["knn"]["filter"] = filters

        return request

    def _parse_candidates(self, response: Dict[str, Any]) -> List[Candidate]:
        require_complete_response(response)
        hits_section = response.get("hits", {})
        hits = hits_section.get("hits", [])
        max_score = hits_section.get("max_score") or 0.0
        candidates: List[Candidate] = []

        for hit in hits:
            source = hit.get("_source", {})
            metadata = source_metadata(source)
            score = float(hit.get("_score") or 0.0)
            confidence = 0.0
            confidence = min(1.0, max(0.0, 2.0 * score - 1.0))
            metadata = dict(metadata)
            metadata["raw_es_score"] = score

            candidates.append(
                Candidate(
                    doc_id=hit.get("_id", ""),
                    score=confidence,
                    text=source.get("normalized_text")
                    or source.get("legal_names")
                    or source.get("name")
                    or "",
                    entity_type=source.get("entity_type", metadata.get("entity_type", "unknown")),
                    metadata=metadata,
                    search_mode=SearchMode.VECTOR,
                    match_fields=[self.config.vector_search.vector_field],
                    confidence=confidence,
                )
            )

        return candidates

    @retry_elasticsearch(max_retries=3, delay=0.5, backoff=2.0)
    async def search(
        self,
        query: Any,
        opts: SearchOpts,
        index_name: Optional[str] = None,
    ) -> List[Candidate]:
        query_vector = self._validate_query_vector(query)
        client = await self._ensure_connection()
        start_time = time.time()

        try:
            body = self._build_vector_query(query_vector, opts)
            response = await client.search(index=index_name or self.config.elasticsearch.vector_index, body=body)
            candidates = self._parse_candidates(response)
        except ElasticsearchException as exc:
            self.logger.error(f"Elasticsearch error during vector search: {exc}")
            self._connected = False
            raise
        except (ConnectionError, TimeoutError) as exc:
            self.logger.error(f"Connection error during vector search: {exc}")
            self._connected = False
            raise
        except Exception as exc:
            self.logger.error(f"Unexpected error during vector search: {exc}")
            self._connected = False
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency_stats(latency_ms)

        return candidates

    def _update_latency_stats(self, latency_ms: float) -> None:
        self._latency_stats["total_requests"] += 1
        self._latency_stats["request_times"].append(latency_ms)

        if len(self._latency_stats["request_times"]) > 1000:
            self._latency_stats["request_times"].pop(0)

        if self._latency_stats["request_times"]:
            times = self._latency_stats["request_times"]
            self._latency_stats["avg_latency_ms"] = sum(times) / len(times)
            sorted_times = sorted(times)
            p95_index = int(len(sorted_times) * 0.95)
            if p95_index >= len(sorted_times):
                p95_index = len(sorted_times) - 1
            self._latency_stats["p95_latency_ms"] = sorted_times[p95_index]

    async def health_check(self) -> Dict[str, Any]:
        try:
            health = await self._client_factory.health_check()
            health["connected"] = health.get("status") == "healthy"
            
            # Add index health information
            if self._index_manager:
                index_health = await self._index_manager.get_index_health()
                health["indices"] = index_health["indices"]
            
            health.update(
                {
                    "adapter": "elasticsearch_vector",
                    "latency_stats": self.get_latency_stats(),
                }
            )
            return health
        except Exception as exc:
            return {
                "adapter": "elasticsearch_vector",
                "status": "unhealthy",
                "connected": False,
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            }

    def get_latency_stats(self) -> Dict[str, float]:
        return {
            "total_requests": self._latency_stats["total_requests"],
            "avg_latency_ms": self._latency_stats["avg_latency_ms"],
            "p95_latency_ms": self._latency_stats["p95_latency_ms"],
        }
