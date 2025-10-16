"""Hybrid Search Service - combines AC (exact) and Vector (kNN) search modes."""

import asyncio
import json
import math
import time
from contextlib import aclosing
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...core.base_service import BaseService
from ...config import EmbeddingConfig
from ...utils.logging_config import get_logger
from ...contracts.base_contracts import NormalizationResult

from .contracts import (
    Candidate, 
    SearchOpts, 
    SearchService, 
    SearchMode, 
    SearchMetrics
)
from ...contracts.trace_models import SearchTrace, SearchTraceHit, SearchTraceStep
from .config import HybridSearchConfig
from .elasticsearch_adapters import ElasticsearchACAdapter, ElasticsearchVectorAdapter
from .elasticsearch_client import ElasticsearchClientFactory
from .fuzzy_search_service import FuzzySearchService, FuzzyConfig
from .search_integrity import (
    TAX_IDENTIFIER_TYPES, best_per_entity, candidate_identity, source_metadata, source_tax_ids, metadata_matches,
)

class HybridSearchService(BaseService, SearchService):
    """
    Hybrid search service combining AC and Vector search modes.
    
    Implements escalation strategy:
    - First attempt: AC search for exact/almost-exact matches
    - Escalation: If AC results are weak or empty, trigger vector search
    - Dependency failure: propagate an error; no clearance from unrelated local indexes
    """
    
    def __init__(self, config: Optional[HybridSearchConfig] = None):
        """
        Initialize hybrid search service.
        
        Args:
            config: Search configuration, uses default if None
        """
        super().__init__("hybrid_search")
        self.config = HybridSearchConfig.validated_copy(config)

        # Search adapters
        self._ac_adapter: Optional[ElasticsearchACAdapter] = None
        self._vector_adapter: Optional[ElasticsearchVectorAdapter] = None
        self._client_factory: Optional[ElasticsearchClientFactory] = None

        # Metrics tracking
        self._metrics = SearchMetrics()
        self._request_times: List[float] = []
        self._ac_request_times: List[float] = []
        self._vector_request_times: List[float] = []

        # Service state
        self._initialized = False
        self._last_health_check = None
        self._closed = False
        self._close_lock = asyncio.Lock()

        # Embedding service for vector queries (lazy init)
        self._embedding_service = None
        self._owned_embedding_service = None
        self._embedding_config = EmbeddingConfig()

        # Fuzzy search service for typo handling
        fuzzy_config = FuzzyConfig(
            min_score_threshold=0.5,   # Lower threshold for better fuzzy coverage
            high_confidence_threshold=0.80,
            partial_match_threshold=0.70,
            enable_name_fuzzy=True,
            name_boost_factor=1.2
        )
        self._fuzzy_service = FuzzySearchService(fuzzy_config)
        # Embedding cache
        self._embedding_cache: Dict[str, Tuple[List[float], datetime]] = {}
        self._cache_lock = asyncio.Lock()
        
        # Search result cache
        self._search_cache: Dict[str, Tuple[List[Candidate], datetime]] = {}
        self._search_cache_lock = asyncio.Lock()
        
        # Query performance monitoring
        self._query_performance: List[Dict[str, Any]] = []
        self._performance_lock = asyncio.Lock()
        
        # Query caching
        self._query_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self._query_cache_lock = asyncio.Lock()
        
        # Rate limiting
        self._rate_limiter: Dict[str, List[datetime]] = {}
        self._rate_limit_lock = asyncio.Lock()

        # Fusion weights/boosts
        self._fusion_weights, self._fusion_boosts = self._load_fusion_weights()
    
    def _do_initialize(self) -> None:
        """Initialize adapters for the configured source snapshot."""
        self._ensure_open()
        try:
            # Try to initialize Elasticsearch components
            try:
                self._client_factory = ElasticsearchClientFactory(self.config)
                self._ac_adapter = ElasticsearchACAdapter(
                    self.config,
                    client_factory=self._client_factory,
                )
                self._vector_adapter = ElasticsearchVectorAdapter(
                    self.config,
                    client_factory=self._client_factory,
                )
                self.logger.info("[OK] Elasticsearch adapters initialized successfully")
            except Exception as es_e:
                raise RuntimeError("Elasticsearch adapters could not initialize") from es_e

            self._initialized = True
            self.logger.info("[OK] Hybrid search service initialized successfully (with available components)")

        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize hybrid search service: {e}")
            raise

    async def _get_embedding_service(self):
        """Use the same pinned model and preprocessing as ingestion."""
        self._ensure_open()
        if self._embedding_service is None:
            from ..embeddings.embedding_service import EmbeddingService
            self._embedding_service = EmbeddingService(self._embedding_config)
            self._owned_embedding_service = self._embedding_service
        return self._embedding_service

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Sanctions search service is closed")

    def _verify_embedding_contract(self):
        self._ensure_open()
        expected = self._embedding_config.embedding_contract()
        if EmbeddingConfig().embedding_contract() != expected:
            raise RuntimeError("Embedding contract changed; restart and reindex before vector screening")
        if self._embedding_service is not None:
            actual = getattr(self._embedding_service, "config", None)
            if not isinstance(actual, EmbeddingConfig) or actual.embedding_contract() != expected:
                raise RuntimeError("Embedding provider contract differs from the configured index contract")
            if getattr(self._embedding_service, "embedding_contract", expected) != expected:
                raise RuntimeError("Embedding loader contract differs from the configured index contract")

    async def _build_query_vector(self, normalized: NormalizationResult, text: str) -> List[float]:
        service = await self._get_embedding_service()
        if service is None:
            raise RuntimeError("Configured embedding service is unavailable")
        self._verify_embedding_contract()
        query_text = normalized.normalized or text
        cache_key = f"{service.config.model_name}:{service.config.revision}:{service.config.preprocessing_version}:{query_text}"
        cached = await self._get_cached_embedding(cache_key)
        if cached is not None:
            self._verify_embedding_contract()
            return cached
        import inspect
        encode_async = getattr(service, "encode_one_async", None)
        if inspect.iscoroutinefunction(encode_async):
            vector = await encode_async(query_text)
        else:
            vector = await asyncio.to_thread(service.encode_one, query_text)
        self._verify_embedding_contract()
        if self._embedding_service is not service:
            raise RuntimeError("Embedding provider changed during contract verification")
        dimension = self.config.vector_search.vector_dimension
        if len(vector) != dimension or not all(math.isfinite(v) for v in vector):
            raise ValueError("Embedding does not match the configured vector contract")
        if not any(vector):
            raise ValueError("Zero embeddings cannot be used for screening")
        await self._cache_embedding(cache_key, vector)
        return vector

    async def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if available and not expired."""
        if not self.config.enable_embedding_cache:
            return None
            
        async with self._cache_lock:
            if text in self._embedding_cache:
                vector, timestamp = self._embedding_cache[text]
                age_seconds = (datetime.now() - timestamp).total_seconds()
                if age_seconds < self.config.embedding_cache_ttl_seconds:
                    return list(vector)
                else:
                    # Remove expired entry
                    del self._embedding_cache[text]
        return None

    async def _cache_embedding(self, text: str, vector: List[float]) -> None:
        """Cache embedding with TTL."""
        if not self.config.enable_embedding_cache:
            return
            
        async with self._cache_lock:
            self._ensure_open()
            # Remove oldest entries if cache is full
            if len(self._embedding_cache) >= self.config.embedding_cache_size:
                # Remove oldest entry
                oldest_key = min(self._embedding_cache.keys(), 
                               key=lambda k: self._embedding_cache[k][1])
                del self._embedding_cache[oldest_key]
            
            self._embedding_cache[text] = (list(vector), datetime.now())

    def _preprocess_query_for_embedding(self, text: str) -> str:
        """Preprocess query text for better embedding generation."""
        if not self.config.enable_embedding_preprocessing:
            return text
            
        # Basic preprocessing: normalize whitespace, remove extra punctuation
        import re
        processed = re.sub(r'\s+', ' ', text.strip())
        processed = re.sub(r'[^\w\s\-\.]', '', processed)
        return processed

    def _load_fusion_weights(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Load fusion weights and boosts from configuration."""
        default_weights = {"ac": 0.6, "vector": 0.4}
        default_boosts = {
            "shared_hit_bonus": 0.1,
            "metadata_match_bonus": 0.05,
        }

        from ...data.resources import CONFIG_DIR
        path_candidates = [
            Path("config/weights.json"),
            Path("weights.json"),
            CONFIG_DIR / "weights.json",
        ]

        for candidate in path_candidates:
            if candidate.exists():
                try:
                    with candidate.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    fusion = data.get("search_fusion", {})
                    weights = fusion.get("weights", {})
                    boosts = fusion.get("boosts", {})

                    if weights:
                        normalized = {**default_weights}
                        normalized.update(weights)
                        total = sum(normalized.values())
                        if total > 0:
                            normalized = {k: v / total for k, v in normalized.items()}
                        else:
                            normalized = default_weights
                        default_weights = normalized

                    if boosts:
                        merged_boosts = {**default_boosts}
                        merged_boosts.update(boosts)
                        default_boosts = merged_boosts
                except Exception as exc:
                    self.logger.warning(f"Failed to load fusion weights from {candidate}: {exc}")
                finally:
                    break

        return default_weights, default_boosts
    
    async def find_candidates(
        self,
        normalized: NormalizationResult,
        text: str,
        opts: SearchOpts,
        search_trace: Optional[SearchTrace] = None,
    ) -> List[Candidate]:
        """Bound readiness, model waiting and every search stage by one deadline."""
        started = time.perf_counter()
        self._metrics.total_requests += 1
        candidates = None
        try:
            async with asyncio.timeout(opts.timeout_ms / 1000):
                self._ensure_open()
                result = await self._find_candidates_within_deadline(normalized, text, opts, search_trace)
                self._ensure_open()
                candidates = result
                return candidates
        except TimeoutError as exc:
            raise RuntimeError("Configured sanctions search exceeded its deadline") from exc
        finally:
            # Account for readiness failures, cancellation and verified cache hits
            # exactly once, as well as requests that execute a search strategy.
            success = candidates is not None
            result_count = len(candidates) if success else 0
            avg_score = sum(c.score for c in candidates) / result_count if result_count else 0.0
            self._update_metrics(success, (time.perf_counter() - started) * 1000, result_count, avg_score)

    async def _find_candidates_within_deadline(
        self, 
        normalized: NormalizationResult, 
        text: str, 
        opts: SearchOpts,
        search_trace: Optional[SearchTrace] = None
    ) -> List[Candidate]:
        """
        Find search candidates using hybrid search strategy.
        
        Args:
            normalized: Normalized text result from normalization layer
            text: Original input text
            opts: Search options and parameters
            
        Returns:
            List of search candidates sorted by score (descending)
        """
        # Validate search mode (no modification needed)

        if not self._initialized:
            self.initialize()

        # Create dummy trace if none provided
        if search_trace is None:
            search_trace = SearchTrace(enabled=False)
        
        start_time = time.time()
        # Validate and sanitize query
        text = self._validate_query(text)
        
        # Check rate limit
        client_id = getattr(opts, 'client_id', 'default')
        if not await self._check_rate_limit(client_id):
            raise Exception("Rate limit exceeded")
        
        # Structured logging for search operation
        search_log_data = {
            "operation": "find_candidates",
            "query": text,
            "normalized_text": normalized.normalized_text,
            "search_mode": opts.search_mode,
            "top_k": opts.top_k,
            "threshold": opts.threshold,
            "timestamp": datetime.now().isoformat(),
            "language": normalized.language,
            "confidence": normalized.confidence,
            "client_id": client_id
        }
        
        # Check cache first
        require_vectors = opts.search_mode not in {SearchMode.AC, SearchMode.FUZZY}
        dataset_version = await self.readiness(require_vectors=require_vectors)
        cache_key = self._generate_search_cache_key(
            text, opts, normalized=normalized.normalized, dataset_version=dataset_version
        )
        cached_candidates = await self._get_cached_search_result(cache_key)
        
        if cached_candidates is not None:
            await self._verify_dataset_version(dataset_version, require_vectors=require_vectors)
            # Return cached results
            search_log_data.update({
                "status": "cache_hit",
                "processing_time_ms": 0,
                "result_count": len(cached_candidates),
                "avg_score": sum(c.score for c in cached_candidates) / len(cached_candidates) if cached_candidates else 0.0,
                "search_modes_used": ["cache"]
            })
            self.logger.info("Search completed from cache", extra=search_log_data)
            
            # Record performance for cache hit
            await self._record_query_performance(
                query=text,
                search_mode=opts.search_mode,
                processing_time_ms=0,
                result_count=len(cached_candidates),
                cache_hit=True
            )
            
            return cached_candidates
        
        try:
            # Determine search strategy based on options
            if opts.search_mode == SearchMode.AC:
                candidates = await self._ac_search_only(normalized, text, opts, search_trace)
            elif opts.search_mode == SearchMode.VECTOR:
                candidates = await self._vector_search_only(normalized, text, opts, search_trace)
            elif opts.search_mode == SearchMode.FUZZY:
                candidates = await self._fuzzy_search(normalized.normalized or text, opts, search_trace)
            elif opts.search_mode == SearchMode.HYBRID:
                candidates = await self._hybrid_search(normalized, text, opts, search_trace)
            else:
                raise ValueError("Unsupported public search mode")
            await self._verify_dataset_version(dataset_version, require_vectors=require_vectors)
            
            # Process and rank results
            candidates = self._process_results(candidates, opts)
            
            # Filter sensitive data
            candidates = self._filter_sensitive_data(candidates)
            
            # Limit payload size to prevent excessive memory usage
            search_trace.limit_payload_size(max_size_kb=200)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            avg_score = sum(c.score for c in candidates) / len(candidates) if candidates else 0.0
            
            # Cache search results
            await self._cache_search_result(cache_key, candidates)
            
            # Record query performance
            await self._record_query_performance(
                query=text,
                search_mode=opts.search_mode,
                processing_time_ms=processing_time,
                result_count=len(candidates),
                cache_hit=False
            )
            
            # Log successful search
            search_log_data.update({
                "status": "success",
                "processing_time_ms": processing_time,
                "result_count": len(candidates),
                "avg_score": avg_score,
                "search_modes_used": [opts.search_mode]
            })
            self.logger.info("Search completed successfully", extra=search_log_data)
            
            # Log audit event
            self._log_audit_event("search_success", text, len(candidates), client_id)
            
            self.logger.info(
                f"Search completed: {len(candidates)} candidates found in {processing_time:.2f}ms"
            )

            self.logger.debug("%s", " ".join(map(str, [f"[TARGET] find_candidates RESULT: {len(candidates)} candidates, {processing_time:.2f}ms"])))
            return candidates
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            # Log failed search
            search_log_data.update({
                "status": "error",
                "processing_time_ms": processing_time,
                "error": str(e),
                "error_type": type(e).__name__
            })
            self.logger.error("Search failed", extra=search_log_data)

            raise RuntimeError("Configured sanctions search is unavailable") from e
    
    async def _ac_search_only(
        self, 
        normalized: NormalizationResult, 
        text: str, 
        opts: SearchOpts,
        search_trace: Optional[SearchTrace] = None
    ) -> List[Candidate]:
        """Execute AC search only."""
        self._metrics.ac_requests += 1
        start_time = time.time()

        # Create dummy trace if none provided
        if search_trace is None:
            search_trace = SearchTrace(enabled=False)
        
        # Performance logging for AC search
        ac_log_data = {
            "operation": "ac_search",
            "query": text,
            "normalized_text": normalized.normalized_text,
            "top_k": opts.top_k,
            "threshold": opts.threshold,
            "timestamp": datetime.now().isoformat()
        }

        try:
            query_text = normalized.normalized or text
            start_time = time.perf_counter()
            
            candidates = await self._ac_adapter.search(
                query=query_text,
                opts=opts,
                index_name=self.config.elasticsearch.ac_index
            )

            search_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

            if getattr(self._ac_adapter, "_connected", True) is False:
                raise RuntimeError("AC adapter did not complete a connected search")

            # Update AC-specific metrics
            self._update_ac_metrics(search_time, len(candidates))

            # Convert candidates to SearchTraceHit
            ac_hits = []
            for rank, candidate in enumerate(candidates, 1):
                signals = {
                    'dob_match': self._check_dob_match(candidate.metadata, query_text),
                    'doc_id_match': self._check_doc_id_match(candidate.doc_id, query_text),
                    'entity_type': candidate.entity_type,
                    'match_fields': candidate.match_fields,
                    'confidence': candidate.confidence
                }
                
                hit = SearchTraceHit(
                    doc_id=candidate.doc_id,
                    score=candidate.score,
                    rank=rank,
                    source="AC",
                    signals=signals
                )
                ac_hits.append(hit)
            
            # Add AC search step to trace
            search_trace.add_step(SearchTraceStep(
                stage="AC",
                query=query_text,
                topk=opts.top_k,
                took_ms=search_time,
                hits=ac_hits,
                meta={
                    "index_name": self.config.elasticsearch.ac_index,
                    "search_mode": "exact",
                    "fallback_enabled": False,
                    "adapter_connected": getattr(self._ac_adapter, "_connected", True)
                }
            ))
            
            # Log AC search results
            ac_log_data.update({
                "status": "success",
                "processing_time_ms": search_time,
                "result_count": len(candidates),
                "avg_score": sum(c.score for c in candidates) / len(candidates) if candidates else 0.0
            })
            self.logger.info("AC search completed", extra=ac_log_data)

            return candidates

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            ac_log_data.update({
                "status": "error",
                "processing_time_ms": processing_time,
                "error": str(e),
                "error_type": type(e).__name__
            })
            self.logger.error("AC search failed", extra=ac_log_data)
            search_trace.note(f"AC search failed: {str(e)}")
            raise

    async def _vector_search_only(
        self, 
        normalized: NormalizationResult, 
        text: str, 
        opts: SearchOpts,
        search_trace: Optional[SearchTrace] = None
    ) -> List[Candidate]:
        """Execute vector search only."""
        self._metrics.vector_requests += 1
        start_time = time.time()

        # Create dummy trace if none provided
        if search_trace is None:
            search_trace = SearchTrace(enabled=False)

        # Performance logging for vector search
        vector_log_data = {
            "operation": "vector_search",
            "query": text,
            "normalized_text": normalized.normalized_text,
            "top_k": opts.top_k,
            "threshold": opts.threshold,
            "timestamp": datetime.now().isoformat()
        }

        try:
            query_text = normalized.normalized or text
            start_time = time.perf_counter()
            
            query_vector = await self._build_query_vector(normalized, text)
            candidates = await self._vector_adapter.search(
                query=query_vector,
                opts=opts,
                index_name=self.config.elasticsearch.vector_index
            )
            
            search_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

            if getattr(self._vector_adapter, "_connected", True) is False:
                raise RuntimeError("Vector adapter did not complete a connected search")

            # Update Vector-specific metrics
            self._update_vector_metrics(search_time, len(candidates))

            # Convert candidates to SearchTraceHit
            vector_hits = []
            for rank, candidate in enumerate(candidates, 1):
                signals = {
                    'dob_match': self._check_dob_match(candidate.metadata, query_text),
                    'doc_id_match': self._check_doc_id_match(candidate.doc_id, query_text),
                    'entity_type': candidate.entity_type,
                    'match_fields': candidate.match_fields,
                    'confidence': candidate.confidence,
                    'vector_similarity': candidate.score
                }
                
                hit = SearchTraceHit(
                    doc_id=candidate.doc_id,
                    score=candidate.score,
                    rank=rank,
                    source="SEMANTIC",
                    signals=signals
                )
                vector_hits.append(hit)
            
            # Add vector search step to trace
            search_trace.add_step(SearchTraceStep(
                stage="SEMANTIC",
                query=query_text,
                topk=opts.top_k,
                took_ms=search_time,
                hits=vector_hits,
                meta={
                    "index_name": self.config.elasticsearch.vector_index,
                    "search_mode": "vector_similarity",
                    "fallback_enabled": False,
                    "adapter_connected": getattr(self._vector_adapter, "_connected", True),
                    "embedding_model": getattr(self._embedding_service, 'model_name', 'unknown')
                }
            ))
            
            # Log vector search results
            vector_log_data.update({
                "status": "success",
                "processing_time_ms": search_time,
                "result_count": len(candidates),
                "avg_score": sum(c.score for c in candidates) / len(candidates) if candidates else 0.0
            })
            self.logger.info("Vector search completed", extra=vector_log_data)

            return candidates

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            vector_log_data.update({
                "status": "error",
                "processing_time_ms": processing_time,
                "error": str(e),
                "error_type": type(e).__name__
            })
            self.logger.error("Vector search failed", extra=vector_log_data)
            search_trace.note(f"Vector search failed: {str(e)}")
            raise
    
    async def _hybrid_search(self, normalized, text, opts, search_trace=None):
        """Escalate once per strategy and retain every strategy's accepted matches."""
        self._metrics.hybrid_requests += 1
        search_trace = search_trace or SearchTrace(enabled=False)
        started = time.perf_counter()
        query_text = normalized.normalized or text
        ac_candidates = await self._ac_search_only(normalized, text, opts, search_trace)
        fuzzy_candidates = []
        vector_candidates = []
        escalate = self._should_escalate(ac_candidates, opts)
        if escalate:
            self._metrics.escalation_triggered += 1
            fuzzy_candidates = await self._fuzzy_search(query_text, opts, search_trace)
            if not self._fuzzy_results_sufficient(fuzzy_candidates, opts):
                vector_candidates = await self._vector_search_only(normalized, text, opts, search_trace)
        lexical = best_per_entity(ac_candidates + fuzzy_candidates)
        candidates = self._combine_results(lexical, vector_candidates, opts)
        self._add_hybrid_trace_step(
            search_trace, query_text, opts, candidates,
            (time.perf_counter() - started) * 1000,
            {"escalation_triggered": escalate, "fuzzy_search_used": escalate,
             "ac_candidates": len(ac_candidates), "fuzzy_candidates": len(fuzzy_candidates),
             "vector_candidates": len(vector_candidates), "final_candidates": len(candidates)},
        )
        return candidates

    def _should_escalate(self, ac_candidates: List[Candidate], opts: SearchOpts) -> bool:
        """Determine if escalation to vector search is needed."""
        self.logger.info(f"Checking escalation: enable={opts.enable_escalation}, ac_count={len(ac_candidates)}, threshold={opts.escalation_threshold}")

        if not opts.enable_escalation:
            self.logger.info("Escalation disabled in SearchOpts")
            return False

        if not ac_candidates:
            self.logger.info("No AC candidates found - escalating to fuzzy/vector search")
            return True

        # Check if best AC score is below escalation threshold
        best_score = max(candidate.score for candidate in ac_candidates)
        escalate = best_score < opts.escalation_threshold
        self.logger.info(f"AC best score: {best_score:.3f}, threshold: {opts.escalation_threshold:.3f}, escalate: {escalate}")
        return escalate

    def _should_use_vector_fallback(
        self, 
        ac_candidates: List[Candidate], 
        vector_candidates: List[Candidate], 
        opts: SearchOpts
    ) -> bool:
        """Determine if vector fallback should be used."""
        if not getattr(self.config, 'enable_vector_fallback', True):
            return False
        
        # If no AC results at all, use fallback
        if not ac_candidates:
            return True
        
        # If AC results are very weak, use fallback
        best_ac_score = max(candidate.score for candidate in ac_candidates)
        if best_ac_score < 0.3:  # Very low confidence
            return True
        
        # If vector results are significantly better, use fallback
        if vector_candidates:
            best_vector_score = max(candidate.score for candidate in vector_candidates)
            if best_vector_score > best_ac_score * 1.5:  # 50% better
                return True
        
        return False

    async def _vector_fallback_search(self, normalized, text, opts, search_trace=None):
        """Vector fallback shares the main model, index and similarity contract."""
        query_vector = await self._build_query_vector(normalized, text)
        return await self._vector_adapter.search_vector_fallback(
            query_vector=query_vector, query_text=text, opts=opts,
        )

    def _deduplicate_and_rerank(self, candidates, opts):
        return best_per_entity(candidates)[:opts.top_k]

    def _combine_results(self, ac_candidates, vector_candidates, opts):
        """Fuse evidence per source entity; absence of another hit is not a penalty."""
        combined = {candidate_identity(c): replace(c, metadata=dict(c.metadata))
                    for c in best_per_entity(ac_candidates)}
        for candidate in best_per_entity(vector_candidates):
            key = candidate_identity(candidate)
            existing = combined.get(key)
            if existing is None:
                combined[key] = replace(candidate, metadata=dict(candidate.metadata))
                continue
            # AC and fuzzy scores measure the same lexical evidence. They must
            # not receive an agreement bonus or be counted as separate identities.
            if candidate.search_mode != SearchMode.VECTOR:
                combined[key] = best_per_entity([existing, candidate])[0]
                continue
            ac_weight = self._fusion_weights["ac"]
            vector_weight = self._fusion_weights["vector"]
            score = (existing.score * ac_weight + candidate.score * vector_weight) / (ac_weight + vector_weight)
            metadata = {**candidate.metadata, **existing.metadata,
                        "retrieval_scores": {"lexical": existing.score, "vector": candidate.score}}
            combined[key] = replace(
                existing, score=min(1.0, score), metadata=metadata,
                confidence=max(existing.confidence, candidate.confidence),
                search_mode=SearchMode.HYBRID,
                match_fields=sorted(set(existing.match_fields + candidate.match_fields)),
            )
        return best_per_entity(combined.values())[:opts.top_k]

    def _process_results(self, candidates: List[Candidate], opts: SearchOpts) -> List[Candidate]:
        """Process and filter search results."""
        # Apply threshold filtering
        filtered_candidates = [
            c for c in candidates 
            if c.score >= opts.threshold
        ]
        
        # Apply entity type filtering
        if opts.entity_types:
            filtered_candidates = [
                c for c in filtered_candidates
                if c.entity_type in opts.entity_types
            ]
        
        # Apply metadata filtering
        if opts.metadata_filters:
            filtered_candidates = [
                c for c in filtered_candidates
                if self._matches_metadata_filters(c, opts.metadata_filters)
            ]
        
        return best_per_entity(filtered_candidates)[:opts.top_k]

    def _matches_metadata_filters(self, candidate, filters):
        return metadata_matches(candidate.doc_id, candidate.metadata, filters)

    async def clear_embedding_cache(self) -> None:
        """Clear cached query embeddings."""
        async with self._cache_lock:
            self._embedding_cache.clear()
            self.logger.info("Embedding cache cleared")

    async def _get_cached_search_result(self, cache_key: str) -> Optional[List[Candidate]]:
        """Get cached search result if available and not expired."""
        if not self.config.enable_search_cache:
            return None
            
        async with self._search_cache_lock:
            if cache_key in self._search_cache:
                candidates, timestamp = self._search_cache[cache_key]
                
                # Check if cache entry is expired
                age_seconds = (datetime.now() - timestamp).total_seconds()
                if age_seconds < self.config.search_cache_ttl_seconds:
                    self.logger.debug(f"Cache hit for search key: {cache_key[:20]}...")
                    return candidates
                else:
                    # Remove expired entry
                    del self._search_cache[cache_key]
                    self.logger.debug(f"Cache expired for search key: {cache_key[:20]}...")
            
            return None
    
    async def _cache_search_result(self, cache_key: str, candidates: List[Candidate]) -> None:
        """Cache search result."""
        if not self.config.enable_search_cache:
            return
            
        async with self._search_cache_lock:
            self._ensure_open()
            # Check cache size limit
            if len(self._search_cache) >= self.config.search_cache_size:
                # Remove oldest entry
                oldest_key = min(self._search_cache.keys(), 
                               key=lambda k: self._search_cache[k][1])
                del self._search_cache[oldest_key]
                self.logger.debug(f"Cache evicted oldest entry: {oldest_key[:20]}...")
            
            self._search_cache[cache_key] = (candidates, datetime.now())
            self.logger.debug(f"Cached search result for key: {cache_key[:20]}...")
    
    def _generate_search_cache_key(self, query: str, opts: SearchOpts,
                                   *, normalized: str = "", dataset_version=None) -> str:
        """Generate cache key for search query."""
        import hashlib
        
        # Create a hash of the query and options
        key_data = {"query": query, "normalized": normalized,
                    "options": opts.model_dump(mode="json"), "dataset": dataset_version,
                    "config": self.config.model_dump(mode="json")}

        key_string = json.dumps(key_data, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def readiness(self, *, require_vectors: bool = True) -> Dict[str, str]:
        """Verify the selected index generations before returning screening results."""
        self._ensure_open()
        from .index_schema import index_mapping, validate_mapping, SOURCE_COVERAGE_VERSION
        if require_vectors:
            self._verify_embedding_contract()
        if not self._initialized:
            self.initialize()
        client = await self._client_factory.get_client()
        client = client.options(request_timeout=5, max_retries=0)
        indices = [(self.config.elasticsearch.ac_index, False)]
        if require_vectors:
            indices.append((self.config.elasticsearch.vector_index, True))
        generations = {}
        manifests = []
        for index, vectors in indices:
            mappings = await client.indices.get_mapping(index=index)
            if len(mappings) != 1:
                raise RuntimeError(f"Index {index} must resolve to one concrete source snapshot")
            expected_mapping = index_mapping(self.config, vectors=vectors)["mappings"]
            expected = expected_mapping["_meta"]
            for actual_name, mapping in mappings.items():
                validate_mapping(mapping.get("mappings", {}), expected_mapping)
                meta = mapping.get("mappings", {}).get("_meta", {})
                if any(meta.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(f"Index {index} has an incompatible data contract")
                if meta.get("ingestion_status") != "completed" or not meta.get("generation"):
                    raise RuntimeError(f"Index {index} has no completed ingestion")
                if vectors and meta.get("source_coverage_version") != SOURCE_COVERAGE_VERSION:
                    raise RuntimeError(f"Index {index} requires source coverage verification before screening")
                if (await client.count(index=actual_name))["count"] == 0:
                    raise RuntimeError(f"Index {index} is empty")
                generations[actual_name] = meta["generation"]
                manifests.append(meta.get("source_manifest"))
        if require_vectors and (len(set(generations.values())) != 1 or
                                any(manifest != manifests[0] for manifest in manifests[1:])):
            raise RuntimeError("AC and vector indices do not share a coherent snapshot generation and source manifest")
        if require_vectors:
            self._verify_embedding_contract()
        self._ensure_open()
        return generations

    async def find_by_identifier(self, value, identifier_type, opts=None):
        """Exact tax-identifier evidence comes from the same active source as names."""
        if identifier_type.lower() not in TAX_IDENTIFIER_TYPES:
            return []
        opts = opts or SearchOpts(search_mode=SearchMode.AC)
        generation = await self.readiness(require_vectors=False)
        candidates = []
        async with asyncio.timeout(opts.timeout_ms / 1000):
            async with aclosing(self._ac_adapter.iter_documents(opts)) as pages:
                async for hits in pages:
                    for hit in hits:
                        source = hit["_source"]
                        metadata = source_metadata(source)
                        if value not in source_tax_ids(metadata):
                            continue
                        candidates.append(Candidate(
                            doc_id=hit["_id"], score=1.0,
                            text=source.get("name") or source.get("normalized_text") or source["pattern"],
                            entity_type=source["entity_type"], metadata=metadata,
                            search_mode=SearchMode.AC, match_fields=[identifier_type], confidence=1.0,
                            trace={"id_match": True, "reason": "confirmed_source_identifier"},
                        ))
                    candidates = best_per_entity(candidates)[:opts.top_k]
                    await asyncio.sleep(0)
        await self._verify_dataset_version(generation, require_vectors=False)
        return candidates

    async def _verify_dataset_version(self, expected, *, require_vectors):
        actual = await self.readiness(require_vectors=require_vectors)
        if actual != expected:
            raise RuntimeError("Sanctions dataset changed during screening; retry the request")

    async def close(self) -> None:
        """Permanently close this service without closing a borrowed runtime model."""
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._initialized = False
            encoder = self._owned_embedding_service
            self._owned_embedding_service = None
            self._embedding_service = None
            self._search_cache.clear()
            self._embedding_cache.clear()
            self._query_cache.clear()
            self._query_performance.clear()
            self._rate_limiter.clear()
            failures = []
            if encoder is not None:
                try:
                    encoder.close()
                except Exception:
                    failures.append("embedding")
            if self._client_factory is not None:
                try:
                    await self._client_factory.close()
                except Exception:
                    failures.append("client")
            if failures:
                raise RuntimeError("Search service resource cleanup failed")
    
    async def clear_search_cache(self) -> None:
        """Clear the search result cache."""
        async with self._search_cache_lock:
            self._search_cache.clear()
            self.logger.info("Search result cache cleared")
    
    async def invalidate_search_cache(self, pattern: Optional[str] = None) -> int:
        """Invalidate search cache entries matching pattern or all if None."""
        async with self._search_cache_lock:
            if pattern is None:
                # Clear all cache
                count = len(self._search_cache)
                self._search_cache.clear()
                self.logger.info(f"Invalidated all {count} search cache entries")
                return count
            else:
                # Remove entries matching pattern
                keys_to_remove = [key for key in self._search_cache.keys() if pattern in key]
                for key in keys_to_remove:
                    del self._search_cache[key]
                self.logger.info(f"Invalidated {len(keys_to_remove)} search cache entries matching pattern: {pattern}")
                return len(keys_to_remove)
    
    async def cleanup_expired_cache_entries(self) -> int:
        """Clean up expired cache entries."""
        now = datetime.now()
        expired_count = 0
        
        async with self._search_cache_lock:
            keys_to_remove = []
            for key, (_, timestamp) in self._search_cache.items():
                age_seconds = (now - timestamp).total_seconds()
                if age_seconds >= self.config.search_cache_ttl_seconds:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._search_cache[key]
                expired_count += 1
            
            if expired_count > 0:
                self.logger.info(f"Cleaned up {expired_count} expired search cache entries")
        
        return expired_count
    
    async def _record_query_performance(self, query: str, search_mode: str, processing_time_ms: float, result_count: int, cache_hit: bool = False) -> None:
        """Record query performance metrics."""
        async with self._performance_lock:
            self._ensure_open()
            performance_record = {
                "timestamp": datetime.now().isoformat(),
                "query": query[:100],  # Truncate long queries
                "search_mode": search_mode,
                "processing_time_ms": processing_time_ms,
                "result_count": result_count,
                "cache_hit": cache_hit
            }
            
            self._query_performance.append(performance_record)
            
            # Keep only last 1000 records to prevent memory issues
            if len(self._query_performance) > 1000:
                self._query_performance = self._query_performance[-1000:]
    
    async def get_query_performance_stats(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        async with self._performance_lock:
            if not self._query_performance:
                return {
                    "total_queries": 0,
                    "avg_processing_time_ms": 0,
                    "avg_result_count": 0,
                    "cache_hit_rate": 0,
                    "search_mode_distribution": {}
                }
            
            # Calculate statistics
            processing_times = [record["processing_time_ms"] for record in self._query_performance]
            result_counts = [record["result_count"] for record in self._query_performance]
            cache_hits = [record["cache_hit"] for record in self._query_performance]
            search_modes = [record["search_mode"] for record in self._query_performance]
            
            # Calculate averages
            avg_processing_time = sum(processing_times) / len(processing_times)
            avg_result_count = sum(result_counts) / len(result_counts)
            cache_hit_rate = sum(cache_hits) / len(cache_hits) if cache_hits else 0
            
            # Search mode distribution
            mode_distribution = {}
            for mode in search_modes:
                mode_distribution[mode] = mode_distribution.get(mode, 0) + 1
            
            return {
                "total_queries": len(self._query_performance),
                "avg_processing_time_ms": avg_processing_time,
                "avg_result_count": avg_result_count,
                "cache_hit_rate": cache_hit_rate,
                "search_mode_distribution": mode_distribution,
                "min_processing_time_ms": min(processing_times),
                "max_processing_time_ms": max(processing_times),
                "min_result_count": min(result_counts),
                "max_result_count": max(result_counts)
            }
    
    async def clear_query_performance(self) -> None:
        """Clear query performance records."""
        async with self._performance_lock:
            self._query_performance.clear()
            self.logger.info("Query performance records cleared")
    
    async def _get_cached_query(self, query_key: str) -> Optional[Dict[str, Any]]:
        """Get cached query if available and not expired."""
        if not self.config.enable_query_caching:
            return None
            
        async with self._query_cache_lock:
            if query_key in self._query_cache:
                query_data, timestamp = self._query_cache[query_key]
                
                # Check if cache entry is expired
                age_seconds = (datetime.now() - timestamp).total_seconds()
                if age_seconds < self.config.query_cache_ttl_seconds:
                    self.logger.debug(f"Query cache hit for key: {query_key[:20]}...")
                    return query_data
                else:
                    # Remove expired entry
                    del self._query_cache[query_key]
                    self.logger.debug(f"Query cache expired for key: {query_key[:20]}...")
            
            return None
    
    async def _cache_query(self, query_key: str, query_data: Dict[str, Any]) -> None:
        """Cache query data."""
        if not self.config.enable_query_caching:
            return
            
        async with self._query_cache_lock:
            self._ensure_open()
            # Check cache size limit
            if len(self._query_cache) >= self.config.query_cache_size:
                # Remove oldest entry
                oldest_key = min(self._query_cache.keys(), 
                               key=lambda k: self._query_cache[k][1])
                del self._query_cache[oldest_key]
                self.logger.debug(f"Query cache evicted oldest entry: {oldest_key[:20]}...")
            
            self._query_cache[query_key] = (query_data, datetime.now())
            self.logger.debug(f"Cached query data for key: {query_key[:20]}...")
    
    def _generate_query_cache_key(self, query: str, search_mode: str) -> str:
        """Generate cache key for query data."""
        import hashlib
        
        key_data = {
            "query": query,
            "search_mode": search_mode
        }
        
        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def clear_query_cache(self) -> None:
        """Clear the query cache."""
        async with self._query_cache_lock:
            self._query_cache.clear()
            self.logger.info("Query cache cleared")
    
    async def get_query_cache_stats(self) -> Dict[str, Any]:
        """Get query cache statistics."""
        async with self._query_cache_lock:
            if not self._query_cache:
                return {
                    "cache_size": 0,
                    "cache_enabled": self.config.enable_query_caching,
                    "max_cache_size": self.config.query_cache_size,
                    "cache_ttl_seconds": self.config.query_cache_ttl_seconds
                }
            
            now = datetime.now()
            ages = [(now - timestamp).total_seconds() for _, (_, timestamp) in self._query_cache.items()]
            
            return {
                "cache_size": len(self._query_cache),
                "cache_enabled": self.config.enable_query_caching,
                "max_cache_size": self.config.query_cache_size,
                "cache_ttl_seconds": self.config.query_cache_ttl_seconds,
                "avg_age_seconds": sum(ages) / len(ages) if ages else 0,
                "max_age_seconds": max(ages) if ages else 0,
                "min_age_seconds": min(ages) if ages else 0
            }
    
    def _validate_query(self, query: str) -> str:
        """Validate and sanitize search query."""
        if not self.config.enable_query_validation:
            return query
        
        # Remove potentially dangerous characters
        import re
        sanitized = re.sub(r'[<>"\']', '', query)
        
        # Limit query length
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            self.logger.warning(f"Query truncated to {max_length} characters")
        
        # Check for SQL injection patterns
        sql_patterns = [
            r'(union|select|insert|update|delete|drop|create|alter)\s+',
            r'(or|and)\s+\d+\s*=\s*\d+',
            r';\s*(drop|delete|insert|update)',
            r'--\s*',
            r'/\*.*?\*/'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                self.logger.warning(f"Potential SQL injection pattern detected in query: {pattern}")
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    async def _check_rate_limit(self, client_id: str = "default") -> bool:
        """Check if client has exceeded rate limit."""
        if not self.config.enable_rate_limiting:
            return True
        
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        async with self._rate_limit_lock:
            if client_id not in self._rate_limiter:
                self._rate_limiter[client_id] = []
            
            # Remove old requests
            self._rate_limiter[client_id] = [
                req_time for req_time in self._rate_limiter[client_id]
                if req_time > minute_ago
            ]
            
            # Check if limit exceeded
            if len(self._rate_limiter[client_id]) >= self.config.rate_limit_requests_per_minute:
                return False
            
            # Add current request
            self._rate_limiter[client_id].append(now)
            return True
    
    def _filter_sensitive_data(self, candidates: List[Candidate]) -> List[Candidate]:
        """Filter sensitive data from search results."""
        if not self.config.enable_sensitive_data_filtering:
            return candidates
        
        filtered_candidates = []
        for candidate in candidates:
            # Create a copy to avoid modifying original
            filtered_candidate = Candidate(
                doc_id=candidate.doc_id,
                score=candidate.score,
                text=candidate.text,
                entity_type=candidate.entity_type,
                metadata=candidate.metadata.copy() if candidate.metadata else {},
                search_mode=candidate.search_mode,
                match_fields=candidate.match_fields,
                confidence=candidate.confidence
                , trace=candidate.trace
            )
            
            # Remove sensitive fields from metadata
            sensitive_fields = ['ssn', 'passport', 'credit_card', 'bank_account', 'phone', 'email']
            for field in sensitive_fields:
                if field in filtered_candidate.metadata:
                    filtered_candidate.metadata[field] = "***"
            
            filtered_candidates.append(filtered_candidate)
        
        return filtered_candidates
    
    def _log_audit_event(self, event_type: str, query: str, result_count: int, client_id: str = "default") -> None:
        """Log audit event for security monitoring."""
        if not self.config.enable_audit_logging:
            return
        
        audit_data = {
            "event_type": event_type,
            "query": query[:100],  # Truncate for privacy
            "result_count": result_count,
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "service": "hybrid_search"
        }
        
        self.logger.info("Audit event", extra=audit_data)
    
    async def get_search_cache_stats(self) -> Dict[str, Any]:
        """Get search cache statistics."""
        async with self._search_cache_lock:
            if not self._search_cache:
                return {
                    "cache_size": 0,
                    "cache_enabled": self.config.enable_search_cache,
                    "max_cache_size": self.config.search_cache_size,
                    "cache_ttl_seconds": self.config.search_cache_ttl_seconds
                }
            
            now = datetime.now()
            ages = [(now - timestamp).total_seconds() for _, (_, timestamp) in self._search_cache.items()]
            
            return {
                "cache_size": len(self._search_cache),
                "cache_enabled": self.config.enable_search_cache,
                "max_cache_size": self.config.search_cache_size,
                "cache_ttl_seconds": self.config.search_cache_ttl_seconds,
                "avg_age_seconds": sum(ages) / len(ages) if ages else 0,
                "max_age_seconds": max(ages) if ages else 0,
                "min_age_seconds": min(ages) if ages else 0
            }
    
    async def get_embedding_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        async with self._cache_lock:
            cache_size = len(self._embedding_cache)
            max_size = self.config.embedding_cache_size
            ttl_seconds = self.config.embedding_cache_ttl_seconds
            
            # Calculate cache age statistics
            now = datetime.now()
            ages = []
            for _, (_, timestamp) in self._embedding_cache.items():
                age = (now - timestamp).total_seconds()
                ages.append(age)
            
            avg_age = sum(ages) / len(ages) if ages else 0
            max_age = max(ages) if ages else 0
            
            return {
                "cache_size": cache_size,
                "max_size": max_size,
                "utilization": cache_size / max_size if max_size > 0 else 0,
                "ttl_seconds": ttl_seconds,
                "avg_age_seconds": avg_age,
                "max_age_seconds": max_age,
                "cache_enabled": self.config.enable_embedding_cache
            }
    
    def _update_metrics(self, success: bool, processing_time_ms: float, result_count: int, avg_score: float = 0.0) -> None:
        """Update search metrics."""
        if success:
            self._metrics.successful_requests += 1
        else:
            self._metrics.failed_requests += 1
        
        # Update latency tracking
        self._request_times.append(processing_time_ms)
        if len(self._request_times) > self.config.metrics_window_size:
            self._request_times.pop(0)

        # Calculate average latency
        if self._request_times:
            self._metrics.avg_hybrid_latency_ms = sum(self._request_times) / len(self._request_times)

            # Calculate P95 latency
            sorted_times = sorted(self._request_times)
            p95_index = int(len(sorted_times) * 0.95)
            
            # Update metrics service if available
            if hasattr(self, 'metrics_service') and self.metrics_service:
                try:
                    # Record search metrics
                    self.metrics_service.record_metric(
                        name="search_requests_total",
                        value=1,
                        metric_type="counter",
                        tags={"status": "success" if success else "failed"}
                    )
                    
                    self.metrics_service.record_metric(
                        name="search_latency_ms",
                        value=processing_time_ms,
                        metric_type="histogram",
                        tags={"operation": "hybrid_search"}
                    )
                    
                    self.metrics_service.record_metric(
                        name="search_results_count",
                        value=result_count,
                        metric_type="histogram",
                        tags={"operation": "hybrid_search"}
                    )
                    
                    if avg_score > 0:
                        self.metrics_service.record_metric(
                            name="search_avg_score",
                            value=avg_score,
                            metric_type="histogram",
                            tags={"operation": "hybrid_search"}
                        )
                        
                except Exception as e:
                    self.logger.warning(f"Failed to record metrics: {e}")
            self._metrics.p95_latency_ms = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]

        # Update result quality metrics
        self._metrics.avg_results_per_request = (
            (self._metrics.avg_results_per_request * (self._metrics.total_requests - 1) + result_count)
            / self._metrics.total_requests
        )

        # Update average score
        if result_count > 0:
            self._metrics.avg_score = (
                (self._metrics.avg_score * (self._metrics.total_requests - 1) + avg_score)
                / self._metrics.total_requests
            )
        
        # Update hit rate (simplified calculation)
        if result_count > 0:
            self._metrics.hybrid_hit_rate = (
                (self._metrics.hybrid_hit_rate * (self._metrics.total_requests - 1) + 1.0) 
                / self._metrics.total_requests
            )
        else:
            self._metrics.hybrid_hit_rate = (
                (self._metrics.hybrid_hit_rate * (self._metrics.total_requests - 1) + 0.0)
                / self._metrics.total_requests
            )

    def _update_ac_metrics(self, processing_time_ms: float, result_count: int) -> None:
        """Update AC-specific search metrics."""
        # Update AC latency tracking
        self._ac_request_times.append(processing_time_ms)
        if len(self._ac_request_times) > self.config.metrics_window_size:
            self._ac_request_times.pop(0)

        # Calculate average AC latency
        if self._ac_request_times:
            self._metrics.avg_ac_latency_ms = sum(self._ac_request_times) / len(self._ac_request_times)

        # Update AC hit rate
        if result_count > 0:
            self._metrics.ac_hit_rate = (
                (self._metrics.ac_hit_rate * (self._metrics.ac_requests - 1) + 1.0)
                / self._metrics.ac_requests
            )
        else:
            self._metrics.ac_hit_rate = (
                (self._metrics.ac_hit_rate * (self._metrics.ac_requests - 1) + 0.0)
                / self._metrics.ac_requests
            )

    def _update_vector_metrics(self, processing_time_ms: float, result_count: int) -> None:
        """Update Vector-specific search metrics."""
        # Update Vector latency tracking
        self._vector_request_times.append(processing_time_ms)
        if len(self._vector_request_times) > self.config.metrics_window_size:
            self._vector_request_times.pop(0)

        # Calculate average Vector latency
        if self._vector_request_times:
            self._metrics.avg_vector_latency_ms = sum(self._vector_request_times) / len(self._vector_request_times)

        # Update Vector hit rate
        if result_count > 0:
            self._metrics.vector_hit_rate = (
                (self._metrics.vector_hit_rate * (self._metrics.vector_requests - 1) + 1.0)
                / self._metrics.vector_requests
            )
        else:
            self._metrics.vector_hit_rate = (
                (self._metrics.vector_hit_rate * (self._metrics.vector_requests - 1) + 0.0)
                / self._metrics.vector_requests
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check search service health status."""
        if self._closed:
            return {
                "service": "hybrid_search", "status": "unhealthy", "closed": True,
                "initialized": False, "fallback_enabled": False,
                "metrics": self._metrics.to_dict(),
            }
        health_status = {
            "service": "hybrid_search",
            "status": "healthy" if self._initialized else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "initialized": self._initialized,
            "metrics": self._metrics.to_dict(),
            "fallback_enabled": False,
            "fallback_services": {"status": "disabled", "reason": "Active source snapshot required"},
        }
        
        try:
            for name, adapter in (("ac_adapter", self._ac_adapter), ("vector_adapter", self._vector_adapter)):
                if adapter is None:
                    adapter_health = {"status": "unhealthy", "error": "Adapter is not initialized"}
                else:
                    adapter_health = await adapter.health_check()
                health_status[name] = adapter_health
                if (not isinstance(adapter_health, dict)
                        or adapter_health.get("status") != "healthy"
                        or adapter_health.get("connected") is False):
                    health_status["status"] = "unhealthy"

            # Add embedding cache information
            health_status["embedding_cache"] = await self.get_embedding_cache_stats()
            
            # Add search cache information
            health_status["search_cache"] = await self.get_search_cache_stats()
            
            # Add connection pool statistics
            if self._client_factory:
                health_status["connection_pool"] = await self._client_factory.get_connection_stats()
            
            # Add query performance statistics
            health_status["query_performance"] = await self.get_query_performance_stats()
            
            # Add query cache statistics
            health_status["query_cache"] = await self.get_query_cache_stats()

        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            self.logger.error(f"Health check failed: {e}")
        
        self._last_health_check = health_status
        return health_status
    
    def get_metrics(self) -> SearchMetrics:
        """Get current search metrics."""
        return self._metrics

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including adapter-specific stats."""
        base_metrics = self._metrics.to_dict()

        # Add adapter-specific latency stats
        comprehensive = {
            **base_metrics,
            "adapter_stats": {},
            "request_counts": {
                "total": self._metrics.total_requests,
                "ac_only": self._metrics.ac_requests,
                "vector_only": self._metrics.vector_requests,
                "hybrid": self._metrics.hybrid_requests,
                "escalations": self._metrics.escalation_triggered
            }
        }

        # Get AC adapter stats
        if self._ac_adapter and hasattr(self._ac_adapter, 'get_latency_stats'):
            try:
                comprehensive["adapter_stats"]["ac"] = self._ac_adapter.get_latency_stats()
            except Exception as e:
                self.logger.warning(f"Failed to get AC adapter stats: {e}")

        # Get Vector adapter stats
        if self._vector_adapter and hasattr(self._vector_adapter, 'get_latency_stats'):
            try:
                comprehensive["adapter_stats"]["vector"] = self._vector_adapter.get_latency_stats()
            except Exception as e:
                self.logger.warning(f"Failed to get Vector adapter stats: {e}")

        return comprehensive
    
    def reset_metrics(self) -> None:
        """Reset search metrics."""
        self._metrics = SearchMetrics()
        self._request_times.clear()
        self._ac_request_times.clear()
        self._vector_request_times.clear()
        self.logger.info("Search metrics reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed service status."""
        return {
            "service_name": self.service_name,
            "initialized": self._initialized,
            "config": self.config.to_dict(),
            "metrics": self._metrics.to_dict(),
            "last_health_check": self._last_health_check,
            "fallback_services": {
                "watchlist": False,
                "vector": False,
            }
        }
    
    def _add_hybrid_trace_step(
        self, 
        search_trace: SearchTrace, 
        query: str, 
        opts: SearchOpts, 
        candidates: List[Candidate], 
        took_ms: float, 
        meta: Dict[str, Any]
    ) -> None:
        """Add hybrid search step to trace."""
        # Convert candidates to SearchTraceHit
        hybrid_hits = []
        for rank, candidate in enumerate(candidates, 1):
            signals = {
                'dob_match': self._check_dob_match(candidate.metadata, query),
                'doc_id_match': self._check_doc_id_match(candidate.doc_id, query),
                'entity_type': candidate.entity_type,
                'match_fields': candidate.match_fields,
                'confidence': candidate.confidence,
                'search_mode': candidate.search_mode.value
            }
            
            hit = SearchTraceHit(
                doc_id=candidate.doc_id,
                score=candidate.score,
                rank=rank,
                source="HYBRID",
                signals=signals
            )
            hybrid_hits.append(hit)
        
        # Add hybrid step to trace
        search_trace.add_step(SearchTraceStep(
            stage="HYBRID",
            query=query,
            topk=opts.top_k,
            took_ms=took_ms,
            hits=hybrid_hits,
            meta=meta
        ))
    
    def _check_dob_match(self, metadata: Dict[str, Any], query: str) -> bool:
        """Check if date of birth matches query."""
        if not metadata or not query:
            return False
        
        dob = metadata.get('dob')
        if not dob:
            return False
        
        query_lower = query.lower()
        dob_lower = str(dob).lower()
        
        # Check for year match
        if len(str(dob)) >= 4 and len(query) >= 4:
            dob_year = str(dob)[-4:] if str(dob)[-4:].isdigit() else ""
            query_year = "".join([c for c in query if c.isdigit()])
            if dob_year and query_year and dob_year in query_year:
                return True
        
        return dob_lower in query_lower or query_lower in dob_lower
    
    def _check_doc_id_match(self, doc_id: str, query: str) -> bool:
        """Check if document ID matches query."""
        if not doc_id or not query:
            return False
        
        query_lower = query.lower()
        doc_id_lower = doc_id.lower()
        
        return doc_id_lower in query_lower or query_lower in doc_id_lower
    
    async def update_configuration(self, new_config: HybridSearchConfig) -> None:
        """Replace pending settings before initialization; live changes require restart.

        Adapters, clients, caches and in-flight requests must share one configuration.
        Replacing just the config object cannot provide that guarantee at runtime.
        """
        self._ensure_open()
        if self._initialized or self._client_factory is not None or self._ac_adapter is not None or self._vector_adapter is not None:
            raise RuntimeError("Configuration changes require recreating the search service")
        self.config = self._validate_configuration(new_config)

    def _validate_configuration(self, config: HybridSearchConfig) -> HybridSearchConfig:
        """Validate and copy the complete canonical configuration without mutation."""
        return HybridSearchConfig.validated_copy(config)

    # ==========================================
    # Fuzzy Search Methods
    # ==========================================

    async def _fuzzy_search(self, query_text, opts, search_trace=None):
        """Screen the active index, with a deadline and no unrelated local dataset."""
        async with asyncio.timeout(opts.timeout_ms / 1000):
            return await self._in_memory_fuzzy_search(query_text, opts, search_trace)

    async def _elasticsearch_fuzzy_search(self, query_text, opts):
        """Compatibility entry point using the canonical AC fields and parser."""
        if self._ac_adapter is None:
            raise RuntimeError("AC search is unavailable")
        candidates = await self._ac_adapter.search(query_text, opts)
        return [replace(candidate, search_mode=SearchMode.FUZZY) for candidate in candidates]

    async def _in_memory_fuzzy_search(self, query_text, opts, search_trace=None):
        """Score bounded pages of a consistent snapshot of the active AC index."""
        if self._ac_adapter is None or not self._fuzzy_service.enabled:
            raise RuntimeError("Fuzzy screening is unavailable")
        started = time.perf_counter()
        candidates = []
        scanned = 0
        pages = self._ac_adapter.iter_documents(
            opts, batch_size=self._fuzzy_service.config.max_candidates
        )
        async with aclosing(pages):
            async for hits in pages:
                names = {}
                for hit in hits:
                    source = hit["_source"]
                    aliases = source.get("aliases") or []
                    if isinstance(aliases, str):
                        aliases = [aliases]
                    for name in [source.get("pattern"), source.get("normalized_text"),
                                 source.get("name"), *aliases]:
                        if isinstance(name, str) and name.strip():
                            names.setdefault(name, {})[hit["_id"]] = hit
                # Every matched alias is considered before selecting top entities.
                matches = await self._fuzzy_service.search_async(
                    query_text, list(names), max_results=len(names)
                )
                for match in matches:
                    if match.score < opts.threshold:
                        continue
                    for hit in names[match.matched_text].values():
                        source = hit["_source"]
                        metadata = source_metadata(source)
                        metadata["fuzzy_algorithm"] = match.algorithm
                        candidates.append(Candidate(
                            doc_id=hit["_id"], score=min(1.0, match.score),
                            text=source.get("name") or match.matched_text,
                            entity_type=source["entity_type"], metadata=metadata,
                            search_mode=SearchMode.FUZZY, match_fields=["fuzzy_name"],
                            confidence=min(1.0, match.score),
                            trace={"reason": "fuzzy_match", "matched_alias": match.matched_text},
                        ))
                candidates = best_per_entity(candidates)[:opts.top_k]
                scanned += len(hits)
                await asyncio.sleep(0)
        if search_trace and search_trace.enabled:
            search_trace.add_step(SearchTraceStep(
                stage="LEXICAL", query=query_text, topk=opts.top_k,
                took_ms=(time.perf_counter() - started) * 1000,
                hits=[SearchTraceHit(doc_id=c.doc_id, score=c.score, rank=i + 1, source="LEXICAL")
                      for i, c in enumerate(candidates)],
                meta={"index": self.config.elasticsearch.ac_index, "scanned_documents": scanned},
            ))
        return candidates

    def _fuzzy_results_sufficient(self, fuzzy_candidates: List[Candidate], opts: SearchOpts) -> bool:
        """
        Determine if fuzzy search results are good enough to skip vector search.

        Args:
            fuzzy_candidates: Results from fuzzy search
            opts: Search options

        Returns:
            True if fuzzy results are sufficient, False otherwise
        """
        if not fuzzy_candidates:
            return False

        # Check if we have enough high-confidence results
        high_confidence_results = [
            c for c in fuzzy_candidates
            if c.score >= self._fuzzy_service.config.high_confidence_threshold
        ]

        if len(high_confidence_results) >= 1:  # At least one high-confidence match
            self.logger.debug(f"Fuzzy search found {len(high_confidence_results)} high-confidence matches")
            return True

        # Check if best score is above minimum threshold
        best_score = max(c.score for c in fuzzy_candidates)
        if best_score >= self._fuzzy_service.config.min_score_threshold * 1.1:  # 10% above minimum
            self.logger.debug(f"Fuzzy search best score {best_score:.3f} is sufficient")
            return True

        return False
