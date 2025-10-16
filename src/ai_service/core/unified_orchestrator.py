"""
Unified Orchestrator Service - Implementation of the layered architecture.

This is the single, authoritative orchestrator that implements the layer specification
from CLAUDE.md. It replaces all other orchestrator implementations.

Layers implemented:
1. Validation & Sanitization
2. Smart Filter (optional skip)
3. Language Detection
4. Unicode Normalization
5. Name Normalization (morph)
6. Signals (enrichment)
7. Variants (optional)
8. Embeddings (optional)
9. Search (optional)
10. Decision & Response
"""

import asyncio
import logging
import time
import unicodedata
import uuid
from copy import deepcopy
from ..data.patterns.identifiers import get_compiled_patterns_cached, normalize_identifier
from ..layers.search.search_integrity import TAX_IDENTIFIER_TYPES, metadata_matches, source_identity, source_tax_ids
from ..utils.source_text_view import without_format_controls
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from ..config import SERVICE_CONFIG
from ..utils.feature_flags import FeatureFlags, get_feature_flag_manager, validated_feature_flags, merge_feature_flags
from ..contracts.base_contracts import (
    EmbeddingsServiceInterface,
    LanguageDetectionInterface,
    NormalizationResult,
    NormalizationServiceInterface,
    ProcessingContext,
    ProcessingStage,
    SignalsResult,
    SignalsServiceInterface,
    SmartFilterInterface,
    UnicodeServiceInterface,
    UnifiedProcessingResult,
    ValidationServiceInterface,
    VariantsServiceInterface,
)
# Import HybridSearchService conditionally to avoid elasticsearch dependency issues
try:
    from ..layers.search.hybrid_search_service import HybridSearchService
    from ..layers.search.contracts import SearchOpts
except ImportError as e:
    logger.warning(f"Failed to import HybridSearchService: {e}")
    HybridSearchService = None
    SearchOpts = None
from ..contracts.decision_contracts import (
    DecisionInput,
    DecisionOutput,
    RiskLevel,
    SmartFilterInfo,
    SignalsInfo,
    SimilarityInfo,
)
from ..contracts.trace_models import SearchTrace, SearchTraceBuilder
from ..core.decision_engine import DecisionEngine
from ..config.settings import DecisionConfig
from ..layers.normalization.homoglyph_detector import HomoglyphDetector
from ..exceptions import InternalServerError, ServiceInitializationError
from ..utils import get_logger
from ..monitoring.metrics_service import MetricsService, MetricType, AlertSeverity

logger = get_logger(__name__)


class UnifiedOrchestrator:
    """
    Unified orchestrator implementing the 9-layer processing model.

    This is the SINGLE orchestrator - all other orchestrator implementations
    should be deprecated in favor of this one.
    """

    def __init__(
        self,
        # Required services
        validation_service: ValidationServiceInterface,
        language_service: LanguageDetectionInterface,
        unicode_service: UnicodeServiceInterface,
        normalization_service: NormalizationServiceInterface,
        signals_service: SignalsServiceInterface,
        # Optional services
        smart_filter_service: Optional[SmartFilterInterface] = None,
        variants_service: Optional[VariantsServiceInterface] = None,
        embeddings_service: Optional[EmbeddingsServiceInterface] = None,
        decision_engine: Optional[DecisionEngine] = None,
        metrics_service: Optional[MetricsService] = None,
        search_service: Optional[Any] = None,  # HybridSearchService temporarily disabled
        default_feature_flags: Optional[FeatureFlags] = None,
        # Configuration - defaults from SERVICE_CONFIG
        enable_smart_filter: Optional[bool] = None,
        enable_variants: Optional[bool] = None,
        enable_embeddings: Optional[bool] = None,
        enable_decision_engine: Optional[bool] = None,
        enable_search: Optional[bool] = False,
        allow_smart_filter_skip: Optional[bool] = None,
    ):
        # Validate required services are not None
        if validation_service is None:
            raise ServiceInitializationError("validation_service cannot be None")
        if language_service is None:
            raise ServiceInitializationError("language_service cannot be None")
        if unicode_service is None:
            raise ServiceInitializationError("unicode_service cannot be None")
        if normalization_service is None:
            raise ServiceInitializationError("normalization_service cannot be None")
        if signals_service is None:
            raise ServiceInitializationError("signals_service cannot be None")

        self.validation_service = validation_service
        self.smart_filter_service = smart_filter_service
        self.language_service = language_service
        self.unicode_service = unicode_service
        self.normalization_service = normalization_service
        self.signals_service = signals_service
        self.variants_service = variants_service
        self.embeddings_service = embeddings_service
        self.decision_engine = decision_engine
        self.metrics_service = metrics_service
        self.search_service = search_service

        # Auto-initialize search only when this orchestrator instance enables it.
        # Using the global setting here made explicitly minimal/test instances
        # connect to Elasticsearch during construction.
        requested_search = (
            enable_search if enable_search is not None else SERVICE_CONFIG.enable_search
        )
        if self.search_service is None and requested_search:
            try:
                from ai_service.layers.search.hybrid_search_service import HybridSearchService
                from ai_service.layers.search.config import HybridSearchConfig

                search_config = HybridSearchConfig.from_env()
                self.search_service = HybridSearchService(config=search_config)

                # Initialize the service (this may throw if Elasticsearch unavailable)
                self.search_service.initialize()

                logger.info("[OK] Auto-initialized HybridSearchService (search enabled, no service provided)")
            except Exception as e:
                raise ServiceInitializationError("Configured search service could not initialize") from e

        self.default_feature_flags = (
            validated_feature_flags(default_feature_flags)
            if default_feature_flags is not None else get_feature_flag_manager().get_flags()
        )

        # Initialize homoglyph detector for search query normalization
        self.homoglyph_detector = HomoglyphDetector()

        # Legacy compatibility attributes for old tests
        self.cache_service = getattr(self, "cache_service", None)
        self._cache_namespace = uuid.uuid4().hex
        self.embedding_service = getattr(self, "embedding_service", None) or embeddings_service
        self.signal_service = getattr(self, "signal_service", None) or signals_service
        self.pattern_service = getattr(self, "pattern_service", None)
        self.template_builder = getattr(self, "template_builder", None)
        
        # Legacy processing stats for old tests
        self.processing_stats = getattr(self, "processing_stats", None) or {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache": 0,  # Legacy compatibility
            "errors": 0,
            "processing_times": []
        }

        # Configuration flags - use SERVICE_CONFIG defaults if not provided
        self.enable_smart_filter = (
            (enable_smart_filter if enable_smart_filter is not None else SERVICE_CONFIG.enable_smart_filter)
            and smart_filter_service is not None
        )
        self.enable_variants = (
            (enable_variants if enable_variants is not None else SERVICE_CONFIG.enable_variants)
            and variants_service is not None
        )
        self.enable_embeddings = (
            (enable_embeddings if enable_embeddings is not None else SERVICE_CONFIG.enable_embeddings)
            and embeddings_service is not None
        )
        self.enable_decision_engine = (
            (enable_decision_engine if enable_decision_engine is not None else SERVICE_CONFIG.enable_decision_engine)
            and decision_engine is not None
        )
        self.enable_search = (
            (enable_search if enable_search is not None else SERVICE_CONFIG.enable_search)
            and self.search_service is not None  # Use self.search_service (after auto-init)
        )
        self.allow_smart_filter_skip = (
            allow_smart_filter_skip if allow_smart_filter_skip is not None else SERVICE_CONFIG.allow_smart_filter_skip
        )

        # Log search service type for debugging
        search_service_type = "None"
        if self.search_service:
            search_service_type = type(self.search_service).__name__

        logger.info(
            f"UnifiedOrchestrator initialized with stages: "
            f"validation=True, smart_filter={self.enable_smart_filter}, "
            f"language=True, unicode=True, normalization=True, signals=True, "
            f"variants={self.enable_variants}, embeddings={self.enable_embeddings}, "
            f"search={self.enable_search}, search_service={search_service_type}"
        )

    async def _maybe_await(self, x):
        """Helper to await if needed"""
        import inspect
        return await x if inspect.isawaitable(x) else x

    def _safe_len(self, x):
        """Safe length calculation that handles mocks and other objects"""
        try:
            return len(x)
        except Exception:
            return 0

    def _coerce_lang(self, res) -> Dict[str, Any]:
        """Coerce language detection result to dict with language/confidence keys"""
        if res is None:
            return {"language": "en", "confidence": 0.0}  # Fallback to English
        
        if isinstance(res, dict):
            return res
        
        if hasattr(res, 'language') and hasattr(res, 'confidence'):
            return {"language": res.language, "confidence": res.confidence}
        
        # If it's a string, assume it's the language code
        if isinstance(res, str):
            return {"language": res, "confidence": 0.5}
        
        # If it's a tuple, assume (language, confidence)
        if isinstance(res, tuple) and len(res) >= 2:
            return {"language": res[0], "confidence": res[1]}
        
        # Default fallback
        return {"language": "en", "confidence": 0.0}

    async def _handle_validation_layer(
        self, text: str, context: ProcessingContext, start_time: float
    ) -> Optional[UnifiedProcessingResult]:
        """
        Handle Layer 1: Validation & Sanitization

        Returns:
            UnifiedProcessingResult if early termination needed, None otherwise
        """
        logger.debug("Stage 1: Validation & Sanitization")
        layer_start = time.time()

        validation_result = await self._maybe_await(self.validation_service.validate_and_sanitize(text))
        context.sanitized_text = validation_result.get("sanitized_text", text)

        # Debug trace for lengths
        logger.debug(f"Validation: input_len={self._safe_len(text)}, sanitized_len={self._safe_len(context.sanitized_text)}")

        if self.metrics_service:
            self.metrics_service.record_timer('processing.layer.validation', time.time() - layer_start)

        if not validation_result.get("should_process", True):
            if self.metrics_service:
                self.metrics_service.record_counter('processing.validation.failed', 1)
            return self._create_early_response(context, "Input validation failed", start_time)

        return None  # Continue processing

    async def _handle_smart_filter_layer(
        self, context: ProcessingContext, start_time: float
    ) -> Optional[UnifiedProcessingResult]:
        """
        Handle Layer 2: Smart Filter (optional skip)

        Returns:
            UnifiedProcessingResult if early termination needed, None otherwise
        """
        if self.enable_smart_filter:
            logger.debug("Stage 2: Smart Filter")
            layer_start = time.time()

            filter_result = await self._maybe_await(self.smart_filter_service.should_process(
                context.sanitized_text
            ))
            context.should_process = filter_result.should_process
            context.metadata["smart_filter"] = {
                "should_process": filter_result.should_process,
                "confidence": filter_result.confidence,
                "classification": filter_result.classification,
                "detected_signals": filter_result.detected_signals,
                "details": filter_result.details,
            }

            if self.metrics_service:
                self.metrics_service.record_timer('processing.layer.smart_filter', time.time() - layer_start)
                self.metrics_service.record_histogram('smart_filter.confidence', filter_result.confidence)

            if not context.should_process and self.allow_smart_filter_skip:
                if self.metrics_service:
                    self.metrics_service.record_counter('processing.smart_filter.skipped', 1)
                return self._create_filtered_response(
                    context, filter_result, start_time
                )

        return None  # Continue processing

    async def _handle_language_detection_layer(
        self, context: ProcessingContext, language_hint: Optional[str]
    ) -> None:
        """
        Handle Layer 3: Language Detection

        Updates the context with detected language and confidence.
        """
        logger.debug("Stage 3: Language Detection")
        layer_start = time.time()

        from ..config import LANGUAGE_CONFIG
        lang_raw = await self._maybe_await(self.language_service.detect_language_config_driven(
            context.sanitized_text,  # Use original text to preserve Ukrainian/Russian markers
            LANGUAGE_CONFIG
        ))

        # Coerce language result to dict format
        lang = self._coerce_lang(lang_raw)
        context.language = language_hint or lang["language"]
        context.language_confidence = lang["confidence"]

        # Debug trace for language detection
        try:
            confidence_val = float(context.language_confidence)
            logger.debug(f"Language: detected='{context.language}', confidence={confidence_val:.3f}")
        except (ValueError, TypeError):
            logger.debug(f"Language: detected='{context.language}', confidence={context.language_confidence}")

        if self.metrics_service:
            self.metrics_service.record_timer('processing.layer.language_detection', time.time() - layer_start)
            self.metrics_service.record_histogram('language_detection.confidence', context.language_confidence)
            self.metrics_service.record_counter(f'language_detection.detected.{context.language}', 1)

    async def _handle_unicode_normalization_layer(
        self, context: ProcessingContext
    ) -> str:
        """
        Handle Layer 4: Unicode Normalization

        Returns:
            Unicode normalized text
        """
        logger.debug("Stage 4: Unicode Normalization")
        layer_start = time.time()

        # Unicode normalization after language detection
        text_in = context.sanitized_text
        unicode_result = await self._maybe_await(self.unicode_service.normalize_unicode(text_in))

        # Handle both legacy string return and new dict return
        if isinstance(unicode_result, str):
            text_u = unicode_result
        else:
            text_u = unicode_result.get("normalized", text_in)

        # Debug trace for lengths
        logger.debug(f"Unicode: input_len={self._safe_len(text_in)}, normalized_len={self._safe_len(text_u)}")

        if self.metrics_service:
            self.metrics_service.record_timer('processing.layer.unicode_normalization', time.time() - layer_start)

        return text_u

    async def _handle_name_normalization_layer(
        self,
        text_u: str,
        context: ProcessingContext,
        remove_stop_words: bool,
        preserve_names: bool,
        enable_advanced_features: bool,
        feature_flags: FeatureFlags,
        errors: list
    ) -> Any:
        """
        Handle Layer 5: Name Normalization (morph) - THE CORE

        Args:
            text_u: Unicode normalized text
            context: Processing context
            remove_stop_words: Clean STOP_ALL tokens in normalization
            preserve_names: Keep `. - '` for initials/compound names
            enable_advanced_features: Use morphology + diminutives + gender
            errors: List to append errors to

        Returns:
            Normalization result from the service
        """
        logger.debug("Stage 5: Name Normalization")
        layer_start = time.time()

        # Initialize metrics tracking for this layer
        metrics = None
        try:
            from ..monitoring.prometheus_exporter import get_exporter
            metrics = get_exporter()
        except Exception as e:
            logger.debug(f"Metrics not available in normalization layer: {e}")
            metrics = None

        # Use unicode-normalized text for normalization
        norm_start = time.time()
        norm_result = await self._maybe_await(self.normalization_service.normalize_async(
            text_u,  # Use unicode-normalized text
            language=context.language,
            remove_stop_words=remove_stop_words,
            preserve_names=preserve_names,
            enable_advanced_features=enable_advanced_features,
            feature_flags=feature_flags,
        ))

        # Record normalization metrics
        if metrics:
            norm_duration = (time.time() - norm_start) * 1000  # Convert to ms
            metrics.record_pipeline_stage_duration("normalization", norm_duration)
        
        # Add flag reasons to trace if debug_trace is enabled
        if hasattr(norm_result, 'debug_trace') and norm_result.debug_trace:
            from ..utils.flag_propagation import create_flag_context
            flag_context = create_flag_context(feature_flags, "normalization", True)
            
            # Note: Flag reasons are logged separately, not added to trace
            # as trace should only contain TokenTrace objects

        # Note: Feature flags are logged separately, not added to trace
        # as trace should only contain TokenTrace objects

        if self.metrics_service:
            self.metrics_service.record_timer('processing.layer.normalization', time.time() - layer_start)
            if hasattr(norm_result, 'confidence') and norm_result.confidence is not None:
                self.metrics_service.record_histogram('normalization.confidence', norm_result.confidence)
            self.metrics_service.record_histogram('normalization.token_count', self._safe_len(norm_result.tokens))

        if not norm_result.success:
            if self.metrics_service:
                self.metrics_service.record_counter('processing.normalization.failed', 1)
            errors.extend(norm_result.errors)

        return norm_result

    async def _handle_signals_layer(
        self, text_u: str, norm_result: Any, context: ProcessingContext
    ) -> Any:
        """
        Handle Layer 6: Signals (enrichment)

        Args:
            text_u: Unicode normalized text
            norm_result: Result from name normalization
            context: Processing context

        Returns:
            Signals extraction result
        """
        logger.debug("Stage 6: Signals Extraction")
        layer_start = time.time()

        # Initialize metrics tracking for this layer
        metrics = None
        try:
            from ..monitoring.prometheus_exporter import get_exporter
            metrics = get_exporter()
        except Exception as e:
            logger.debug(f"Metrics not available in signals layer: {e}")
            metrics = None

        signals_result = await self._maybe_await(self.signals_service.extract_signals(
            text=context.original_text, normalization_result=norm_result, language=context.language
        ))

        # Debug logging
        logger.info(f"Signals result: {signals_result}")
        logger.info(f"Signals organizations: {signals_result.organizations}")
        logger.info(f"Signals persons: {signals_result.persons}")

        if self.metrics_service:
            self.metrics_service.record_timer('processing.layer.signals', time.time() - layer_start)

        # Record signals processing metrics
        if metrics:
            signals_duration = (time.time() - layer_start) * 1000  # Convert to ms
            metrics.record_pipeline_stage_duration("signals", signals_duration)

        return signals_result

    async def _handle_variants_layer(
        self, norm_result: Any, context: ProcessingContext, generate_variants: Optional[bool], errors: list
    ) -> Optional[list]:
        """
        Handle Layer 7: Variants (optional)

        Args:
            norm_result: Result from name normalization
            context: Processing context
            generate_variants: Override for variants generation
            errors: List to append errors to

        Returns:
            Generated variants or None if disabled/failed
        """
        variants = None
        if (generate_variants is True) or (
            generate_variants is None and self.enable_variants
        ):
            logger.debug("Stage 7: Variant Generation")
            layer_start = time.time()
            try:
                if self.variants_service is not None:
                    import inspect
                    generate_async = getattr(self.variants_service, "generate_variants_async", None)
                    if inspect.iscoroutinefunction(generate_async):
                        generated = await generate_async(norm_result.normalized, context.language)
                    else:
                        generated = await self._maybe_await(self.variants_service.generate_variants(
                            norm_result.normalized, context.language))
                    variants = generated.get("variants") if isinstance(generated, dict) else generated
                    if not isinstance(variants, list) or any(not isinstance(value, str) for value in variants):
                        raise ValueError("Variant generation returned an invalid list")
                else:
                    raise RuntimeError("Requested variant generation is unavailable")
                if self.metrics_service:
                    self.metrics_service.record_timer('processing.layer.variants', time.time() - layer_start)
                    if variants:
                        self.metrics_service.record_histogram('variants.count', self._safe_len(variants))
            except Exception as e:
                logger.warning(f"Variant generation failed: {e}")
                if self.metrics_service:
                    self.metrics_service.record_counter('processing.variants.failed', 1)
                variants = None
                errors.append(f"Variant generation failed ({type(e).__name__})")

        return variants

    async def _handle_embeddings_layer(
        self, norm_result: Any, generate_embeddings: Optional[bool], errors: list
    ) -> Optional[list]:
        """
        Handle Layer 8: Embeddings (optional)

        Args:
            norm_result: Result from name normalization
            generate_embeddings: Override for embeddings generation
            errors: List to append errors to

        Returns:
            Generated embeddings or None if disabled/failed
        """
        embeddings = None
        if (generate_embeddings is True) or (
            generate_embeddings is None and self.enable_embeddings
        ):
            logger.debug("Stage 8: Embedding Generation")
            layer_start = time.time()
            try:
                if self.embeddings_service is not None:
                    import inspect
                    generate = self.embeddings_service.generate_embeddings
                    generate_async = getattr(self.embeddings_service, "generate_embeddings_async", None)
                    if inspect.iscoroutinefunction(generate_async):
                        embeddings = await generate_async(norm_result.normalized)
                    elif inspect.iscoroutinefunction(generate):
                        embeddings = await generate(norm_result.normalized)
                    else:
                        embeddings = await asyncio.to_thread(generate, norm_result.normalized)
                    if not embeddings:
                        raise RuntimeError("Embedding generation returned no vector")
                else:
                    raise RuntimeError("Requested embedding generation is unavailable")
                if self.metrics_service:
                    self.metrics_service.record_timer('processing.layer.embeddings', time.time() - layer_start)
                    if embeddings is not None:
                        self.metrics_service.record_histogram('embeddings.dimension', self._safe_len(embeddings))
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")
                if self.metrics_service:
                    self.metrics_service.record_counter('processing.embeddings.failed', 1)
                embeddings = None
                errors.append(f"Embedding generation failed ({type(e).__name__})")

        return embeddings

    async def _handle_search_layer(
        self,
        norm_result: Any,
        embeddings: Optional[list],
        errors: list,
        original_text: str,
        search_trace: Optional[SearchTrace] = None,
        signals_result: Optional[Any] = None,
        search_options: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Handle Layer 9: Search (optional)

        Args:
            norm_result: Result from name normalization
            embeddings: Generated embeddings (optional)
            errors: List to accumulate errors
            search_trace: Search trace for debugging

        Returns:
            Search results or None if disabled/failed
        """
        search_results = None

        # Check if search should be forced due to ID match (critical for sanctions screening)
        force_search_for_id_match = False
        if signals_result and hasattr(signals_result, 'persons'):
            for person in signals_result.persons:
                if hasattr(person, 'ids') and person.ids:
                    force_search_for_id_match = True
                    logger.info(f"🚨 CRITICAL: ID match detected - forcing search regardless of risk score: {person.ids}")
                    break

        if (self.enable_search or force_search_for_id_match) and self.search_service:
            layer_start = time.time()
            try:
                if self.metrics_service:
                    self.metrics_service.record_counter('processing.search.started', 1)

                # Perform search using normalized text, but fallback to original for organizations
                query = norm_result.normalized if norm_result.normalized else ""

                # FIXED: If normalized is empty (organizations), use original text or org parts
                if not query.strip():
                    # For organizations, try to build query from signals
                    if signals_result and hasattr(signals_result, 'organizations') and signals_result.organizations:
                        # Use the most confident organization
                        best_org = max(signals_result.organizations, key=lambda o: getattr(o, 'confidence', 0))
                        if hasattr(best_org, 'core') and best_org.core:
                            query = best_org.core
                            if hasattr(best_org, 'legal_form') and best_org.legal_form:
                                query = f"{best_org.core} {best_org.legal_form}"
                            logger.debug("%s", " ".join(map(str, [f"🏢 ORG SEARCH: Using organization parts as query: '{query}'"])))

                    # Fallback to original text if no org parts
                    if not query.strip():
                        query = original_text
                        logger.debug("%s", " ".join(map(str, [f"[CHECK] FALLBACK SEARCH: Using original text as query: '{query}'"])))

                # ENHANCED: Check for homoglyphs and generate permutations for better detection
                search_queries = [query]  # Default to original query
                is_homoglyph_case = False

                if self.homoglyph_detector and query.strip():
                    original_query = query
                    # Detect homoglyphs first
                    homoglyph_result = self.homoglyph_detector.detect_homoglyphs(query)
                    if homoglyph_result and homoglyph_result.get('has_homoglyphs', False):
                        # Normalize homoglyphs for search
                        normalized_query, transformations = self.homoglyph_detector.normalize_homoglyphs(query)
                        if normalized_query != original_query:
                            query = normalized_query
                            is_homoglyph_case = True
                            logger.warning(f"🔧 HOMOGLYPH NORMALIZATION FOR SEARCH: '{original_query}' → '{query}' (transformations: {len(transformations)})")
                            logger.debug("%s", " ".join(map(str, [f"🔧 HOMOGLYPH SEARCH: '{original_query}' → '{query}' - normalized for search"])))

                            # Generate permutations for improved detection
                            from ..utils.name_permutations import generate_homoglyph_permutations
                            search_queries = generate_homoglyph_permutations(original_query, normalized_query)
                            logger.debug("%s", " ".join(map(str, [f"[PROGRESS] HOMOGLYPH PERMUTATIONS: Trying {len(search_queries)} variants: {search_queries}"])))
                        else:
                            logger.debug("Homoglyphs detected but no normalization needed")
                    else:
                        logger.debug("No homoglyphs detected in search query")
                logger.debug("%s", " ".join(map(str, [f"[CHECK] SEARCH DEBUG: query='{query}', search_service={self.search_service is not None}, SearchOpts={SearchOpts is not None}"])))

                if query.strip() and SearchOpts:
                    logger.debug("%s", " ".join(map(str, [f"[INIT] CALLING SEARCH: query='{query.strip()}'"])))
                    search_opts = search_options or SearchOpts(
                        top_k=10,
                        threshold=0.7,
                        enable_escalation=True,
                        escalation_threshold=0.6,
                    )
                    logger.debug("%s", " ".join(map(str, [f"🔧 SEARCH OPTS: escalation={search_opts.enable_escalation}, threshold={search_opts.escalation_threshold}"])))

                    search_start_time = time.time()
                    candidates = []

                    require_vectors = search_opts.search_mode.value not in {"ac", "fuzzy"}
                    dataset_version = await self.search_service.readiness(require_vectors=require_vectors)

                    # Check for sanctioned IDs first (critical security check)
                    id_candidates = await self._search_by_extracted_ids(signals_result, search_opts)
                    if id_candidates:
                        candidates.extend(id_candidates)
                        logger.debug("%s", " ".join(map(str, [f"🚨 SANCTIONED ID DETECTED: {len(id_candidates)} matches found"])))

                    # Identifier-only inputs have no semantic name to embed. Require
                    # complete source-span coverage before omitting name retrieval.
                    identifier_only = not norm_result.normalized.strip() and self._has_only_identifier_evidence(
                        original_text, signals_result
                    )
                    if identifier_only:
                        search_processing_time = (time.time() - search_start_time) * 1000
                        if search_trace:
                            search_trace.note("Identifier-only input: exact identifier screening completed")
                    elif self.search_service:
                        try:
                            if is_homoglyph_case and len(search_queries) > 1:
                                # Try all permutations for homoglyph cases
                                logger.debug("%s", " ".join(map(str, [f"[PROGRESS] HOMOGLYPH MULTI-SEARCH: Trying {len(search_queries)} permutations"])))
                                all_results = []
                                best_candidates = []
                                best_score = 0.0

                                for i, search_query in enumerate(search_queries):
                                    # Create modified normalization result for this query
                                    modified_norm_result = self._create_modified_norm_result(norm_result, search_query)

                                    try:
                                        perm_candidates = await self.search_service.find_candidates(
                                            normalized=modified_norm_result,
                                            text=original_text,
                                            opts=search_opts,
                                            search_trace=search_trace,
                                        )

                                        if perm_candidates:
                                            all_results.extend(perm_candidates)
                                            # Get best score from this permutation
                                            max_score = max((getattr(c, 'score', 0.0) or getattr(c, 'final_score', 0.0))
                                                          for c in perm_candidates)
                                            logger.debug("%s", " ".join(map(str, [f"   Permutation {i+1}: '{search_query}' → {len(perm_candidates)} results, best score: {max_score:.3f}"])))

                                            # Keep best results
                                            if max_score > best_score:
                                                best_score = max_score
                                                best_candidates = perm_candidates
                                                logger.debug("%s", " ".join(map(str, [f"   🏆 NEW BEST: '{search_query}' with score {max_score:.3f}"])))
                                        else:
                                            logger.debug("%s", " ".join(map(str, [f"   Permutation {i+1}: '{search_query}' → No results"])))

                                    except Exception as perm_e:
                                        raise RuntimeError("A required search permutation failed") from perm_e

                                candidates.extend(all_results)
                                logger.debug("%s", " ".join(map(str, [f"[OK] HOMOGLYPH SEARCH COMPLETED: {len(best_candidates)} candidates, best score: {best_score:.3f}"])))
                            else:
                                # Normal search
                                name_candidates = await self.search_service.find_candidates(
                                    normalized=norm_result,
                                    text=original_text,
                                    opts=search_opts,
                                            search_trace=search_trace,
                                )
                                candidates.extend(name_candidates)
                                logger.debug("%s", " ".join(map(str, [f"[OK] NORMAL SEARCH COMPLETED: {len(name_candidates)} candidates"])))

                            search_processing_time = (time.time() - search_start_time) * 1000
                            logger.debug("%s", " ".join(map(str, [f"[OK] FULL SEARCH COMPLETED: {len(candidates) - len(id_candidates)} name candidates + {len(id_candidates)} ID candidates in {search_processing_time:.2f}ms"])))
                        except Exception as e:
                            search_processing_time = (time.time() - search_start_time) * 1000
                            logger.debug("%s", " ".join(map(str, [f"[ERROR] FULL SEARCH FAILED: {e} after {search_processing_time:.2f}ms"])))
                            raise RuntimeError("Name screening is unavailable") from e
                    else:
                        logger.debug("%s", " ".join(map(str, [f"[WARN] SEARCH SERVICE IS NONE - using fallback fuzzy search"])))
                        search_processing_time = (time.time() - search_start_time) * 1000

                    await self.search_service._verify_dataset_version(dataset_version, require_vectors=require_vectors)
                    serialized = [candidate.to_dict() if hasattr(candidate, "to_dict") else candidate for candidate in candidates]
                    if search_opts.entity_types:
                        serialized = [candidate for candidate in serialized if candidate.get("entity_type") in search_opts.entity_types]
                    serialized = [candidate for candidate in serialized if metadata_matches(
                        candidate.get("doc_id"), candidate.get("metadata", {}), search_opts.metadata_filters or {}
                    )]
                    unique = {}
                    for candidate in serialized:
                        identity = source_identity(candidate.get("doc_id"), candidate.get("entity_type"), candidate.get("metadata", {}))
                        if identity not in unique or candidate.get("confidence", 0) > unique[identity].get("confidence", 0):
                            unique[identity] = candidate
                    serialized = sorted(unique.values(), key=lambda item: (-item.get("confidence", 0), item.get("doc_id", "")))[:search_opts.top_k]
                    search_results = {"query": query, "results": serialized,
                                      "total_hits": len(serialized),
                                      "search_type": search_opts.search_mode.value,
                                      "processing_time_ms": search_processing_time}

                    # Add search trace if available
                    if search_trace and search_trace.enabled:
                        search_results["trace"] = search_trace.to_dict()

                    if search_trace:
                        result_count = search_results.get('total_hits', 0) if search_results else 0
                        search_trace.note(f"Search completed: {result_count} results for '{query}'")

                    logger.debug(f"Search completed for '{query}': {search_results.get('total_hits', 0) if search_results else 0} results")

                else:
                    if search_trace:
                        search_trace.note("Search skipped - empty normalized text")
                    logger.debug("Search skipped - empty normalized text")

                # Update metrics
                if self.metrics_service:
                    self.metrics_service.record_timer('processing.layer.search', time.time() - layer_start)
                    if search_results:
                        hit_count = search_results.get('total_hits', 0)
                        self.metrics_service.record_histogram('search.results.count', hit_count)
                        if hit_count > 0:
                            self.metrics_service.record_counter('processing.search.found_results', 1)
                        else:
                            self.metrics_service.record_counter('processing.search.no_results', 1)

            except Exception as e:
                logger.warning(f"Search failed: {e}")
                if search_trace:
                    search_trace.note(f"Search failed: {str(e)}")
                if self.metrics_service:
                    self.metrics_service.record_counter('processing.search.failed', 1)
                errors.append(f"Search: {str(e)}")

        else:
            logger.debug("Search layer skipped - service disabled or unavailable")
            if search_trace:
                search_trace.note("Search layer skipped - service disabled or unavailable")

        return search_results

    async def _handle_decision_layer(
        self,
        context: ProcessingContext,
        norm_result: Any,
        signals_result: Any,
        variants: Optional[list],
        embeddings: Optional[list],
        search_results: Optional[dict],
        errors: list,
        search_trace: Optional[SearchTrace] = None
    ) -> Optional[Any]:
        """
        Handle Layer 9: Decision & Response

        Args:
            context: Processing context
            norm_result: Result from name normalization
            signals_result: Result from signals extraction
            variants: Generated variants (optional)
            embeddings: Generated embeddings (optional)
            errors: List of errors

        Returns:
            Decision result or None if disabled/failed
        """
        decision_result = None

        # Initialize metrics tracking for this layer
        metrics = None
        try:
            from ..monitoring.prometheus_exporter import get_exporter
            metrics = get_exporter()
        except Exception as e:
            logger.debug(f"Metrics not available in decision layer: {e}")
            metrics = None

        # Create processing result for decision engine
        temp_processing_result = UnifiedProcessingResult(
            original_text=context.original_text,
            language=context.language,
            language_confidence=context.language_confidence,
            normalized_text=norm_result.normalized,
            tokens=norm_result.tokens,
            trace=norm_result.trace,
            signals=signals_result,
            variants=variants,
            embeddings=embeddings,
            processing_time=0.0,  # Will be set below
            success=self._safe_len(errors) == 0,
            errors=errors,
            # Copy homoglyph fields from normalization result
            homoglyph_detected=getattr(norm_result, 'homoglyph_detected', False),
            homoglyph_analysis=getattr(norm_result, 'homoglyph_analysis', None),
        )

        # Run decision engine if enabled
        if self.enable_decision_engine and not errors:
            logger.debug("Stage 9: Decision Engine")
            layer_start = time.time()
            try:
                # Create DecisionInput from processing results
                decision_input = self._create_decision_input(
                    context, temp_processing_result, signals_result, search_results
                )

                # Make decision using our new DecisionEngine
                decision_result = self.decision_engine.decide(decision_input, search_trace)

                logger.debug(
                    f"Decision made: {decision_result.risk.value} "
                    f"(score: {decision_result.score:.2f})"
                )
                if self.metrics_service:
                    self.metrics_service.record_timer('processing.layer.decision', time.time() - layer_start)
                    self.metrics_service.record_histogram('decision.score', decision_result.score)
                    self.metrics_service.record_counter(f'decision.result.{decision_result.risk.value.lower()}', 1)

                # Record sanctions decision metrics
                if metrics:
                    # Detect if fast path was used by checking for fast path sanctions in signals
                    fast_path_used = any(
                        getattr(result, 'fast_path_sanctions', {}).get('cache_hit', False)
                        for result in [signals_result] if hasattr(result, '__dict__')
                    )
                    metrics.record_sanctions_decision(decision_result.risk.value, fast_path_used)
            except Exception as e:
                logger.warning(f"Decision engine failed: {e}")
                if self.metrics_service:
                    self.metrics_service.record_counter('processing.decision.failed', 1)
                errors.append(f"Decision engine: {str(e)}")

        return decision_result

    async def process(
        self,
        text: str,
        *,
        # Normalization flags (must have real effect per CLAUDE.md)
        remove_stop_words: bool = True,
        preserve_names: bool = True,
        enable_advanced_features: bool = True,
        # Processing hints
        language_hint: Optional[str] = None,
        screen: bool = True,
        clean_unicode: bool = True,
        generate_variants: Optional[bool] = None,
        generate_embeddings: Optional[bool] = None,
        feature_flags: Optional[FeatureFlags] = None,
        # Search tracing
        search_trace_enabled: bool = True,  # Enable by default for debugging
        search_options: Optional[Any] = None,
        force_full_pipeline: bool = False,
        # Legacy compatibility kwargs (ignored but accepted)
        cache_result: Optional[bool] = None,
        embeddings: Optional[bool] = None,
        variants: Optional[bool] = None,
        **legacy_kwargs,
    ) -> UnifiedProcessingResult:
        """
        Process text through the unified 9-layer pipeline.

        Args:
            text: Input text to process
            remove_stop_words: Clean STOP_ALL tokens in normalization
            preserve_names: Keep `. - '` for initials/compound names
            enable_advanced_features: Use morphology + diminutives + gender
            language_hint: Optional language hint
            generate_variants: Override variants generation
            generate_embeddings: Override embeddings generation

        Returns:
            UnifiedProcessingResult with all layers' output
        """
        start_time = time.time()
        context = ProcessingContext(original_text=text)
        errors = []

        # Initialize metrics tracking
        metrics = None
        try:
            from ..monitoring.prometheus_exporter import get_exporter
            metrics = get_exporter()
        except Exception as e:
            # Catch all exceptions to ensure metrics is always defined
            logger.debug(f"Metrics not available: {e}")
            metrics = None
        
        # Initialize search trace if enabled
        search_trace = None
        if search_trace_enabled:
            search_trace = SearchTrace(enabled=True)

        # Defensive handling of feature flags
        effective_flags = self._validate_and_normalize_flags(feature_flags)
        context.processing_flags["feature_flags"] = effective_flags.to_dict()
        context.metadata["feature_flags"] = effective_flags.to_dict()

        # Handle legacy kwargs mapping
        if embeddings is not None:
            generate_embeddings = embeddings
        if variants is not None:
            generate_variants = variants

        # Handle caching if enabled and cache_service is available
        cache_key = None
        if cache_result and self.cache_service and not (screen and self.enable_search):
            cache_key = self._generate_cache_key(text, remove_stop_words, preserve_names,
                language_hint=language_hint, clean_unicode=clean_unicode,
                enable_advanced_features=enable_advanced_features,
                feature_flags=effective_flags.to_dict(), screen=screen,
                generate_variants=self.enable_variants if generate_variants is None else generate_variants,
                generate_embeddings=self.enable_embeddings if generate_embeddings is None else generate_embeddings,
                search_trace_enabled=search_trace_enabled)
            try:
                cached_result = self.cache_service.get(cache_key)
                if cached_result:
                    # Update stats for cache hit
                    self.update_stats(0.001, cache_hit=True, error=False)
                    return deepcopy(cached_result)
            except Exception as e:
                logger.debug(f"Cache get failed: {e}")

        # Initialize metrics collection
        if self.metrics_service:
            self.metrics_service.record_counter('processing.requests.total', 1)
            self.metrics_service.record_gauge('processing.requests.active', 1)

        try:
            # ================================================================
            # Layer 1: Validation & Sanitization
            # ================================================================
            validation_result = await self._handle_validation_layer(text, context, start_time)
            if validation_result is not None:  # Early return if validation failed
                return validation_result

            # ================================================================
            # Layer 2: Smart Filter (optional skip)
            # ================================================================
            smart_filter_result = await self._handle_smart_filter_layer(context, start_time)
            if smart_filter_result is not None:  # Early return if filtered
                return smart_filter_result
            
            # Add trace note for smart filter
            if search_trace:
                search_trace.note("Smart filter passed - continuing with processing")

            # ================================================================
            # Layer 3: Language Detection (on original text to preserve language markers)
            # ================================================================
            await self._handle_language_detection_layer(context, language_hint)

            # ================================================================
            # Layer 4: Unicode Normalization (after language detection)
            # ================================================================
            text_u = await self._handle_unicode_normalization_layer(context) if clean_unicode else context.original_text

            # ================================================================
            # Layer 5: Name Normalization (morph) - THE CORE
            # ================================================================
            norm_result = await self._handle_name_normalization_layer(
                text_u, context, remove_stop_words, preserve_names, enable_advanced_features, effective_flags, errors
            )
            
            # Add trace note for AC patterns after normalization
            if search_trace and hasattr(norm_result, 'tokens') and norm_result.tokens:
                # Check if we have meaningful name tokens for AC search
                # AC search should run for any person names (surnames, given names)
                has_searchable_patterns = any(
                    len(token) >= 2 and token.isalpha()  # Any alphabetic token ≥2 chars
                    for token in norm_result.tokens
                )
                if not has_searchable_patterns:
                    search_trace.note("AC skipped - no searchable patterns detected")
                else:
                    search_trace.note("AC patterns detected - proceeding with search")

            # ================================================================
            # Layer 6: Signals (enrichment)
            # ================================================================
            signals_result = await self._handle_signals_layer(text_u, norm_result, context)

            # ================================================================
            # Layer 7: Variants (optional)
            # ================================================================
            variants = await self._handle_variants_layer(norm_result, context, generate_variants, errors)

            # ================================================================
            # Layer 8: Embeddings (optional)
            # ================================================================
            embeddings = await self._handle_embeddings_layer(norm_result, generate_embeddings, errors)

            # Add trace note for vector fallback
            if search_trace and embeddings:
                search_trace.note("Vector fallback engaged - embeddings generated for search")

            # ================================================================
            # Layer 9: Search (optional)
            # ================================================================
            search_results = None if not screen else await self._handle_search_layer(
                norm_result,
                embeddings,
                errors,
                text,
                search_trace,
                signals_result,
                search_options,
            )

            # ================================================================
            # Layer 10: Decision & Response
            # ================================================================
            decision_result = None if not screen else await self._handle_decision_layer(
                context, norm_result, signals_result, variants, embeddings, search_results, errors, search_trace
            )

            processing_time = time.time() - start_time

            # Update final metrics
            if self.metrics_service:
                self.metrics_service.record_timer('processing.total_time', processing_time)
                self.metrics_service.record_gauge('processing.requests.active', -1)  # Decrement active requests
                if len(errors) == 0:
                    self.metrics_service.record_counter('processing.requests.successful', 1)
                else:
                    self.metrics_service.record_counter('processing.requests.failed', 1)
                    self.metrics_service.record_histogram('processing.error_count', self._safe_len(errors))

            # Warn if processing is slow
            if processing_time > 0.1:  # 100ms threshold per CLAUDE.md
                logger.warning(
                    f"Slow processing: {processing_time:.3f}s for text: {text[:50]}..."
                )
                if self.metrics_service:
                    self.metrics_service.record_counter('processing.slow_requests', 1)

            # Update legacy stats
            self.update_stats(processing_time, cache_hit=False, error=self._safe_len(errors) > 0)

            # Create the result
            result = UnifiedProcessingResult(
                original_text=context.original_text,
                language=context.language,
                language_confidence=context.language_confidence,
                normalized_text=norm_result.normalized,
                tokens=norm_result.tokens,
                trace=norm_result.trace,
                signals=signals_result,
                variants=variants,
                embeddings=embeddings,
                search_results=search_results,
                decision=decision_result,
                processing_time=processing_time,
                success=self._safe_len(errors) == 0,
                errors=errors,
                # Copy homoglyph fields from normalization result
                homoglyph_detected=getattr(norm_result, 'homoglyph_detected', False),
                homoglyph_analysis=getattr(norm_result, 'homoglyph_analysis', None),
            )

            # Cache the result if caching is enabled and successful
            if cache_result and self.cache_service and cache_key and result.success:
                try:
                    self.cache_service.set(cache_key, deepcopy(result))
                except Exception as e:
                    logger.debug(f"Cache set failed: {e}")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Processing failed: {e}", exc_info=True)

            # Update error metrics
            if self.metrics_service:
                self.metrics_service.record_counter('processing.requests.failed', 1)
                self.metrics_service.record_counter('processing.exceptions', 1)
                self.metrics_service.record_gauge('processing.requests.active', -1)  # Decrement active requests
                self.metrics_service.record_timer('processing.total_time', processing_time)

            return UnifiedProcessingResult(
                original_text=context.original_text,
                language=context.language or "unknown",
                language_confidence=context.language_confidence or 0.0,
                normalized_text="",
                tokens=[],
                trace=[],
                signals=SignalsResult(),
                processing_time=processing_time,
                success=False,
                errors=[str(e)],
            )

    def _create_early_response(
        self, context: ProcessingContext, reason: str, start_time: float
    ) -> UnifiedProcessingResult:
        """Create response for early termination"""
        return UnifiedProcessingResult(
            original_text=context.original_text,
            language="unknown",
            language_confidence=0.0,
            normalized_text="",
            tokens=[],
            trace=[],
            signals=SignalsResult(),
            processing_time=time.time() - start_time,
            success=False,
            errors=[reason],
        )

    def _create_filtered_response(
        self,
        context: ProcessingContext,
        filter_result,  # SmartFilterResult object
        start_time: float,
    ) -> UnifiedProcessingResult:
        """Create response when smart filter suggests skipping"""
        return UnifiedProcessingResult(
            original_text=context.original_text,
            language=context.language or "unknown",
            language_confidence=context.language_confidence or 0.0,
            normalized_text=context.sanitized_text or context.original_text,
            tokens=[],
            trace=[],
            signals=SignalsResult(confidence=filter_result.confidence),
            processing_time=time.time() - start_time,
            success=True,
            errors=[],
        )

    def _create_decision_input(
        self,
        context: ProcessingContext,
        processing_result: UnifiedProcessingResult,
        signals_result: SignalsResult,
        search_results: Optional[dict] = None
    ) -> DecisionInput:
        """Create DecisionInput from processing results"""
        
        # Extract smart filter information
        smart_filter_info = context.metadata.get("smart_filter", {})
        smart_filter = SmartFilterInfo(
            should_process=smart_filter_info.get("should_process", True),
            confidence=smart_filter_info.get("confidence", 1.0),
            estimated_complexity=smart_filter_info.get("classification")
        )
        
        def value(obj, key, default=None):
            return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

        def dates(raw):
            import re
            from datetime import datetime
            result = set()
            for token in re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}[./]\d{2}[./]\d{4}\b", str(raw or "")):
                for form in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
                    try:
                        result.add(datetime.strptime(token, form).date().isoformat())
                        break
                    except ValueError:
                        pass
            return result

        persons = signals_result.persons or []
        organizations = signals_result.organizations or []
        entities = [*persons, *organizations]
        def tax_ids(entity):
            return {str(item.get("value", "")).strip()
                    for item in (value(entity, "ids", []) or [])
                    if isinstance(item, dict) and item.get("value") and
                    item.get("type", "").casefold() in {"inn", "inn_ua", "inn_ru", "itn", "tin", "tax_id", "edrpou"}}

        identity_groups = [(tax_ids(person), dates(value(person, "dob"))) for person in persons]
        identity_groups.extend((tax_ids(org), set()) for org in organizations)
        query_ids = set().union(*(ids for ids, _ in identity_groups))
        extras = value(signals_result, "extras", {}) or {}
        unassigned_ids = tax_ids({"ids": value(extras, "unassigned_ids", [])})
        query_ids.update(unassigned_ids)
        query_dobs = set().union(*(dates(value(person, "dob")) for person in persons))
        search_info = self._create_search_info_from_results(search_results) if search_results else None
        if search_info:
            for candidate in search_info.fusion_candidates:
                import re
                candidate_ids = source_tax_ids(candidate.meta)
                candidate_dates = dates(candidate.dob or candidate.meta.get("birthdate"))
                candidate.features["id_match"] = bool(query_ids & candidate_ids)
                candidate.features["unassigned_id_match"] = bool(unassigned_ids & candidate_ids)
                candidate.features["date_match"] = bool(query_dobs & candidate_dates)
                candidate.features["identity_pair_match"] = any(
                    bool(ids & candidate_ids) and bool(dobs & candidate_dates)
                    for ids, dobs in identity_groups
                )
                candidate.features["identity_conflict"] = any(
                    bool(ids & candidate_ids) and bool(dobs) and bool(candidate_dates)
                    and not bool(dobs & candidate_dates)
                    for ids, dobs in identity_groups
                )
        fusion = search_info.fusion_candidates if search_info else []
        evidence = {"persons_count": len(persons), "organizations_count": len(organizations),
                    "signals_confidence": signals_result.confidence,
                    "extracted_ids": ["tin"] if query_ids else [],
                    "extracted_dates": ["dob"] if query_dobs else []}
        signals = SignalsInfo(
            person_confidence=max((value(person, "confidence", 0.0) for person in persons), default=0.0),
            org_confidence=max((value(org, "confidence", 0.0) for org in organizations), default=0.0),
            id_match=any(candidate.features.get("id_match") for candidate in fusion),
            date_match=any(candidate.features.get("date_match") for candidate in fusion), evidence=evidence,
        )
        similarity = SimilarityInfo()

        # Create normalization object with homoglyph detection
        normalization_obj = type('NormalizationObj', (), {
            'homoglyph_detected': getattr(processing_result, 'homoglyph_detected', False),
            'homoglyph_analysis': getattr(processing_result, 'homoglyph_analysis', None),
            'normalized_text': getattr(processing_result, 'normalized_text', ''),
        })()

        return DecisionInput(
            text=context.original_text,
            language=context.language,
            smartfilter=smart_filter,
            signals=signals,
            similarity=similarity,
            search=search_info,
            normalization=normalization_obj  # Pass normalization with homoglyph data
        )

    def _create_search_info_from_results(self, search_results: dict):
        """Create SearchInfo from search_results dict format"""
        try:
            from ..contracts.search_contracts import (
                SearchResult, Candidate, SearchType, SearchInfo, create_search_info
            )

            candidates = []
            query = " ".join(search_results.get("query", "").casefold().split())
            for result in search_results.get("results", []):
                fields = result.get("match_fields", [])
                metadata = result.get("metadata") or {}
                text = result.get("text", "")
                mode = result.get("search_mode")
                exact_text = query and query == " ".join(text.casefold().split())
                if mode == "vector":
                    search_type = SearchType.VECTOR
                elif exact_text and mode in {"ac", "hybrid", "fallback_ac"}:
                    search_type = SearchType.EXACT
                elif "phrase" in fields:
                    search_type = SearchType.PHRASE
                elif "ngram" in fields or mode == "fuzzy" or "fuzzy_name" in fields:
                    search_type = SearchType.NGRAM
                else:
                    search_type = SearchType.WEAK
                score = min(1.0, max(0.0, float(result.get("confidence", result.get("score", 0.0)))))
                trace = result.get("trace") or {}
                candidates.append(Candidate(
                    entity_id=metadata.get("entity_id", result.get("doc_id", "")),
                    entity_type=result.get("entity_type", ""), normalized_name=text,
                    aliases=metadata.get("aliases") or [], country=metadata.get("country", ""),
                    dob=metadata.get("dob"), meta=metadata, final_score=score,
                    ac_score=score if mode != "vector" else 0.0,
                    vector_score=score if mode == "vector" else 0.0,
                    features={"match_fields": fields, "id_match": trace.get("id_match"),
                              "date_match": trace.get("date_match")}, search_type=search_type,
                ))

            # Create SearchResult
            search_result = SearchResult(
                candidates=candidates,
                ac_results=[],  # Empty as in production
                vector_results=[],  # Empty as in production
                search_metadata=search_results.get("search_metadata", {}),
                processing_time=search_results.get("processing_time_ms", 0) / 1000.0,
                success=True
            )

            # Create SearchInfo using existing function
            logger.debug(f"Creating SearchInfo from {len(candidates)} candidates")
            search_info = create_search_info(search_result)
            logger.debug(f"SearchInfo created: high_confidence_matches={search_info.high_confidence_matches}")
            return search_info

        except Exception as exc:
            raise RuntimeError("Search evidence could not be converted into a decision") from exc

    def _create_basic_search_info(self, has_matches: bool, total_matches: int):
        """Create basic SearchInfo when full contracts unavailable"""
        try:
            from ..contracts.search_contracts import SearchInfo
            return SearchInfo(
                has_exact_matches=has_matches,
                exact_confidence=0.9 if has_matches else 0.0,
                total_matches=total_matches,
                high_confidence_matches=total_matches if has_matches else 0
            )
        except ImportError:
            # Return a dict that looks like SearchInfo for compatibility
            return type('SearchInfo', (), {
                'has_exact_matches': has_matches,
                'exact_confidence': 0.9 if has_matches else 0.0,
                'total_matches': total_matches,
                'high_confidence_matches': total_matches if has_matches else 0,
                'has_phrase_matches': False,
                'has_ngram_matches': False,
                'has_vector_matches': False,
                'phrase_confidence': 0.0,
                'ngram_confidence': 0.0,
                'vector_confidence': 0.0
            })()

    def _is_simple_case(self, text: str) -> bool:
        """Check if text is simple enough for fast path processing"""
        if not text or not text.strip():
            return False

        if len(text.strip()) < 3:
            return True

        # Single word, likely just a name
        words = text.strip().split()
        if len(words) == 1 and len(words[0]) < 20:
            return True

        # Only digits or punctuation
        if not any(c.isalpha() for c in text):
            return True

        # Very short text with no complex patterns
        if len(text.strip()) < 10 and not any(pattern in text.lower() for pattern in
                                             ['оао', 'ооо', 'тов', 'llc', 'inc', 'ltd']):
            return True

        return False

    def _create_simple_response(self, text: str, context: ProcessingContext, start_time: float) -> UnifiedProcessingResult:
        """Create fast response for simple cases"""
        processing_time = time.time() - start_time

        # Basic cleanup for simple text
        cleaned = text.strip().title() if text.strip() else ""

        return UnifiedProcessingResult(
            original_text=context.original_text,
            language="en",  # Default for simple cases
            language_confidence=0.5,
            normalized_text=cleaned,
            tokens=[cleaned] if cleaned else [],
            trace=[],
            signals=SignalsResult(confidence=0.3),  # Low confidence for simple cases
            processing_time=processing_time,
            success=True,
            errors=[],
        )

    # Convenience methods for backward compatibility

    async def normalize_async(self, text: str, **kwargs) -> NormalizationResult:
        """
        Backward compatibility: direct normalization.
        For new code, use process() instead.
        """
        logger.warning("normalize_async is deprecated. Use process() instead.")

        # Extract normalization-specific flags
        norm_flags = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "language",
                "remove_stop_words",
                "preserve_names",
                "enable_advanced_features",
            ]
        }

        # Run minimal pipeline: validation -> unicode -> normalization
        validation_result = await self._maybe_await(self.validation_service.validate_and_sanitize(text))
        sanitized = validation_result.get("sanitized_text", text)

        # Fixed order: Unicode normalization first
        unicode_result = await self._maybe_await(self.unicode_service.normalize_unicode(sanitized))
        
        # Handle both legacy string return and new dict return
        if isinstance(unicode_result, str):
            unicode_normalized = unicode_result
        else:
            unicode_normalized = unicode_result.get("normalized", sanitized)

        return await self._maybe_await(self.normalization_service.normalize_async(
            unicode_normalized, **norm_flags
        ))

    async def extract_signals(
        self, original_text: str, normalization_result: NormalizationResult
    ) -> SignalsResult:
        """Backward compatibility: direct signals extraction"""
        logger.warning("extract_signals is deprecated. Use process() instead.")
        return await self._maybe_await(self.signals_service.extract_async(
            text=original_text, normalization_result=normalization_result
        ))

    # Legacy compatibility methods for old tests
    def clear_cache(self):
        """Legacy method for cache clearing"""
        if hasattr(self.cache_service, 'clear'):
            self.cache_service.clear()
        logger.warning("clear_cache is deprecated. Use cache_service directly.")

    def _generate_cache_key(self, text: str, remove_stop_words: bool, preserve_names: bool, **options) -> str:
        """Cache only equivalent requests within this configured service instance."""
        import hashlib
        import json
        key_data = json.dumps({"instance": self._cache_namespace, "text": text,
            "remove_stop_words": remove_stop_words, "preserve_names": preserve_names,
            **options}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        hash_part = hashlib.sha256(key_data.encode()).hexdigest()
        return f"orchestrator_{hash_part}"

    def _calculate_complexity_score(self, unicode_complexity, language_complexity, name_complexity) -> float:
        """Legacy method for complexity calculation"""
        # Extract confidence values from dictionaries if needed
        unicode_val = unicode_complexity.get('confidence', 0.0) if isinstance(unicode_complexity, dict) else unicode_complexity
        language_val = language_complexity.get('confidence', 0.0) if isinstance(language_complexity, dict) else language_complexity
        name_val = name_complexity.get('confidence', 0.0) if isinstance(name_complexity, dict) else name_complexity
        
        # Combine different complexity factors
        return min(1.0, (unicode_val + language_val + name_val) / 3.0)

    def _generate_complexity_recommendations(self, score: float) -> List[str]:
        """Legacy method for complexity recommendations"""
        if score < 0.3:
            return ["Text is simple", "Consider adding more context"]
        elif score < 0.7:
            return ["Text complexity is moderate", "Good balance", "Consider reviewing structure"]
        else:
            return ["Text is complex", "Consider simplifying", "Break into smaller parts", "Use simpler language"]

    def get_processing_stats(self) -> Dict[str, Any]:
        """Legacy method for getting processing statistics"""
        return self.processing_stats.copy()

    def reset_stats(self):
        """Legacy method for resetting statistics"""
        self.processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache": 0,  # Legacy compatibility
            "errors": 0,
            "processing_times": []
        }

    def update_stats(self, processing_time: float, cache_hit: bool = False, error: bool = False):
        """Legacy method for updating statistics"""
        self.processing_stats["total_processed"] += 1
        if cache_hit:
            self.processing_stats["cache_hits"] += 1
        else:
            self.processing_stats["cache_misses"] += 1
        if error:
            self.processing_stats["errors"] += 1
        self.processing_stats["successful" if not error else "failed"] += 1
        self.processing_stats["processing_times"].append(processing_time)
        self.processing_stats["processing_times"] = self.processing_stats["processing_times"][-1000:]

    def _update_stats(self, processing_time: float, cache_hit: bool = False, error: bool = False):
        """Legacy method alias for updating statistics"""
        self.update_stats(processing_time, cache_hit, error)

    async def process_batch(self, texts: List[str], max_concurrent: int = 10, **kwargs) -> List[UnifiedProcessingResult]:
        if not 1 <= max_concurrent <= 32:
            raise ValueError("max_concurrent must be between 1 and 32")
        semaphore = asyncio.Semaphore(max_concurrent)
        async def process_one(text):
            async with semaphore:
                try:
                    return await self.process(text, **kwargs)
                except Exception as exc:
                    return UnifiedProcessingResult(original_text=text, language="en",
                        language_confidence=0.0, normalized_text=text, success=False, errors=[str(exc)])
        return await asyncio.gather(*(process_one(text) for text in texts))

    def _validate_and_normalize_flags(self, feature_flags: Optional[FeatureFlags]) -> FeatureFlags:
        return merge_feature_flags(self.default_feature_flags, feature_flags)

    async def search_similar_names(self, query: str, limit: int = 10,
            candidates: Optional[List[str]] = None, use_embeddings: bool = True,
            threshold: float = 0.7, top_k: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Compare the supplied candidate list with a query and apply all options."""
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        count = top_k if top_k is not None else limit
        if count < 1:
            raise ValueError("top_k must be positive")
        texts = list(candidates or [])
        if use_embeddings and texts:
            if self.embeddings_service is None:
                raise RuntimeError("Configured embedding service is unavailable")
            import inspect
            encode_async = getattr(self.embeddings_service, "encode_batch_async", None)
            if inspect.iscoroutinefunction(encode_async):
                vectors = await encode_async([query, *texts])
            else:
                vectors = await asyncio.to_thread(self.embeddings_service.encode_batch, [query, *texts])
            if len(vectors) != len(texts) + 1:
                raise RuntimeError("Embedding generation did not preserve candidate rows")
            import numpy as np
            query_vector = np.asarray(vectors[0])
            scores = [float(np.dot(query_vector, v) / (np.linalg.norm(query_vector) * np.linalg.norm(v))) for v in vectors[1:]]
        else:
            from rapidfuzz.fuzz import ratio
            scores = [ratio(query.casefold(), text.casefold()) / 100 for text in texts]
        results = [{"text": text, "score": score} for text, score in zip(texts, scores) if score >= threshold]
        results.sort(key=lambda item: (-item["score"], item["text"]))
        return {"method": "embeddings" if use_embeddings else "fuzzy", "query": query, "results": results[:count]}

    async def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """Legacy method for analyzing text complexity"""
        # Calculate individual complexity factors
        words = text.split()
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        unicode_complexity = min(1.0, len(text) / 100.0)
        language_complexity = min(1.0, word_count / 20.0)
        name_complexity = min(1.0, avg_word_length / 10.0)
        
        score = self._calculate_complexity_score(unicode_complexity, language_complexity, name_complexity)
        recommendations = self._generate_complexity_recommendations(score)
        return {
            "complexity_score": score,
            "recommendations": recommendations,
            "word_count": word_count,
            "character_count": len(text)
        }

    @staticmethod
    def _has_only_identifier_evidence(text, signals_result) -> bool:
        """Allow omitting name retrieval only for supported, source-verified IDs."""
        if not signals_result:
            return False
        extras = getattr(signals_result, "extras", {}) or {}
        identifiers = list(extras.get("unassigned_ids", []) if isinstance(extras, dict)
                           else getattr(extras, "unassigned_ids", []))
        for entity in [*(signals_result.persons or []), *(signals_result.organizations or [])]:
            identifiers.extend(entity.get("ids", []) if isinstance(entity, dict) else getattr(entity, "ids", []))

        covered = [False] * len(text)
        for identifier in identifiers:
            value = str(identifier.get("value") or "").strip()
            if str(identifier.get("type") or "").lower() not in TAX_IDENTIFIER_TYPES:
                continue
            if value not in source_tax_ids({"tin": value}):
                continue
            span = identifier.get("position")
            if not isinstance(span, (tuple, list)) or len(span) != 2 or any(type(n) is not int for n in span):
                continue
            start, end = span
            if not 0 <= start < end <= len(text) or text[start:end] != identifier.get("raw"):
                continue
            raw = without_format_controls(text[start:end])
            for pattern, regex in get_compiled_patterns_cached():
                if pattern.type not in TAX_IDENTIFIER_TYPES:
                    continue
                match = regex.fullmatch(raw)
                if match and normalize_identifier(match.group(1), pattern.type) == value:
                    covered[start:end] = [True] * (end - start)
                    break

        remainder = without_format_controls("".join(char for char, included in zip(text, covered) if not included))
        return any(covered) and all(
            char.isspace() or unicodedata.category(char).startswith("P") or char == "|"
            for char in remainder
        )

    async def _search_by_extracted_ids(self, signals_result, search_opts) -> List[Dict[str, Any]]:
        """Search for sanctioned persons by extracted IDs (INN, EDRPOU, etc.)."""
        try:
            if not signals_result:
                return []

            id_candidates = []

            entities = [*(signals_result.persons or []), *(signals_result.organizations or [])]
            extras = getattr(signals_result, "extras", {}) or {}
            unassigned = extras.get("unassigned_ids", []) if isinstance(extras, dict) else getattr(extras, "unassigned_ids", [])
            identifiers = list(unassigned)
            for entity in entities:
                identifiers.extend(entity.get("ids", []) if isinstance(entity, dict) else getattr(entity, "ids", []))
            seen = set()
            for id_info in identifiers:
                id_value = str(id_info.get("value") or "").strip()
                id_type = str(id_info.get("type") or "").lower()
                if not id_value or (id_type, id_value) in seen:
                    continue
                seen.add((id_type, id_value))
                id_candidates.extend(await self._find_candidates_by_id(id_value, id_type, search_opts))

            return id_candidates
        except Exception as e:
            raise RuntimeError("Identifier screening failed") from e

    async def _find_candidates_by_id(self, id_value, id_type, search_opts=None):
        """Use the configured active sanctions snapshot for identifier evidence."""
        if self.search_service is None:
            raise RuntimeError("Identifier screening is unavailable")
        candidates = await self.search_service.find_by_identifier(id_value, id_type, search_opts)
        return [{**candidate.to_dict(), "search_mode": "id_exact"} for candidate in candidates]

    def _create_modified_norm_result(self, norm_result, new_normalized_text):
        """
        Create modified normalization result with different normalized text.

        Used for homoglyph permutation searches to try different name orders.
        """
        from copy import deepcopy

        # Create a copy of the normalization result
        modified_result = deepcopy(norm_result)

        # Update the normalized text
        modified_result.normalized = new_normalized_text

        # Update tokens if needed
        if hasattr(modified_result, 'tokens'):
            modified_result.tokens = new_normalized_text.split() if new_normalized_text else []

        return modified_result
