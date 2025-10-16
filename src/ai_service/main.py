#!/usr/bin/env python3
"""
AI Service for normalization and variant generation
for sanctions data verification
"""

import os
import asyncio
import inspect
import logging
import subprocess
import sys
import time
import secrets
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, ValidationError, validator

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_service.config import (
    DEPLOYMENT_CONFIG,
    INTEGRATION_CONFIG,
    SECURITY_CONFIG,
    SERVICE_CONFIG,
)
from ai_service import __version__
from ai_service.api.error_contracts import RequestValidationIssue, RequestValidationResponse
from ai_service.contracts.base_contracts import (
    NormalizationResponse,
    ProcessResponse,
    ProcessBatchItem,
    ProcessBatchResponse,
    UnifiedProcessingResult,
)
from ai_service.core.orchestrator_factory import OrchestratorFactory
from ai_service.exceptions import (
    AuthenticationError,
    InternalServerError,
    ServiceUnavailableError,
    ValidationAPIError,
)
from ai_service.monitoring.prometheus_exporter import get_exporter
from ai_service.utils import get_logger, setup_logging

from ai_service.layers.search.contracts import SearchMode, SearchOpts
from ai_service.utils.source_text_view import without_format_controls
from ai_service.layers.search.config import HybridSearchConfig
from ai_service.utils.feature_flags import FeatureFlags, get_feature_flag_manager
from ai_service.utils.response_formatter import format_processing_result
from ai_service.utils.inference_queue import InferenceUnavailableError

# Setup centralized logging
setup_logging()
logger = get_logger(__name__)

# Import lazy_imports module to trigger initialization
from ai_service.utils.lazy_imports import NAMEPARSER, NLP_EN, NLP_RU, NLP_UK, RAPIDFUZZ

# Create FastAPI application


def _require_text_content(value: str) -> str:
    """Reject effectively empty input without changing its source offsets."""
    if not without_format_controls(value).strip():
        raise ValueError("Text must contain visible content")
    return value


class NormalizationOptions(BaseModel):
    """Normalization options including feature flags"""

    flags: Optional[Dict[str, Any] | FeatureFlags] = None


class TextNormalizationRequest(BaseModel):
    """Request model for text normalization"""

    text: str = Field(..., max_length=SERVICE_CONFIG.max_input_length, min_length=1)
    language: str = Field(default="auto", pattern="^(auto|ru|uk|en)$")
    remove_stop_words: bool = False  # For names, don't remove stop words
    apply_stemming: bool = False  # For names, don't apply stemming
    apply_lemmatization: bool = True  # For names, apply lemmatization
    clean_unicode: bool = True
    preserve_names: bool = True  # Preserve names and surnames
    options: Optional[NormalizationOptions] = None

    @validator("text")
    def validate_text_content(cls, v):
        """Additional text validation for security"""
        _require_text_content(v)

        # Check for excessive special characters (potential attack)
        special_char_count = sum(1 for c in v if not c.isalnum() and not c.isspace())
        if special_char_count > len(v) * 0.5:  # More than 50% special chars
            raise ValueError("Text contains too many special characters")

        return v


class ProcessTextRequest(BaseModel):
    """Request model for text processing"""

    text: str = Field(..., max_length=SERVICE_CONFIG.max_input_length, min_length=1)
    generate_variants: bool = True
    generate_embeddings: bool = False
    cache_result: bool = True
    options: Optional[NormalizationOptions] = None

    @validator("text")
    def validate_text_length(cls, v):
        """Validate text length"""
        _require_text_content(v)
        if len(v) > SERVICE_CONFIG.max_input_length:
            raise ValueError(
                f"Text too long: {len(v)} > {SERVICE_CONFIG.max_input_length}"
            )
        return v


class ProcessBatchRequest(BaseModel):
    """Request model for batch text processing"""

    texts: List[str] = Field(..., min_length=1, max_length=100)
    generate_variants: bool = True
    generate_embeddings: bool = False
    max_concurrent: int = Field(default=10, ge=1, le=32)

    @validator("texts")
    def validate_texts(cls, v):
        """Validate each text in the list"""
        for text in v:
            _require_text_content(text)
            if len(text) > SERVICE_CONFIG.max_input_length:
                raise ValueError(
                    f"Text too long: {len(text)} > {SERVICE_CONFIG.max_input_length}"
                )
        return v


class SearchSimilarRequest(BaseModel):
    """Request model for searching similar names"""

    query: str = Field(..., min_length=1, max_length=SERVICE_CONFIG.max_input_length)
    candidates: List[str] = Field(..., min_length=1, max_length=1000)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=100)
    use_embeddings: bool = True

    @validator("candidates")
    def validate_candidates(cls, v):
        """Bound individual candidate size to the normal input limit."""
        for candidate in v:
            if not candidate.strip():
                raise ValueError("Candidates cannot be empty")
            if len(candidate) > SERVICE_CONFIG.max_input_length:
                raise ValueError("Candidate is too long")
        return v


class SearchRequest(BaseModel):
    """Request model for sanctions-list search."""

    query: str = Field(..., max_length=SERVICE_CONFIG.max_input_length, min_length=1)
    search_mode: SearchMode = SearchMode.HYBRID
    top_k: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_escalation: bool = True

    @validator("query")
    def validate_query(cls, v):
        """Reject whitespace-only search requests."""
        _require_text_content(v)
        return v.strip()


class ComplexityAnalysisRequest(BaseModel):
    """Request model for text processing"""

    text: str = Field(..., max_length=SERVICE_CONFIG.max_input_length)


app = FastAPI(
    title="Multilingual Sanctions Search",
    description="Multilingual normalization and hybrid search for sanctions screening",
    version=__version__,
    docs_url=INTEGRATION_CONFIG.docs_url if INTEGRATION_CONFIG.enable_docs else None,
    redoc_url=INTEGRATION_CONFIG.redoc_url if INTEGRATION_CONFIG.enable_docs else None,
    responses={422: {"model": RequestValidationResponse, "description": "Request validation failed"}},
)

# Import admin endpoints
from ai_service.api.admin_endpoints import router as admin_router

# Configure CORS
if INTEGRATION_CONFIG.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=INTEGRATION_CONFIG.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

# Rate limiting for DoS protection
class RateLimitingMiddleware:
    """Simple in-memory rate limiting middleware for DoS protection"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.max_clients = 10_000
        self.last_cleanup = 0.0

    async def __call__(self, request: Request, call_next):
        if request.url.path in {"/health", "/health/live", "/health/ready", "/metrics"}:
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.monotonic()
        if current_time - self.last_cleanup >= self.window_seconds:
            self.requests = defaultdict(list, {
                client: timestamps for client, timestamps in self.requests.items()
                if timestamps and current_time - timestamps[-1] < self.window_seconds
            })
            self.last_cleanup = current_time
        if client_ip not in self.requests and len(self.requests) >= self.max_clients:
            return JSONResponse(status_code=429, content={"detail": "Rate limit capacity reached"})

        # Clean old requests
        self.requests[client_ip] = [
            req_time
            for req_time in self.requests[client_ip]
            if current_time - req_time < self.window_seconds
        ]

        # Check rate limit
        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Too Many Requests - Rate limit exceeded"},
            )

        # Add current request
        self.requests[client_ip].append(current_time)

        response = await call_next(request)
        return response


# Use the configured limit so documented security settings actually take effect.
if SECURITY_CONFIG.rate_limit_enabled:
    app.middleware("http")(
        RateLimitingMiddleware(
            max_requests=SECURITY_CONFIG.max_requests_per_minute,
            window_seconds=60,
        )
    )

from ai_service.api.request_limits import RequestLimitsConfig, RequestLimitsMiddleware
app.add_middleware(
    RequestLimitsMiddleware,
    get_admin_key=lambda: SECURITY_CONFIG.admin_api_key,
    **RequestLimitsConfig().model_dump(),
)

# Initialize orchestrator
orchestrator = None

# Security
security = HTTPBearer(auto_error=False)


def _extract_signals_dict(result: UnifiedProcessingResult) -> Dict[str, Any]:
    """Extract signals information as dictionary"""
    if not result.signals:
        return None

    extras = getattr(result.signals, "extras", None)
    public_extras = {
        key: (extras.get(key, []) if isinstance(extras, dict) else getattr(extras, key, []))
        for key in ("dates", "amounts", "unassigned_ids")
    }

    return {
        "persons": [
            {
                "core": person.core,
                "full_name": person.full_name,
                "dob": person.dob,
                "dob_raw": getattr(person, "dob_raw", None),
                "dob_position": getattr(person, "dob_position", None),
                "ids": person.ids,
                "confidence": person.confidence,
                "evidence": person.evidence,
            }
            for person in result.signals.persons
        ],
        "organizations": [
            {
                "core": org.core,
                "legal_form": org.legal_form,
                "full_name": org.full_name,
                "ids": org.ids,
                "confidence": org.confidence,
                "evidence": org.evidence,
            }
            for org in result.signals.organizations
        ],
        "confidence": result.signals.confidence,
        "extras": public_extras,
    }


def _extract_decision_dict(result: UnifiedProcessingResult) -> Dict[str, Any]:
    """Extract decision information as dictionary"""
    if not result.decision:
        return None

    return {
        "risk_level": result.decision.risk.value,
        "risk_score": result.decision.score,
        "decision_reasons": result.decision.reasons,
        "decision_details": result.decision.details,
        "review_required": result.decision.review_required,
        "required_additional_fields": result.decision.required_additional_fields,
    }


def _processing_response(result, *, generate_variants: bool, generate_embeddings: bool) -> ProcessResponse:
    """One serialization contract for single results and each batch row."""
    if not result.success:
        # Derived partial fields cannot establish a successful screening result.
        return ProcessResponse(normalized_text="", tokens=[], trace=[], language="unknown",
            success=False, errors=["Processing could not complete"], processing_time=result.processing_time)
    return ProcessResponse(
        normalized_text=result.normalized_text, tokens=result.tokens or [], trace=result.trace or [],
        language=result.language, success=True, errors=[], processing_time=result.processing_time,
        signals=_extract_signals_dict(result) if result.signals else None,
        decision=_extract_decision_dict(result) if result.decision else None,
        search_results=result.search_results,
        variants=result.variants if generate_variants else None,
        embedding=result.embeddings if generate_embeddings else None,
        homoglyph_detected=getattr(result, "homoglyph_detected", False),
        homoglyph_analysis=getattr(result, "homoglyph_analysis", None),
    )


def _merge_feature_flags(request_flags) -> FeatureFlags:
    try:
        return get_feature_flag_manager().get_flags(request_flags)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid feature flag configuration") from exc


def verify_admin_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    Verify admin API token

    Args:
        credentials: HTTP authorization credentials

    Returns:
        Verified token string

    Raises:
        AuthenticationError: If token is invalid or not configured
    """
    if credentials is None:
        raise HTTPException(status_code=403, detail="Not authenticated")
    expected_token = SECURITY_CONFIG.admin_api_key

    # Enhanced token validation
    if not expected_token or expected_token == "your-secure-api-key-here":
        logger.warning("Admin API key not configured properly")
        raise AuthenticationError("Admin API key not configured")

    # Check minimum token length and complexity
    if len(expected_token) < 32:
        logger.warning("Admin API key is too short (minimum 32 characters)")
        raise AuthenticationError("Admin API key not configured properly")

    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(credentials.credentials, expected_token):
        logger.warning("Invalid admin API key attempt")
        raise AuthenticationError("Invalid API key")

    return credentials.credentials


# Protect every endpoint exposed by the administrative router. Keeping the
# dependency at the router boundary makes it much harder to accidentally add a
# destructive endpoint without authentication.
app.include_router(admin_router, dependencies=[Depends(verify_admin_token)])


async def _close_runtime(runtime):
    """Release every owned resource even if another provider fails to close."""
    if runtime is not None:
        for name in ("search_service", "embeddings_service", "variants_service"):
            service = getattr(runtime, name, None)
            if service is not None:
                try:
                    result = service.close()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.warning("Service cleanup failed for %s", name)
    from ai_service.layers.normalization.ner_gateways import close_global_gateway
    from ai_service.utils.async_model_loader import _model_loader
    try:
        close_global_gateway()
    except Exception:
        logger.warning("NER cleanup failed")
    try:
        await asyncio.to_thread(_model_loader.close)
    except Exception:
        logger.warning("Model loader cleanup failed")


@app.on_event("startup")
async def startup_event():
    global orchestrator
    from ai_service.api.snapshot_verification import VerificationLimits
    from ai_service.api.runtime_health import initialize_runtime_models
    VerificationLimits()
    candidate = None
    orchestrator = None
    try:
        candidate = await OrchestratorFactory.create_production_orchestrator()
        if candidate.enable_search:
            from ai_service.layers.search.sanctions_data_loader import SanctionsDataLoader
            loader = SanctionsDataLoader()
            await loader.load_dataset()
            candidate.search_service._sanctions_loader = loader
        await initialize_runtime_models(candidate)
        # Empty indices may be populated through the admin API after startup.
        # Traffic readiness verifies their complete generation on every probe.
        orchestrator = candidate
    except BaseException:
        await _close_runtime(candidate)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    global orchestrator
    previous, orchestrator = orchestrator, None
    await _close_runtime(previous)


@app.get("/health", responses={503: {"description": "Required dependencies are unavailable"}})
async def health_check():
    """Dependency health; liveness is provided separately by /health/live."""
    from ai_service.api.runtime_health import collect_runtime_health
    snapshot = await collect_runtime_health(orchestrator)
    return JSONResponse(status_code=200 if snapshot["status"] == "healthy" else 503,
        content={"status": snapshot["status"], "service": "AI Service",
                 "version": __version__, "timestamp": time.time()})


@app.get("/health/detailed", responses={503: {"description": "Dependencies or diagnostics are unavailable"}})
async def detailed_health_check(token: str = Depends(verify_admin_token)):
    """The same required dependencies, plus authenticated diagnostics."""
    from ai_service.api.runtime_health import collect_runtime_health
    runtime = orchestrator
    snapshot = await collect_runtime_health(runtime)
    diagnostics_ok = True
    stats = {}
    if runtime is not None:
        try:
            stats = runtime.get_processing_stats()
            if not isinstance(stats, dict):
                raise TypeError("Invalid statistics")
        except Exception:
            stats = {}
            diagnostics_ok = False
    try:
        from ai_service.utils.http_client_pool import get_http_pool
        pool_stats = get_http_pool().get_stats()
        snapshot["components"]["http_client_pool"] = {
            "status": "healthy", "active_clients": pool_stats.get("async_client_created", False),
            "connections": pool_stats,
        }
    except Exception:
        diagnostics_ok = False
        snapshot["components"]["http_client_pool"] = {
            "status": "unhealthy", "error": "Component health check failed"}
    status = snapshot["status"]
    if status == "healthy" and not diagnostics_ok:
        status = "degraded"
    total = stats.get("total_processed", 0)
    return JSONResponse(status_code=200 if status == "healthy" else 503, content=jsonable_encoder({
        "status": status, "service": "AI Service", "version": __version__,
        "timestamp": time.time(), "implementation": "full",
        "orchestrator": {
            "initialized": runtime is not None, "processed_total": total,
            "success_rate": stats.get("successful", 0) / total if total else 0,
            "cache_hit_rate": stats.get("cache", {}).get("hit_rate", 0)
                if isinstance(stats.get("cache"), dict) else 0,
            "services": stats.get("services", {}),
        },
        "components": snapshot["components"],
        "index_generations": snapshot["index_generations"],
    }))


@app.get("/health/live")
async def liveness_check():
    """Process liveness remains independent of model and backend availability."""
    return {"status": "alive", "timestamp": time.time()}


@app.get("/health/ready", responses={503: {"description": "Required dependencies are not ready"}})
async def readiness_check():
    """Validated models, open workers and completed active index generations."""
    from ai_service.api.runtime_health import collect_runtime_health
    snapshot = await collect_runtime_health(orchestrator)
    if snapshot["status"] == "healthy":
        return {"status": "ready", "timestamp": time.time(),
                "index_generations": snapshot["index_generations"]}
    return JSONResponse(status_code=503, content={"status": "not_ready",
        "message": "Required screening dependencies are not ready"})


@app.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus text format
    """
    try:
        # Get the Prometheus exporter
        exporter = get_exporter()

        # Update service status metrics based on orchestrator availability
        if orchestrator:
            stats = orchestrator.get_processing_stats()

            # Update success rate if available
            total_requests = stats.get("total_processed", 0)
            successful_requests = stats.get("successful", 0)
            if total_requests > 0:
                success_rate = successful_requests / total_requests
                exporter.update_success_rate(success_rate)

            # Update cache hit rate if available
            if "cache" in stats:
                cache_hit_rate = stats["cache"].get("hit_rate", 0.0)
                exporter.update_cache_hit_rate(cache_hit_rate)

            # Update active connections estimate (based on service availability)
            if "services" in stats:
                active_services = sum(
                    1 for s in stats["services"].values() if s.get("available", False)
                )
                exporter.update_active_connections(active_services)

        # Return metrics in Prometheus format with correct Content-Type
        metrics_content = exporter.get_metrics().decode("utf-8")
        return Response(
            content=metrics_content, media_type=exporter.get_metrics_content_type()
        )

    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return Response("# Metrics generation failed\nai_service_up 0\n", media_type="text/plain", status_code=503)


@app.post("/process", response_model=ProcessResponse)
async def process_text(request: ProcessTextRequest):
    """
    Complete text processing through orchestrator

    Args:
        request: Text processing request

    Returns:
        Processing result with normalized text, tokens, trace, and optional sections

    Raises:
        HTTPException: 503 if orchestrator not initialized, 500 for internal errors
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Merge feature flags from request with global configuration
        merged_flags = _merge_feature_flags(
            request.options.flags if request.options else None
        )

        # Log feature flags for tracing
        logger.info(f"Processing with feature flags: {merged_flags.to_dict()}")

        result = await orchestrator.process(
            text=request.text,
            generate_variants=request.generate_variants,
            generate_embeddings=request.generate_embeddings,
            # Normalization flags from updated spec
            remove_stop_words=True,
            preserve_names=True,
            enable_advanced_features=True,
            # Pass feature flags to orchestrator
            feature_flags=merged_flags,
            cache_result=request.cache_result,
        )
        if not result.success and orchestrator.enable_search:
            raise ServiceUnavailableError("Screening could not complete; no clearance decision is available")
        if not result.success:
            raise HTTPException(status_code=500, detail="Processing failed")

        # Note: Feature flags are logged separately, not added to trace
        # as trace should only contain TokenTrace objects

        # Convert to ProcessResponse model
        return _processing_response(result, generate_variants=request.generate_variants,
                                    generate_embeddings=request.generate_embeddings)
    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e}")
        raise HTTPException(status_code=503, detail="Screening could not complete; no clearance decision is available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")


@app.post("/normalize", response_model=NormalizationResponse)
async def normalize_text(request: TextNormalizationRequest):
    """
    Normalize text for search without running screening

    Args:
        request: Text normalization request

    Returns:
        Normalized text result with tokens and trace

    Raises:
        HTTPException: 503 if orchestrator not initialized, 500 for internal errors
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Debug logging
        logger.debug("Normalizing text of length %d", len(request.text))
        logger.info(f"Orchestrator initialized: {orchestrator is not None}")

        # Merge feature flags from request with global configuration
        merged_flags = _merge_feature_flags(
            request.options.flags if request.options else None
        )

        # Log feature flags for tracing
        logger.info(f"Normalizing with feature flags: {merged_flags.to_dict()}")

        if request.apply_stemming:
            raise HTTPException(status_code=422, detail="Stemming is not supported for personal names; use apply_lemmatization")
        # Use unified orchestrator for normalization only
        result = await orchestrator.process(
            text=request.text,
            generate_variants=False,
            generate_embeddings=False,
            # Use request parameters for normalization flags
            remove_stop_words=request.remove_stop_words,
            preserve_names=request.preserve_names,
            enable_advanced_features=request.apply_lemmatization,
            # Pass feature flags to orchestrator
            feature_flags=merged_flags,
            language_hint=None if request.language == "auto" else request.language,
            screen=False,
            clean_unicode=request.clean_unicode,
        )
        if not result.success:
            raise HTTPException(status_code=500, detail="Text normalization failed")

        # Note: Feature flags are logged separately, not added to trace
        # as trace should only contain TokenTrace objects

        # Debug logging
        logger.info(
            f"Result: success={result.success}, tokens={result.tokens}, language={result.language}"
        )
        logger.debug(f"Result type: {type(result)}")
        logger.debug(f"Result attributes: {dir(result)}")
        logger.debug(f"Result normalized_text: {result.normalized_text}")
        logger.debug(f"Result tokens: {result.tokens}")
        logger.debug(f"Result language: {result.language}")

        return NormalizationResponse(
            normalized_text=result.normalized_text,
            tokens=result.tokens,
            trace=result.trace,
            language=result.language,
            success=result.success,
            errors=result.errors,
            processing_time=result.processing_time,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        raise HTTPException(
            status_code=500, detail="Text normalization failed"
        )


@app.post("/process-batch", response_model=ProcessBatchResponse)
async def process_batch(request: ProcessBatchRequest):
    """Batch text processing through orchestrator"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        results = await orchestrator.process_batch(
            texts=request.texts,
            generate_variants=request.generate_variants,
            generate_embeddings=request.generate_embeddings,
            max_concurrent=request.max_concurrent,
        )

        if len(results) != len(request.texts):
            raise RuntimeError("Batch processing did not preserve submitted rows")
        processed_results = []
        for text, result in zip(request.texts, results):
            if result.original_text != text:
                raise RuntimeError("Batch processing changed source-row association")
            response = _processing_response(result, generate_variants=request.generate_variants,
                                            generate_embeddings=request.generate_embeddings)
            processed_results.append(ProcessBatchItem(**response.model_dump(), original_text=text,
                language_confidence=result.language_confidence if result.success else 0.0,
                variants_count=len(response.variants or [])))

        return {
            "results": processed_results,
            "total_texts": len(request.texts),
            "successful": sum(1 for r in results if r.success),
            "total_processing_time": sum(r.processing_time for r in results),
        }
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/search-similar")
async def search_similar_names(request: SearchSimilarRequest):
    """Search for similar names"""
    if not orchestrator:

        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        result = await orchestrator.search_similar_names(
            query=request.query,
            candidates=request.candidates,
            threshold=request.threshold,
            top_k=request.top_k,
            use_embeddings=request.use_embeddings,
        )

        return result
    except InferenceUnavailableError:
        raise HTTPException(status_code=503, detail="Embedding service temporarily unavailable")
    except Exception as e:
        logger.error(f"Error searching similar names: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/analyze-complexity")
async def analyze_complexity(request: ComplexityAnalysisRequest):
    """Text complexity analysis"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        result = await orchestrator.analyze_text_complexity(request.text)
        return result
    except Exception as e:
        logger.error(f"Error analyzing complexity: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/search")
async def search_names(request: SearchRequest):
    """Run the complete normalization and sanctions-search pipeline."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    if not getattr(orchestrator, "search_service", None):
        raise HTTPException(status_code=503, detail="Search service not available")

    try:
        search_options = SearchOpts(
            top_k=request.top_k,
            threshold=request.threshold,
            search_mode=request.search_mode,
            enable_escalation=request.enable_escalation,
        )
        result = await orchestrator.process(
            text=request.query,
            generate_variants=False,
            generate_embeddings=False,
            search_options=search_options,
            force_full_pipeline=True,
        )

        if not result.success:
            raise HTTPException(status_code=503, detail="Screening could not complete; no clearance decision is available")
        payload = result.search_results or {
            "query": request.query,
            "results": [],
            "total_hits": 0,
            "search_type": request.search_mode.value,
            "processing_time_ms": result.processing_time * 1000,
        }
        payload["normalized_query"] = result.normalized_text
        payload["success"] = result.success
        payload["errors"] = result.errors or []
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching names: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/stats")
async def get_statistics():
    """Get service operation statistics"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        stats = orchestrator.get_processing_stats()
        return {
            "processing": {
                "total_processed": stats["total_processed"],
                "successful": stats["successful"],
                "failed": stats["failed"],
                "success_rate": (
                    stats["successful"] / stats["total_processed"]
                    if stats["total_processed"] > 0
                    else 0
                ),
                "average_processing_time": stats["average_time"],
            },
            "cache": stats.get("cache", {}),
            "services": stats.get("services", {}),
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/clear-cache")
async def clear_cache(token: str = Depends(verify_admin_token)):
    """Clear cache - Admin only"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        orchestrator.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/reset-stats")
async def reset_statistics(token: str = Depends(verify_admin_token)):
    """Reset statistics - Admin only"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        orchestrator.reset_stats()
        return {"message": "Statistics reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/reload-config")
async def reload_configuration(token: str = Depends(verify_admin_token)):
    """Report the required restart instead of claiming an unapplied reload."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    raise HTTPException(
        status_code=409,
        detail="Configuration is loaded at startup. Apply settings and recreate the API service.",
    )


@app.get("/config-status")
async def get_configuration_status(token: str = Depends(verify_admin_token)):
    """Get configuration status - Admin only"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    return {
        "search_service": {
            "enabled": getattr(orchestrator, "search_service", None) is not None,
            "hot_reload": False,
            "reload_stats": {},
            "change_application": "restart_required",
        }
    }


@app.post("/validate-config")
async def validate_configuration(token: str = Depends(verify_admin_token)):
    """Validate current configuration - Admin only"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    service = getattr(orchestrator, "search_service", None)
    result = {
        "enabled": service is not None,
        "validation_passed": False,
        "runtime_ready": False,
        "errors": [],
        "warnings": [],
    }
    if service is None:
        return {"search_service": result}

    try:
        # Reconstruct nested models: validating an existing Pydantic instance
        # can otherwise accept fields mutated after its initial validation.
        if not isinstance(service.config, HybridSearchConfig):
            raise TypeError("Unsupported search configuration")
        HybridSearchConfig.model_validate(service.config.model_dump(mode="python"))
    except ValidationError as exc:
        # Values and exception contexts may contain credentials. Expose only
        # field locations and machine-readable validation codes.
        result["errors"] = [
            f"Invalid configuration at {'.'.join(map(str, error['loc'])) or 'root'}: {error['type']}"
            for error in exc.errors(include_input=False, include_context=False, include_url=False)
        ]
    except Exception as exc:
        logger.error("Configuration validation failed (%s)", type(exc).__name__)
        result["errors"].append("Search configuration could not be validated")
    else:
        result["validation_passed"] = True
        try:
            # Cluster connectivity alone cannot prove that the configured
            # sanctions indices are usable and belong to one loaded generation.
            await service.readiness()
            result["runtime_ready"] = True
        except Exception as exc:
            logger.warning("Configuration readiness check failed (%s)", type(exc).__name__)
            result["warnings"].append("Configured search indices are not ready; inspect service diagnostics")
    return {"search_service": result}


@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    return {
        "supported_languages": {
            "en": {"supported": True, "name": "English"},
            "ru": {"supported": True, "name": "Russian"},
            "uk": {"supported": True, "name": "Ukrainian"},
        },
        "auto_detection": True,
        "fallback_language": "en",
    }


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Multilingual Sanctions Search",
        "version": __version__,
        "description": "Multilingual normalization and hybrid search for sanctions screening",
        "implementation": "full_with_orchestrator",
        "orchestrator": orchestrator is not None,
        "endpoints": {
            "health": "/health",
            "health_detailed": "/health/detailed",
            "health_live": "/health/live",
            "health_ready": "/health/ready",
            "metrics": "/metrics",
            "process": "/process",
            "process_batch": "/process-batch",
            "search": "/search",
            "search_similar": "/search-similar",
            "analyze_complexity": "/analyze-complexity",
            "stats": "/stats",
            "clear_cache": "/clear-cache",
            "reset_stats": "/reset-stats",
            "normalize": "/normalize",
            "languages": "/languages",
        },
        "features": [
            "Text normalization",
            "Variant generation",
            "Signal detection",
            "Hybrid search",
            "Similarity search",
            "Complexity analysis",
            "Batch processing",
            "Caching",
            "Multi-language support",
            "Real-time statistics",
        ],
    }


# Admin endpoints
@app.get("/admin/status")
async def admin_status(
    token: str = Depends(verify_admin_token),
):
    """Admin status endpoint with authentication"""
    # Get orchestrator statistics
    stats = {
        "total_processed": 0,
        "successful": 0,
        "failed": 0,
        "cache": {"hits": 0, "misses": 0},
    }

    detailed_stats = stats.copy()

    if orchestrator:
        # Get basic stats from orchestrator if available
        stats.update(
            {
                "orchestrator_initialized": True,
                "processing_time": 0.0,  # Simple value to avoid recursion
            }
        )

        # Try to get detailed stats if method exists
        if hasattr(orchestrator, "get_detailed_stats"):
            try:
                detailed_stats = orchestrator.get_detailed_stats()
            except Exception:
                pass  # Use default stats if method fails
    else:
        stats["orchestrator_initialized"] = False

    return {
        "status": "operational",
        "version": __version__,
        "timestamp": time.time(),
        "statistics": stats,
        "detailed_stats": detailed_stats,
    }


# Exception handlers
@app.exception_handler(AuthenticationError)
async def authentication_exception_handler(request, exc):
    """Handle authentication errors"""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(ValidationAPIError)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(ServiceUnavailableError)
async def service_unavailable_exception_handler(request, exc):
    """Dependency messages can contain private connection details."""
    logger.error("Required service unavailable", exc_info=exc)
    return JSONResponse(status_code=503, content={"detail": "Service temporarily unavailable"})


@app.exception_handler(InternalServerError)
async def internal_server_exception_handler(request, exc):
    """Keep private exception payloads inside the protected diagnostic boundary."""
    logger.error("Internal request failure", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return the published schema while excluding private parser fields."""
    response = RequestValidationResponse(errors=[
        RequestValidationIssue(loc=error["loc"], msg=error["msg"], type=error["type"])
        for error in exc.errors()
    ])
    return JSONResponse(
        status_code=422,
        content=response.model_dump(mode="json"),
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """Validation outside request parsing is an internal failure, not caller data."""
    logger.error("Internal schema validation failed", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled request failure", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=DEPLOYMENT_CONFIG.host,
        port=DEPLOYMENT_CONFIG.port,
        reload=DEPLOYMENT_CONFIG.auto_reload,
        log_level=DEPLOYMENT_CONFIG.log_level,
    )
