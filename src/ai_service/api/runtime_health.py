"""One dependency contract for public health, readiness and admin diagnostics."""
import asyncio


# The container probe gives HTTP four seconds and Docker five seconds.
# Dependency diagnostics must finish within that budget, including cancellation.
HEALTH_TIMEOUT_SECONDS = 3.0


def _unhealthy():
    return {"status": "unhealthy", "error": "Required dependency is unavailable"}


def _local_health(service):
    try:
        health = service.runtime_health_check()
        if not isinstance(health, dict) or health.get("status") not in {"healthy", "unhealthy"}:
            return _unhealthy()
        return health
    except Exception:
        return _unhealthy()


async def initialize_runtime_models(runtime):
    """Warm the actual providers used by this orchestrator before publishing it."""
    # The transformer loader has a transient allocation peak. Load it before
    # retaining the three language pipelines, then verify NER on the same workers.
    if runtime.enable_embeddings:
        await runtime.embeddings_service.initialize_runtime()
        if runtime.search_service is not None:
            runtime.search_service._embedding_service = runtime.embeddings_service
    await runtime.normalization_service.initialize_runtime()
    for name, enabled in (("normalization_service", True),
                          ("embeddings_service", runtime.enable_embeddings),
                          ("variants_service", runtime.enable_variants)):
        if enabled and _local_health(getattr(runtime, name, None))["status"] != "healthy":
            raise RuntimeError("Required model worker is not ready")


async def _search_health(runtime):
    try:
        async with asyncio.timeout(HEALTH_TIMEOUT_SECONDS):
            service = runtime.search_service
            # Readiness initializes adapters and verifies the completed active
            # snapshot; a connected server alone is insufficient.
            generations = await service.readiness(require_vectors=runtime.enable_embeddings)
            count = 2 if runtime.enable_embeddings else 1
            if (not isinstance(generations, dict) or len(generations) != count
                    or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                           for k, v in generations.items())
                    or len(set(generations.values())) != 1):
                return _unhealthy(), {}
            health = await service.health_check()
            if (not isinstance(health, dict) or health.get("status") != "healthy"
                    or health.get("connected") is False):
                return _unhealthy(), {}
            return health, generations
    except Exception:
        return _unhealthy(), {}


async def collect_runtime_health(runtime):
    if runtime is None:
        return {"status": "initializing", "components": {}, "index_generations": {}}
    components = {}
    # The constructors own initialization of these lightweight services. Missing
    # providers are failures; their presence is not a recognition-quality claim.
    required = ["validation_service", "language_service", "unicode_service", "signals_service"]
    for flag, name in (("enable_smart_filter", "smart_filter_service"),
                       ("enable_decision_engine", "decision_engine")):
        if getattr(runtime, flag, False):
            required.append(name)
    components["core_services"] = {
        "status": "healthy" if all(getattr(runtime, name, None) is not None for name in required)
        else "unhealthy"
    }
    generations = {}
    if getattr(runtime, "enable_search", False):
        components["search_service"], generations = await _search_health(runtime)
    else:
        components["search_service"] = {"status": "disabled"}
    # Read worker state after the last await: a worker can close while the
    # backend probe is in flight.
    for name, key, enabled in (
        ("normalization_service", "normalization", True),
        ("embeddings_service", "embedding_inference", getattr(runtime, "enable_embeddings", False)),
        ("variants_service", "variant_generation", getattr(runtime, "enable_variants", False)),
    ):
        components[key] = _local_health(getattr(runtime, name, None)) if enabled else {"status": "disabled"}
    ready = all(part["status"] in {"healthy", "disabled"} for part in components.values())
    return {"status": "healthy" if ready else "unhealthy", "components": components,
            "index_generations": generations}
