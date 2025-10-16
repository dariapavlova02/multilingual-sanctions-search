"""
Admin API endpoints for data loading and management.
"""

import json
import logging
import asyncio
from contextlib import AsyncExitStack
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .elasticsearch_wrapper import ElasticsearchClient
from .ingestion_jobs import IngestionJobStore, IngestionBusy
from .snapshot_verification import SnapshotVerifier
from ..layers.search.config import HybridSearchConfig
from ..layers.search.index_schema import ensure_index, pattern_document, vector_document, bulk_counts, set_ingestion_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
VALID_CATEGORIES = {"person", "company", "terrorism"}

# Request/Response models
class ACPatternsBulkRequest(BaseModel):
    """Request model for bulk AC patterns loading."""
    patterns: List[Dict[str, Any]] = Field(
        ..., min_length=1, max_length=50_000, description="List of AC patterns to load"
    )
    category: str = Field(..., description="Pattern category (person/company/terrorism)")
    tier: str = Field(..., description="Pattern tier (tier_0_exact, tier_1_high, etc.)")
    batch_size: int = Field(default=1000, ge=1, le=5000, description="Bulk indexing batch size")

class VectorsBulkRequest(BaseModel):
    """Request model for bulk vectors loading."""
    vectors: List[Dict[str, Any]] = Field(
        ..., min_length=1, max_length=50_000, description="List of name-vector pairs"
    )
    category: str = Field(..., description="Vector category (person/company)")
    model_name: str = Field(..., description="Embedding model name")
    batch_size: int = Field(default=500, ge=1, le=5000, description="Bulk indexing batch size")

class LoadingStatusResponse(BaseModel):
    """Response model for loading status."""
    success: bool
    message: str
    loaded_count: int
    errors: List[str] = []
    processing_time: float
    job_id: Optional[str] = None

# Global loading status tracking
loading_status = {
    "ac_patterns": {"status": "idle", "progress": 0, "total": 0},
    "vectors": {"status": "idle", "progress": 0, "total": 0}
}

def _queue_job(background_tasks, function, *args, kind, total):
    if total < 1:
        raise HTTPException(status_code=400, detail="Ingestion must contain at least one document")
    config = HybridSearchConfig.from_env()
    index = config.elasticsearch.vector_index if kind == "vectors" else config.elasticsearch.ac_index
    try:
        related = [config.elasticsearch.ac_index] if kind == "vectors" else []
        job = IngestionJobStore().reserve(kind, index, total, related_indices=related)
    except IngestionBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(function, *args, job=job)
    return job.job_id


@router.post("/ac-patterns/bulk", response_model=LoadingStatusResponse)
async def load_ac_patterns_bulk(
    background_tasks: BackgroundTasks,
    request: ACPatternsBulkRequest
):
    """
    Load AC patterns in bulk to Elasticsearch.

    Supports loading patterns by categories and tiers with configurable batch sizes.
    """
    try:
        # Validate category and tier
        valid_categories = sorted(VALID_CATEGORIES)
        valid_tiers = ["tier_0_exact", "tier_1_high", "tier_2_medium", "tier_3_low", "tier_4_experimental"]

        if request.category not in valid_categories:
            raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of: {valid_categories}")

        if request.tier not in valid_tiers:
            raise HTTPException(status_code=400, detail=f"Invalid tier. Must be one of: {valid_tiers}")

        # Start background loading
        job_id = _queue_job(
            background_tasks, _load_ac_patterns_background, request.patterns,
            request.category, request.tier, request.batch_size,
            kind="ac_patterns", total=len(request.patterns),
        )

        return LoadingStatusResponse(
            success=True,
            message=f"Started loading {len(request.patterns)} AC patterns for {request.category}/{request.tier}",
            loaded_count=0,
            job_id=job_id,
            processing_time=0.0
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start AC patterns loading: {e}")
        raise HTTPException(status_code=500, detail="Failed to start loading")

@router.post("/vectors/bulk", response_model=LoadingStatusResponse)
async def load_vectors_bulk(
    background_tasks: BackgroundTasks,
    request: VectorsBulkRequest
):
    """
    Load name vectors in bulk to Elasticsearch.

    Supports loading pre-computed embeddings for semantic search.
    """
    try:
        # Validate category
        valid_categories = ["person", "company"]

        if request.category not in valid_categories:
            raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of: {valid_categories}")

        # Start background loading
        job_id = _queue_job(
            background_tasks, _load_vectors_background, request.vectors,
            request.category, request.model_name, request.batch_size,
            kind="vectors", total=len(request.vectors),
        )

        return LoadingStatusResponse(
            success=True,
            message=f"Started loading {len(request.vectors)} vectors for {request.category}",
            loaded_count=0,
            job_id=job_id,
            processing_time=0.0
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start vectors loading: {e}")
        raise HTTPException(status_code=500, detail="Failed to start loading")

@router.post("/ac-patterns/upload")
async def upload_ac_patterns_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(...),
    batch_size: int = Form(default=1000, ge=1, le=5000)
):
    """
    Upload AC patterns file and load to Elasticsearch.

    Accepts JSON files with AC patterns structure.
    """
    try:
        # Validate file type
        if not (file.filename or "").endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be a JSON file")

        # Read and parse file
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds the 25 MiB upload limit")
        patterns_data = json.loads(content)

        # Validate structure
        if not isinstance(patterns_data, dict):
            raise HTTPException(status_code=400, detail="File must contain a JSON object with tier data")

        if category not in VALID_CATEGORIES:
            raise HTTPException(status_code=400, detail="Invalid category")

        total_patterns = sum(len(patterns) for patterns in patterns_data.values() if isinstance(patterns, list))

        # Start background loading for all tiers
        job_id = _queue_job(
            background_tasks, _load_ac_patterns_file_background, patterns_data,
            category, batch_size, kind="ac_patterns", total=total_patterns,
        )

        return LoadingStatusResponse(
            success=True,
            message=f"Started loading {total_patterns} patterns from {file.filename}",
            loaded_count=0,
            job_id=job_id,
            processing_time=0.0
        )

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        logger.error(f"Failed to upload AC patterns file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file")

@router.post("/vectors/upload")
async def upload_vectors_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(...),
    model_name: str = Form(...),
    batch_size: int = Form(default=500, ge=1, le=5000)
):
    """
    Upload vectors file and load to Elasticsearch.

    Accepts JSON files with name-vector pairs.
    """
    try:
        # Validate file type
        if not (file.filename or "").endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be a JSON file")

        # Read and parse file
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds the 25 MiB upload limit")
        vectors_data = json.loads(content)

        # Validate structure
        if not isinstance(vectors_data, list):
            raise HTTPException(status_code=400, detail="File must contain a JSON array of vectors")
        if category not in {"person", "company"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        # Start background loading
        job_id = _queue_job(
            background_tasks, _load_vectors_file_background, vectors_data,
            category, model_name, batch_size, kind="vectors", total=len(vectors_data),
        )

        return LoadingStatusResponse(
            success=True,
            message=f"Started loading {len(vectors_data)} vectors from {file.filename}",
            loaded_count=0,
            job_id=job_id,
            processing_time=0.0
        )

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload vectors file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file")

@router.get("/loading-status")
async def get_loading_status():
    """Get current loading status for all operations."""
    return JSONResponse(content={**loading_status, **IngestionJobStore().latest()})

@router.get("/loading-status/{job_id}")
async def get_loading_job(job_id: str):
    status = IngestionJobStore().get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return status


@router.delete("/indices/{index_name}")
async def delete_index(index_name: str):
    """Delete an Elasticsearch index."""
    config = HybridSearchConfig.from_env()
    if index_name not in {config.elasticsearch.ac_index, config.elasticsearch.vector_index}:
        raise HTTPException(status_code=400, detail="Only configured sanctions indices can be deleted")
    try:
        job = IngestionJobStore().reserve("maintenance", index_name, 0)
    except IngestionBusy:
        raise HTTPException(status_code=409, detail="Index has an active ingestion job") from None
    try:
        job.update(status="loading")
        async with ElasticsearchClient() as es_client:
            if not await es_client.client.indices.exists(index=index_name):
                job.update(status="completed", result="already_absent")
                raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
            await es_client.client.indices.delete(index=index_name)
        job.update(status="completed")
        return {"success": True, "message": f"Deleted index: {index_name}"}
    except HTTPException:
        raise
    except Exception:
        job.update(status="error", error="Index deletion failed")
        logger.exception("Failed to delete configured index")
        raise HTTPException(status_code=503, detail="Index deletion failed") from None
    finally:
        job.close()

@router.get("/indices")
async def list_indices():
    """List the configured sanctions indices only."""
    try:
        config = HybridSearchConfig.from_env().elasticsearch
        indices = []
        async with ElasticsearchClient() as es_client:
            for index in (config.ac_index, config.vector_index):
                if await es_client.client.indices.exists(index=index):
                    indices.append(index)
        return {"indices": indices}
    except Exception:
        logger.exception("Failed to list configured indices")
        raise HTTPException(status_code=503, detail="Index listing failed") from None

# Background task functions
async def _load_documents(documents, kind, *, vectors=False, batch_size=1000, job=None):
    if not 1 <= batch_size <= 5000:
        raise ValueError("Batch size must be between 1 and 5000")
    status = {"status": "loading", "progress": 0, "total": len(documents), "failed": 0}
    loading_status[kind] = status
    es_client = None
    try:
        config = HybridSearchConfig.from_env()
        index = config.elasticsearch.vector_index if vectors else config.elasticsearch.ac_index
        if job is None:
            related = [config.elasticsearch.ac_index] if vectors else []
            job = IngestionJobStore().reserve(kind, index, len(documents), related_indices=related)
        status["job_id"] = job.job_id
        job.update(**status)
        es_client = ElasticsearchClient()
        await ensure_index(es_client.client, index, config, vectors=vectors)
        import uuid
        generation = str(uuid.uuid4())
        source_manifest = {"kind": "api_upsert", "generation": generation}
        async with AsyncExitStack() as resources:
            verifier = None
            if vectors:
                verifier = resources.enter_context(SnapshotVerifier(es_client.client, config))
                await resources.enter_async_context(asyncio.timeout(verifier.limits.timeout_seconds))
                documents = await verifier.prepare(documents)
                status["total"] = len(documents)
                generation = verifier.generation
                source_manifest = verifier.source_manifest
            await set_ingestion_status(es_client.client, index, "loading", generation)
            for start in range(0, len(documents), batch_size):
                batch = documents[start:start + batch_size]
                operations = []
                for doc_id, doc in batch:
                    doc = dict(doc)
                    if vectors and config.vector_search.vector_field != "vector":
                        doc[config.vector_search.vector_field] = doc.pop("vector")
                    operations.extend([{"index": {"_index": index, "_id": doc_id}}, doc])
                response = await es_client.client.bulk(operations=operations)
                succeeded, failed = bulk_counts(response, len(batch))
                status["progress"] += succeeded
                status["failed"] += failed
                job.update(**status)
                if failed:
                    raise RuntimeError(f"Bulk indexing rejected {failed} document(s)")
            await es_client.client.indices.refresh(index=index)
            index_status = "completed"
            if verifier is not None:
                status.update(await verifier.coverage())
                if not status["snapshot_ready"]:
                    index_status = "incomplete"
            await set_ingestion_status(es_client.client, index, index_status, generation,
                source_manifest=source_manifest,
                source_coverage_verified=vectors and status.get("snapshot_ready", False))
            status["status"] = "completed"
    except Exception as exc:
        status["status"] = "error"
        status["error"] = f"Ingestion failed ({type(exc).__name__})"
        logger.exception("Sanctions ingestion failed")
    finally:
        try:
            if job is not None:
                job.update(**status)
        finally:
            if job is not None:
                job.close()
            if es_client is not None:
                await es_client.close()


def _record_validation_error(kind, error, job):
    if job is not None:
        try:
            job.update(status="error", error=f"Ingestion validation failed ({type(error).__name__})")
        finally:
            job.close()


async def _load_ac_patterns_background(patterns, category, tier, batch_size, *, job=None):
    try:
        documents = [pattern_document(item, category, tier) for item in patterns]
    except Exception as exc:
        loading_status["ac_patterns"] = {"status": "error", "progress": 0,
                                         "total": len(patterns), "error": f"Ingestion validation failed ({type(exc).__name__})"}
        _record_validation_error("ac_patterns", exc, job)
        return
    await _load_documents(documents, "ac_patterns", batch_size=batch_size, job=job)


async def _load_vectors_background(vectors, category, model_name, batch_size, *, job=None):
    try:
        documents = [vector_document(item, category, model_name) for item in vectors]
    except Exception as exc:
        loading_status["vectors"] = {"status": "error", "progress": 0,
                                     "total": len(vectors), "error": f"Ingestion validation failed ({type(exc).__name__})"}
        _record_validation_error("vectors", exc, job)
        return
    await _load_documents(documents, "vectors", vectors=True, batch_size=batch_size, job=job)


async def _load_ac_patterns_file_background(
    patterns_data: Dict[str, Any],
    category: str,
    batch_size: int,
    *, job=None
):
    """Background task to load AC patterns from file data."""
    try:
        documents = [pattern_document(item, category, tier) for tier, items in patterns_data.items()
                     if isinstance(items, list) for item in items]
    except Exception as exc:
        loading_status["ac_patterns"] = {"status": "error", "progress": 0, "error": f"Ingestion validation failed ({type(exc).__name__})"}
        _record_validation_error("ac_patterns", exc, job)
        return
    await _load_documents(documents, "ac_patterns", batch_size=batch_size, job=job)

async def _load_vectors_file_background(
    vectors_data: List[Dict[str, Any]],
    category: str,
    model_name: str,
    batch_size: int,
    *, job=None
):
    """Background task to load vectors from file data."""
    await _load_vectors_background(vectors_data, category, model_name, batch_size, job=job)

async def _ensure_ac_patterns_index(es_client, index_name):
    await ensure_index(es_client.client, index_name, HybridSearchConfig.from_env())


async def _ensure_vectors_index(es_client, index_name):
    await ensure_index(es_client.client, index_name, HybridSearchConfig.from_env(), vectors=True)


def _tier_to_confidence(tier_name: str) -> float:
    """Convert tier name to confidence score."""
    tier_mapping = {
        'tier_0_exact': 1.0,
        'tier_1_high': 0.9,
        'tier_2_medium': 0.7,
        'tier_3_low': 0.5,
        'tier_4_experimental': 0.3
    }
    return tier_mapping.get(tier_name, 0.5)
