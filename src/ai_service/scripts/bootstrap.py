"""Create configured indices or explicitly replace a complete sanctions snapshot."""

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import ExitStack

from ..api.elasticsearch_wrapper import ElasticsearchClient
from ..api.ingestion_jobs import IngestionJobStore
from ..config import EmbeddingConfig
from ..layers.embeddings.embedding_service import EmbeddingService
from ..layers.search.config import HybridSearchConfig
from ..layers.search.index_schema import (
    bulk_counts,
    ensure_index,
    pattern_document,
    set_ingestion_status,
    vector_document,
)
from ..layers.search.sanctions_data_loader import SanctionsDataLoader

logger = logging.getLogger(__name__)


def snapshot_documents(dataset):
    documents = {}
    for entry in dataset.persons + dataset.organizations:
        metadata = dict(entry.metadata or {})
        source_id = next(
            (
                metadata[key]
                for key in ("source_id", "person_id", "number_entry")
                if metadata.get(key) is not None
            ),
            None,
        )
        if source_id is None:
            raise ValueError("Source entry has no stable identifier")
        metadata.update(
            {
                "source": entry.source,
                "list_name": entry.list_name,
                "dob": entry.birth_date,
                "nationality": entry.nationality,
            }
        )
        category = "company" if entry.entity_type == "organization" else "person"
        for name in dict.fromkeys([entry.name, *entry.aliases]):
            doc_id, doc = pattern_document(
                {
                    "pattern": name,
                    "canonical": entry.name,
                    "entity_id": str(source_id),
                    "entity_type": entry.entity_type,
                    "source_list": entry.source,
                    "metadata": metadata,
                    "aliases": entry.aliases,
                },
                category,
                0,
            )
            documents[doc_id] = doc
    return list(documents.items())


async def bootstrap(
    *, ingest=False, vectors=False, data_dir=None, batch_size=128, replace=False
):
    if not 1 <= batch_size <= 5000:
        raise ValueError("batch-size must be between 1 and 5000")
    if (vectors or replace) and not ingest:
        raise ValueError("--vectors and --replace require --ingest")
    config = HybridSearchConfig.from_env()
    indices = (config.elasticsearch.ac_index, config.elasticsearch.vector_index)
    async with ElasticsearchClient() as wrapper:
        client = wrapper.client
        deadline = time.monotonic() + float(os.getenv("ES_STARTUP_TIMEOUT", "60"))
        while True:
            try:
                await client.options(request_timeout=5, max_retries=0).info()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Elasticsearch startup deadline exceeded")
                await asyncio.sleep(1)
        for index in indices:
            await ensure_index(client, index, config, vectors=index == indices[1])
        if not ingest:
            print(
                "Configured indices exist. Empty indices remain unready until ingestion completes."
            )
            return
        dataset = await SanctionsDataLoader(data_dir).load_dataset(force_reload=True)
        if not dataset.total_entries:
            raise ValueError("Cannot ingest an empty sanctions snapshot")
        documents = snapshot_documents(dataset)
        encoder = EmbeddingService(EmbeddingConfig()) if vectors else None
        store = IngestionJobStore()
        jobs = {}
        generation = str(uuid.uuid4())
        changed = False
        with ExitStack() as locks:
            try:
                for index in sorted(indices):
                    jobs[index] = store.reserve(
                        "snapshot_vectors" if index == indices[1] else "snapshot_ac",
                        index,
                        len(documents),
                    )
                    locks.callback(jobs[index].close)
                counts = [
                    (await client.count(index=index))["count"] for index in indices
                ]
                if any(counts) and not replace:
                    raise ValueError(
                        "Existing snapshot is nonempty; use --replace to remove obsolete records"
                    )
                # Both locks are held, so an AC-only replacement also invalidates old vectors.
                for index in indices:
                    await set_ingestion_status(client, index, "loading", generation)
                    jobs[index].update(status="loading")
                changed = True
                if replace:
                    for index, count in zip(indices, counts):
                        if count:
                            response = await client.delete_by_query(
                                index=index,
                                query={"match_all": {}},
                                refresh=True,
                                wait_for_completion=True,
                                conflicts="abort",
                            )
                            if response.get("failures") or response.get("timed_out"):
                                raise RuntimeError(
                                    "Could not remove the previous snapshot completely"
                                )
                total = 0
                for start in range(0, len(documents), batch_size):
                    batch = documents[start : start + batch_size]
                    operations = []
                    for doc_id, doc in batch:
                        operations.extend(
                            [{"index": {"_index": indices[0], "_id": doc_id}}, doc]
                        )
                    _, failed = bulk_counts(
                        await client.bulk(operations=operations), len(batch)
                    )
                    if failed:
                        raise RuntimeError(f"AC ingestion rejected {failed} documents")
                    jobs[indices[0]].update(progress=start + len(batch))
                    if encoder is not None:
                        embeddings = await asyncio.to_thread(
                            encoder.encode_batch, [doc["pattern"] for _, doc in batch]
                        )
                        if len(embeddings) != len(batch):
                            raise RuntimeError(
                                "Embedding generation did not preserve source rows"
                            )
                        operations = []
                        for (doc_id, doc), embedding in zip(batch, embeddings):
                            _, vector_doc = vector_document(
                                {
                                    **doc,
                                    "name": doc["pattern"],
                                    "vector": embedding,
                                    "embedding_contract": encoder.embedding_contract,
                                },
                                doc["category"],
                                encoder.config.model_name,
                            )
                            # The source row is authoritative, including alias canonical
                            # fields and tier. Only embedding fields come from validation.
                            vector_doc = {**doc, **{key: vector_doc[key] for key in (
                                "vector", "model_name", "embedding_contract")}}
                            if config.vector_search.vector_field != "vector":
                                vector_doc[config.vector_search.vector_field] = (
                                    vector_doc.pop("vector")
                                )
                            operations.extend(
                                [
                                    {"index": {"_index": indices[1], "_id": doc_id}},
                                    vector_doc,
                                ]
                            )
                        _, failed = bulk_counts(
                            await client.bulk(operations=operations), len(batch)
                        )
                        if failed:
                            raise RuntimeError(
                                f"Vector ingestion rejected {failed} documents"
                            )
                        jobs[indices[1]].update(progress=start + len(batch))
                    total += len(batch)
                    print(
                        json.dumps({"indexed": total, "total": len(documents)}),
                        flush=True,
                    )
                await client.indices.refresh(index=list(indices))
                for index in indices:
                    status = "completed" if index == indices[0] or vectors else "empty"
                    await set_ingestion_status(
                        client,
                        index,
                        status,
                        generation,
                        source_manifest=dataset.source_manifest,
                        source_coverage_verified=index == indices[1] and vectors,
                    )
                    jobs[index].update(status=status)
                print(
                    json.dumps(
                        {
                            "status": "completed",
                            "records": dataset.total_entries,
                            "documents": total,
                            "source_manifest": dataset.source_manifest,
                        }
                    )
                )
            except BaseException as exc:
                for index, job in jobs.items():
                    job.update(status="error", error=f"Snapshot ingestion failed ({type(exc).__name__})")
                    if changed:
                        try:
                            await set_ingestion_status(
                                client, index, "failed", generation
                            )
                        except Exception:
                            logger.exception("Could not record failed snapshot status")
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--vectors", action="store_true")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Remove the previous snapshot under exclusive locks",
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 5000:
        parser.error("batch-size must be between 1 and 5000")
    if (args.vectors or args.replace) and not args.ingest:
        parser.error("--vectors and --replace require --ingest")
    asyncio.run(bootstrap(**vars(args)))


if __name__ == "__main__":
    main()
