"""Join imported vectors to AC source records and verify complete coverage.

Scratch rows live in a private temporary SQLite database, not an unbounded Python
dictionary. Callers hold both index writer locks for this object's lifetime.
"""

from contextlib import aclosing
import hashlib
import json
import os
import sqlite3
import tempfile

from pydantic import BaseModel, ConfigDict, Field

from ..layers.search.index_schema import index_mapping, validate_mapping
from ..layers.search.search_integrity import require_complete_response


SOURCE_FIELDS = (
    "pattern", "normalized_text", "original_text", "name", "entity_id",
    "entity_type", "aliases", "tier", "category", "source_list", "confidence",
    "metadata", "country", "dob",
)


class VerificationLimits(BaseModel):
    model_config = ConfigDict(validate_default=True)
    timeout_seconds: float = Field(default_factory=lambda: float(os.getenv(
        "INGESTION_VERIFY_TIMEOUT_SECONDS", "300")), gt=0, le=3600)


def source_key(document):
    return json.dumps([document.get(key) for key in
        ("source_list", "category", "entity_type", "entity_id", "pattern")],
        ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def source_payload(document):
    return json.dumps({key: document.get(key) for key in SOURCE_FIELDS},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


class SnapshotVerifier:
    def __init__(self, client, config):
        self.client = client.options(request_timeout=config.elasticsearch.timeout, max_retries=0)
        self.config = config
        self.limits = VerificationLimits()
        self.scratch = None
        self.database = None
        self.generation = None
        self.source_manifest = None

    def __enter__(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="snapshot-verify-",
            dir=os.getenv("APP_STATE_DIR") or None)
        try:
            self.database = sqlite3.connect(os.path.join(self.scratch.name, "rows.sqlite3"))
            self.database.execute("PRAGMA journal_mode=OFF")
            self.database.execute("PRAGMA synchronous=OFF")
            self.database.execute("CREATE TABLE ac (id TEXT PRIMARY KEY, source_key TEXT, signature TEXT, document TEXT)")
            self.database.execute("CREATE INDEX ac_source_key ON ac(source_key)")
            self.database.execute("CREATE TABLE vectors (id TEXT PRIMARY KEY, signature TEXT)")
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        if self.database is not None:
            self.database.close()
        if self.scratch is not None:
            self.scratch.cleanup()

    async def records(self, index):
        opened = await self.client.open_point_in_time(index=index, keep_alive="1m",
                                                       allow_partial_search_results=False)
        pit_id = opened["id"]
        try:
            if opened.get("_shards", {}).get("failed", 0):
                raise RuntimeError("Cannot verify a partial source snapshot")
            after = None
            while True:
                body = {"pit": {"id": pit_id, "keep_alive": "1m"},
                        "query": {"match_all": {}}, "size": 1000,
                        "sort": ["_shard_doc"], "track_total_hits": False,
                        "_source": list(SOURCE_FIELDS)}
                if after is not None:
                    body["search_after"] = after
                response = await self.client.search(body=body, allow_partial_search_results=False)
                pit_id = response.get("pit_id", pit_id)
                require_complete_response(response)
                hits = response["hits"]["hits"]
                if not hits:
                    return
                next_page = hits[-1].get("sort")
                if not next_page or next_page == after:
                    raise RuntimeError("Invalid snapshot verification pagination")
                after = next_page
                yield hits
        finally:
            await self.client.options(request_timeout=2, max_retries=0).close_point_in_time(id=pit_id)

    async def prepare(self, vector_documents):
        """Bind each vector to its existing AC identity, name and source metadata."""
        index = self.config.elasticsearch.ac_index
        mappings = await self.client.indices.get_mapping(index=index)
        if len(mappings) != 1:
            raise RuntimeError("Vector ingestion requires one concrete AC index")
        mapping = next(iter(mappings.values()))["mappings"]
        validate_mapping(mapping, index_mapping(self.config)["mappings"])
        meta = mapping.get("_meta", {})
        if meta.get("ingestion_status") != "completed" or not meta.get("generation"):
            raise RuntimeError("AC ingestion must complete before importing vectors")
        self.generation = meta["generation"]
        self.source_manifest = meta.get("source_manifest")
        async with aclosing(self.records(index)) as pages:
            async for page in pages:
                rows = []
                for hit in page:
                    document = hit["_source"]
                    payload = source_payload(document)
                    rows.append((hit["_id"], source_key(document),
                                 hashlib.sha256(payload.encode()).hexdigest(), payload))
                self.database.executemany("INSERT INTO ac VALUES (?, ?, ?, ?)", rows)
        if not self.database.execute("SELECT COUNT(*) FROM ac").fetchone()[0]:
            raise RuntimeError("AC source snapshot is empty")
        bound_documents = {}
        for _, vector in vector_documents:
            matches = self.database.execute("SELECT id, document FROM ac WHERE source_key=?",
                                            (source_key(vector),)).fetchall()
            if not matches:
                raise ValueError("Imported vector has no matching AC source record")
            for doc_id, raw in matches:
                source = json.loads(raw)
                source.update({key: vector[key] for key in ("vector", "model_name", "embedding_contract")})
                bound_documents[doc_id] = source
        return list(bound_documents.items())

    async def coverage(self):
        """Require source fields for every AC row, with no stale/foreign vectors."""
        async with aclosing(self.records(self.config.elasticsearch.vector_index)) as pages:
            async for page in pages:
                rows = [(hit["_id"], hashlib.sha256(source_payload(hit["_source"]).encode()).hexdigest())
                        for hit in page]
                self.database.executemany("INSERT INTO vectors VALUES (?, ?)", rows)
        missing = self.database.execute(
            "SELECT COUNT(*) FROM ac LEFT JOIN vectors USING(id) "
            "WHERE vectors.id IS NULL OR ac.signature != vectors.signature"
        ).fetchone()[0]
        extra = self.database.execute(
            "SELECT COUNT(*) FROM vectors LEFT JOIN ac USING(id) WHERE ac.id IS NULL"
        ).fetchone()[0]
        # Defense against an unsupported writer bypassing the shared local locks.
        mapping = await self.client.indices.get_mapping(index=self.config.elasticsearch.ac_index)
        metas = [entry.get("mappings", {}).get("_meta", {}) for entry in mapping.values()]
        if (len(metas) != 1 or metas[0].get("generation") != self.generation
                or metas[0].get("ingestion_status") != "completed"
                or metas[0].get("source_manifest") != self.source_manifest):
            raise RuntimeError("AC source changed during vector ingestion")
        return {"snapshot_ready": missing == extra == 0,
                "missing_or_changed_vectors": missing, "extra_vectors": extra}
