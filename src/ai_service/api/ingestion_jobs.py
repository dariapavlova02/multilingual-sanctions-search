"""Durable ingestion status and per-index locks for a shared local state volume.

The supported deployment uses one API replica. Multiple worker processes sharing
APP_STATE_DIR also coordinate through advisory file locks. Separate hosts need a
distributed job runner before they may write the same index.
"""

import fcntl
import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class IngestionBusy(RuntimeError):
    pass


class JobReservation:
    def __init__(self, store, job_id, lock, related_locks=()):
        self.store, self.job_id, self.lock = store, job_id, lock
        self.related_locks = list(related_locks)

    def update(self, **values):
        values.pop("job_id", None)
        self.store.update(self.job_id, **values)

    def close(self):
        if self.lock is not None:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None
        for lock in self.related_locks:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        self.related_locks.clear()


class IngestionJobStore:
    def __init__(self, directory=None):
        self.directory = Path(
            directory
            or os.getenv(
                "APP_STATE_DIR",
                str(
                    Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
                    / "hybrid-sanctions"
                    / "state"
                ),
            )
        )
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.database = self.directory / "ingestion.sqlite3"
        with self._connection() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS jobs "
                "(id TEXT PRIMARY KEY, kind TEXT NOT NULL, index_name TEXT NOT NULL, "
                "updated REAL NOT NULL, payload TEXT NOT NULL)"
            )

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.database, timeout=5)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _lock(self, index):
        filename = hashlib.sha256(index.encode()).hexdigest() + ".lock"
        lock = (self.directory / filename).open("a")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise IngestionBusy(
                "An ingestion or maintenance job already holds this index"
            )
        return lock

    def reserve(self, kind, index, total, *, related_indices=()):
        locks = {}
        try:
            # Acquisition is nonblocking and consistently ordered. A vector
            # writer also owns AC, so its source cannot change during validation.
            for name in sorted({index, *related_indices}):
                locks[name] = self._lock(name)
            for name in locks:
                self._recover_index(name)
            job_id = str(uuid.uuid4())
            payload = dict(
                job_id=job_id,
                kind=kind,
                index=index,
                status="queued",
                progress=0,
                total=total,
                failed=0,
                created_at=time.time(),
                locked_indices=list(locks),
            )
            with self._connection() as db:
                db.execute(
                    "INSERT INTO jobs VALUES (?, ?, ?, ?, ?)",
                    (job_id, kind, index, time.time(), json.dumps(payload)),
                )
            return JobReservation(self, job_id, locks[index],
                                  [lock for name, lock in locks.items() if name != index])
        except BaseException:
            for lock in locks.values():
                lock.close()
            raise

    def _recover_index(self, index):
        # The caller holds the index lock, proving any previous owner has exited.
        with self._connection() as db:
            rows = db.execute(
                "SELECT id, payload FROM jobs WHERE index_name=?", (index,)
            ).fetchall()
            for job_id, raw in rows:
                payload = json.loads(raw)
                if payload["status"] in {"queued", "loading"}:
                    payload.update(
                        status="interrupted",
                        error="Worker exited before completing ingestion",
                    )
                    db.execute(
                        "UPDATE jobs SET payload=?, updated=? WHERE id=?",
                        (json.dumps(payload), time.time(), job_id),
                    )

    def update(self, job_id, **values):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            payload = json.loads(row[0])
            payload.update(values)
            payload["updated_at"] = time.time()
            db.execute(
                "UPDATE jobs SET payload=?, updated=? WHERE id=?",
                (json.dumps(payload), time.time(), job_id),
            )

    def get(self, job_id):
        with self._connection() as db:
            row = db.execute(
                "SELECT payload FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        if payload["status"] in {"queued", "loading"}:
            try:
                lock = self._lock(payload["index"])
            except IngestionBusy:
                pass
            else:
                try:
                    self._recover_index(payload["index"])
                finally:
                    lock.close()
                return self.get(job_id)
        return payload

    def latest(self):
        with self._connection() as db:
            rows = db.execute(
                "SELECT id, kind FROM jobs ORDER BY updated DESC LIMIT 1000"
            ).fetchall()
        result = {}
        for job_id, kind in rows:
            if kind not in result:
                result[kind] = self.get(job_id)
        return result
