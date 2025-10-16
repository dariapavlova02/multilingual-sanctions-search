"""Bounded data-only JSON snapshots. Digests detect corruption, not source trust."""

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

FORMAT = "hybrid-sanctions-watchlist"
VERSION = 1
FILENAME = "snapshot.json"


def canonical_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Non-finite JSON value")


def read_json(path, max_bytes, expected_sha256=None):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError("Snapshot file exceeds supported limits")
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Snapshot file exceeds supported limits")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Snapshot artifact digest mismatch")
    return (
        json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        ),
        digest,
    )


def write_snapshot(directory, payload, max_bytes, *, commit_guard=None):
    payload_data = canonical_bytes(payload)
    envelope = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(payload_data).hexdigest(),
    }
    data = canonical_bytes(envelope) + b"\n"
    if len(data) > max_bytes:
        raise ValueError("Snapshot exceeds configured byte limit")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".snapshot-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        with commit_guard if commit_guard is not None else nullcontext():
            os.replace(temporary, directory / FILENAME)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(data).hexdigest()


def read_snapshot(directory, max_bytes, expected_sha256=None):
    envelope, digest = read_json(Path(directory) / FILENAME, max_bytes, expected_sha256)
    if type(envelope) is not dict or set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError("Unsupported snapshot envelope")
    payload = envelope["payload"]
    if (
        type(payload) is not dict
        or hashlib.sha256(canonical_bytes(payload)).hexdigest()
        != envelope["payload_sha256"]
    ):
        raise ValueError("Snapshot content digest mismatch")
    if set(payload) != {
        "format",
        "version",
        "config",
        "embedding_contract",
        "index_id",
        "documents",
    }:
        raise ValueError("Unsupported snapshot fields")
    if (
        payload["format"] != FORMAT
        or type(payload["version"]) is not int
        or payload["version"] != VERSION
    ):
        raise ValueError("Unsupported snapshot version")
    return payload, digest
