# HTTP API

Default local address: `http://127.0.0.1:8001`. The running version's `/docs` and
`/openapi.json` provide the complete request and response schemas.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /normalize` | Normalize text without requiring loaded sanctions indices |
| `POST /process` | Normalize, extract signals, search and return a decision |
| `POST /process-batch` | Process 1–100 texts; inspect each item's `success` and `errors` |
| `POST /search` | Search the imported sanctions snapshot |
| `POST /search-similar` | Compare a query with 1–1,000 supplied candidates |
| `GET /health/live` | Process liveness |
| `GET /health/ready` | Models and complete screening snapshot readiness |
| `GET /health` | Basic service and dependency state |
| `GET /metrics` | Prometheus metrics |

Processing endpoints have no application authentication; the supplied Compose
configuration binds the API to loopback. Administrative routes require
`Authorization: Bearer <ADMIN_API_KEY>` with a key of at least 32 characters.

## Search

```bash
curl --fail http://127.0.0.1:8001/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"John Smith","search_mode":"hybrid","top_k":10,"threshold":0.7}'
```

| Field | Default | Accepted values |
| --- | --- | --- |
| `query` | Required | Nonblank text; default limit 10,000 characters |
| `search_mode` | `hybrid` | `ac`, `fuzzy`, `vector`, `hybrid` |
| `top_k` | `10` | Integer 1–100 |
| `threshold` | `0.7` | Number 0–1 |
| `enable_escalation` | `true` | Boolean |

The response includes `query`, `normalized_query`, `results`, `total_hits`,
`search_type`, `processing_time_ms`, `success` and `errors`. Candidates contain
source metadata, scores and match evidence. `total_hits` counts returned
candidates, not the full corpus. An empty successful result is specific to the
query, threshold and loaded snapshot; it is not a clearance decision.

Use the documented request fields. `language`, `entity_types`, `client_id` and
internal fusion settings are not public `/search` options. Unknown fields may be
ignored by the current request models.

## Normalization and processing

```bash
curl --fail http://127.0.0.1:8001/normalize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Ивана Петрова","language":"ru"}'
```

`/normalize` accepts `language` (`auto`, `ru`, `uk`, `en`), `remove_stop_words`,
`apply_lemmatization`, `clean_unicode`, `preserve_names` and `options.flags`.
`apply_stemming: true` is unsupported and returns 422.

`/process` detects language and accepts `text`, `generate_variants` (default true),
`generate_embeddings` (false), `cache_result` (true) and optional `options.flags`.
It returns normalized text, tokens and traces, signals, search results, decision,
variants when requested and an embedding when requested. Explicitly requesting an
unavailable generation stage fails the request.

Decision fields include `risk_level`, `risk_score`, `decision_reasons`,
`decision_details`, `review_required` and `required_additional_fields`. An empty
additional-fields list does not cancel an ownership-review requirement. Source
positions and identifier association follow the [evidence contract](ARCHITECTURE.md#normalization-and-evidence).

`/process-batch` accepts generation flags and `max_concurrent` from 1–32, default
10. Results retain input order. HTTP 200 does not mean every item succeeded;
failed items omit partial results. `total_processing_time` sums item times and
is not parallel wall-clock time.

## Administration

| Route | Behavior |
| --- | --- |
| `/admin/ac-patterns/bulk`, `/admin/vectors/bulk` | Submit incremental upserts |
| `/admin/ac-patterns/upload`, `/admin/vectors/upload` | Submit uploaded data |
| `/admin/loading-status/{job_id}` | Read job progress and completion |
| `/admin/indices`, `/admin/indices/{index_name}` | Inspect or delete configured indices; check HTTP methods in OpenAPI |
| `/clear-cache`, `/health/detailed` | Cache maintenance and component diagnostics |
| `/config-status`, `/validate-config` | Active settings and runtime-readiness diagnostics |
| `/reload-config` | Returns 409; settings require service recreation |

Accepted ingestion is not completed ingestion. For vector imports also inspect
`snapshot_ready`, `missing_or_changed_vectors` and `extra_vectors`; a completed
partial job may leave the whole snapshot unavailable. See [snapshot loading](DEPLOYMENT.md#load-a-snapshot).

## Failures and limits

Blank and invisible-only inputs return 422. Missing or incomplete indices,
backend failures, generation changes and exhausted processing capacity return 503.
The default body limits are 1 MiB for ordinary requests and 26 MiB for uploads
including multipart overhead, with a 25 MiB file limit.

| Status | Meaning |
| --- | --- |
| `401` / `403` | Invalid or missing administrative credentials |
| `408` | Body-read deadline exceeded |
| `409` | Operation conflicts with the current state, including live reload |
| `413` | Body too large |
| `422` | Invalid input or unsupported option |
| `429` | Request rate limit exceeded |
| `500` | Internal processing failure |
| `503` | Dependency, snapshot, capacity or processing-deadline failure |

Errors omit internal exception payloads. [Configuration](CONFIGURATION.md) describes
the admission and model-queue limits; [Security](../SECURITY.md) describes the local
access and logging boundary.
