# Configuration

Settings load at startup. Recreate the API to apply changes; `/reload-config`
returns 409 and does not alter running clients or caches.

## Local environment

Run `python3 scripts/create_local_env.py` to generate a private `.env`.
[env.example](../env.example) and [env.production.example](../env.production.example)
document the Compose settings. Compose forwards variables explicitly: adding an
arbitrary application variable to `.env` does not put it in the container.

| Variable | Purpose / default |
| --- | --- |
| `ELASTIC_PASSWORD` | Elasticsearch administrator password; provisioning only |
| `ES_SERVICE_PASSWORD` | Distinct restricted application-account password |
| `ADMIN_API_KEY` | Administrative HTTP token, at least 32 characters |
| `API_PORT` | Loopback host port, `8001` |
| `ES_INDEX_PREFIX` | Shared index prefix, `sanctions` |
| `SANCTIONS_IMAGE`, `SANCTIONS_ES_IMAGE` | API and Elasticsearch image names |
| `CORS_ORIGINS` | Allowed origins as a JSON array, `[]`; not authentication |
| `API_MEMORY_LIMIT_BYTES` | Container memory budget, 2 GiB by default |
| `ENABLE_VARIANTS` | Enables variant generation; true in Compose |
| `MAX_REQUEST_BYTES`, `MAX_UPLOAD_BYTES` | Body limits, 1 MiB and 26 MiB |
| `HTTP_MAX_INFLIGHT` | Active HTTP request slots, `4` |
| `HTTP_BODY_TIMEOUT_SECONDS` | Body-read deadline |
| `HTTP_PROCESSING_TIMEOUT_SECONDS` | Ordinary request processing deadline, `30` |
| `EMBEDDING_MAX_PENDING`, `EMBEDDING_TIMEOUT_SECONDS` | Embedding queue bound and caller deadline |
| `VARIANTS_MAX_PENDING`, `VARIANTS_TIMEOUT_SECONDS` | Variant queue bound and deadline, `16` and `30` |
| `INGESTION_VERIFY_TIMEOUT_SECONDS` | Vector-import verification deadline, `300`; maximum `3600` |

Invalid capacity settings fail startup. A cancelled queued job is removed; a native
model call already running retains its slot until it completes. Increasing the
waiting queue does not increase model throughput. Liveness remains available when
ordinary HTTP admission slots are occupied.

## Search settings

The active schema is [`HybridSearchConfig`](../src/ai_service/layers/search/config.py),
with nested `elasticsearch`, `ac_search` and `vector_search` models. Defaults come
from these models. `AI_SEARCH_SETTINGS_PATH` explicitly selects a YAML file;
supported environment overrides apply afterward. No arbitrary working-directory
file loads implicitly.

```yaml
search:
  ac_search:
    min_score: 0.6
  vector_search:
    similarity_threshold: 0.7
```

For Docker, mount the file read-only and add `AI_SEARCH_SETTINGS_PATH` to the API
environment through a Compose override. Standard Compose does not forward it.

Direct runtimes use `ES_HOSTS`, `ES_USERNAME` / `ES_PASSWORD` or `ES_API_KEY`, and
`ES_CA_CERTS` for certificate validation where configured. Host URLs cannot embed
credentials. Conflicting authentication modes, partial basic credentials and
malformed settings are rejected. Clear both basic fields when switching to an API
key. All configured nodes must belong to the intended cluster.

## Normalization flags

Precedence: runtime defaults → explicitly selected YAML profile → environment →
fields explicitly supplied in the request. A partial request override does not
reset unspecified defaults or change other requests. The processing cache includes
the complete effective flag configuration.

`AISVC_FEATURE_FLAGS_FILE` selects a file and `APP_ENV` selects its profile:

```yaml
production:
  feature_flags:
    strict_stopwords: true
    fix_initials_double_dot: true
    preserve_hyphenated_case: true
```

Mount and forward the file explicitly in Docker. A selected file must contain the
chosen profile and valid boolean fields. Duplicate keys, unknown flags and malformed
values are errors. Supported fields are defined by
[`FeatureFlags`](../src/ai_service/utils/feature_flags.py).

Each field accepts `AISVC_FLAG_<UPPERCASE_FIELD_NAME>`, for example
`AISVC_FLAG_ENABLE_SPACY_EN_NER=false`. Boolean environment values accept
`true/false`, `1/0`, `yes/no`, `y/n`, `on/off`. Empty or misspelled values fail loading.
HTTP `options.flags` requires actual JSON booleans. Retired implementation, rollout
and fallback selectors are rejected.

The runtime manager defaults `use_diminutives_dictionary_only` to true; the explicit
library constructor `FeatureFlags()` defaults it to false. Supplying a library
`FeatureFlags` object selects a complete configuration. Recreate components when
changing deployment defaults.

## Model contract

The [embedding manifest](../src/ai_service/data/config/embedding_model.json) pins
`paraphrase-multilingual-MiniLM-L12-v2`, its revision, dimension 384, normalized
vectors and preprocessing version. [requirements-models.txt](../requirements-models.txt)
pins the spaCy wheel hashes. Docker includes these artifacts for offline runtime use.

Changing the embedding model, dimension or preprocessing requires regenerated
vectors and a compatible image. Changing a dimension variable alone cannot make
an old vector snapshot compatible.

## Diagnostics

Authenticated `/config-status` reports the restart policy. `/validate-config`
checks active search settings and separately reports runtime readiness; it does
not validate an unmounted future environment file. `/health/ready` checks models,
index schemas and completed compatible generations, not connectivity alone.

See [Local operation](DEPLOYMENT.md) for applying settings and
[Security](../SECURITY.md) for credentials and logging.
