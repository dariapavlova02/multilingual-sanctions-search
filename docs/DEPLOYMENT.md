# Run locally

The supported example uses one host, one API replica and authenticated
Elasticsearch. Start with the [quick start](../README.md#quick-start).
`docker-compose.yml` reuses the service definitions in `docker-compose.prod.yml`;
the latter's historical filename does not imply verified production operation.

## Startup and credentials

`python3 scripts/create_local_env.py` writes a private `.env` with mode 0600 and
three distinct random secrets. It refuses to overwrite an existing file. Compose
provisions an Elasticsearch application account restricted to the configured
indices; the API does not receive the administrator password.

Allow at least 4 GB for Docker, plus disk space for images, models and data. The
supplied limits are 768 MiB for Elasticsearch and 2 GiB for the API. These are
local startup budgets, not measured dataset or request capacity.

The API listens on `127.0.0.1:8001`; Elasticsearch has no published host port.
A fresh stack has empty indices. Normalization can work while screening and
`/health/ready` return 503 until a complete snapshot is loaded.

## Load a snapshot

The bootstrap command loads a selected source directory through the same ingestion
implementation used by the application. Review [Data provenance](DATA_PROVENANCE.md)
before using historical or external lists.

For the bundled historical files, omit the quick start's demo mount and `--data-dir`:

```bash
docker compose up -d elasticsearch provision
docker compose run --rm -T --entrypoint python ai-service \
  -m ai_service.scripts.bootstrap --ingest --vectors --batch-size 16
docker compose up -d ai-service
```

This requires empty indices. The full historical snapshot has not received the
same end-to-end validation as the small demo. Batch size 16 is a starting setting,
not a throughput guarantee.

For your own source files, mount a reviewed directory and select it explicitly:

```bash
docker compose run --rm -T \
  -v "$PWD/my-data:/input:ro" --entrypoint python ai-service \
  -m ai_service.scripts.bootstrap --ingest --vectors --data-dir /input --batch-size 16
```

The [loader](../src/ai_service/layers/search/sanctions_data_loader.py) defines the accepted source
formats; the [demo JSON](../examples/demo/custom_portfolio_demo.json) is a small
custom-source example. Merely placing a file in the repository does not import it.

### Replace an existing snapshot

Stop the API before loading so a second model process does not compete with the
serving process. Add `--replace` only when deliberately replacing the current
snapshot. This removes old indexed records and creates a new generation.

```bash
(
  set -e
  trap 'docker compose start ai-service' EXIT
  docker compose stop ai-service
  docker compose run --rm --no-deps -T --entrypoint python ai-service \
    -m ai_service.scripts.bootstrap --ingest --vectors --replace --batch-size 16
)
```

This example replaces the current snapshot with the bundled historical files.
For a chosen external or demo directory, include its mount and `--data-dir` as
above. Screening stays unavailable until both lexical and vector generations are
complete. An interrupted load must be retried; restarting the API does not repair it.

Administrative bulk/upload endpoints perform incremental upserts, not full-list
replacement. Follow their job IDs and check snapshot readiness. All writers must
share the same state volume; cross-host writers are unsupported.

## Check and stop

```bash
curl --fail http://127.0.0.1:8001/health/live
curl --fail http://127.0.0.1:8001/health/ready
docker compose ps
docker compose logs --tail 100 ai-service
docker compose down
```

`down` preserves volumes. Reusing them preserves the imported snapshot and
Elasticsearch credentials. Available disk space matters: a full Docker disk can
make Elasticsearch reject writes. Liveness alone does not prove that screening works.

## Development

Python 3.11–3.13 and Poetry 2.1.x are required for local development; the recorded
reference run used Python 3.12.

```bash
poetry install --with dev
make download-models
make test-unit
```

For API development with the configured backend and source reload:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d
```

The override mounts `src` read-only and enables reload for the API. Model downloads
are pinned in [requirements-models.txt](../requirements-models.txt) and the
[embedding manifest](../src/ai_service/data/config/embedding_model.json).

Apply settings by recreating the API. Changes to index names or credentials also
require compatible provisioning and data. Changing `.env` alone does not rotate an
existing Elasticsearch administrator password. See [Configuration](CONFIGURATION.md)
and [Security](../SECURITY.md).
