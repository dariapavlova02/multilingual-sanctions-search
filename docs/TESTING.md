# Testing

The project separates maintained public contracts from a broader research suite.
The exact maintained scope is versioned in
[`tests/reference_suite.txt`](../tests/reference_suite.txt).

## Run checks

Requires Python 3.11–3.13, Poetry 2.1.x and Docker. The recorded local run used
Python 3.12. Model installation uses the same pinned artifacts as the image.

```bash
poetry install --with dev
make check
```

| Command | Scope |
| --- | --- |
| `make check` | Documentation links, package metadata, reference suite, complete collection, wheel/sdist and Compose |
| `make test-reference` | Maintained contracts with real models and isolated Elasticsearch |
| `make test-unit` | Selected core tests |
| `make test-integration` | Integration and end-to-end tests, including research cases |
| `make test-all` | Entire suite, including unresolved research expectations and timing checks |
| `make check-docs` | Local Markdown links and anchors against the publication file set |
| `make lint` | Black and isort checks |

The reference runner builds an authenticated disposable Elasticsearch container,
verifies document write/read/delete, runs pytest and removes its own backend.
It does not use or clear the application's data volumes. Required models or backend
failures fail the check. Two FAISS checks may skip when that optional backend is absent.

## Maintained contracts

Coverage includes API validation, configuration, dependency failures, source
identity, aliases, snapshot consistency, lexical/vector/hybrid search, identity
review rules, cache isolation, bounded workers and normalization golden cases.
The demo and screening HTTP workflows use real models and Elasticsearch. Unit
tests also inject fakes to exercise failure paths.

The [CI workflow](../.github/workflows/ci.yml) runs this selection and uploads JUnit
and runner results as artifacts. `workflow_dispatch` can additionally run the full
research suite; its failures are not masked. Local output goes to `.artifacts/`
and is not committed.

## Recorded verification

Recorded environment: macOS ARM64 / Python 3.12, Elasticsearch in Docker:

| Check | Result |
| --- | --- |
| Reference gate | 1,739 passed, 2 optional FAISS skips, zero failures/errors |
| Entire final collection | 3,883 cases, no collection errors |
| Package and dependencies | Metadata, dependency consistency, compilation, wheel and sdist passed |
| Isolated wheel | Three normalization examples passed, including middle-name preservation |
| HTTP demo | Bootstrap, aliases, real-model retrieval and replacement protection passed |
| Compose | Default, base and development configurations validated |

These measurements describe a local reference run. They do not claim a successful
remote CI run or quality on the complete historical dataset.

## Research limits

Before targeted test repairs, the complete baseline had 3,918 named cases:
3,600 passed, 300 failed and 18 skipped/expected failures, with zero setup errors.
The full suite was not rerun after those repairs. The baseline failure count must
not be presented as either fixed or a current exact remaining count.

The current collection contains 3,883 cases. Obsolete simulated checks and empty
tests are excluded; retired cases do not count as repaired behavior. Remaining
research tests retain experimental language expectations, private legacy interfaces
and machine-dependent timing targets.

A 50 ms search timing threshold remains in the performance suite. No service
latency guarantee, precision, recall or calibrated risk probability is inferred
from these engineering tests. See [Architecture](ARCHITECTURE.md) for the current
name and identity contracts.
