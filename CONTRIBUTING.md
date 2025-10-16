# Contributing

Keep changes small, explain observable behavior and include relevant validation.
Discuss architecture, data-source and matching-policy changes in an issue first.

## Setup

Requires Python 3.11–3.13, Poetry 2.1.x and Docker.

```bash
poetry install --with dev
make download-models
make check
poetry run pre-commit install
```

[Testing](docs/TESTING.md) describes the maintained reference selection and known
research-suite failures. Run relevant research tests when changing experimental
behavior and report their failures separately.

## Pull requests

- Add meaningful tests for changed behavior and document public API/configuration changes.
- Format touched Python files with `poetry run black` and `poetry run isort`.
  Pre-commit uses the project's locked versions of both tools.
- Include a rationale and before/after examples for matching or decision-policy changes.
- Keep generated reports, model caches, live credentials and customer data out of Git.
- Record source provenance, retrieval date, terms and checksums for dataset changes.

Follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities through
[Security](SECURITY.md). Contributions are licensed under the [MIT License](LICENSE).
