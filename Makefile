.PHONY: help install install-dev build run stop test test-unit test-reference test-all test-model test-integration regression-backend lint format check check-docs logs status health setup dev restart shell

CORE_TEST_PATHS = \
	tests/unit/api \
	tests/unit/cache \
	tests/unit/core \
	tests/unit/data \
	tests/unit/decision \
	tests/unit/embedding/test_embedding_preprocessor.py \
	tests/unit/layers/search/test_search_imports.py \
	tests/unit/search/test_search_contracts.py \
	tests/unit/signals/test_signals_service_async.py \
	tests/unit/unicode/test_apostrophe_normalization.py \
	tests/unit/unicode/test_unicode_policy.py \
	tests/unit/validation

REGRESSION_ES_IMAGE ?= hybrid-sanctions-elasticsearch:regression
REFERENCE_TEST_PATHS = $(shell cat tests/reference_suite.txt)

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install runtime dependencies
	poetry install --only main

install-dev: ## Install runtime and development dependencies
	poetry install --with dev

build: ## Build API and Elasticsearch images
	docker compose build

run: ## Run container locally
	docker compose up -d

stop: ## Stop container
	docker compose down

test: test-unit ## Run core tests with the pinned NLP models

test-unit: download-models ## Run core tests with the required NLP models
	poetry run -- pytest $(CORE_TEST_PATHS) -m "not model"

regression-backend: ## Build the isolated test backend
	docker build -f Dockerfile.elasticsearch -t $(REGRESSION_ES_IMAGE) .

test-reference: download-models regression-backend ## Check supported reference contracts with real models and isolated Elasticsearch
	poetry run -- python scripts/run_regression_gate.py --image $(REGRESSION_ES_IMAGE) --result-dir .artifacts/reference -- $(REFERENCE_TEST_PATHS) -m "not performance and not perf_micro"

test-all: download-models regression-backend ## Run every test, including research contracts and machine-dependent timing checks
	poetry run -- python scripts/run_regression_gate.py --image $(REGRESSION_ES_IMAGE) --result-dir .artifacts/regression -- tests

test-model: ## Run additional standalone model tests
	poetry run -- pytest -m model

test-integration: download-models regression-backend ## Run integration tests with isolated Elasticsearch
	poetry run -- python scripts/run_regression_gate.py --image $(REGRESSION_ES_IMAGE) -- tests/integration/ tests/e2e/

lint: ## Check formatting and import order
	poetry run -- black --check src tests
	poetry run -- isort --check-only src tests

format: ## Format Python sources and tests
	poetry run -- black src tests
	poetry run -- isort src tests

check-docs: ## Check documentation links against files in Git
	python3 scripts/check_docs.py

check: check-docs ## Validate metadata, reference contracts, collection, package and Compose
	poetry check
	poetry run -- python -m compileall -q src tests
	$(MAKE) test-reference
	poetry run -- pytest --collect-only -q
	poetry build
	poetry run -- python scripts/validate_compose.py

test-micro: ## Run micro-benchmarks
	poetry run -- python -m pytest tests/performance/test_micro_benchmarks.py -v -m perf_micro

test-perf: ## Run all performance tests
	poetry run -- python -m pytest tests/ -v -m "performance or perf_micro"

test-ascii: ## Run ASCII fastpath tests
	poetry run -- python -m pytest tests/integration/test_ascii_fastpath_equivalence.py tests/integration/test_ascii_fastpath_golden_integration.py -v

test-ascii-perf: ## Run ASCII fastpath performance tests
	poetry run -- python -m pytest tests/performance/test_ascii_fastpath_performance.py -v -m performance

logs: ## Show container logs
	docker compose logs -f ai-service

status: ## Show container status
	docker compose ps

health: ## Check service health
	curl -f http://localhost:8001/health/ready

download-models: ## Install the pinned language and embedding models
	poetry run -- python -m pip install --no-deps --require-hashes -r requirements-models.txt
	poetry run -- python -m pip check
	poetry run -- python -c "from ai_service.config import EmbeddingConfig; from ai_service.layers.embeddings.embedding_service import EmbeddingService; EmbeddingService(EmbeddingConfig()).encode_one('model setup verification')"

setup: install-dev download-models ## Setup development environment
	@echo "Development environment ready!"

dev: ## Run in development mode
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

restart: ## Restart service
	docker compose restart ai-service

shell: ## Access container shell
	docker compose exec ai-service /bin/bash
