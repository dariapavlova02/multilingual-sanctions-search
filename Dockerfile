FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true POETRY_INSTALLER_RE_RESOLVE=false \
    HF_HOME=/opt/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence_transformers
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
ARG POETRY_VERSION=2.1.4
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"
COPY pyproject.toml poetry.lock ./
RUN poetry install --only=main --no-root --no-interaction --no-ansi
RUN .venv/bin/python -m pip install --no-cache-dir "pip==26.2.1"
COPY requirements-models.txt ./
RUN .venv/bin/pip install --no-cache-dir --no-deps --require-hashes -r requirements-models.txt
COPY src/ai_service/data/config/embedding_model.json /tmp/embedding_model.json
RUN .venv/bin/python -c "import json; from sentence_transformers import SentenceTransformer; c=json.load(open('/tmp/embedding_model.json')); SentenceTransformer(c['model_name'], revision=c['revision'], trust_remote_code=False, model_kwargs={'use_safetensors': True})"
COPY README.md ./
COPY src/ ./src/
RUN poetry build --format wheel && .venv/bin/pip install --no-deps dist/*.whl
RUN .venv/bin/python -c "from ai_service.config import EmbeddingConfig; from ai_service.layers.embeddings.embedding_service import EmbeddingService; EmbeddingService(EmbeddingConfig()).encode_one('model build verification')"
RUN .venv/bin/pip check && .venv/bin/python -m pip uninstall -y pip

FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_ENV=production \
    PATH=/app/.venv/bin:$PATH \
    HF_HOME=/opt/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence_transformers \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ENABLE_SEARCH=true ENABLE_EMBEDDINGS=true \
    XDG_CACHE_HOME=/app/cache APP_STATE_DIR=/app/state
WORKDIR /app
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && /usr/local/bin/python -m pip uninstall -y pip \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/cache /app/logs /app/state && chown -R app:app /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/models /opt/models
COPY scripts/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod 755 /app/docker-entrypoint.sh
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-m", "ai_service.scripts.healthcheck"]
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "ai_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
