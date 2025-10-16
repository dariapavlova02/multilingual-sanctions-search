#!/usr/bin/env bash
set -euo pipefail

if [[ "${ENABLE_SEARCH:-true}" == "true" && "${SKIP_ES_INIT:-false}" != "true" ]]; then
    python -m ai_service.scripts.bootstrap
fi

if [[ "$#" -eq 0 ]]; then
    set -- python -m uvicorn ai_service.main:app --host 0.0.0.0 --port 8000
fi
exec "$@"
