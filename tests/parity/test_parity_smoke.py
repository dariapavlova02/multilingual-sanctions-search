"""Synchronous and asynchronous entry points preserve the same name evidence."""

import json
from pathlib import Path
from statistics import quantiles
from time import perf_counter

from ai_service.layers.normalization.normalization_service import NormalizationService

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "parity_golden.jsonl"


async def test_sync_async_parity_smoke():
    cases = [json.loads(line) for line in FIXTURE_PATH.read_text().splitlines() if line.strip()]
    assert len(cases) == 10, "Fixture should contain 10 parity cases"
    service = NormalizationService()
    timings = []
    for case in cases:
        sync = service.normalize_sync(case["text"], language="auto")
        start = perf_counter()
        asynchronous = await service.normalize_async(case["text"], language="auto")
        timings.append((perf_counter() - start) * 1000)
        assert sync.success and asynchronous.success
        assert sync.normalized == asynchronous.normalized
        assert sync.tokens == asynchronous.tokens
        assert sync.persons_core == asynchronous.persons_core
    assert quantiles(timings, n=100, method="inclusive")[94] <= 20.0, "Normalization p95 must remain within 20ms"
