#!/usr/bin/env python3
"""Create or validate the canonical indices without making empty data ready."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elasticsearch import AsyncElasticsearch
from ai_service.layers.search.config import HybridSearchConfig
from ai_service.layers.search.elasticsearch_client import build_client_kwargs
from ai_service.layers.search.index_schema import ensure_index, embedding_contract


def configure_target(es_host=None, index_prefix=None):
    if es_host:
        os.environ["ES_HOSTS"] = es_host
    if index_prefix:
        os.environ["ES_INDEX_PREFIX"] = index_prefix
        os.environ["ES_AC_INDEX"] = f"{index_prefix}_ac_patterns"
        os.environ["ES_VECTOR_INDEX"] = f"{index_prefix}_vectors"
    return HybridSearchConfig.from_env()


async def _create_index(es_host, index_name, *, vectors=False, vector_dim=None):
    config = configure_target(es_host)
    if vector_dim is not None and vector_dim != embedding_contract()["dimension"]:
        raise ValueError("Vector dimension must match the pinned query model")
    async with AsyncElasticsearch(
        **build_client_kwargs(config.elasticsearch)
    ) as client:
        await ensure_index(client, index_name, config, vectors=vectors)
    return True


async def create_ac_patterns_index(es_host, index_name):
    return await _create_index(es_host, index_name)


async def create_vectors_index(es_host, index_name, vector_dim=None):
    return await _create_index(es_host, index_name, vectors=True, vector_dim=vector_dim)


async def main_async(args):
    config = configure_target(args.es_host, args.index_prefix)
    await create_ac_patterns_index(None, config.elasticsearch.ac_index)
    await create_vectors_index(None, config.elasticsearch.vector_index, args.vector_dim)
    print("Indices validated. Empty indices remain unready until ingestion completes.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--es-host", help="URL or host:port; defaults to ES_HOSTS")
    parser.add_argument("--index-prefix", help="Override both configured index names")
    parser.add_argument("--vector-dim", type=int, default=None)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
