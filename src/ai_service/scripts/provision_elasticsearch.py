"""Provision the single-host service account without sharing cluster credentials."""

import asyncio
import os

from elasticsearch import AsyncElasticsearch

from ..layers.search.config import HybridSearchConfig
from ..layers.search.elasticsearch_client import build_client_kwargs


async def provision():
    config = HybridSearchConfig.from_env().elasticsearch
    username = os.environ.get("ES_SERVICE_USERNAME", "sanctions_service")
    password = os.environ.get("ES_SERVICE_PASSWORD", "")
    if username == "elastic" or len(password) < 32:
        raise ValueError(
            "Use a separate service user and a password of at least 32 characters"
        )
    if not config.password or password == config.password:
        raise ValueError("Cluster and service passwords must be distinct")
    async with AsyncElasticsearch(**build_client_kwargs(config)) as client:
        await client.security.put_role(
            name="sanctions_service",
            cluster=["monitor"],
            indices=[
                {
                    "names": [config.ac_index, config.vector_index],
                    "privileges": ["manage", "read", "write", "view_index_metadata"],
                    "allow_restricted_indices": False,
                }
            ],
        )
        await client.security.put_user(
            username=username,
            password=password,
            roles=["sanctions_service"],
            full_name="Sanctions screening service",
        )
    print("Provisioned service account for the configured sanctions indices")


if __name__ == "__main__":
    asyncio.run(provision())
