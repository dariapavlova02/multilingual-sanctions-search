"""Legacy import names for the canonical Elasticsearch search service.

The former prototype used random vectors, a separate persons/organizations schema
and successful local fallback on backend failures. It is retired. These names now
use HybridSearchConfig and HybridSearchService without a second implementation.

Call find_candidates/readiness/health_check with the canonical search contracts.
For source loading, use the supported ingestion API or bootstrap command; for
local snapshots, use WatchlistIndexService. Prototype mutation/snapshot methods
are intentionally not exposed. See docs/ARCHITECTURE.md.
"""

from typing import Optional

from ...search.config import HybridSearchConfig
from ...search.hybrid_search_service import HybridSearchService

ElasticsearchWatchlistConfig = HybridSearchConfig
ElasticsearchWatchlistAdapter = HybridSearchService


def create_elasticsearch_watchlist_adapter(
    config: Optional[HybridSearchConfig] = None,
    fallback_config=None,
) -> HybridSearchService:
    """Resolve a legacy factory call to canonical screening with no local fallback."""
    if fallback_config is not None:
        raise ValueError(
            "Local fallback is not supported for sanctions screening; "
            "use the canonical search configuration and migrate source data"
        )
    return HybridSearchService(config)


def create_elasticsearch_enhanced_adapter(
    config: Optional[HybridSearchConfig] = None,
    fallback_config=None,
) -> HybridSearchService:
    """Use the same canonical search service as the watchlist factory."""
    return create_elasticsearch_watchlist_adapter(config, fallback_config)
