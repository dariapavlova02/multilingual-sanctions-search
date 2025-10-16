"""Legacy import for the canonical sanctions search implementation.

All callers use the same source readiness, vector contract, cache, deadline and
failure behavior. The former incomplete class and pseudo-vector path are removed.
"""
from .hybrid_search_service import HybridSearchService as HybridSearchServiceRefactored

__all__ = ["HybridSearchServiceRefactored"]
