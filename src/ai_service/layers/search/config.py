"""Search layer configuration models.

Defines configuration structures for hybrid search functionality and provides
helpers to load settings from environment variables or YAML files.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...config.env_values import parse_boolean


def _configuration_mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a configuration mapping")
    return dict(value)


def _translate_configuration_aliases(value):
    payload = _configuration_mapping(value, "search")
    aliases = {
        "elasticsearch": {"es_hosts": "hosts", "es_timeout": "timeout",
            "es_username": "username", "es_password": "password", "es_api_key": "api_key",
            "es_ca_certs": "ca_certs", "enable_ssl_verification": "verify_certs"},
        "vector_search": {"vector_dimension": "vector_dimension",
            "vector_similarity_threshold": "similarity_threshold"},
    }
    for section, fields in aliases.items():
        nested = payload.get(section, {})
        if isinstance(nested, BaseModel):
            nested = nested.model_dump(mode="python")
        nested = _configuration_mapping(nested, section)
        for alias, canonical in fields.items():
            if alias not in payload:
                continue
            supplied = payload.pop(alias)
            if canonical in nested and nested[canonical] != supplied:
                label = "vector" if section == "vector_search" else "connection"
                raise ValueError(f"Conflicting {label} configuration: {alias}")
            nested[canonical] = supplied
        if nested or section in payload:
            payload[section] = nested
    return payload


class SearchConfigurationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True, hide_input_in_errors=True)


class ElasticsearchConfig(SearchConfigurationModel):
    """Elasticsearch connection configuration"""

    # Connection settings
    hosts: List[str] = Field(default=["localhost:9200"], description="Elasticsearch hosts")
    username: Optional[str] = Field(default=None, description="Elasticsearch username")
    password: Optional[str] = Field(default=None, repr=False, description="Elasticsearch password")
    api_key: Optional[str] = Field(default=None, repr=False, description="Elasticsearch API key")
    ca_certs: Optional[str] = Field(default=None, description="Path to CA certificates")
    verify_certs: bool = Field(default=True, description="Verify SSL certificates")
    scheme: Optional[Literal["http", "https"]] = Field(default=None, description="Explicit connection scheme (http/https)")

    # Connection pool settings
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    timeout: int = Field(default=30, ge=1, le=300, description="Connection timeout in seconds")
    healthcheck_path: str = Field(default="/_cluster/health", description="Path used for health checks")
    smoke_test_timeout: float = Field(default=5.0, ge=0.1, le=30.0, description="Timeout (seconds) for smoke tests")

    # Index settings
    default_index: str = Field(default="watchlist", description="Default index name")
    ac_index: str = Field(default="sanctions_ac_patterns", description="AC search index name")
    vector_index: str = Field(default="sanctions_vectors", description="Vector search index name")
    
    @field_validator("hosts")
    @classmethod
    def validate_hosts(cls, v):
        from ...config.elasticsearch_hosts import validate_elasticsearch_hosts
        return validate_elasticsearch_hosts(v)

    @model_validator(mode="after")
    def validate_credentials(self):
        has_basic = self.username is not None or self.password is not None
        if has_basic and (not self.username or not self.username.strip() or not self.password):
            raise ValueError("Elasticsearch basic authentication requires username and password")
        if self.api_key is not None and not self.api_key.strip():
            raise ValueError("Elasticsearch API key cannot be empty")
        if has_basic and self.api_key is not None:
            raise ValueError("Configure either Elasticsearch basic authentication or an API key")
        return self

    def normalized_hosts(self) -> List[str]:
        """Return hosts with explicit scheme."""
        normalized = []
        for host in self.hosts:
            if host.startswith("http://") or host.startswith("https://"):
                normalized.append(host.rstrip("/"))
                continue
            # Use explicit scheme if set, otherwise default to http
            # Note: verify_certs is about certificate validation, not connection scheme
            base_scheme = self.scheme or "http"
            normalized.append(f"{base_scheme}://{host.strip('/')}".rstrip("/"))
        return normalized

    @classmethod
    def from_sources(
        cls,
        data: Optional[Dict[str, Any]] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "ElasticsearchConfig":
        """Create configuration from combined YAML/env sources."""

        env = dict(env or {})
        payload: Dict[str, Any] = {} if data is None else _configuration_mapping(data, "elasticsearch")

        # Environment overrides
        if "ES_HOSTS" in env:
            payload["hosts"] = [h.strip() for h in env["ES_HOSTS"].split(",")]
        elif "ELASTICSEARCH_HOSTS" in env:
            payload["hosts"] = [h.strip() for h in env["ELASTICSEARCH_HOSTS"].split(",")]
        elif "ELASTICSEARCH_HOST" in env or "ES_HOST" in env:
            host = env["ELASTICSEARCH_HOST"] if "ELASTICSEARCH_HOST" in env else env["ES_HOST"]
            port = env.get("ELASTICSEARCH_PORT") or env.get("ES_PORT", "9200")
            payload["hosts"] = [host if ":" in host else f"{host}:{port}"]
        # With no environment override, preserve YAML and model defaults.

        if env.get("ES_INDEX_PREFIX"):
            payload["ac_index"] = f"{env['ES_INDEX_PREFIX']}_ac_patterns"
            payload["vector_index"] = f"{env['ES_INDEX_PREFIX']}_vectors"

        str_overrides = {
            "username": "ES_USERNAME",
            "password": "ES_PASSWORD",
            "api_key": "ES_API_KEY",
            "ca_certs": "ES_CA_CERTS",
            "default_index": "ES_DEFAULT_INDEX",
            "ac_index": "ES_AC_INDEX",
            "vector_index": "ES_VECTOR_INDEX",
            "scheme": "ES_SCHEME",
            "healthcheck_path": "ES_HEALTHCHECK_PATH",
        }

        for field_name, env_key in str_overrides.items():
            if env_key in env:
                if field_name in {"username", "password", "api_key", "ca_certs"}:
                    payload[field_name] = env[env_key] or None
                elif env[env_key] or env_key == "ES_SCHEME":
                    payload[field_name] = env[env_key]

        int_overrides = {
            "timeout": "ES_TIMEOUT",
            "max_retries": "ES_MAX_RETRIES",
        }
        for field_name, env_key in int_overrides.items():
            if env_key in env:
                try:
                    payload[field_name] = int(env[env_key])
                except ValueError:
                    raise ValueError(f"Invalid integer value for {env_key}") from None

        float_overrides = {"smoke_test_timeout": "ES_SMOKE_TEST_TIMEOUT"}
        for field_name, env_key in float_overrides.items():
            if env_key in env:
                try:
                    payload[field_name] = float(env[env_key])
                except ValueError:
                    raise ValueError(f"Invalid float value for {env_key}") from None

        bool_overrides = {
            "verify_certs": "ES_VERIFY_CERTS",
            "retry_on_timeout": "ES_RETRY_ON_TIMEOUT",
        }
        for field_name, env_key in bool_overrides.items():
            if env_key in env:
                payload[field_name] = parse_boolean(env[env_key], name=env_key)

        return cls(**payload)


class ACSearchConfig(SearchConfigurationModel):
    """AC (exact/almost-exact) search configuration"""
    
    # Search parameters
    boost: float = Field(default=1.2, ge=0.1, le=5.0, description="Score boost for AC matches")
    fuzziness: int = Field(default=1, ge=0, le=3, description="Fuzziness level (0=exact, 1-3=fuzzy)")
    min_score: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum score threshold")
    
    # Field weights for multi-field search
    field_weights: Dict[str, float] = Field(
        default={
            "normalized_text": 2.0,
            "aliases": 1.5,
            "legal_names": 1.8,
            "original_text": 1.0,
        },
        description="Field weights for scoring"
    )

    @field_validator("field_weights")
    @classmethod
    def validate_field_weights(cls, weights):
        if not weights or not any(weight > 0 for weight in weights.values()):
            raise ValueError("At least one positive AC field weight is required")
        if any(not name.strip() or not math.isfinite(weight) or weight < 0
               for name, weight in weights.items()):
            raise ValueError("AC field weights must have names and finite nonnegative values")
        return weights
    
    # Query settings
    enable_phrase_queries: bool = Field(default=True, description="Enable phrase queries")
    enable_wildcard_queries: bool = Field(default=False, description="Enable wildcard queries")
    enable_regex_queries: bool = Field(default=False, description="Enable regex queries")
    
    # Performance settings
    max_query_terms: int = Field(default=25, ge=1, le=100, description="Maximum query terms")
    tie_breaker: float = Field(default=0.3, ge=0.0, le=1.0, description="DisMax tie breaker")


class VectorSearchConfig(SearchConfigurationModel):
    """Vector (kNN) search configuration"""
    
    # Vector search parameters
    boost: float = Field(default=1.0, ge=0.1, le=5.0, description="Score boost for vector matches")
    min_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum score threshold")
    ef_search: int = Field(default=100, ge=10, le=1000, description="HNSW ef_search parameter")
    
    # Vector field settings
    vector_field: str = Field(default="vector", description="Vector field name")
    vector_dimension: int = Field(default=384, ge=64, le=4096, description="Vector dimension")
    
    # Similarity settings
    similarity_type: str = Field(default="cosine", description="Similarity type (cosine, dot_product, l2_norm)")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    
    # Performance settings
    max_candidates: int = Field(default=1000, ge=10, le=10000, description="Maximum candidates to evaluate")
    enable_reranking: bool = Field(default=True, description="Enable result reranking")


class HybridSearchConfig(SearchConfigurationModel):
    """Hybrid search configuration combining AC and Vector search"""
    
    @model_validator(mode="before")
    @classmethod
    def translate_compatibility_fields(cls, value):
        if not isinstance(value, dict):
            return value
        return _translate_configuration_aliases(value)

    @classmethod
    def validated_copy(cls, config=None):
        if config is None:
            return cls.from_env()
        if not isinstance(config, cls):
            raise TypeError("Expected HybridSearchConfig")
        return cls.model_validate(config.model_dump(mode="python"))

    @property
    def vector_dimension(self):
        return self.vector_search.vector_dimension

    @property
    def vector_similarity_threshold(self):
        return self.vector_search.similarity_threshold

    # Service configuration
    service_name: str = Field(default="hybrid_search", description="Service name for logging")
    enable_logging: bool = Field(default=True, description="Enable detailed logging")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    
    # Search mode settings
    default_mode: str = Field(default="hybrid", description="Default search mode")
    enable_escalation: bool = Field(default=True, description="Enable AC->Vector escalation")
    escalation_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="AC score threshold for escalation")
    max_escalation_results: int = Field(default=100, ge=10, le=500, description="Max results for escalation")
    
    # AC patterns in Elasticsearch
    enable_ac_es: bool = Field(default=True, description="Enable AC patterns search in Elasticsearch")
    
    # Vector fallback settings
    enable_vector_fallback: bool = Field(default=True, description="Enable vector fallback when AC search fails")
    vector_cos_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="Cosine similarity threshold for vector fallback")
    vector_fallback_max_results: int = Field(default=50, ge=5, le=200, description="Maximum results for vector fallback")
    enable_rapidfuzz_rerank: bool = Field(default=True, description="Enable RapidFuzz reranking for vector results")
    enable_dob_id_anchors: bool = Field(default=True, description="Enable DoB/ID anchor checking for vector results")
    
    # Contract validation
    strict_candidate_contract: bool = Field(default=True, description="Enforce strict candidate contract validation")
    
    # Result processing
    enable_deduplication: bool = Field(default=True, description="Enable result deduplication")
    dedup_field: str = Field(default="doc_id", description="Field to use for deduplication")
    enable_reranking: bool = Field(default=True, description="Enable final result reranking")
    
    # Performance settings
    request_timeout_ms: int = Field(default=5000, ge=100, le=30000, description="Request timeout in milliseconds")
    max_concurrent_requests: int = Field(default=10, ge=1, le=100, description="Maximum concurrent requests")
    
    # Fallback settings
    enable_fallback: bool = Field(default=True, description="Legacy compatibility flag; local fallback is disabled for screening")
    fallback_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Score threshold for fallback")
    fallback_timeout_ms: int = Field(default=2000, ge=100, le=10000, description="Fallback search timeout in milliseconds")
    fallback_max_results: int = Field(default=100, ge=10, le=500, description="Maximum results from fallback search")
    enable_fallback_health_check: bool = Field(default=True, description="Enable health checks for fallback services")
    
    # Vector fallback settings
    enable_vector_fallback: bool = Field(default=True, description="Enable vector search fallback")
    vector_fallback_threshold: float = Field(default=0.4, ge=0.0, le=1.0, description="Score threshold for vector fallback")
    vector_fallback_timeout_ms: int = Field(default=3000, ge=100, le=15000, description="Vector fallback search timeout in milliseconds")
    vector_fallback_max_results: int = Field(default=50, ge=5, le=200, description="Maximum results from vector fallback search")
    
    # Embeddings integration settings
    enable_embedding_cache: bool = Field(default=True, description="Enable caching for generated query vectors")
    embedding_cache_size: int = Field(default=1000, ge=100, le=10000, description="Maximum number of cached embeddings")
    embedding_cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400, description="Embedding cache TTL in seconds")
    enable_embedding_preprocessing: bool = Field(default=True, description="Enable query preprocessing for embeddings")
    embedding_batch_size: int = Field(default=1, ge=1, le=32, description="Batch size for embedding generation")
    
    # Search result caching settings
    enable_search_cache: bool = Field(default=True, description="Enable caching for search results")
    search_cache_size: int = Field(default=500, ge=50, le=5000, description="Maximum number of cached search results")
    search_cache_ttl_seconds: int = Field(default=1800, ge=60, le=86400, description="Search result cache TTL in seconds")
    
    # Query optimization settings
    enable_query_optimization: bool = Field(default=True, description="Enable query optimization features")
    ac_query_boost_factor: float = Field(default=1.0, ge=0.1, le=5.0, description="AC query boost factor")
    vector_query_boost_factor: float = Field(default=1.0, ge=0.1, le=5.0, description="Vector query boost factor")
    bm25_query_boost_factor: float = Field(default=1.0, ge=0.1, le=5.0, description="BM25 query boost factor")
    enable_query_caching: bool = Field(default=True, description="Enable query result caching")
    query_cache_size: int = Field(default=1000, ge=100, le=10000, description="Maximum number of cached queries")
    query_cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400, description="Query cache TTL in seconds")
    
    # Security settings
    # Connection credentials and TLS settings live only in `elasticsearch`.
    enable_audit_logging: bool = Field(default=False, description="Enable audit logging for search operations")
    enable_rate_limiting: bool = Field(default=False, description="Enable rate limiting for search requests")
    rate_limit_requests_per_minute: int = Field(default=100, ge=10, le=10000, description="Rate limit requests per minute")
    enable_query_validation: bool = Field(default=True, description="Enable query validation and sanitization")
    enable_sensitive_data_filtering: bool = Field(default=True, description="Enable sensitive data filtering in results")
    
    # Elasticsearch configuration
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    
    # Search mode configurations
    ac_search: ACSearchConfig = Field(default_factory=ACSearchConfig)
    vector_search: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    
    # Metrics configuration
    metrics_window_size: int = Field(default=1000, ge=100, le=10000, description="Metrics rolling window size")
    metrics_retention_hours: int = Field(default=24, ge=1, le=168, description="Metrics retention in hours")
    
    @field_validator("default_mode")
    @classmethod
    def validate_default_mode(cls, v):
        """Validate default search mode"""
        valid_modes = ["ac", "vector", "hybrid"]
        if v not in valid_modes:
            raise ValueError(f"default_mode must be one of {valid_modes}")
        return v
    
    def get_elasticsearch_config(self) -> Dict[str, Any]:
        """Get Elasticsearch configuration as dictionary"""
        return self.elasticsearch.model_dump()
    
    def get_ac_config(self) -> Dict[str, Any]:
        """Get AC search configuration as dictionary"""
        return self.ac_search.model_dump()
    
    def get_vector_config(self) -> Dict[str, Any]:
        """Get vector search configuration as dictionary"""
        return self.vector_search.model_dump()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary"""
        return self.model_dump()

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        settings_path: Optional[Union[str, Path]] = None,
    ) -> "HybridSearchConfig":
        """Load configuration from YAML and environment overrides."""

        env_map = dict(os.environ if env is None else env)
        yaml_payload: Dict[str, Any] = {}
        selected_path = settings_path if settings_path is not None else env_map.get("AI_SEARCH_SETTINGS_PATH")
        if selected_path is not None:
            if not str(selected_path).strip():
                raise ValueError("AI_SEARCH_SETTINGS_PATH cannot be empty")
            with Path(selected_path).open("r", encoding="utf-8") as file:
                raw = yaml.safe_load(file)
            raw = _configuration_mapping({} if raw is None else raw, "settings")
            yaml_payload = _configuration_mapping(raw.get("search", raw), "search")

        yaml_payload = _translate_configuration_aliases(yaml_payload)

        es_settings = _configuration_mapping(yaml_payload.get("elasticsearch", {}), "elasticsearch")
        ac_settings = _configuration_mapping(yaml_payload.get("ac_search", {}), "ac_search")
        vector_settings = _configuration_mapping(yaml_payload.get("vector_search", {}), "vector_search")

        config_payload: Dict[str, Any] = yaml_payload.copy()
        config_payload["elasticsearch"] = ElasticsearchConfig.from_sources(
            es_settings,
            env=env_map,
        )

        if "ES_AC_FIELD_WEIGHTS" in env_map:
            weights = {}
            for item in env_map["ES_AC_FIELD_WEIGHTS"].split(","):
                name, delimiter, weight = item.partition(":")
                name = name.strip()
                if not name or not delimiter or name in weights:
                    raise ValueError("Invalid field mapping for ES_AC_FIELD_WEIGHTS")
                try:
                    weights[name] = float(weight)
                except ValueError:
                    raise ValueError("Invalid numeric value for ES_AC_FIELD_WEIGHTS") from None
            ac_settings["field_weights"] = weights

        if "ES_VECTOR_DIMENSION" in env_map:
            try:
                vector_settings["vector_dimension"] = int(env_map["ES_VECTOR_DIMENSION"])
            except ValueError:
                raise ValueError("Invalid integer value for ES_VECTOR_DIMENSION") from None

        if ac_settings:
            config_payload["ac_search"] = {**ac_settings}
        if vector_settings:
            config_payload["vector_search"] = {**vector_settings}

        return cls(**config_payload)


# Default configuration instance
DEFAULT_HYBRID_SEARCH_CONFIG = HybridSearchConfig()
