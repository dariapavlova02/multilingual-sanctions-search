"""Immutable legacy model descriptions resolved through the shared model catalog."""
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional

from ....config import EmbeddingConfig


class ModelType(str, Enum):
    SENTENCE_TRANSFORMER = "sentence_transformer"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_type: ModelType = ModelType.SENTENCE_TRANSFORMER
    model_path: str = ""
    dimension: Optional[int] = None
    max_sequence_length: Optional[int] = None
    device: Optional[str] = None
    batch_size: int = 32
    normalize_embeddings: bool = True
    use_fp16: bool = False
    cache_dir: Optional[str] = None
    enable_gpu: bool = False
    thread_pool_size: int = 4
    max_memory_usage: float = 0.8
    model_kwargs: Optional[Mapping[str, Any]] = None
    revision: Optional[str] = None

    def __post_init__(self):
        if self.model_type != ModelType.SENTENCE_TRANSFORMER:
            raise ValueError("Only pinned sentence-transformer models are supported")
        if self.model_path and self.model_path != self.name:
            raise ValueError("Model path must identify the configured repository")
        if dict(self.model_kwargs or {}) not in ({}, {"use_safetensors": True}):
            raise ValueError("Model loader options cannot override the safe loading contract")
        values = {"model_name": self.name, "batch_size": self.batch_size}
        if self.revision is not None:
            values["revision"] = self.revision
        if self.dimension is not None:
            values["dimension"] = self.dimension
        resolved = EmbeddingConfig(**values)
        object.__setattr__(self, "revision", resolved.revision)
        object.__setattr__(self, "dimension", resolved.dimension)
        object.__setattr__(self, "model_path", resolved.model_name)
        object.__setattr__(self, "model_kwargs", MappingProxyType({"use_safetensors": True}))
        if self.max_sequence_length is not None and (type(self.max_sequence_length) is not int or self.max_sequence_length < 1):
            raise ValueError("Sequence length must be a positive integer")
        # These historical knobs never implemented memory scheduling or loading
        # concurrency. Reject custom values instead of claiming to apply them.
        if self.thread_pool_size != 4 or self.max_memory_usage != 0.8:
            raise ValueError("Configure loading capacity on EmbeddingModelManager")

    def embedding_config(self, default_device="cpu"):
        return EmbeddingConfig(model_name=self.name, revision=self.revision,
                               dimension=self.dimension, device=self.device or default_device,
                               batch_size=self.batch_size)

    def to_dict(self):
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["model_kwargs"] = dict(self.model_kwargs)
        return result


DEFAULT_MODELS = MappingProxyType({
    "multilingual": ModelConfig(name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", max_sequence_length=128),
    "english": ModelConfig(name="sentence-transformers/all-MiniLM-L6-v2", max_sequence_length=256),
    "multilingual_large": ModelConfig(name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", max_sequence_length=128),
})


def get_model_config(model_name: str) -> ModelConfig:
    return DEFAULT_MODELS[model_name] if model_name in DEFAULT_MODELS else ModelConfig(name=model_name)
