"""Startup limits for the serial variant-generation worker."""

import os

from pydantic import BaseModel, ConfigDict, Field


class VariantExecutionConfig(BaseModel):
    model_config = ConfigDict(validate_default=True)
    max_pending: int = Field(default_factory=lambda: int(os.getenv("VARIANTS_MAX_PENDING", "16")), ge=0, le=128)
    timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("VARIANTS_TIMEOUT_SECONDS", "30")), gt=0, le=300)
