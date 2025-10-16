"""Public request-validation errors, without submitted values or parser context."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class RequestValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loc: list[str | int]
    msg: str
    type: str


class RequestValidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: Literal["Validation error"] = "Validation error"
    errors: list[RequestValidationIssue]
