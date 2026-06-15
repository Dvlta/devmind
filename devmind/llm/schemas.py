from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Confidence = Literal["high", "medium", "low"]


class PlanResult(BaseModel):
    steps: list[str] = Field(min_length=1, max_length=6)


class ModelCitation(BaseModel):
    file_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class SynthesisResult(BaseModel):
    answer: str
    citations: list[ModelCitation]
    confidence: Confidence
    gaps: list[str] = Field(default_factory=list)


class GroundingResult(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)

