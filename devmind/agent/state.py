from __future__ import annotations

from typing import Any, Literal, TypedDict

from devmind.indexer.metadata import RetrievedChunk, SourceExcerpt


Intent = Literal["architecture", "flow", "tests", "bugs", "docs", "general"]
Confidence = Literal["high", "medium", "low"]


class ToolCallRecord(TypedDict):
    tool: str
    input: dict[str, Any]
    output_count: int


class Citation(TypedDict):
    file_path: str
    start_line: int
    end_line: int


class AgentState(TypedDict, total=False):
    query: str
    repo: str | None
    db_path: str
    max_sources: int
    intent: Intent
    plan: list[str]
    retrieved_chunks: list[RetrievedChunk]
    inspected_sources: list[SourceExcerpt]
    citations: list[Citation]
    answer: str
    confidence: Confidence
    gaps: list[str]
    unsupported_claims: list[str]
    llm_provider: str
    llm_model: str
    llm_mode: Literal["fast", "full"]
    trace: list[dict[str, Any]]
