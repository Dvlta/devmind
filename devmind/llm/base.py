from __future__ import annotations

from typing import Protocol

from devmind.agent.state import Intent
from devmind.indexer.metadata import SourceExcerpt
from devmind.llm.schemas import GroundingResult, PlanResult, SynthesisResult


class LLMProvider(Protocol):
    name: str
    model: str

    def plan(self, query: str, intent: Intent) -> PlanResult:
        """Create a concise investigation plan."""

    def synthesize(
        self,
        query: str,
        intent: Intent,
        plan: list[str],
        sources: list[SourceExcerpt],
    ) -> SynthesisResult:
        """Answer the query using only the supplied source excerpts."""

    def validate(
        self,
        query: str,
        answer: str,
        sources: list[SourceExcerpt],
    ) -> GroundingResult:
        """Judge whether the answer is supported by the supplied sources."""

