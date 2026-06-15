from __future__ import annotations

from devmind.agent.state import Intent
from devmind.indexer.metadata import SourceExcerpt
from devmind.llm.schemas import GroundingResult, ModelCitation, PlanResult, SynthesisResult


class DeterministicProvider:
    name = "deterministic"
    model = "none"

    def plan(self, query: str, intent: Intent) -> PlanResult:
        steps = [
            "Run hybrid retrieval over semantic, keyword, path, symbol, and import signals.",
            "Inspect the highest-ranked source chunks with exact line numbers.",
            "Synthesize a grounded answer using only inspected source evidence.",
        ]
        if intent == "tests":
            steps.insert(1, "Prioritize tests and test helpers related to the queried behavior.")
        elif intent == "flow":
            steps.insert(1, "Prioritize implementation chunks that show control flow and call order.")
        elif intent == "architecture":
            steps.insert(1, "Prioritize module boundaries, core classes, and design documentation.")
        return PlanResult(steps=steps)

    def synthesize(
        self,
        query: str,
        intent: Intent,
        plan: list[str],
        sources: list[SourceExcerpt],
    ) -> SynthesisResult:
        if not sources:
            return SynthesisResult(
                answer="I could not find enough indexed source evidence to answer that question.",
                citations=[],
                confidence="low",
                gaps=["No relevant indexed sources were retrieved."],
            )

        citations = [
            ModelCitation(
                file_path=source.file_path,
                start_line=source.start_line,
                end_line=source.end_line,
            )
            for source in sources
        ]
        labels = ", ".join(
            f"[{citation.file_path}:{citation.start_line}-{citation.end_line}]"
            for citation in citations
        )
        return SynthesisResult(
            answer=(
                f"The most relevant evidence for this {intent} question is in {labels}. "
                "Inspect these excerpts to trace the implementation details. "
                "Enable an LLM provider for a synthesized natural-language explanation."
            ),
            citations=citations,
            confidence="high" if len(citations) >= 2 else "medium",
            gaps=[],
        )

    def validate(
        self,
        query: str,
        answer: str,
        sources: list[SourceExcerpt],
    ) -> GroundingResult:
        labels = [
            f"[{source.file_path}:{source.start_line}-{source.end_line}]"
            for source in sources
        ]
        grounded = bool(labels) and all(label in answer for label in labels)
        return GroundingResult(
            grounded=grounded,
            unsupported_claims=[] if grounded else ["The answer is missing one or more source citations."],
        )

