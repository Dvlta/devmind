from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from devmind.agent.state import Intent
from devmind.indexer.metadata import SourceExcerpt
from devmind.llm.prompts import (
    GROUNDING_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)
from devmind.llm.schemas import GroundingResult, PlanResult, SynthesisResult


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str = "gpt-5.5", client: Any | None = None) -> None:
        self.model = model
        if client is not None:
            self.client = client
            return

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                'OpenAI support is not installed. Run: pip install -e ".[llm]"'
            ) from exc
        self.client = OpenAI()

    def plan(self, query: str, intent: Intent) -> PlanResult:
        response = self.client.responses.parse(
            model=self.model,
            max_output_tokens=1500,
            input=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Intent: {intent}\nDeveloper question: {query}",
                },
            ],
            text_format=PlanResult,
        )
        return require_parsed(response)

    def synthesize(
        self,
        query: str,
        intent: Intent,
        plan: list[str],
        sources: list[SourceExcerpt],
    ) -> SynthesisResult:
        request = {
            "model": self.model,
            "input": [
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Intent: {intent}\n"
                        f"Developer question: {query}\n\n"
                        f"Investigation plan:\n{format_plan(plan)}\n\n"
                        f"Source excerpts:\n{format_sources(sources)}"
                    ),
                },
            ],
            "text_format": SynthesisResult,
        }
        try:
            response = self.client.responses.parse(
                **request,
                max_output_tokens=3000,
            )
        except ValidationError as exc:
            if not is_truncated_json_error(exc):
                raise
            response = self.client.responses.parse(
                **request,
                max_output_tokens=6000,
            )
        return require_parsed(response)

    def validate(
        self,
        query: str,
        answer: str,
        sources: list[SourceExcerpt],
    ) -> GroundingResult:
        response = self.client.responses.parse(
            model=self.model,
            max_output_tokens=1500,
            input=[
                {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Developer question: {query}\n\n"
                        f"Answer:\n{answer}\n\n"
                        f"Source excerpts:\n{format_sources(sources)}"
                    ),
                },
            ],
            text_format=GroundingResult,
        )
        return require_parsed(response)


def require_parsed(response: Any):
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("The model did not return a parsed structured response.")
    return parsed


def is_truncated_json_error(exc: ValidationError) -> bool:
    return any(error.get("type") == "json_invalid" for error in exc.errors())


def format_plan(plan: list[str]) -> str:
    return "\n".join(f"{index}. {step}" for index, step in enumerate(plan, start=1))


def format_sources(sources: list[SourceExcerpt]) -> str:
    blocks = []
    for source in sources:
        blocks.append(
            f"--- {source.file_path}:{source.start_line}-{source.end_line} ---\n"
            f"{source.content}"
        )
    return "\n\n".join(blocks)
