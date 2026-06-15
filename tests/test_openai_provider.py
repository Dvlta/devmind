from types import SimpleNamespace

from pydantic import ValidationError

from devmind.indexer.metadata import SourceExcerpt
from devmind.llm.openai_provider import OpenAIProvider
from devmind.llm.schemas import GroundingResult, ModelCitation, PlanResult, SynthesisResult


class FakeResponses:
    def __init__(self):
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs["text_format"]
        if schema is PlanResult:
            parsed = PlanResult(steps=["Search code.", "Inspect sources."])
        elif schema is SynthesisResult:
            parsed = SynthesisResult(
                answer="Auth is implemented here [auth.py:1-3].",
                citations=[ModelCitation(file_path="auth.py", start_line=1, end_line=3)],
                confidence="high",
            )
        elif schema is GroundingResult:
            parsed = GroundingResult(grounded=True)
        else:
            raise AssertionError(f"Unexpected schema: {schema}")
        return SimpleNamespace(output_parsed=parsed)


def test_openai_provider_uses_structured_responses_api():
    responses = FakeResponses()
    provider = OpenAIProvider(
        model="test-model",
        client=SimpleNamespace(responses=responses),
    )
    sources = [
        SourceExcerpt(
            repo_path="/repo",
            file_path="auth.py",
            start_line=1,
            end_line=3,
            content="1: class AuthMiddleware:\n2:     pass",
        )
    ]

    plan = provider.plan("How does auth work?", "flow")
    synthesis = provider.synthesize("How does auth work?", "flow", plan.steps, sources)
    validation = provider.validate("How does auth work?", synthesis.answer, sources)

    assert len(responses.calls) == 3
    assert all(call["model"] == "test-model" for call in responses.calls)
    assert responses.calls[0]["text_format"] is PlanResult
    assert responses.calls[1]["text_format"] is SynthesisResult
    assert responses.calls[2]["text_format"] is GroundingResult
    assert responses.calls[1]["max_output_tokens"] == 3000
    assert validation.grounded is True


def test_openai_provider_retries_truncated_structured_output():
    class TruncatedResponses(FakeResponses):
        def __init__(self):
            super().__init__()
            self.synthesis_attempts = 0

        def parse(self, **kwargs):
            if kwargs["text_format"] is SynthesisResult:
                self.synthesis_attempts += 1
                if self.synthesis_attempts == 1:
                    self.calls.append(kwargs)
                    try:
                        SynthesisResult.model_validate_json('{"answer":"cut off')
                    except ValidationError as exc:
                        raise exc
            return super().parse(**kwargs)

    responses = TruncatedResponses()
    provider = OpenAIProvider(
        model="test-model",
        client=SimpleNamespace(responses=responses),
    )
    sources = [
        SourceExcerpt(
            repo_path="/repo",
            file_path="auth.py",
            start_line=1,
            end_line=3,
            content="1: class AuthMiddleware:\n2:     pass",
        )
    ]

    result = provider.synthesize(
        "How does auth work?",
        "flow",
        ["Inspect auth."],
        sources,
    )

    synthesis_calls = [
        call for call in responses.calls
        if call["text_format"] is SynthesisResult
    ]
    assert result.confidence == "high"
    assert [call["max_output_tokens"] for call in synthesis_calls] == [3000, 6000]
