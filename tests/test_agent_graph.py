from pathlib import Path

from devmind.agent.graph import run_agent
from devmind.indexer.code_indexer import index_repository
from devmind.llm.deterministic import DeterministicProvider
from devmind.llm.schemas import GroundingResult, ModelCitation, PlanResult, SynthesisResult


class FakeLLM:
    name = "fake"
    model = "fake-model"

    def plan(self, query, intent):
        return PlanResult(steps=["Find the implementation.", "Inspect exact source lines."])

    def synthesize(self, query, intent, plan, sources):
        source = sources[0]
        label = f"[{source.file_path}:{source.start_line}-{source.end_line}]"
        return SynthesisResult(
            answer=f"Authentication is handled by middleware {label}.",
            citations=[
                ModelCitation(
                    file_path=source.file_path,
                    start_line=source.start_line,
                    end_line=source.end_line,
                )
            ],
            confidence="high",
        )

    def validate(self, query, answer, sources):
        return GroundingResult(grounded=True)


class HallucinatingLLM(FakeLLM):
    def synthesize(self, query, intent, plan, sources):
        return SynthesisResult(
            answer="Authentication is handled elsewhere [missing.py:1-5].",
            citations=[
                ModelCitation(
                    file_path="missing.py",
                    start_line=1,
                    end_line=5,
                )
            ],
            confidence="high",
        )


def test_agent_graph_returns_cited_answer_and_trace(tmp_path: Path):
    repo = tmp_path / "sample"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "\n".join(
            [
                "class AuthMiddleware:",
                "    def authenticate(self, token):",
                "        return token == 'valid'",
            ]
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "devmind.sqlite"
    index_repository(repo, db_path=db_path, repo_name="sample")

    result = run_agent(
        "How does authentication flow work?",
        repo="sample",
        db_path=db_path,
        llm=DeterministicProvider(),
    )

    assert result["intent"] == "flow"
    assert result["retrieved_chunks"]
    assert result["citations"][0]["file_path"] == "auth.py"
    assert "auth.py:" in result["answer"]
    assert result["confidence"] in {"medium", "high"}
    assert [entry["node"] for entry in result["trace"]] == [
        "classify_intent",
        "plan_investigation",
        "retrieve_context",
        "inspect_sources",
        "synthesize_answer",
        "validate_grounding",
    ]


def test_agent_graph_handles_missing_evidence(tmp_path: Path):
    repo = tmp_path / "sample"
    repo.mkdir()
    (repo / "billing.py").write_text(
        "\n".join(
            [
                "class InvoiceCalculator:",
                "    def total(self, prices):",
                "        return sum(prices)",
            ]
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "devmind.sqlite"
    index_repository(repo, db_path=db_path, repo_name="sample")

    result = run_agent(
        "zqxj unknown subsystem",
        repo="sample",
        db_path=db_path,
        llm=DeterministicProvider(),
    )

    assert result["confidence"] == "low"
    assert result["citations"] == []
    assert "could not find enough indexed source evidence" in result["answer"]


def test_agent_graph_uses_llm_plan_and_synthesis(tmp_path: Path):
    repo = tmp_path / "sample"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "class AuthMiddleware:\n    def authenticate(self, token):\n        return bool(token)\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "devmind.sqlite"
    index_repository(repo, db_path=db_path, repo_name="sample")

    result = run_agent(
        "How does authentication work?",
        repo="sample",
        db_path=db_path,
        llm_mode="full",
        llm=FakeLLM(),
    )

    assert result["plan"] == ["Find the implementation.", "Inspect exact source lines."]
    assert "Authentication is handled by middleware" in result["answer"]
    assert result["confidence"] == "high"
    assert result["llm_provider"] == "fake"


def test_agent_graph_rejects_citations_outside_inspected_sources(tmp_path: Path):
    repo = tmp_path / "sample"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "class AuthMiddleware:\n    def authenticate(self, token):\n        return bool(token)\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "devmind.sqlite"
    index_repository(repo, db_path=db_path, repo_name="sample")

    result = run_agent(
        "How does authentication work?",
        repo="sample",
        db_path=db_path,
        llm_mode="full",
        llm=HallucinatingLLM(),
    )

    assert result["confidence"] == "low"
    validation = result["trace"][-1]["details"]
    assert validation["local_grounded"] is False


def test_fast_mode_skips_llm_planning_and_validation(tmp_path: Path):
    repo = tmp_path / "sample"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "class AuthMiddleware:\n    def authenticate(self, token):\n        return bool(token)\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "devmind.sqlite"
    index_repository(repo, db_path=db_path, repo_name="sample")

    class CountingLLM(FakeLLM):
        def __init__(self):
            self.plan_calls = 0
            self.synthesis_calls = 0
            self.validation_calls = 0

        def plan(self, query, intent):
            self.plan_calls += 1
            return super().plan(query, intent)

        def synthesize(self, query, intent, plan, sources):
            self.synthesis_calls += 1
            return super().synthesize(query, intent, plan, sources)

        def validate(self, query, answer, sources):
            self.validation_calls += 1
            return super().validate(query, answer, sources)

    llm = CountingLLM()
    result = run_agent(
        "How does authentication work?",
        repo="sample",
        db_path=db_path,
        llm_mode="fast",
        llm=llm,
    )

    assert llm.plan_calls == 0
    assert llm.synthesis_calls == 1
    assert llm.validation_calls == 0
    assert result["llm_mode"] == "fast"
    assert result["confidence"] == "high"
