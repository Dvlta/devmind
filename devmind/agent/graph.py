from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from devmind.agent.simple_agent import get_repo_path
from devmind.agent.state import AgentState, Citation
from devmind.indexer.metadata import RetrievedChunk, SourceExcerpt
from devmind.llm.base import LLMProvider
from devmind.llm.deterministic import DeterministicProvider
from devmind.llm.factory import create_llm_provider
from devmind.tools.file_inspector import inspect_file
from devmind.tools.hybrid_search import hybrid_code_search


def build_graph(llm: LLMProvider, llm_mode: str):
    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node(
        "plan_investigation",
        lambda state: plan_investigation(state, llm, llm_mode),
    )
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("inspect_sources", inspect_sources)
    graph.add_node("synthesize_answer", lambda state: synthesize_answer(state, llm))
    graph.add_node(
        "validate_grounding",
        lambda state: validate_grounding(state, llm, llm_mode),
    )

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "plan_investigation")
    graph.add_edge("plan_investigation", "retrieve_context")
    graph.add_edge("retrieve_context", "inspect_sources")
    graph.add_edge("inspect_sources", "synthesize_answer")
    graph.add_edge("synthesize_answer", "validate_grounding")
    graph.add_edge("validate_grounding", END)
    return graph.compile()


def run_agent(
    query: str,
    repo: str | None = None,
    db_path: Path | str = ".devmind/devmind.sqlite",
    max_sources: int = 3,
    provider: str | None = None,
    model: str | None = None,
    llm_mode: str | None = None,
    llm: LLMProvider | None = None,
) -> AgentState:
    llm_provider = llm or create_llm_provider(provider=provider, model=model)
    execution_mode = resolve_llm_mode(llm_mode)
    initial_state: AgentState = {
        "query": query,
        "repo": repo,
        "db_path": str(db_path),
        "max_sources": max_sources,
        "llm_provider": llm_provider.name,
        "llm_model": llm_provider.model,
        "llm_mode": execution_mode,
        "trace": [],
    }
    return build_graph(llm_provider, execution_mode).invoke(initial_state)


def classify_intent(state: AgentState) -> AgentState:
    query = state["query"].lower()
    if any(term in query for term in ["test", "tests", "pytest", "coverage", "spec"]):
        intent = "tests"
    elif any(term in query for term in ["bug", "issue", "failure", "error", "broken"]):
        intent = "bugs"
    elif any(term in query for term in ["doc", "docs", "readme", "design"]):
        intent = "docs"
    elif any(term in query for term in ["flow", "path", "request", "lifecycle"]):
        intent = "flow"
    elif any(term in query for term in ["architecture", "component", "module", "system"]):
        intent = "architecture"
    else:
        intent = "general"
    return {"intent": intent, "trace": append_trace(state, "classify_intent", {"intent": intent})}


def plan_investigation(
    state: AgentState,
    llm: LLMProvider,
    llm_mode: str,
) -> AgentState:
    planner = DeterministicProvider() if llm_mode == "fast" else llm
    result = planner.plan(state["query"], state["intent"])
    return {
        "plan": result.steps,
        "trace": append_trace(
            state,
            "plan_investigation",
            {
                "steps": len(result.steps),
                "provider": planner.name,
                "model": planner.model,
                "mode": llm_mode,
            },
        ),
    }


def retrieve_context(state: AgentState) -> AgentState:
    db_path = Path(state["db_path"])
    max_sources = state.get("max_sources", 3)
    hits = hybrid_code_search(state["query"], repo=state.get("repo"), k=max_sources, db_path=db_path)
    tool_call = {
        "tool": "hybrid_code_search",
        "input": {"query": state["query"], "repo": state.get("repo"), "k": max_sources},
        "output_count": len(hits),
    }
    return {
        "retrieved_chunks": hits,
        "trace": append_trace(
            state,
            "retrieve_context",
            {
                "tool_call": tool_call,
                "top_sources": [chunk_label(hit) for hit in hits],
            },
        ),
    }


def inspect_sources(state: AgentState) -> AgentState:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {
            "inspected_sources": [],
            "citations": [],
            "trace": append_trace(state, "inspect_sources", {"source_count": 0}),
        }

    db_path = Path(state["db_path"])
    repo_path = Path(get_repo_path(chunks[0].repo, db_path))
    excerpts: list[SourceExcerpt] = []
    citations: list[Citation] = []

    for chunk in chunks:
        excerpt = inspect_file(
            repo_path,
            chunk.file_path,
            start_line=chunk.start_line,
            end_line=min(chunk.end_line, chunk.start_line + 20),
        )
        excerpts.append(excerpt)
        citations.append(
            {
                "file_path": excerpt.file_path,
                "start_line": excerpt.start_line,
                "end_line": excerpt.end_line,
            }
        )

    return {
        "inspected_sources": excerpts,
        "citations": citations,
        "trace": append_trace(
            state,
            "inspect_sources",
            {"source_count": len(excerpts), "citations": citations},
        ),
    }


def synthesize_answer(state: AgentState, llm: LLMProvider) -> AgentState:
    excerpts = state.get("inspected_sources", [])
    if not excerpts:
        answer = "I could not find enough indexed source evidence to answer that question."
        return {
            "answer": answer,
            "confidence": "low",
            "citations": [],
            "gaps": ["No relevant indexed sources were retrieved."],
            "trace": append_trace(
                state,
                "synthesize_answer",
                {"confidence": "low", "provider": "none", "reason": "no_evidence"},
            ),
        }

    result = llm.synthesize(
        query=state["query"],
        intent=state["intent"],
        plan=state.get("plan", []),
        sources=excerpts,
    )
    citations: list[Citation] = [
        {
            "file_path": citation.file_path,
            "start_line": citation.start_line,
            "end_line": citation.end_line,
        }
        for citation in result.citations
    ]
    return {
        "answer": result.answer,
        "citations": citations,
        "confidence": result.confidence,
        "gaps": result.gaps,
        "trace": append_trace(
            state,
            "synthesize_answer",
            {
                "confidence": result.confidence,
                "citation_count": len(citations),
                "provider": llm.name,
                "model": llm.model,
            },
        ),
    }


def validate_grounding(
    state: AgentState,
    llm: LLMProvider,
    llm_mode: str,
) -> AgentState:
    citations = state.get("citations", [])
    answer = state.get("answer", "")
    sources = state.get("inspected_sources", [])
    local_grounded = citations_are_valid(citations, sources, answer)
    if not sources:
        model_grounded = False
        unsupported_claims = ["No source evidence was available."]
    elif llm_mode == "fast":
        model_grounded = True
        unsupported_claims = []
    else:
        result = llm.validate(state["query"], answer, sources)
        model_grounded = result.grounded
        unsupported_claims = result.unsupported_claims
    grounded = local_grounded and model_grounded
    confidence = state.get("confidence", "low") if grounded else "low"
    return {
        "confidence": confidence,
        "unsupported_claims": unsupported_claims,
        "trace": append_trace(
            state,
            "validate_grounding",
            {
                "grounded": grounded,
                "local_grounded": local_grounded,
                "model_grounded": model_grounded,
                "citation_count": len(citations),
                "confidence": confidence,
                "unsupported_claims": unsupported_claims,
                "provider": llm.name if sources and llm_mode == "full" else "local",
                "mode": llm_mode,
            },
        ),
    }


def append_trace(state: AgentState, node: str, details: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("trace", []), {"node": node, "details": details}]


def chunk_label(chunk: RetrievedChunk) -> str:
    return f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"


def resolve_llm_mode(value: str | None) -> str:
    import os

    mode = (value or os.getenv("DEVMIND_LLM_MODE", "fast")).lower()
    if mode not in {"fast", "full"}:
        raise ValueError(f"Unsupported LLM mode: {mode}")
    return mode


def citations_are_valid(
    citations: list[Citation],
    sources: list[SourceExcerpt],
    answer: str,
) -> bool:
    allowed = {
        (source.file_path, source.start_line, source.end_line)
        for source in sources
    }
    if not citations:
        return False
    for citation in citations:
        key = (
            citation["file_path"],
            citation["start_line"],
            citation["end_line"],
        )
        label = (
            f"[{citation['file_path']}:"
            f"{citation['start_line']}-{citation['end_line']}]"
        )
        if key not in allowed or label not in answer:
            return False
    return True
