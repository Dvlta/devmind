from __future__ import annotations

from pathlib import Path

from devmind.config import DEFAULT_DB_PATH
from devmind.indexer.code_indexer import connect
from devmind.tools.file_inspector import inspect_file
from devmind.tools.hybrid_search import hybrid_code_search


def answer_with_citations(query: str, repo: str | None = None, db_path: Path = DEFAULT_DB_PATH, max_sources: int = 3) -> str:
    hits = hybrid_code_search(query, repo=repo, k=max_sources, db_path=db_path)
    if not hits:
        return "I could not find enough indexed source evidence to answer that question."

    repo_path = get_repo_path(hits[0].repo, db_path)
    lines = [
        "I found the most relevant indexed source evidence below.",
        "",
        "Sources:",
    ]

    for index, hit in enumerate(hits, start=1):
        excerpt = inspect_file(
            Path(repo_path),
            hit.file_path,
            start_line=hit.start_line,
            end_line=min(hit.end_line, hit.start_line + 20),
        )
        lines.append(f"{index}. {hit.file_path}:{excerpt.start_line}-{excerpt.end_line}")
        lines.append("```")
        lines.append(excerpt.content)
        lines.append("```")

    lines.extend(
        [
            "",
            "Next step: use these cited files as the grounding context for LLM synthesis.",
        ]
    )
    return "\n".join(lines)


def get_repo_path(repo: str, db_path: Path = DEFAULT_DB_PATH) -> str:
    with connect(db_path) as conn:
        row = conn.execute("SELECT repo_path FROM chunks WHERE repo = ? LIMIT 1", (repo,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown repo: {repo}")
    return str(row["repo_path"])
