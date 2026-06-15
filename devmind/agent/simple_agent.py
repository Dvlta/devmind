from __future__ import annotations

from pathlib import Path

from devmind.config import DEFAULT_DB_PATH
from devmind.indexer.code_indexer import connect


def answer_with_citations(query: str, repo: str | None = None, db_path: Path = DEFAULT_DB_PATH, max_sources: int = 3) -> str:
    from devmind.agent.graph import run_agent

    return run_agent(query, repo=repo, db_path=db_path, max_sources=max_sources)["answer"]


def get_repo_path(repo: str, db_path: Path = DEFAULT_DB_PATH) -> str:
    with connect(db_path) as conn:
        row = conn.execute("SELECT repo_path FROM chunks WHERE repo = ? LIMIT 1", (repo,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown repo: {repo}")
    return str(row["repo_path"])
