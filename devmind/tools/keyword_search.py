from __future__ import annotations

from pathlib import Path

from devmind.config import DEFAULT_DB_PATH
from devmind.indexer.code_indexer import connect
from devmind.indexer.metadata import RetrievedChunk


def keyword_code_search(query: str, repo: str | None = None, k: int = 10, db_path: Path = DEFAULT_DB_PATH) -> list[RetrievedChunk]:
    fts_query = normalize_fts_query(query)
    if not fts_query:
        return []

    repo_clause = ""
    params: list[object] = [fts_query]
    if repo:
        repo_clause = "AND chunks.repo = ?"
        params.append(repo)
    params.append(k)

    sql = f"""
        SELECT
            chunks.*,
            bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
        WHERE chunks_fts MATCH ?
        {repo_clause}
        ORDER BY score
        LIMIT ?
    """

    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    return [
        RetrievedChunk(
            chunk_id=row["chunk_id"],
            repo=row["repo"],
            commit_sha=row["commit_sha"],
            file_path=row["file_path"],
            language=row["language"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            symbol_name=row["symbol_name"],
            symbol_type=row["symbol_type"],
            imports=tuple(filter(None, row["imports"].split(","))),
            content=row["content"],
            score=float(row["score"]),
            source="keyword",
        )
        for row in rows
    ]


def normalize_fts_query(query: str) -> str:
    terms = []
    for raw in query.replace('"', " ").replace("'", " ").split():
        term = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
        if len(term) >= 2:
            terms.append(term)
    return " OR ".join(terms[:12])
