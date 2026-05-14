from __future__ import annotations

from pathlib import Path

from devmind.config import DEFAULT_DB_PATH
from devmind.embeddings.local import cosine_similarity, deserialize_vector, embed_text
from devmind.indexer.code_indexer import connect
from devmind.indexer.metadata import RetrievedChunk


def semantic_code_search(query: str, repo: str | None = None, k: int = 10, db_path: Path = DEFAULT_DB_PATH) -> list[RetrievedChunk]:
    query_vector = embed_text(query)
    repo_clause = ""
    params: list[object] = []
    if repo:
        repo_clause = "WHERE chunks.repo = ?"
        params.append(repo)

    sql = f"""
        SELECT chunks.*, chunk_embeddings.vector
        FROM chunk_embeddings
        JOIN chunks ON chunks.chunk_id = chunk_embeddings.chunk_id
        {repo_clause}
    """

    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    hits: list[RetrievedChunk] = []
    for row in rows:
        score = cosine_similarity(query_vector, deserialize_vector(row["vector"]))
        if score <= 0:
            continue
        hits.append(row_to_retrieved_chunk(row, score=score))

    return sorted(hits, key=lambda hit: hit.score, reverse=True)[:k]


def row_to_retrieved_chunk(row, score: float) -> RetrievedChunk:
    return RetrievedChunk(
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
        score=score,
        source="semantic",
    )

