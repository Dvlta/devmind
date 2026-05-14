from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from devmind.config import DEFAULT_DB_PATH
from devmind.indexer.code_indexer import connect
from devmind.indexer.metadata import RetrievedChunk
from devmind.tools.keyword_search import keyword_code_search


def hybrid_code_search(query: str, repo: str | None = None, k: int = 10, db_path: Path = DEFAULT_DB_PATH) -> list[RetrievedChunk]:
    terms = query_terms(query)
    if not terms:
        return []

    candidates: dict[str, RetrievedChunk] = {}

    for rank, hit in enumerate(keyword_code_search(query, repo=repo, k=max(k * 4, 20), db_path=db_path), start=1):
        rank_score = 1.0 / rank
        score = rank_score + path_score(hit, terms) + symbol_score(hit, terms)
        candidates[hit.chunk_id] = replace(hit, score=score, source="hybrid")

    for hit in path_symbol_candidates(terms, repo=repo, k=max(k * 4, 20), db_path=db_path):
        score = 0.25 + path_score(hit, terms) + symbol_score(hit, terms)
        current = candidates.get(hit.chunk_id)
        if current is None or score > current.score:
            candidates[hit.chunk_id] = replace(hit, score=score, source="hybrid")

    return sorted(candidates.values(), key=lambda hit: hit.score, reverse=True)[:k]


def path_symbol_candidates(
    terms: list[str],
    repo: str | None = None,
    k: int = 20,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[RetrievedChunk]:
    clauses = []
    params: list[object] = []
    for term in terms:
        pattern = f"%{term}%"
        clauses.append("(LOWER(file_path) LIKE ? OR LOWER(COALESCE(symbol_name, '')) LIKE ? OR LOWER(imports) LIKE ?)")
        params.extend([pattern, pattern, pattern])

    repo_clause = ""
    if repo:
        repo_clause = "AND repo = ?"
        params.append(repo)
    params.append(k)

    sql = f"""
        SELECT *
        FROM chunks
        WHERE ({' OR '.join(clauses)})
        {repo_clause}
        ORDER BY file_path, start_line
        LIMIT ?
    """

    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    return [row_to_retrieved_chunk(row, score=0.0, source="path_symbol") for row in rows]


def row_to_retrieved_chunk(row, score: float, source: str) -> RetrievedChunk:
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
        source=source,
    )


def query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for raw in query.lower().replace("/", " ").replace(".", " ").replace("_", " ").split():
        term = "".join(ch for ch in raw if ch.isalnum() or ch == "-")
        if len(term) >= 2:
            terms.append(term)
    return list(dict.fromkeys(terms))[:12]


def path_score(hit: RetrievedChunk, terms: list[str]) -> float:
    path = hit.file_path.lower()
    basename = Path(path).name
    score = 0.0
    for term in terms:
        if term in basename:
            score += 0.35
        elif term in path:
            score += 0.2
    return score


def symbol_score(hit: RetrievedChunk, terms: list[str]) -> float:
    symbol = (hit.symbol_name or "").lower()
    imports = " ".join(hit.imports).lower()
    score = 0.0
    for term in terms:
        if symbol and term in symbol:
            score += 0.3
        if imports and term in imports:
            score += 0.1
    return score

