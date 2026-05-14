from __future__ import annotations

from pathlib import Path

from devmind.config import DEFAULT_DB_PATH
from devmind.indexer.code_indexer import connect


def path_search(pattern: str, repo: str | None = None, k: int = 20, db_path: Path = DEFAULT_DB_PATH) -> list[str]:
    like_pattern = f"%{pattern.strip()}%"
    repo_clause = ""
    params: list[object] = [like_pattern]
    if repo:
        repo_clause = "AND repo = ?"
        params.append(repo)
    params.append(k)

    sql = f"""
        SELECT DISTINCT file_path
        FROM chunks
        WHERE file_path LIKE ?
        {repo_clause}
        ORDER BY file_path
        LIMIT ?
    """

    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [row["file_path"] for row in rows]

