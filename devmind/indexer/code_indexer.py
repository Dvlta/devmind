from __future__ import annotations

import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from devmind.config import DEFAULT_DB_PATH, IGNORED_DIR_NAMES, SUPPORTED_EXTENSIONS
from devmind.embeddings.local import DEFAULT_DIMENSION, embed_text, serialize_vector
from devmind.indexer.chunkers import chunk_text, language_for_path
from devmind.indexer.metadata import CodeChunk


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    file_path TEXT NOT NULL,
    language TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    symbol_name TEXT,
    symbol_type TEXT,
    imports TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    file_path,
    language,
    symbol_name,
    imports,
    content
);

CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(repo, file_path);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
);
"""


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    ensure_schema(conn)
    return conn


def index_repository(repo_path: Path, db_path: Path = DEFAULT_DB_PATH, repo_name: str | None = None) -> int:
    repo_path = repo_path.resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        raise ValueError(f"Repository path does not exist or is not a directory: {repo_path}")

    repo = repo_name or repo_path.name
    commit_sha = current_commit_sha(repo_path)
    indexed_at = datetime.now(timezone.utc).isoformat()
    chunks: list[CodeChunk] = []

    for path in iter_indexable_files(repo_path):
        relative_path = path.relative_to(repo_path).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        language = language_for_path(path)
        chunks.extend(
            chunk_text(
                repo=repo,
                commit_sha=commit_sha,
                file_path=relative_path,
                text=text,
                language=language,
            )
        )

    with connect(db_path) as conn:
        conn.execute(
            "DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE repo = ?)",
            (repo,),
        )
        conn.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE repo = ?)",
            (repo,),
        )
        conn.execute("DELETE FROM chunks WHERE repo = ?", (repo,))
        for chunk in chunks:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, repo, repo_path, commit_sha, file_path, language,
                    start_line, end_line, symbol_name, symbol_type, imports, content, indexed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.repo,
                    str(repo_path),
                    chunk.commit_sha,
                    chunk.file_path,
                    chunk.language,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.symbol_name,
                    chunk.symbol_type,
                    ",".join(chunk.imports),
                    chunk.content,
                    indexed_at,
                ),
            )
            conn.execute(
                """
                INSERT INTO chunks_fts (chunk_id, file_path, language, symbol_name, imports, content)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.file_path,
                    chunk.language,
                    chunk.symbol_name or "",
                    " ".join(chunk.imports),
                    chunk.content,
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO chunk_embeddings (chunk_id, provider, dimension, vector)
                VALUES (?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    "local-hash",
                    DEFAULT_DIMENSION,
                    serialize_vector(embed_text(embedding_text(chunk))),
                ),
            )

    return len(chunks)


def ensure_schema(conn: sqlite3.Connection) -> None:
    chunk_columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    if "imports" not in chunk_columns:
        conn.execute("ALTER TABLE chunks ADD COLUMN imports TEXT NOT NULL DEFAULT ''")

    fts_columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks_fts)").fetchall()}
    if "imports" not in fts_columns:
        conn.execute("DROP TABLE chunks_fts")
        conn.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                file_path,
                language,
                symbol_name,
                imports,
                content
            )
            """
        )
        conn.execute(
            """
            INSERT INTO chunks_fts (chunk_id, file_path, language, symbol_name, imports, content)
            SELECT chunk_id, file_path, language, COALESCE(symbol_name, ''), imports, content
            FROM chunks
            """
        )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            vector TEXT NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
        )
        """
    )


def embedding_text(chunk: CodeChunk) -> str:
    metadata = " ".join(
        part
        for part in [
            chunk.file_path,
            chunk.language,
            chunk.symbol_name or "",
            chunk.symbol_type or "",
            " ".join(chunk.imports),
        ]
        if part
    )
    return f"{metadata}\n{chunk.content}"


def iter_indexable_files(repo_path: Path):
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIR_NAMES for part in path.relative_to(repo_path).parts):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        yield path


def current_commit_sha(repo_path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"
