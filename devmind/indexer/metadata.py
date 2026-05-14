from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodeChunk:
    repo: str
    commit_sha: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    symbol_name: str | None
    symbol_type: str | None
    imports: tuple[str, ...]
    content: str

    @property
    def chunk_id(self) -> str:
        return f"{self.repo}:{self.commit_sha}:{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    repo: str
    commit_sha: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    symbol_name: str | None
    symbol_type: str | None
    imports: tuple[str, ...]
    content: str
    score: float
    source: str


@dataclass(frozen=True)
class SourceExcerpt:
    repo_path: str
    file_path: str
    start_line: int
    end_line: int
    content: str
