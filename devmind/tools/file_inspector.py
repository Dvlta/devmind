from __future__ import annotations

from pathlib import Path

from devmind.indexer.metadata import SourceExcerpt


def inspect_file(repo_path: Path, file_path: str, start_line: int | None = None, end_line: int | None = None) -> SourceExcerpt:
    repo_path = repo_path.resolve()
    target = (repo_path / file_path).resolve()
    if not str(target).startswith(str(repo_path)):
        raise ValueError("File path escapes repository root")
    if not target.exists() or not target.is_file():
        raise ValueError(f"File does not exist: {file_path}")

    lines = target.read_text(encoding="utf-8").splitlines()
    if not lines:
        return SourceExcerpt(str(repo_path), file_path, 1, 1, "")

    start = max(1, start_line or 1)
    end = min(len(lines), end_line or len(lines))
    if start > end:
        raise ValueError("start_line must be less than or equal to end_line")

    numbered = [f"{line_no}: {lines[line_no - 1]}" for line_no in range(start, end + 1)]
    return SourceExcerpt(
        repo_path=str(repo_path),
        file_path=file_path,
        start_line=start,
        end_line=end,
        content="\n".join(numbered),
    )

