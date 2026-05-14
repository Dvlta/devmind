from __future__ import annotations

import ast
from pathlib import Path

from devmind.indexer.metadata import CodeChunk


LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".h": "c/cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".md": "markdown",
    ".rst": "markdown",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}


def language_for_path(path: Path) -> str:
    return LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "text")


def chunk_text(
    *,
    repo: str,
    commit_sha: str,
    file_path: str,
    text: str,
    language: str,
    max_lines: int = 80,
    overlap: int = 15,
) -> list[CodeChunk]:
    if language == "python":
        python_chunks = chunk_python_text(
            repo=repo,
            commit_sha=commit_sha,
            file_path=file_path,
            text=text,
            max_lines=max_lines,
            overlap=overlap,
        )
        if python_chunks:
            return python_chunks

    return chunk_by_line_window(
        repo=repo,
        commit_sha=commit_sha,
        file_path=file_path,
        text=text,
        language=language,
        max_lines=max_lines,
        overlap=overlap,
        imports=(),
    )


def chunk_by_line_window(
    *,
    repo: str,
    commit_sha: str,
    file_path: str,
    text: str,
    language: str,
    max_lines: int = 80,
    overlap: int = 15,
    imports: tuple[str, ...] = (),
) -> list[CodeChunk]:
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[CodeChunk] = []
    step = max(1, max_lines - overlap)

    for start_index in range(0, len(lines), step):
        end_index = min(start_index + max_lines, len(lines))
        selected = lines[start_index:end_index]
        content = "\n".join(selected).strip()
        if content:
            chunks.append(
                CodeChunk(
                    repo=repo,
                    commit_sha=commit_sha,
                    file_path=file_path,
                    language=language,
                    start_line=start_index + 1,
                    end_line=end_index,
                    symbol_name=None,
                    symbol_type=None,
                    imports=imports,
                    content=content,
                )
            )
        if end_index == len(lines):
            break

    return chunks


def chunk_python_text(
    *,
    repo: str,
    commit_sha: str,
    file_path: str,
    text: str,
    max_lines: int = 80,
    overlap: int = 15,
) -> list[CodeChunk]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_by_line_window(
            repo=repo,
            commit_sha=commit_sha,
            file_path=file_path,
            text=text,
            language="python",
            max_lines=max_lines,
            overlap=overlap,
        )

    lines = text.splitlines()
    imports = extract_python_imports(tree)
    chunks: list[CodeChunk] = []

    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start_line = node_start_line(node)
        end_line = getattr(node, "end_lineno", None)
        if end_line is None:
            continue

        content = "\n".join(lines[start_line - 1 : end_line]).strip()
        if not content:
            continue

        chunks.append(
            CodeChunk(
                repo=repo,
                commit_sha=commit_sha,
                file_path=file_path,
                language="python",
                start_line=start_line,
                end_line=end_line,
                symbol_name=node.name,
                symbol_type=python_symbol_type(node),
                imports=imports,
                content=content,
            )
        )

    if chunks:
        return chunks

    return chunk_by_line_window(
        repo=repo,
        commit_sha=commit_sha,
        file_path=file_path,
        text=text,
        language="python",
        max_lines=max_lines,
        overlap=overlap,
        imports=imports,
    )


def extract_python_imports(tree: ast.AST) -> tuple[str, ...]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                names.append(f"{module}.{alias.name}" if module else alias.name)
    return tuple(dict.fromkeys(names))


def node_start_line(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    decorator_lines = [decorator.lineno for decorator in node.decorator_list]
    return min([node.lineno, *decorator_lines]) if decorator_lines else node.lineno


def python_symbol_type(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        return "class"
    if isinstance(node, ast.AsyncFunctionDef):
        return "async_function"
    return "function"
