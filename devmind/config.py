from __future__ import annotations

from pathlib import Path


DEFAULT_STATE_DIR = Path(".devmind")
DEFAULT_DB_PATH = DEFAULT_STATE_DIR / "devmind.sqlite"

IGNORED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    "target",
    ".next",
    ".turbo",
    "coverage",
}

SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
    ".md",
    ".rst",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}


def load_environment() -> bool:
    """Load a local .env file without overriding exported variables."""
    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        return False
    return bool(load_dotenv(dotenv_path=dotenv_path, override=False))
