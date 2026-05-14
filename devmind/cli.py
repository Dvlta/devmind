from __future__ import annotations

import argparse
from pathlib import Path

from devmind.agent.simple_agent import answer_with_citations
from devmind.config import DEFAULT_DB_PATH
from devmind.indexer.code_indexer import index_repository
from devmind.tools.file_inspector import inspect_file
from devmind.tools.hybrid_search import hybrid_code_search
from devmind.tools.path_search import path_search


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="devmind")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to DevMind SQLite index")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a local repository")
    index_parser.add_argument("repo_path")
    index_parser.add_argument("--repo", help="Override repository name")

    search_parser = subparsers.add_parser("search", help="Search indexed code")
    search_parser.add_argument("query")
    search_parser.add_argument("--repo")
    search_parser.add_argument("-k", type=int, default=5)

    path_parser = subparsers.add_parser("path", help="Search indexed file paths")
    path_parser.add_argument("pattern")
    path_parser.add_argument("--repo")
    path_parser.add_argument("-k", type=int, default=10)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a source file with line numbers")
    inspect_parser.add_argument("repo_path")
    inspect_parser.add_argument("file_path")
    inspect_parser.add_argument("--start", type=int)
    inspect_parser.add_argument("--end", type=int)

    ask_parser = subparsers.add_parser("ask", help="Answer a question using indexed source citations")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--repo")

    args = parser.parse_args(argv)
    db_path = Path(args.db)

    if args.command == "index":
        count = index_repository(Path(args.repo_path), db_path=db_path, repo_name=args.repo)
        print(f"Indexed {count} chunks into {db_path}")
        return 0

    if args.command == "search":
        hits = hybrid_code_search(args.query, repo=args.repo, k=args.k, db_path=db_path)
        for hit in hits:
            print(f"{hit.file_path}:{hit.start_line}-{hit.end_line} [{hit.source}] score={hit.score:.3f}")
        return 0

    if args.command == "path":
        for file_path in path_search(args.pattern, repo=args.repo, k=args.k, db_path=db_path):
            print(file_path)
        return 0

    if args.command == "inspect":
        excerpt = inspect_file(Path(args.repo_path), args.file_path, args.start, args.end)
        print(excerpt.content)
        return 0

    if args.command == "ask":
        print(answer_with_citations(args.query, repo=args.repo, db_path=db_path))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
