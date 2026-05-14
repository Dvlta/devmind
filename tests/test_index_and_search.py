from pathlib import Path

from devmind.indexer.code_indexer import index_repository
from devmind.tools.file_inspector import inspect_file
from devmind.tools.hybrid_search import hybrid_code_search
from devmind.tools.keyword_search import keyword_code_search
from devmind.tools.path_search import path_search


def test_index_search_path_and_inspect(tmp_path: Path):
    repo = tmp_path / "sample"
    repo.mkdir()
    source = repo / "auth.py"
    source.write_text(
        "\n".join(
            [
                "class AuthMiddleware:",
                "    def authenticate(self, token):",
                "        return token == 'valid'",
            ]
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "devmind.sqlite"

    count = index_repository(repo, db_path=db_path)

    assert count == 1
    hits = keyword_code_search("authenticate token", db_path=db_path)
    assert hits
    assert hits[0].file_path == "auth.py"
    assert hits[0].symbol_name == "AuthMiddleware"
    assert path_search("auth", db_path=db_path) == ["auth.py"]

    excerpt = inspect_file(repo, "auth.py", 1, 2)
    assert "1: class AuthMiddleware:" in excerpt.content
    assert "2:     def authenticate" in excerpt.content


def test_hybrid_search_boosts_path_and_symbol_matches(tmp_path: Path):
    repo = tmp_path / "sample"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "\n".join(
            [
                "class AuthMiddleware:",
                "    def check(self):",
                "        return True",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "misc.py").write_text(
        "\n".join(
            [
                "def helper():",
                "    message = 'auth auth auth auth'",
                "    return message",
            ]
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "devmind.sqlite"
    index_repository(repo, db_path=db_path)

    hits = hybrid_code_search("auth middleware", db_path=db_path)

    assert hits
    assert hits[0].file_path == "auth.py"
    assert hits[0].symbol_name == "AuthMiddleware"
    assert len({hit.chunk_id for hit in hits}) == len(hits)
