from devmind.indexer.chunkers import chunk_text


def test_chunk_text_preserves_line_ranges():
    text = "\n".join(f"line {i}" for i in range(1, 101))

    chunks = chunk_text(
        repo="repo",
        commit_sha="sha",
        file_path="app.py",
        text=text,
        language="python",
        max_lines=25,
        overlap=5,
    )

    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 25
    assert chunks[1].start_line == 21
    assert chunks[-1].end_line == 100


def test_python_chunking_extracts_top_level_symbols_and_imports():
    text = "\n".join(
        [
            "import os",
            "from pathlib import Path",
            "",
            "class AuthMiddleware:",
            "    def authenticate(self, token):",
            "        return token",
            "",
            "async def load_user():",
            "    return None",
        ]
    )

    chunks = chunk_text(
        repo="repo",
        commit_sha="sha",
        file_path="auth.py",
        text=text,
        language="python",
    )

    assert [chunk.symbol_name for chunk in chunks] == ["AuthMiddleware", "load_user"]
    assert [chunk.symbol_type for chunk in chunks] == ["class", "async_function"]
    assert chunks[0].start_line == 4
    assert chunks[0].end_line == 6
    assert "os" in chunks[0].imports
    assert "pathlib.Path" in chunks[0].imports
