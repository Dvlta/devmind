import os

from devmind.config import load_environment


def test_load_environment_reads_dotenv_without_overriding_shell(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "DEVMIND_LLM_PROVIDER=openai\nDEVMIND_LLM_MODEL=env-model\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEVMIND_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("DEVMIND_LLM_MODEL", "shell-model")

    assert load_environment() is True
    assert os.environ["DEVMIND_LLM_PROVIDER"] == "openai"
    assert os.environ["DEVMIND_LLM_MODEL"] == "shell-model"
