import os
import importlib

from video2tasks.config import Config


def test_config_accepts_openai_backend() -> None:
    cfg = Config(worker={"backend": "openai"})

    assert cfg.worker.backend == "openai"
    assert cfg.worker.openai.model == "gpt-5.2"
    assert cfg.worker.openai.base_url == "https://api.openai.com/v1"


def test_config_reads_openai_values_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_TIMEOUT", "42")

    cfg = Config.from_env()

    assert cfg.worker.backend == "openai"
    assert cfg.worker.openai.api_key == "sk-test"
    assert cfg.worker.openai.model == "gpt-5.2"
    assert cfg.worker.openai.base_url == "https://example.test/v1"
    assert cfg.worker.openai.timeout_sec == 42.0


def test_validate_config_cli_module_is_importable() -> None:
    module = importlib.import_module("video2tasks.cli.validate_config")

    assert module is not None
