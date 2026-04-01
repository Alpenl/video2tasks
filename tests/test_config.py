import os
import importlib

from video2tasks.config import Config


def test_config_accepts_openai_backend() -> None:
    cfg = Config(worker={"backend": "openai"})

    assert cfg.worker.backend == "openai"
    assert cfg.worker.openai.model == "gpt-5.2"
    assert cfg.worker.openai.base_url == "https://api.openai.com/v1"


def test_config_accepts_gemini_backend() -> None:
    cfg = Config(worker={"backend": "gemini"})

    assert cfg.worker.backend == "gemini"
    assert cfg.worker.gemini.api_mode == "native"
    assert cfg.worker.gemini.model == "gemini-3-flash-preview"
    assert cfg.worker.gemini.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert cfg.windowing.adaptive_merge_guard is True
    assert cfg.windowing.adaptive_merge_min_segments == 8
    assert cfg.windowing.adaptive_merge_collapse_ratio == 0.6
    assert cfg.windowing.boundary_support_threshold == 0.9
    assert cfg.windowing.refine_final_instructions is True


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


def test_config_reads_gemini_values_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://example.test/gemini")
    monkeypatch.setenv("GEMINI_API_MODE", "openai_compatible")
    monkeypatch.setenv("GEMINI_TIMEOUT", "18")

    cfg = Config.from_env()

    assert cfg.worker.backend == "gemini"
    assert cfg.worker.gemini.api_key == "gem-test"
    assert cfg.worker.gemini.model == "gemini-2.5-flash"
    assert cfg.worker.gemini.base_url == "https://example.test/gemini"
    assert cfg.worker.gemini.api_mode == "openai_compatible"
    assert cfg.worker.gemini.timeout_sec == 18.0


def test_validate_config_cli_module_is_importable() -> None:
    module = importlib.import_module("video2tasks.cli.validate_config")

    assert module is not None
