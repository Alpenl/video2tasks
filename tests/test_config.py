import importlib
from textwrap import dedent
import pytest

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
    assert cfg.windowing.use_contact_sheets is False
    assert cfg.windowing.contact_sheet_rows == 4
    assert cfg.windowing.contact_sheet_cols == 4
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


def test_worker_count_defaults_to_seven() -> None:
    cfg = Config()

    assert cfg.worker.count == 7


def test_config_reads_worker_count_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("WORKER_COUNT", "3")

    cfg = Config.from_env()

    assert cfg.worker.count == 3


def test_config_reads_gemini_values_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://example.test/gemini")
    monkeypatch.setenv("GEMINI_API_MODE", "openai_compatible")
    monkeypatch.setenv("GEMINI_TIMEOUT", "18")

    cfg = Config.from_env()

    assert cfg.worker.backend == "gemini"
    assert cfg.worker.gemini.api_key == "gem-test"
    assert cfg.worker.gemini.model == "gemini-3-flash-preview"
    assert cfg.worker.gemini.base_url == "https://example.test/gemini"
    assert cfg.worker.gemini.api_mode == "openai_compatible"
    assert cfg.worker.gemini.timeout_sec == 18.0


def test_config_from_yaml_applies_env_overrides(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        dedent(
            """
            run:
              run_id: yaml-run
            server:
              port: 8001
            worker:
              backend: gemini
              server_url: http://127.0.0.1:8001
              gemini:
                api_key: yaml-key
                model: gemini-3-flash-preview
                api_mode: native
                base_url: https://yaml.test/v1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("RUN_ID", "env-run")
    monkeypatch.setenv("PORT", "8123")
    monkeypatch.setenv("SERVER_URL", "http://127.0.0.1:8123")
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("GEMINI_API_MODE", "openai_compatible")

    cfg = Config.from_yaml(cfg_path)

    assert cfg.run.run_id == "env-run"
    assert cfg.server.port == 8123
    assert cfg.worker.server_url == "http://127.0.0.1:8123"
    assert cfg.worker.gemini.api_key == "env-key"
    assert cfg.worker.gemini.base_url == "https://env.test/v1"
    assert cfg.worker.gemini.api_mode == "openai_compatible"


def test_validate_config_cli_module_is_importable() -> None:
    module = importlib.import_module("video2tasks.cli.validate_config")

    assert module is not None


def test_config_accepts_multi_probe_boundary_prompt_mode() -> None:
    cfg = Config(windowing={"boundary_prompt_mode": "multi_probe_scan"})

    assert cfg.windowing.boundary_prompt_mode == "multi_probe_scan"
    assert cfg.windowing.segment_labeling_mode == "inline"


def test_config_accepts_candidate_boundary_prompt_mode() -> None:
    cfg = Config(windowing={"boundary_prompt_mode": "candidate_scan"})

    assert cfg.windowing.boundary_prompt_mode == "candidate_scan"


def test_config_accepts_contact_sheet_settings() -> None:
    cfg = Config(
        windowing={
            "use_contact_sheets": True,
            "contact_sheet_rows": 3,
            "contact_sheet_cols": 5,
        }
    )

    assert cfg.windowing.use_contact_sheets is True
    assert cfg.windowing.contact_sheet_rows == 3
    assert cfg.windowing.contact_sheet_cols == 5


def test_config_accepts_window_repeat_count() -> None:
    cfg = Config(windowing={"window_repeat_count": 2})

    assert cfg.windowing.window_repeat_count == 2


def test_config_rejects_unknown_boundary_prompt_mode() -> None:
    with pytest.raises(ValueError, match="boundary_prompt_mode"):
        Config(windowing={"boundary_prompt_mode": "unknown_mode"})


def test_config_accepts_deferred_segment_labeling_mode() -> None:
    cfg = Config(windowing={"segment_labeling_mode": "deferred"})

    assert cfg.windowing.segment_labeling_mode == "deferred"


def test_config_accepts_boundary_refinement_settings() -> None:
    cfg = Config(
        windowing={
            "enable_boundary_refinement": True,
            "boundary_refinement_window_sec": 4.0,
            "boundary_refinement_frames_per_window": 12,
            "boundary_refinement_abstain_merge_max_support": 0.0,
        }
    )

    assert cfg.windowing.enable_boundary_refinement is True
    assert cfg.windowing.boundary_refinement_window_sec == 4.0
    assert cfg.windowing.boundary_refinement_frames_per_window == 12
    assert cfg.windowing.boundary_refinement_abstain_merge_max_support == 0.0


def test_config_rejects_unknown_segment_labeling_mode() -> None:
    with pytest.raises(ValueError, match="segment_labeling_mode"):
        Config(windowing={"segment_labeling_mode": "unknown_mode"})


def test_config_rejects_invalid_backend_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("BACKEND", "not_a_backend")

    with pytest.raises(ValueError, match="backend"):
        Config.from_env()
