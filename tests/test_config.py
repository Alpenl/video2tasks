import importlib
from textwrap import dedent
import pytest

from video2tasks.config import Config, LLMMergeConfig


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
    assert cfg.llm_merge.enabled is False
    assert cfg.llm_merge.model == "gpt-5.2"
    assert cfg.llm_merge.summary_levels == [1, 1, 1]
    assert cfg.llm_merge.granularity == "guarded"
    assert cfg.llm_merge.max_attempts == 3
    assert cfg.llm_merge.repeat_count == 1
    assert cfg.llm_merge.boundary_vote_threshold == 0.5
    assert cfg.llm_merge.min_output_ratio == 0.35
    assert cfg.llm_merge.coarse_min_output_ratio == 0.18
    assert cfg.llm_merge.coarse_max_supported_anchors_per_range == 1
    assert cfg.llm_merge.coarse_anchor_min_spacing_segments == 3
    assert cfg.llm_merge.coarse_anchor_min_side_segments == 2
    assert cfg.llm_merge.coarse_anchor_min_score == 1.03
    assert cfg.llm_merge.protect_supported_boundaries is True
    assert cfg.llm_merge.protected_boundary_support_threshold == 0.45
    assert cfg.llm_merge.protect_distinct_sequence_markers is True
    assert cfg.llm_merge.protect_instruction_drift is True
    assert cfg.llm_merge.protect_duplicate_tail_anchor is True
    assert cfg.llm_merge.duplicate_tail_anchor_min_frames == 5


def test_export_config_defaults() -> None:
    cfg = Config()

    assert cfg.export.enabled is False
    assert cfg.export.mode == "annotated"
    assert cfg.export.clips_dirname == "clips"
    assert cfg.export.manifest_name == "manifest.json"
    assert cfg.export.annotated_dirname == "exports"
    assert cfg.export.annotated_name == "annotated.mp4"
    assert cfg.export.subtitles.enabled is True
    assert cfg.export.subtitles.language == "zh"
    assert cfg.export.subtitles.position == "top_right"
    assert cfg.export.subtitles.font_file == ""
    assert cfg.export.subtitles.font_size == 28


def test_config_reads_export_values_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EXPORT_ENABLED", "true")
    monkeypatch.setenv("EXPORT_MODE", "both")
    monkeypatch.setenv("EXPORT_ANNOTATED_NAME", "demo.mp4")
    monkeypatch.setenv("EXPORT_SUBTITLE_LANGUAGE", "en")
    monkeypatch.setenv("EXPORT_SUBTITLE_POSITION", "bottom_left")
    monkeypatch.setenv("EXPORT_SUBTITLE_FONT_SIZE", "36")

    cfg = Config.from_env()

    assert cfg.export.enabled is True
    assert cfg.export.mode == "both"
    assert cfg.export.annotated_name == "demo.mp4"
    assert cfg.export.subtitles.language == "en"
    assert cfg.export.subtitles.position == "bottom_left"
    assert cfg.export.subtitles.font_size == 36


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


def test_config_reads_run_force_resume_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("RUN_FORCE_RESUME", "true")

    cfg = Config.from_env()

    assert cfg.run.force_resume is True


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


def test_config_reads_llm_merge_values_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MERGE_ENABLED", "true")
    monkeypatch.setenv("LLM_MERGE_API_KEY", "sk-merge")
    monkeypatch.setenv("LLM_MERGE_MODEL", "gpt-5.2")
    monkeypatch.setenv("LLM_MERGE_BASE_URL", "https://merge.test/v1")
    monkeypatch.setenv("LLM_MERGE_TIMEOUT", "33")
    monkeypatch.setenv("LLM_MERGE_REASONING_EFFORT", "medium")
    monkeypatch.setenv("LLM_MERGE_MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("LLM_MERGE_MAX_ATTEMPTS", "7")
    monkeypatch.setenv("LLM_MERGE_SUMMARY_LEVELS", "[1,0,1]")
    monkeypatch.setenv("LLM_MERGE_REPEAT_COUNT", "3")
    monkeypatch.setenv("LLM_MERGE_BOUNDARY_VOTE_THRESHOLD", "0.6")
    monkeypatch.setenv("LLM_MERGE_GRANULARITY", "coarse")
    monkeypatch.setenv("LLM_MERGE_MIN_INPUT_SEGMENTS", "10")
    monkeypatch.setenv("LLM_MERGE_MAX_INPUT_SEGMENTS", "80")
    monkeypatch.setenv("LLM_MERGE_MIN_OUTPUT_RATIO", "0.5")
    monkeypatch.setenv("LLM_MERGE_COARSE_MIN_OUTPUT_RATIO", "0.2")
    monkeypatch.setenv("LLM_MERGE_COARSE_MAX_SUPPORTED_ANCHORS_PER_RANGE", "2")
    monkeypatch.setenv("LLM_MERGE_COARSE_ANCHOR_MIN_SPACING_SEGMENTS", "5")
    monkeypatch.setenv("LLM_MERGE_COARSE_ANCHOR_MIN_SIDE_SEGMENTS", "4")
    monkeypatch.setenv("LLM_MERGE_COARSE_ANCHOR_MIN_SCORE", "1.4")
    monkeypatch.setenv("LLM_MERGE_PROTECT_SUPPORTED_BOUNDARIES", "false")
    monkeypatch.setenv("LLM_MERGE_PROTECTED_BOUNDARY_SUPPORT_THRESHOLD", "0.8")
    monkeypatch.setenv("LLM_MERGE_PROTECT_DISTINCT_SEQUENCE_MARKERS", "false")
    monkeypatch.setenv("LLM_MERGE_PROTECT_INSTRUCTION_DRIFT", "false")
    monkeypatch.setenv("LLM_MERGE_PROTECT_DUPLICATE_TAIL_ANCHOR", "false")
    monkeypatch.setenv("LLM_MERGE_DUPLICATE_TAIL_ANCHOR_MIN_FRAMES", "9")

    cfg = Config.from_env()

    assert cfg.llm_merge.enabled is True
    assert cfg.llm_merge.api_key == "sk-merge"
    assert cfg.llm_merge.model == "gpt-5.2"
    assert cfg.llm_merge.base_url == "https://merge.test/v1"
    assert cfg.llm_merge.timeout_sec == 33.0
    assert cfg.llm_merge.reasoning_effort == "medium"
    assert cfg.llm_merge.max_output_tokens == 4096
    assert cfg.llm_merge.max_attempts == 7
    assert cfg.llm_merge.summary_levels == [1, 0, 1]
    assert cfg.llm_merge.repeat_count == 3
    assert cfg.llm_merge.boundary_vote_threshold == 0.6
    assert cfg.llm_merge.granularity == "coarse"
    assert cfg.llm_merge.min_input_segments == 10
    assert cfg.llm_merge.max_input_segments == 80
    assert cfg.llm_merge.min_output_ratio == 0.5
    assert cfg.llm_merge.coarse_min_output_ratio == 0.2
    assert cfg.llm_merge.coarse_max_supported_anchors_per_range == 2
    assert cfg.llm_merge.coarse_anchor_min_spacing_segments == 5
    assert cfg.llm_merge.coarse_anchor_min_side_segments == 4
    assert cfg.llm_merge.coarse_anchor_min_score == 1.4
    assert cfg.llm_merge.protect_supported_boundaries is False
    assert cfg.llm_merge.protected_boundary_support_threshold == 0.8
    assert cfg.llm_merge.protect_distinct_sequence_markers is False
    assert cfg.llm_merge.protect_instruction_drift is False
    assert cfg.llm_merge.protect_duplicate_tail_anchor is False
    assert cfg.llm_merge.duplicate_tail_anchor_min_frames == 9


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
            llm_merge:
              enabled: true
              api_key: yaml-merge-key
              model: gpt-5.2
              base_url: https://yaml-merge.test/v1
              granularity: coarse
              coarse_min_output_ratio: 0.12
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
    assert cfg.llm_merge.enabled is True
    assert cfg.llm_merge.api_key == "yaml-merge-key"
    assert cfg.llm_merge.base_url == "https://yaml-merge.test/v1"
    assert cfg.llm_merge.granularity == "coarse"
    assert cfg.llm_merge.coarse_min_output_ratio == 0.12


def test_config_load_does_not_auto_discover_config_yaml_from_cwd(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        dedent(
            """
            run:
              run_id: discovered-config
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = Config.load()

    assert cfg.run.run_id == "default"


def test_config_load_uses_video2tasks_config_env_path(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "named-config.yaml"
    cfg_path.write_text(
        dedent(
            """
            run:
              run_id: env-config
            server:
              port: 8010
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VIDEO2TASKS_CONFIG", str(cfg_path))

    cfg = Config.load()

    assert cfg.run.run_id == "env-config"
    assert cfg.server.port == 8010


def test_max_empty_retries_per_job_defaults_to_finite_value() -> None:
    cfg = Config()

    assert cfg.server.max_empty_retries_per_job == 3


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


def test_llm_merge_config_accepts_named_summary_levels_mapping() -> None:
    cfg = LLMMergeConfig(summary_levels={"coarse": 1, "medium": 0, "fine": 1})

    assert cfg.summary_levels == [1, 0, 1]
    assert cfg.summary_levels_named == {"coarse": 1, "medium": 0, "fine": 1}


def test_llm_merge_config_accepts_partial_named_summary_levels_mapping() -> None:
    cfg = LLMMergeConfig(summary_levels={"fine": 1})

    assert cfg.summary_levels == [0, 0, 1]
    assert cfg.summary_levels_named == {"coarse": 0, "medium": 0, "fine": 1}


def test_config_accepts_named_summary_levels_mapping() -> None:
    cfg = Config(llm_merge={"summary_levels": {"coarse": 0, "medium": 1, "fine": 1}})

    assert cfg.llm_merge.summary_levels == [0, 1, 1]
    assert cfg.llm_merge.summary_levels_named == {"coarse": 0, "medium": 1, "fine": 1}


def test_export_subtitle_language_accepts_language_aliases() -> None:
    cfg = Config(export={"subtitles": {"language": "en-US"}})

    assert cfg.export.subtitles.language == "en"

    cfg = Config(export={"subtitles": {"language": "zh-Hans"}})

    assert cfg.export.subtitles.language == "zh"


def test_config_reads_export_subtitle_language_aliases_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EXPORT_SUBTITLE_LANGUAGE", "en-GB")

    cfg = Config.from_env()

    assert cfg.export.subtitles.language == "en"


def test_config_reads_llm_merge_summary_levels_named_mapping_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MERGE_SUMMARY_LEVELS", '{"coarse": 1, "medium": 0, "fine": 1}')

    cfg = Config.from_env()

    assert cfg.llm_merge.summary_levels == [1, 0, 1]
    assert cfg.llm_merge.summary_levels_named == {"coarse": 1, "medium": 0, "fine": 1}

