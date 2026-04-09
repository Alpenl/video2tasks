import ast
import json
import logging
from io import StringIO
from pathlib import Path

import pytest
import video2tasks.server.app as server_app_module
import video2tasks.vlm.openai_api as openai_api_module
import video2tasks.worker.runner as worker_runner_module
from video2tasks.config import Config
from video2tasks.logging_utils import (
    FROZEN_EVENT_SCHEMAS,
    PACKAGE_LOGGER_NAME,
    log_event,
)


EXPECTED_FROZEN_EVENT_NAMES = {
    "artifact_extract_done",
    "artifact_reuse_hit",
    "job_dispatched",
    "infer_attempt",
    "job_done",
    "result_empty_retry",
    "result_timeout_retry",
    "fallback_applied",
    "sample_failed",
    "finalize_done",
}


def _build_test_logger() -> tuple[logging.Logger, StringIO]:
    stream = StringIO()
    logger = logging.getLogger("video2tasks.tests.logging")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger, stream


def _collect_log_event_calls(*module_paths: Path) -> dict[str, list[set[str]]]:
    calls: dict[str, list[set[str]]] = {}
    for module_path in module_paths:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "log_event":
                continue
            if len(node.args) < 2:
                continue
            event_arg = node.args[1]
            if not isinstance(event_arg, ast.Constant) or not isinstance(event_arg.value, str):
                continue

            event_name = event_arg.value
            explicit_fields = {kw.arg for kw in node.keywords if kw.arg is not None}
            calls.setdefault(event_name, []).append(explicit_fields)
    return calls


def test_server_worker_and_vlm_loggers_share_package_namespace(tmp_path) -> None:
    config = Config(
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={"auto_exit_after_all_done": False},
    )

    server_app_module.create_app(config)

    assert server_app_module.logger.name == "video2tasks.server.app"
    assert worker_runner_module.logger.name == "video2tasks.worker.runner"
    assert openai_api_module.logger.name == "video2tasks.vlm.openai_api"
    assert server_app_module.logger.parent.name == PACKAGE_LOGGER_NAME
    assert worker_runner_module.logger.parent.name == PACKAGE_LOGGER_NAME
    assert openai_api_module.logger.parent.name == PACKAGE_LOGGER_NAME


def test_config_logging_level_suppresses_info_output_across_package(tmp_path, capsys) -> None:
    config = Config(
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={"auto_exit_after_all_done": False},
        logging={"level": "WARNING"},
    )

    server_app_module.create_app(config)
    server_app_module.logger.info("[Server] hidden info")
    worker_runner_module.logger.info("[Worker] hidden info")
    openai_api_module.logger.warning("[OpenAI] visible warning")

    out = capsys.readouterr().out

    assert "[Server] hidden info" not in out
    assert "[Worker] hidden info" not in out
    assert "[OpenAI] visible warning" in out


def test_frozen_event_names_match_contract() -> None:
    assert set(FROZEN_EVENT_SCHEMAS.keys()) == EXPECTED_FROZEN_EVENT_NAMES


def test_log_event_emits_json_with_event_and_fields() -> None:
    logger, stream = _build_test_logger()

    log_event(
        logger,
        "artifact_reuse_hit",
        task_id="demo::sample_w0_r0",
        subset="demo",
        sample_id="sample",
        job_type="window_boundary",
        artifact_reuse=True,
        artifact_reuse_group="reuse::abc123",
        artifact_producer_task_id="demo::sample_w0_r0",
        artifact_consumer_task_id="demo::sample_w0_r1",
    )

    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "artifact_reuse_hit"
    assert payload["subset"] == "demo"
    assert payload["task_id"] == "demo::sample_w0_r0"
    assert payload["job_type"] == "window_boundary"
    assert payload["artifact_reuse"] is True
    assert payload["artifact_producer_task_id"] == "demo::sample_w0_r0"


def test_log_event_rejects_missing_required_frozen_event_fields() -> None:
    logger, _ = _build_test_logger()
    with pytest.raises(ValueError, match="job_dispatched"):
        log_event(logger, "job_dispatched", task_id="demo::sample_w0")


def test_frozen_event_emitters_include_required_fields() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    calls = _collect_log_event_calls(
        repo_root / "src/video2tasks/server/app.py",
        repo_root / "src/video2tasks/server/job_builder.py",
    )

    assert EXPECTED_FROZEN_EVENT_NAMES <= set(calls.keys())
    for event_name, schema in FROZEN_EVENT_SCHEMAS.items():
        assert event_name in calls
        required = set(schema.required_fields)
        for explicit_fields in calls[event_name]:
            assert required <= explicit_fields
