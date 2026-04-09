import ast
import json
import logging
import re
from io import StringIO
from pathlib import Path

import pytest
import video2tasks.server.app as server_app_module
from video2tasks.server.protocol import InlineImageTransport, JobEnvelope
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


def _extract_doc_required_fields(doc_path: Path) -> dict[str, tuple[str, ...]]:
    lines = doc_path.read_text(encoding="utf-8").splitlines()
    parsed: dict[str, tuple[str, ...]] = {}

    for index, line in enumerate(lines):
        heading_match = re.match(r"^###\s+\d+\)\s+`([^`]+)`\s*$", line.strip())
        if not heading_match:
            continue

        event_name = heading_match.group(1)
        required_line = ""
        for seek in range(index + 1, len(lines)):
            candidate = lines[seek].strip()
            if candidate.startswith("### "):
                break
            if candidate.startswith("`"):
                required_line = candidate
                break

        if not required_line:
            continue

        parsed[event_name] = tuple(
            part.strip().strip("`")
            for part in required_line.split(",")
            if part.strip()
        )

    return parsed


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


@pytest.mark.parametrize("field_name", ["subset", "sample_id", "job_type", "task_id", "dispatch_id"])
def test_log_event_rejects_blank_identifier_required_fields(field_name: str) -> None:
    logger, _ = _build_test_logger()
    payload = {
        "task_id": "demo::sample_w0",
        "dispatch_id": "d1",
        "subset": "demo",
        "sample_id": "sample_001",
        "job_type": "window_boundary",
        "source_count": 1,
        "transport_mode": "inline",
        "artifact_reuse": False,
    }
    payload[field_name] = "   "

    with pytest.raises(ValueError, match=field_name):
        log_event(logger, "job_dispatched", **payload)


def test_frozen_event_emitters_include_required_fields() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    calls = _collect_log_event_calls(
        repo_root / "src/video2tasks/server/app.py",
        repo_root / "src/video2tasks/server/routes.py",
        repo_root / "src/video2tasks/server/producer.py",
        repo_root / "src/video2tasks/server/runtime_state.py",
        repo_root / "src/video2tasks/server/job_builder.py",
    )

    assert EXPECTED_FROZEN_EVENT_NAMES <= set(calls.keys())
    for event_name, schema in FROZEN_EVENT_SCHEMAS.items():
        assert event_name in calls
        required = set(schema.required_fields)
        for explicit_fields in calls[event_name]:
            assert required <= explicit_fields


def test_event_schema_doc_matches_frozen_schema_required_fields() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    doc_required_fields = _extract_doc_required_fields(
        repo_root / "docs/observability/event-schema.md",
    )
    code_required_fields = {
        event_name: schema.required_fields
        for event_name, schema in FROZEN_EVENT_SCHEMAS.items()
    }

    assert set(doc_required_fields.keys()) == set(code_required_fields.keys())
    for event_name, expected_required_fields in code_required_fields.items():
        assert doc_required_fields[event_name] == expected_required_fields


def test_get_job_emits_job_dispatched_event_payload(tmp_path, capsys) -> None:
    config = Config(
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={"auto_exit_after_all_done": False},
    )
    app = server_app_module.create_app(config)
    app.state.job_queue.append(
        JobEnvelope(
            task_id="demo_smoke::sample_001_w0",
            meta={
                "subset": "demo_smoke",
                "sample_id": "sample_001",
                "job_type": "window_boundary",
                "artifact_reuse": False,
            },
            image_transport=InlineImageTransport(images=["img_b64"]),
        )
    )

    get_job_route = next(route for route in app.routes if getattr(route, "path", "") == "/get_job")
    result = get_job_route.endpoint()
    assert result["status"] == "ok"

    out_lines = capsys.readouterr().out.splitlines()
    payload = next(
        json.loads(line)
        for line in out_lines
        if line.startswith("{") and '"event": "job_dispatched"' in line
    )

    assert payload["event"] == "job_dispatched"
    assert payload["task_id"] == "demo_smoke::sample_001_w0"
    assert payload["dispatch_id"] == "d1"
    assert payload["subset"] == "demo_smoke"
    assert payload["sample_id"] == "sample_001"
    assert payload["job_type"] == "window_boundary"
