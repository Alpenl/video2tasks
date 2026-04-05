import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from video2tasks.config import Config
from video2tasks.server.app import (
    _count_failed_samples,
    _final_exit_code,
    _normalize_loaded_boundary_refinement_vlm_json,
    _normalize_loaded_window_vlm_json,
    _requeue_empty_result,
    create_app,
)


def _make_app(tmp_path):
    config = Config(
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={"auto_exit_after_all_done": False},
    )
    app = create_app(config)
    return app, TestClient(app)


def test_requeue_empty_result_respects_retry_limit() -> None:
    job_queue = [{"task_id": "existing"}]
    retry_counts = {}
    job = {"task_id": "target"}

    attempt1, requeued1 = _requeue_empty_result(job_queue, retry_counts, "target", job, 2)
    attempt2, requeued2 = _requeue_empty_result(job_queue, retry_counts, "target", job, 2)
    attempt3, requeued3 = _requeue_empty_result(job_queue, retry_counts, "target", job, 2)

    assert (attempt1, requeued1) == (1, True)
    assert (attempt2, requeued2) == (2, True)
    assert (attempt3, requeued3) == (3, False)
    assert retry_counts == {"target": 3}
    assert job_queue == [
        {"task_id": "existing"},
        {"task_id": "target"},
        {"task_id": "target"},
    ]


def test_submit_result_is_idempotent_and_rejects_stale_dispatches(tmp_path) -> None:
    app, client = _make_app(tmp_path)
    task_id = "demo::sample_w0"
    meta = {"subset": "demo", "sample_id": "sample", "window_id": 0, "job_type": "window_boundary"}
    base_job = {"task_id": task_id, "meta": meta}
    app.state.inflight[task_id] = {"job": base_job, "ts": time.time(), "dispatch_id": "d1"}

    received = client.post(
        "/submit_result",
        json={
            "task_id": task_id,
            "dispatch_id": "d1",
            "vlm_json": {"transitions": [], "instructions": ["Add potatoes"]},
            "meta": meta,
        },
    )
    assert received.json()["status"] == "received"

    duplicate = client.post(
        "/submit_result",
        json={
            "task_id": task_id,
            "dispatch_id": "d1",
            "vlm_json": {"transitions": [], "instructions": ["Add potatoes"]},
            "meta": meta,
        },
    )
    assert duplicate.json()["status"] == "already_received"

    app.state.inflight[task_id] = {"job": base_job, "ts": time.time(), "dispatch_id": "d2"}
    stale = client.post(
        "/submit_result",
        json={
            "task_id": task_id,
            "dispatch_id": "d0",
            "vlm_json": {"transitions": [1], "instructions": ["Stir the pot"]},
            "meta": meta,
        },
    )
    assert stale.json()["status"] == "stale_ignored"

    windows_path = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample" / "windows.jsonl"
    lines = windows_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["dispatch_id"] == "d1"
    assert record["window_id"] == 0


def test_submit_result_persists_repeat_index_for_window_boundary(tmp_path) -> None:
    app, client = _make_app(tmp_path)
    task_id = "demo::sample_w0_r1"
    meta = {
        "subset": "demo",
        "sample_id": "sample",
        "window_id": 0,
        "repeat_index": 1,
        "job_type": "window_boundary",
    }
    base_job = {"task_id": task_id, "meta": meta}
    app.state.inflight[task_id] = {"job": base_job, "ts": time.time(), "dispatch_id": "d1"}

    received = client.post(
        "/submit_result",
        json={
            "task_id": task_id,
            "dispatch_id": "d1",
            "vlm_json": {"transitions": [], "instructions": ["Add potatoes"]},
            "meta": meta,
        },
    )

    assert received.json()["status"] == "received"

    windows_path = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample" / "windows.jsonl"
    record = json.loads(windows_path.read_text(encoding="utf-8").strip())
    assert record["window_id"] == 0
    assert record["repeat_index"] == 1


def test_submit_result_exhausts_empty_retry_budget_and_records_terminal_empty(tmp_path) -> None:
    config = Config(
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={"auto_exit_after_all_done": False, "max_empty_retries_per_job": 2},
    )
    app = create_app(config)
    client = TestClient(app)

    task_id = "demo::sample_w0"
    meta = {"subset": "demo", "sample_id": "sample", "window_id": 0, "job_type": "window_boundary"}
    base_job = {"task_id": task_id, "meta": meta}

    app.state.inflight[task_id] = {"job": base_job, "ts": time.time(), "dispatch_id": "d1"}
    attempt1 = client.post(
        "/submit_result",
        json={"task_id": task_id, "dispatch_id": "d1", "vlm_json": {}, "meta": meta},
    )
    assert attempt1.json()["status"] == "retry_triggered"
    assert len(app.state.job_queue) == 1

    requeued_job = app.state.job_queue.pop(0)
    app.state.inflight[task_id] = {"job": requeued_job, "ts": time.time(), "dispatch_id": "d2"}
    attempt2 = client.post(
        "/submit_result",
        json={"task_id": task_id, "dispatch_id": "d2", "vlm_json": {}, "meta": meta},
    )
    assert attempt2.json()["status"] == "retry_triggered"
    assert len(app.state.job_queue) == 1

    requeued_job = app.state.job_queue.pop(0)
    app.state.inflight[task_id] = {"job": requeued_job, "ts": time.time(), "dispatch_id": "d3"}
    attempt3 = client.post(
        "/submit_result",
        json={"task_id": task_id, "dispatch_id": "d3", "vlm_json": {}, "meta": meta},
    )
    assert attempt3.json()["status"] == "empty_retry_exhausted"
    assert len(app.state.job_queue) == 0
    assert app.state.completed_dispatch_ids[task_id] == "d3"

    windows_path = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample" / "windows.jsonl"
    lines = windows_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["dispatch_id"] == "d3"
    assert record["terminal_error"] == "empty_retry_exhausted"
    assert record["vlm_json"] == {}


def test_submit_result_retries_invalid_structured_payload_and_records_terminal_empty(tmp_path) -> None:
    config = Config(
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={"auto_exit_after_all_done": False, "max_empty_retries_per_job": 1},
    )
    app = create_app(config)
    client = TestClient(app)

    task_id = "demo::sample_w0"
    meta = {
        "subset": "demo",
        "sample_id": "sample",
        "window_id": 0,
        "job_type": "window_boundary",
        "logical_frame_count": 4,
    }
    base_job = {"task_id": task_id, "meta": meta}

    app.state.inflight[task_id] = {"job": base_job, "ts": time.time(), "dispatch_id": "d1"}
    attempt1 = client.post(
        "/submit_result",
        json={
            "task_id": task_id,
            "dispatch_id": "d1",
            "vlm_json": {"transitions": [1], "instructions": ["Only one instruction"]},
            "meta": meta,
        },
    )
    assert attempt1.json()["status"] == "retry_triggered"

    requeued_job = app.state.job_queue.pop(0)
    app.state.inflight[task_id] = {"job": requeued_job, "ts": time.time(), "dispatch_id": "d2"}
    attempt2 = client.post(
        "/submit_result",
        json={
            "task_id": task_id,
            "dispatch_id": "d2",
            "vlm_json": {"transitions": [1], "instructions": ["Only one instruction"]},
            "meta": meta,
        },
    )
    assert attempt2.json()["status"] == "empty_retry_exhausted"

    windows_path = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample" / "windows.jsonl"
    record = json.loads(windows_path.read_text(encoding="utf-8").strip())
    assert record["terminal_error"] == "empty_retry_exhausted"


def test_timeout_exhaustion_records_terminal_error(tmp_path) -> None:
    config = Config(
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={
            "auto_exit_after_all_done": False,
            "inflight_timeout_sec": 0.01,
            "max_retries_per_job": 1,
        },
    )
    app = create_app(config)

    task_id = "demo::sample_w0"
    meta = {
        "subset": "demo",
        "sample_id": "sample",
        "window_id": 0,
        "job_type": "window_boundary",
    }
    base_job = {"task_id": task_id, "meta": meta}

    app.state.inflight[task_id] = {"job": base_job, "ts": time.time() - 1.0, "dispatch_id": "d1"}
    time.sleep(1.2)
    assert len(app.state.job_queue) == 1

    requeued_job = app.state.job_queue.pop(0)
    app.state.inflight[task_id] = {"job": requeued_job, "ts": time.time() - 1.0, "dispatch_id": "d2"}
    time.sleep(1.2)

    windows_path = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample" / "windows.jsonl"
    record = json.loads(windows_path.read_text(encoding="utf-8").strip())
    assert record["dispatch_id"] == "d2"
    assert record["terminal_error"] == "timeout_retry_exhausted"
    assert app.state.completed_dispatch_ids[task_id] == "d2"
    assert len(app.state.job_queue) == 0


def test_normalize_loaded_window_vlm_json_rejects_transition_beyond_logical_frame_count() -> None:
    record = {
        "meta": {"logical_frame_count": 4},
        "vlm_json": {"transitions": [99], "instructions": ["Add potatoes", "Stir the pot"]},
    }

    assert _normalize_loaded_window_vlm_json(record) == {}


def test_normalize_loaded_boundary_refinement_vlm_json_rejects_transition_outside_middle_candidates() -> None:
    record = {
        "frame_ids": list(range(8)),
        "vlm_json": {"transitions": [0], "instructions": ["Add potatoes", "Stir the pot"]},
    }

    assert _normalize_loaded_boundary_refinement_vlm_json(record) == {}


def test_final_exit_code_returns_nonzero_when_any_sample_failed() -> None:
    states = {
        "demo": {"sample_status": {"done": 3, "failed": 4}},
        "demo2": {"sample_status": {"pending": 0}},
    }

    assert _count_failed_samples(states) == 1
    assert _final_exit_code(states) == 1
    assert _final_exit_code({"demo": {"sample_status": {"done": 3}}}) == 0
