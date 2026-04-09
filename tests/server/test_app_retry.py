import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

import video2tasks.server.app as app_module
import video2tasks.server.windowing as windowing_module
from video2tasks.config import Config
from video2tasks.server.app import (
    _count_failed_samples,
    _final_exit_code,
    _normalize_loaded_boundary_refinement_vlm_json,
    _normalize_loaded_window_vlm_json,
    _requeue_empty_result,
    create_app,
)
from video2tasks.server.protocol import JobEnvelope, SharedFSImageTransport
from video2tasks.server.run_manifest import build_run_manifest, run_manifest_path
from video2tasks.server.task_artifacts import ArtifactPayloadIssue, ArtifactPayloadValidationError
from video2tasks.server.windowing import BoundaryRefinementWindow, Window


def _make_app(tmp_path):
    config = Config(
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={"auto_exit_after_all_done": False},
    )
    app = create_app(config)
    return app, TestClient(app)


def _start_runtime(app) -> None:
    app.state.runtime.start()


def _make_dataset_app(
    tmp_path,
    *,
    sample_id: str = "sample",
    with_mp4: bool = False,
    windowing: dict | None = None,
    export: dict | None = None,
    llm_merge: dict | None = None,
    start_runtime: bool = True,
):
    subset = "demo"
    data_root = tmp_path / "data"
    sample_dir = data_root / subset / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    if with_mp4:
        (sample_dir / "Frame_demo.mp4").write_bytes(b"not-a-real-video")

    config = Config(
        datasets=[{"root": str(data_root), "subset": subset}],
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={"auto_exit_after_all_done": False},
        windowing=windowing or {},
        export=export or {},
        llm_merge=llm_merge or {},
    )
    app = create_app(config)
    if start_runtime:
        _start_runtime(app)
    sample_out_dir = Path(tmp_path) / subset / "testrun" / "samples" / sample_id
    return app, sample_dir, sample_out_dir


def _seed_dataset_run_manifest(
    tmp_path,
    *,
    windowing: dict | None = None,
    export: dict | None = None,
    llm_merge: dict | None = None,
    required_stages: list[str] | None = None,
) -> None:
    subset = "demo"
    data_root = tmp_path / "data"
    (data_root / subset).mkdir(parents=True, exist_ok=True)
    config = Config(
        datasets=[{"root": str(data_root), "subset": subset}],
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={"auto_exit_after_all_done": False},
        windowing=windowing or {},
        export=export or {},
        llm_merge=llm_merge or {},
    )
    run_dir = Path(tmp_path) / subset / "testrun"
    manifest = build_run_manifest(
        run_dir=run_dir,
        subset=subset,
        data_root=str(data_root),
        config=config,
    )
    if required_stages is not None:
        manifest.required_stages = [str(stage) for stage in required_stages]
    path = run_manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.02) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    assert predicate()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _override_required_stages(app, stages: list[str], *, subset: str = "demo") -> None:
    manifest_path = Path(app.state.run_manifest_paths[subset])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["required_stages"] = [str(stage) for stage in stages]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class _NoopFrameExtractor:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_many_b64_with_artifacts(self, *_args, **_kwargs):
        return [], None


class _InvalidArtifactFrameExtractor:
    reason = "image_decode_failed"
    source = "cv2_frame"

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_many_b64_with_artifacts(self, *_args, **_kwargs):
        raise ArtifactPayloadValidationError(
            [
                ArtifactPayloadIssue(
                    index=0,
                    reason=type(self).reason,
                    byte_size=0,
                    source=type(self).source,
                )
            ]
        )



class _CountingArtifactFrameExtractor:
    calls = 0

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_many_b64_with_artifacts(self, *_args, **_kwargs):
        type(self).calls += 1
        call_index = type(self).calls

        record = type(
            "ArtifactRecord",
            (),
            {"path": f"/tmp/repeat_artifact_{call_index:02d}.png"},
        )()
        batch = type(
            "ArtifactBatch",
            (),
            {
                "records": [record],
                "manifest_path": f"/tmp/repeat_artifact_{call_index:02d}.json",
            },
        )()
        return [], batch


class _ProducerBatchedCountingArtifactFrameExtractor(_CountingArtifactFrameExtractor):
    pass


def _single_window() -> list[Window]:
    return [Window(window_id=0, start_frame=0, end_frame=15, frame_ids=[0, 5, 10, 15])]


def _two_windows_with_same_frames() -> list[Window]:
    return [
        Window(window_id=0, start_frame=0, end_frame=15, frame_ids=[0, 5, 10, 15]),
        Window(window_id=1, start_frame=8, end_frame=23, frame_ids=[0, 5, 10, 15]),
    ]


def _write_valid_mp4(path: Path, *, fps: float = 30.0, frame_count: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (8, 8))
    assert writer.isOpened()
    try:
        for frame_index in range(frame_count):
            frame = np.full((8, 8, 3), frame_index % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records),
        encoding='utf-8',
    )


def _append_jsonl_record(path: Path, record: dict) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def _stage2_result(
    segments: list[dict],
    *,
    hierarchy: dict | None = None,
    subtitle_items: list[dict] | None = None,
    merge_diagnostics: dict | None = None,
    summary_diagnostics: dict | None = None,
    subtitle_diagnostics: dict | None = None,
) -> dict:
    return {
        "stage": "stage2",
        "version": 2,
        "merge": {
            "applied": bool((merge_diagnostics or {}).get("llm_merge_applied", False)),
            "segments": [dict(segment) for segment in segments],
            "diagnostics": dict(merge_diagnostics or {}),
        },
        "summary": {
            "applied": bool((summary_diagnostics or {}).get("llm_summary_applied", False)),
            "hierarchy": None if hierarchy is None else dict(hierarchy),
            "diagnostics": dict(summary_diagnostics or {}),
        },
        "subtitles": {
            "requested_language": str((subtitle_diagnostics or {}).get("llm_subtitle_requested_language", "en")),
            "target_language": str((subtitle_diagnostics or {}).get("llm_subtitle_requested_language", "en")),
            "language": str((subtitle_diagnostics or {}).get("llm_subtitle_language", "en")),
            "output_language": str((subtitle_diagnostics or {}).get("llm_subtitle_output_language", "en")),
            "source_instruction_language": "en",
            "applied": bool((subtitle_diagnostics or {}).get("llm_subtitle_applied", False)),
            "items": [dict(item) for item in (subtitle_items or [])],
            "diagnostics": dict(subtitle_diagnostics or {}),
        },
    }


def _seed_completed_window_result(
    sample_out_dir: Path,
    *,
    repeat_indices: tuple[int, ...] = (0,),
    vlm_json: dict | None = None,
) -> None:
    _write_jsonl(
        sample_out_dir / 'windows.jsonl',
        [
            {
                'task_id': f'demo::sample_w0_r{repeat_index}',
                'dispatch_id': f'd{repeat_index + 1}',
                'window_id': 0,
                'repeat_index': repeat_index,
                'logical_frame_count': 4,
                'vlm_json': vlm_json
                or {'transitions': [1], 'instructions': ['Add potatoes', 'Stir the pot']},
            }
            for repeat_index in repeat_indices
        ],
    )


def _install_basic_finalize_mocks(monkeypatch, *, segments: list[dict] | None = None) -> None:
    monkeypatch.setattr(app_module, 'read_video_info', lambda _mp4: (30.0, 16))
    monkeypatch.setattr(app_module, 'build_windows', lambda *_args, **_kwargs: _single_window())
    monkeypatch.setattr(
        app_module,
        'build_segments_via_cuts',
        lambda *_args, **_kwargs: {
            'segments': segments
            or [{'seg_id': 0, 'start_frame': 0, 'end_frame': 15, 'instruction': 'Add potatoes'}]
        },
    )
    monkeypatch.setattr(app_module, 'FrameExtractor', _NoopFrameExtractor)


def test_get_job_returns_typed_image_transport_with_dispatch_id(tmp_path) -> None:
    app, client = _make_app(tmp_path)
    app.state.job_queue.append(
        JobEnvelope(
            task_id="demo::sample_w0",
            meta={"subset": "demo", "sample_id": "sample", "job_type": "window_boundary"},
            image_transport=SharedFSImageTransport(
                image_paths=["/tmp/frame_000.png"],
                artifact_manifest_path="/tmp/manifest.json",
            ),
        )
    )

    response = client.get("/get_job")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "data": {
            "task_id": "demo::sample_w0",
            "dispatch_id": "d1",
            "meta": {
                "subset": "demo",
                "sample_id": "sample",
                "job_type": "window_boundary",
                "dispatch_id": "d1",
            },
            "image_transport": {
                "mode": "shared_fs",
                "image_paths": ["/tmp/frame_000.png"],
                "artifact_manifest_path": "/tmp/manifest.json",
            },
        },
    }


def test_window_scheduling_duplicate_check_accepts_typed_jobs_in_queue(tmp_path, monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))

    def paused_build_windows(*_args, **_kwargs):
        entered.set()
        assert release.wait(2.0)
        return _single_window()

    monkeypatch.setattr(app_module, "build_windows", paused_build_windows)
    monkeypatch.setattr(app_module, "FrameExtractor", _NoopFrameExtractor)

    app, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=True)
    _wait_until(entered.is_set)

    app.state.job_queue.append(
        JobEnvelope(
            task_id="demo::existing_seg0",
            meta={"subset": "demo", "sample_id": "sample", "job_type": "segment_label"},
            image_transport=SharedFSImageTransport(image_paths=["/tmp/existing.png"]),
        )
    )

    release.set()

    _wait_until(lambda: len(app.state.job_queue) >= 2)

    queued_task_ids = [job.task_id if isinstance(job, JobEnvelope) else job["task_id"] for job in app.state.job_queue]
    assert queued_task_ids == ["demo::existing_seg0", "demo::sample_w0_r0"]

    (sample_out_dir / ".FAILED").touch()
    time.sleep(0.05)
    app.state.runtime.stop()
    app.state.runtime.join(timeout=1.0)



def test_step_a_repeat_jobs_reuse_contact_sheet_artifacts_per_logical_window(tmp_path, monkeypatch) -> None:
    _CountingArtifactFrameExtractor.calls = 0
    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))
    monkeypatch.setattr(app_module, "build_windows", lambda *_args, **_kwargs: _single_window())
    monkeypatch.setattr(app_module, "FrameExtractor", _CountingArtifactFrameExtractor)

    app, _, sample_out_dir = _make_dataset_app(
        tmp_path,
        with_mp4=True,
        windowing={"window_repeat_count": 2, "use_contact_sheets": True},
    )

    _wait_until(lambda: len(app.state.job_queue) >= 2)

    assert _CountingArtifactFrameExtractor.calls == 1
    assert [job.task_id for job in app.state.job_queue[:2]] == [
        "demo::sample_w0_r0",
        "demo::sample_w0_r1",
    ]
    assert [job.meta["repeat_index"] for job in app.state.job_queue[:2]] == [0, 1]
    assert app.state.job_queue[0].image_transport.model_dump() == app.state.job_queue[1].image_transport.model_dump()
    assert app.state.job_queue[0].image_transport.artifact_manifest_path == "/tmp/repeat_artifact_01.json"

    (sample_out_dir / ".FAILED").touch()
    time.sleep(0.05)
    app.state.runtime.stop()
    app.state.runtime.join(timeout=1.0)


def test_step_a_repeat_jobs_do_not_reuse_contact_sheet_artifacts_across_logical_windows(tmp_path, monkeypatch) -> None:
    _CountingArtifactFrameExtractor.calls = 0
    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 24))
    monkeypatch.setattr(app_module, "build_windows", lambda *_args, **_kwargs: _two_windows_with_same_frames())
    monkeypatch.setattr(app_module, "FrameExtractor", _CountingArtifactFrameExtractor)

    app, _, sample_out_dir = _make_dataset_app(
        tmp_path,
        with_mp4=True,
        windowing={"window_repeat_count": 2, "use_contact_sheets": True},
    )

    _wait_until(lambda: len(app.state.job_queue) >= 4)

    assert _CountingArtifactFrameExtractor.calls == 2
    assert [job.task_id for job in app.state.job_queue[:4]] == [
        "demo::sample_w0_r0",
        "demo::sample_w0_r1",
        "demo::sample_w1_r0",
        "demo::sample_w1_r1",
    ]
    assert app.state.job_queue[0].image_transport.model_dump() == app.state.job_queue[1].image_transport.model_dump()
    assert app.state.job_queue[2].image_transport.model_dump() == app.state.job_queue[3].image_transport.model_dump()
    assert (
        app.state.job_queue[0].image_transport.artifact_manifest_path
        != app.state.job_queue[2].image_transport.artifact_manifest_path
    )

    (sample_out_dir / ".FAILED").touch()
    time.sleep(0.05)
    app.state.runtime.stop()
    app.state.runtime.join(timeout=1.0)


def test_step_a_repeat_jobs_reuse_contact_sheet_artifacts_across_producer_batches(tmp_path, monkeypatch) -> None:
    _ProducerBatchedCountingArtifactFrameExtractor.calls = 0
    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))
    monkeypatch.setattr(app_module, "build_windows", lambda *_args, **_kwargs: _single_window())
    monkeypatch.setattr(app_module, "FrameExtractor", _ProducerBatchedCountingArtifactFrameExtractor)

    app, _, sample_out_dir = _make_dataset_app(
        tmp_path,
        with_mp4=True,
        windowing={"window_repeat_count": 3, "use_contact_sheets": True},
        start_runtime=False,
    )
    app.state.step_a_producer_batch_limit = 0
    _start_runtime(app)

    _wait_until(lambda: len(app.state.job_queue) >= 3)

    assert _ProducerBatchedCountingArtifactFrameExtractor.calls == 1
    assert [job.task_id for job in app.state.job_queue[:3]] == [
        "demo::sample_w0_r0",
        "demo::sample_w0_r1",
        "demo::sample_w0_r2",
    ]
    assert app.state.job_queue[1].meta["artifact_reuse"] is True
    assert app.state.job_queue[1].meta["artifact_producer_task_id"] == "demo::sample_w0_r0"
    assert app.state.job_queue[2].meta["artifact_producer_task_id"] == "demo::sample_w0_r0"
    manifest_paths = {job.image_transport.artifact_manifest_path for job in app.state.job_queue[:3]}
    assert manifest_paths == {"/tmp/repeat_artifact_01.json"}

    (sample_out_dir / ".FAILED").touch()
    time.sleep(0.05)
    app.state.runtime.stop()
    app.state.runtime.join(timeout=1.0)


def test_refinement_repeat_jobs_do_not_reuse_step_a_contact_sheet_artifacts(tmp_path, monkeypatch) -> None:
    _CountingArtifactFrameExtractor.calls = 0
    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))
    monkeypatch.setattr(windowing_module, "read_video_info", lambda _mp4: (30.0, 16))
    monkeypatch.setattr(app_module, "build_windows", lambda *_args, **_kwargs: _single_window())
    monkeypatch.setattr(
        app_module,
        "build_refinement_windows",
        lambda *_args, **_kwargs: [
            Window(window_id=100, start_frame=4, end_frame=11, frame_ids=[4, 6, 8, 10])
        ],
    )
    monkeypatch.setattr(app_module, "FrameExtractor", _CountingArtifactFrameExtractor)

    app, sample_dir, sample_out_dir = _make_dataset_app(
        tmp_path,
        with_mp4=True,
        windowing={
            "window_repeat_count": 2,
            "use_contact_sheets": True,
            "enable_refinement_pass": True,
        },
        start_runtime=False,
    )
    _write_valid_mp4(sample_dir / "Frame_demo.mp4")
    _seed_completed_window_result(sample_out_dir, repeat_indices=(0, 1))
    _start_runtime(app)

    _wait_until(lambda: len(app.state.job_queue) >= 2)

    assert _CountingArtifactFrameExtractor.calls == 2
    assert [job.task_id for job in app.state.job_queue[:2]] == [
        "demo::sample_rw100_r0",
        "demo::sample_rw100_r1",
    ]
    assert (
        app.state.job_queue[0].image_transport.artifact_manifest_path
        != app.state.job_queue[1].image_transport.artifact_manifest_path
    )

    (sample_out_dir / ".FAILED").touch()
    time.sleep(0.05)
    app.state.runtime.stop()
    app.state.runtime.join(timeout=1.0)


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


def test_submit_result_uses_authoritative_server_meta_for_window_validation(tmp_path) -> None:
    app, client = _make_app(tmp_path)
    task_id = "demo::sample_w0"
    authoritative_meta = {
        "subset": "demo",
        "sample_id": "sample",
        "window_id": 0,
        "job_type": "window_boundary",
        "logical_frame_count": 4,
    }
    worker_meta = {
        **authoritative_meta,
        "logical_frame_count": 100,
    }
    app.state.inflight[task_id] = {
        "job": JobEnvelope(
            task_id=task_id,
            meta=authoritative_meta,
            image_transport=SharedFSImageTransport(image_paths=["/tmp/frame_000.png"]),
        ),
        "ts": time.time(),
        "dispatch_id": "d1",
    }

    response = client.post(
        "/submit_result",
        json={
            "task_id": task_id,
            "dispatch_id": "d1",
            "vlm_json": {"transitions": [99], "instructions": ["Before", "After"]},
            "meta": worker_meta,
        },
    )

    assert response.json()["status"] == "retry_triggered"
    assert len(app.state.job_queue) == 1


def test_submit_result_rejects_boundary_refinement_transition_outside_candidates(tmp_path) -> None:
    app, client = _make_app(tmp_path)
    task_id = "demo::sample_b7"
    meta = {
        "subset": "demo",
        "sample_id": "sample",
        "job_type": "boundary_refinement",
        "boundary_id": 7,
        "frame_ids": list(range(8)),
        "logical_frame_count": 8,
    }
    app.state.inflight[task_id] = {
        "job": JobEnvelope(
            task_id=task_id,
            meta=meta,
            image_transport=SharedFSImageTransport(image_paths=["/tmp/frame_001.png"]),
        ),
        "ts": time.time(),
        "dispatch_id": "d1",
    }

    response = client.post(
        "/submit_result",
        json={
            "task_id": task_id,
            "dispatch_id": "d1",
            "vlm_json": {"transitions": [1], "instructions": ["Before", "After"]},
            "meta": meta,
        },
    )

    assert response.json()["status"] == "retry_triggered"
    assert len(app.state.job_queue) == 1


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
    _start_runtime(app)

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


def test_normalize_loaded_window_vlm_json_uses_persisted_top_level_logical_frame_count() -> None:
    record = {
        "logical_frame_count": 4,
        "vlm_json": {"transitions": [99], "instructions": ["Add potatoes", "Stir the pot"]},
    }

    assert _normalize_loaded_window_vlm_json(record) == {}


def test_missing_frame_mp4_marks_sample_failed(tmp_path) -> None:
    app, _, sample_out_dir = _make_dataset_app(tmp_path)
    failed_marker = sample_out_dir / ".FAILED"

    _wait_until(lambda: failed_marker.exists())

    assert not (sample_out_dir / ".DONE").exists()
    report = _read_json(sample_out_dir / "failure.json")
    assert report["reason"] == "missing_input_video"
    assert report["sample_id"] == "sample"
    assert app.state.job_queue == []


def test_failure_closure_clears_stale_done_and_segments_and_overwrites_payload(tmp_path, monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))

    def raise_step_a(*_args, **_kwargs):
        entered.set()
        assert release.wait(2.0)
        raise RuntimeError("step a exploded")

    monkeypatch.setattr(app_module, "build_windows", raise_step_a)

    _, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=True)

    _wait_until(entered.is_set)

    (sample_out_dir / "segments.json").write_text(
        json.dumps({"segments": [{"seg_id": 0}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (sample_out_dir / ".DONE").write_text("", encoding="utf-8")
    (sample_out_dir / "failure.json").write_text(
        json.dumps({"subset": "demo", "sample_id": "sample", "reason": "stale_reason", "details": {"stale": True}}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert not (sample_out_dir / ".FAILED").exists()

    release.set()
    _wait_until(lambda: (sample_out_dir / ".FAILED").exists())

    assert not (sample_out_dir / "segments.json").exists()
    assert not (sample_out_dir / ".DONE").exists()

    report = _read_json(sample_out_dir / "failure.json")
    assert report["subset"] == "demo"
    assert report["sample_id"] == "sample"
    assert report["reason"] == "step_a_exception"
    assert report["details"]["error"] == "step a exploded"
    assert "stale" not in report["details"]


def test_output_directory_existing_without_done_marker_is_not_treated_as_done(tmp_path) -> None:
    sample_dir = Path(tmp_path) / "data" / "demo" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path)

    sample_out_dir = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample"
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    (sample_out_dir / "segments.json").write_text(
        json.dumps({"segments": [{"seg_id": 0}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    _, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=False)
    failed_marker = sample_out_dir / ".FAILED"

    _wait_until(lambda: failed_marker.exists())

    assert not (sample_out_dir / ".DONE").exists()
    report = _read_json(sample_out_dir / "failure.json")
    assert report["reason"] == "missing_input_video"


def test_create_app_backfills_done_runtime_and_run_summary_from_existing_terminal_artifacts(tmp_path) -> None:
    sample_id = "sample"
    sample_dir = Path(tmp_path) / "data" / "demo" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path, required_stages=["stage1_segments"])

    sample_out_dir = Path(tmp_path) / "demo" / "testrun" / "samples" / sample_id
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    (sample_out_dir / ".DONE").write_text("", encoding="utf-8")
    (sample_out_dir / "segments.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "nframes": 16,
                "segments": [{"seg_id": 0, "start_frame": 0, "end_frame": 15, "instruction": "Add potatoes"}],
                "diagnostics": {
                    "required_stages": ["stage1_segments"],
                    "completed_stages": ["stage1_segments"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    app, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=False, start_runtime=False)

    sample_runtime = _read_json(sample_out_dir / "sample_runtime.json")
    run_summary = _read_json(Path(tmp_path) / "demo" / "testrun" / "run_summary.json")

    assert sample_runtime["terminal_state"] == "done"
    assert sample_runtime["stages"]["completed"] == ["stage1_segments"]
    assert run_summary["sample_counts"] == {"total": 1, "done": 1, "failed": 0, "pending": 0}
    assert app.state.sample_store.load_sample_runtime("demo", sample_id)["terminal_state"] == "done"


def test_retry_evidence_survives_restart_before_terminalization(tmp_path, monkeypatch) -> None:
    subset = "demo"
    sample_id = "sample"
    data_root = Path(tmp_path) / "data"
    sample_dir = data_root / subset / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "Frame_demo.mp4").write_bytes(b"not-a-real-video")

    config = Config(
        datasets=[{"root": str(data_root), "subset": subset}],
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={"auto_exit_after_all_done": False, "max_empty_retries_per_job": 2},
    )
    app = create_app(config)
    client = TestClient(app)
    resumed_app = None
    try:
        task_id = "demo::sample_w0_r0"
        meta = {
            "subset": subset,
            "sample_id": sample_id,
            "window_id": 0,
            "repeat_index": 0,
            "job_type": "window_boundary",
            "logical_frame_count": 4,
        }
        app.state.inflight[task_id] = {"job": {"task_id": task_id, "meta": meta}, "ts": time.time(), "dispatch_id": "d1"}

        attempt = client.post(
            "/submit_result",
            json={"task_id": task_id, "dispatch_id": "d1", "vlm_json": {}, "meta": meta},
        )
        assert attempt.json()["status"] == "retry_triggered"

        sample_out_dir = Path(tmp_path) / subset / "testrun" / "samples" / sample_id
        retry_runtime = _read_json(sample_out_dir / "sample_runtime.json")
        assert retry_runtime["retry"]["total_retries"] == 1
        assert retry_runtime["retry"]["empty_result_retries"] == 1

        _seed_completed_window_result(sample_out_dir)
        _install_basic_finalize_mocks(monkeypatch)

        resumed_config = Config(
            datasets=[{"root": str(data_root), "subset": subset}],
            run={"base_dir": str(tmp_path), "run_id": "testrun", "force_resume": True},
            server={"auto_exit_after_all_done": False, "max_empty_retries_per_job": 2},
        )
        resumed_app = create_app(resumed_config)
        resumed_app.state.runtime.start()

        _wait_until(lambda: (sample_out_dir / ".DONE").exists())

        sample_runtime = _read_json(sample_out_dir / "sample_runtime.json")
        assert sample_runtime["terminal_state"] == "done"
        assert sample_runtime["retry"]["total_retries"] == 1
        assert sample_runtime["retry"]["empty_result_retries"] == 1
        assert _read_json(Path(tmp_path) / subset / "testrun" / "run_summary.json")["retry"]["total_retries"] == 1
    finally:
        client.close()
        if resumed_app is not None:
            resumed_app.state.runtime.stop()
            resumed_app.state.runtime.join(timeout=1.0)


def test_create_app_backfills_failed_runtime_with_export_failure_details(tmp_path) -> None:
    sample_id = "sample"
    sample_dir = Path(tmp_path) / "data" / "demo" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path, export={"enabled": True})

    sample_out_dir = Path(tmp_path) / "demo" / "testrun" / "samples" / sample_id
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    (sample_out_dir / ".FAILED").write_text("", encoding="utf-8")
    (sample_out_dir / "failure.json").write_text(
        json.dumps(
            {
                "subset": "demo",
                "sample_id": sample_id,
                "reason": "export_failed",
                "details": {
                    "stage": "export",
                    "required_stages": ["stage1_segments", "export"],
                    "completed_stages": ["stage1_segments"],
                    "export_enabled": True,
                    "export_attempted": True,
                    "export_mode": "clips",
                    "export_reason": "failed_before_export_completion",
                    "export_error": "ffmpeg exploded",
                    "export_errors": ["clips:degraded"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _make_dataset_app(tmp_path, with_mp4=False, export={"enabled": True}, start_runtime=False)

    sample_runtime = _read_json(sample_out_dir / "sample_runtime.json")
    run_summary = _read_json(Path(tmp_path) / "demo" / "testrun" / "run_summary.json")

    assert sample_runtime["terminal_state"] == "failed"
    assert sample_runtime["export"] == {
        "required": True,
        "enabled": True,
        "attempted": True,
        "status": "failed",
        "reason": "failed_before_export_completion",
        "mode": "clips",
        "errors": ["clips:degraded"],
        "error": "ffmpeg exploded",
    }
    assert run_summary["export"]["status_counts"] == {"failed": 1}
    assert run_summary["failure_reasons"] == {"export_failed": 1}


def test_done_marker_writes_after_stage1_when_stage1_is_only_required_stage(tmp_path, monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()

    _install_basic_finalize_mocks(monkeypatch)

    def paused_stage2(_sid, segments, _config, *, target_language="en", backend=None):
        del target_language, backend
        entered.set()
        assert release.wait(2.0)
        return _stage2_result(
            segments,
            hierarchy={
                "roots": [
                    {
                        "level": "coarse",
                        "start_seg_id": 0,
                        "end_seg_id": 0,
                        "summary": "Add potatoes",
                        "children": [],
                    }
                ]
            },
            subtitle_items=[{"seg_id": 0, "subtitle": "Add potatoes"}],
            summary_diagnostics={"llm_summary_applied": True},
            subtitle_diagnostics={
                "llm_subtitle_requested_language": "zh",
                "llm_subtitle_language": "en",
                "llm_subtitle_output_language": "en",
                "llm_subtitle_fallback_used": True,
                "llm_subtitle_reason": "source_instruction_reused",
            },
        )

    monkeypatch.setattr(app_module, "run_llm_stage2_pass", paused_stage2)
    monkeypatch.setattr(
        app_module,
        "export_sample_outputs",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("export should not run when not required")),
    )

    app, _, sample_out_dir = _make_dataset_app(
        tmp_path,
        with_mp4=True,
        llm_merge={"enabled": True},
        start_runtime=False,
    )
    _seed_completed_window_result(sample_out_dir)
    _override_required_stages(app, ["stage1_segments"])
    _start_runtime(app)
    done_marker = sample_out_dir / ".DONE"
    failed_marker = sample_out_dir / ".FAILED"

    _wait_until(entered.is_set)

    assert done_marker.exists()
    assert not failed_marker.exists()

    payload = _read_json(sample_out_dir / "segments.json")
    assert payload["segments"][0]["instruction"] == "Add potatoes"
    assert "task_hierarchy" not in payload

    release.set()
    _wait_until(
        lambda: "task_hierarchy" in _read_json(sample_out_dir / "segments.json")
        and _read_json(sample_out_dir / "segments.json")["segments"][0].get("export_subtitle") == "Add potatoes"
    )

    payload = _read_json(sample_out_dir / "segments.json")
    assert payload["task_hierarchy"]["roots"][0]["summary"] == "Add potatoes"
    assert payload["segments"][0]["export_subtitle"] == "Add potatoes"
    assert done_marker.exists()
    assert not failed_marker.exists()

def test_done_marker_waits_for_stage2_required_stage_completion(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample"
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path, llm_merge={"enabled": True})
    _seed_completed_window_result(sample_out_dir)

    entered = threading.Event()
    release = threading.Event()

    _install_basic_finalize_mocks(monkeypatch)

    def paused_stage2(_sid, segments, _config, *, target_language="en", backend=None):
        del target_language, backend
        entered.set()
        assert release.wait(2.0)
        return _stage2_result(segments, subtitle_items=[{"seg_id": 0, "subtitle": "Add potatoes"}])

    monkeypatch.setattr(app_module, "run_llm_stage2_pass", paused_stage2)
    monkeypatch.setattr(
        app_module,
        "export_sample_outputs",
        lambda **_kwargs: {"export_enabled": False, "export_attempted": False},
    )

    _, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=True, llm_merge={"enabled": True})
    done_marker = sample_out_dir / ".DONE"
    failed_marker = sample_out_dir / ".FAILED"

    _wait_until(entered.is_set)

    assert not done_marker.exists()
    assert not failed_marker.exists()

    release.set()
    _wait_until(lambda: done_marker.exists())
    assert not failed_marker.exists()

def test_done_marker_waits_for_export_required_stage_completion_when_enabled(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample"
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path, export={"enabled": True})
    _seed_completed_window_result(sample_out_dir)

    entered = threading.Event()
    release = threading.Event()
    export_inputs: list[list[dict]] = []

    _install_basic_finalize_mocks(monkeypatch)
    monkeypatch.setattr(
        app_module,
        "run_llm_stage2_pass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Stage 2 must not run when llm_merge is disabled")),
    )

    def paused_export(**_kwargs):
        export_inputs.append([dict(segment) for segment in _kwargs["segments"]])
        entered.set()
        assert release.wait(2.0)
        return {
            "export_enabled": True,
            "export_attempted": True,
            "export_reason": "applied",
        }

    monkeypatch.setattr(app_module, "export_sample_outputs", paused_export)

    _, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=True, export={"enabled": True})
    done_marker = sample_out_dir / ".DONE"
    failed_marker = sample_out_dir / ".FAILED"

    _wait_until(entered.is_set)

    assert not done_marker.exists()
    assert not failed_marker.exists()
    assert export_inputs
    assert export_inputs[0][0]["instruction"] == "Add potatoes"
    assert "export_subtitle" not in export_inputs[0][0]

    release.set()
    _wait_until(lambda: done_marker.exists())
    assert not failed_marker.exists()

def test_done_marker_does_not_wait_for_optional_export_when_manifest_does_not_require_it(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample"
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path, export={"enabled": True})
    _seed_completed_window_result(sample_out_dir)

    _install_basic_finalize_mocks(monkeypatch)
    monkeypatch.setattr(
        app_module,
        "run_llm_stage2_pass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("optional Stage 2 should not run when llm_merge is disabled")),
    )
    monkeypatch.setattr(
        app_module,
        "export_sample_outputs",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("optional export should not run after terminal success")),
    )

    app, _, sample_out_dir = _make_dataset_app(
        tmp_path,
        with_mp4=True,
        export={"enabled": True},
        start_runtime=False,
    )
    _override_required_stages(app, ["stage1_segments"])
    _start_runtime(app)
    done_marker = sample_out_dir / ".DONE"
    failed_marker = sample_out_dir / ".FAILED"

    _wait_until(lambda: done_marker.exists() or failed_marker.exists())

    assert done_marker.exists()
    assert not failed_marker.exists()

def test_step_a_exception_marks_sample_failed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))

    def raise_step_a(*_args, **_kwargs):
        raise RuntimeError("step a exploded")

    monkeypatch.setattr(app_module, "build_windows", raise_step_a)

    _, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=True)
    failed_marker = sample_out_dir / ".FAILED"

    _wait_until(lambda: failed_marker.exists())

    assert not (sample_out_dir / ".DONE").exists()
    report = _read_json(sample_out_dir / "failure.json")
    assert report["reason"] == "step_a_exception"
    assert report["details"]["error"] == "step a exploded"


def test_required_export_failure_marks_sample_failed_without_done(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample"
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path, export={"enabled": True})
    _seed_completed_window_result(sample_out_dir)

    _install_basic_finalize_mocks(monkeypatch)
    export_inputs: list[list[dict]] = []
    monkeypatch.setattr(
        app_module,
        "run_llm_stage2_pass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Stage 2 must not run when llm_merge is disabled")),
    )
    monkeypatch.setattr(
        app_module,
        "export_sample_outputs",
        lambda **_kwargs: (
            export_inputs.append([dict(segment) for segment in _kwargs["segments"]]),
            {
                "export_enabled": True,
                "export_attempted": True,
                "export_mode": "clips",
                "export_reason": "failed",
                "export_errors": ["clips:degraded"],
                "export_clips_contract_status": "degraded",
            },
        )[1],
    )

    _, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=True, export={"enabled": True})
    failed_marker = sample_out_dir / ".FAILED"
    done_marker = sample_out_dir / ".DONE"

    _wait_until(lambda: failed_marker.exists() or done_marker.exists())

    assert failed_marker.exists()
    assert not done_marker.exists()
    assert export_inputs
    assert "export_subtitle" not in export_inputs[0][0]
    report = _read_json(sample_out_dir / "failure.json")
    assert report["reason"] == "export_failed"
    assert report["details"]["stage"] == "export"
    assert report["details"]["export_enabled"] is True
    assert report["details"]["export_attempted"] is True
    assert report["details"]["export_mode"] == "clips"
    assert report["details"]["export_reason"] == "failed"
    assert report["details"]["export_errors"] == ["clips:degraded"]

    sample_runtime = _read_json(sample_out_dir / "sample_runtime.json")
    assert sample_runtime["failure"]["reason"] == "export_failed"
    assert sample_runtime["export"] == {
        "required": True,
        "enabled": True,
        "attempted": True,
        "status": "failed",
        "reason": "failed",
        "mode": "clips",
        "errors": ["clips:degraded"],
    }

def test_required_export_failure_preserves_stage2_runtime_fallback_details(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample"
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path, llm_merge={"enabled": True}, export={"enabled": True})
    _seed_completed_window_result(sample_out_dir)

    _install_basic_finalize_mocks(monkeypatch)
    monkeypatch.setattr(
        app_module,
        "run_llm_stage2_pass",
        lambda _sid, segments, _config, **_kwargs: _stage2_result(
            segments,
            subtitle_items=[{"seg_id": 0, "subtitle": "Add potatoes"}],
            summary_diagnostics={"llm_summary_applied": True},
            subtitle_diagnostics={
                "llm_subtitle_requested_language": "zh",
                "llm_subtitle_language": "en",
                "llm_subtitle_output_language": "en",
                "llm_subtitle_fallback_used": True,
                "llm_subtitle_reason": "request_failed:RuntimeError",
            },
        ),
    )
    monkeypatch.setattr(
        app_module,
        "export_sample_outputs",
        lambda **_kwargs: {
            "export_enabled": True,
            "export_attempted": True,
            "export_mode": "clips",
            "export_reason": "failed",
            "export_errors": ["clips:degraded"],
            "export_clips_contract_status": "degraded",
        },
    )

    _, _, sample_out_dir = _make_dataset_app(
        tmp_path,
        with_mp4=True,
        llm_merge={"enabled": True},
        export={"enabled": True},
    )
    failed_marker = sample_out_dir / ".FAILED"
    done_marker = sample_out_dir / ".DONE"
    failure_report = sample_out_dir / "failure.json"

    _wait_until(lambda: failed_marker.exists() or done_marker.exists())
    _wait_until(lambda: failure_report.exists())

    assert failed_marker.exists()
    assert not done_marker.exists()

    report = _read_json(failure_report)
    assert report["details"]["diagnostics"]["llm_subtitle_fallback_used"] is True
    assert report["details"]["diagnostics"]["llm_subtitle_reason"] == "request_failed:RuntimeError"
    assert report["details"]["diagnostics"]["export_reason"] == "failed"

    sample_runtime = _read_json(sample_out_dir / "sample_runtime.json")
    assert sample_runtime["fallback"]["fields"]["llm_subtitle_fallback_used"] is True
    assert sample_runtime["export"] == {
        "required": True,
        "enabled": True,
        "attempted": True,
        "status": "failed",
        "reason": "failed",
        "mode": "clips",
        "errors": ["clips:degraded"],
    }


def test_finalize_exception_marks_sample_failed_without_done_marker(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample"
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path)
    (sample_out_dir / "windows.jsonl").write_text(
        json.dumps(
            {
                "task_id": "demo::sample_w0_r0",
                "dispatch_id": "d1",
                "window_id": 0,
                "repeat_index": 0,
                "logical_frame_count": 4,
                "vlm_json": {"transitions": [1], "instructions": ["Add potatoes", "Stir the pot"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))
    monkeypatch.setattr(
        app_module,
        "build_windows",
        lambda *_args, **_kwargs: [Window(window_id=0, start_frame=0, end_frame=15, frame_ids=[0, 5, 10, 15])],
    )

    def raise_finalize(*_args, **_kwargs):
        raise RuntimeError("finalize exploded")

    monkeypatch.setattr(app_module, "build_segments_via_cuts", raise_finalize)
    monkeypatch.setattr(app_module, "FrameExtractor", _NoopFrameExtractor)

    _, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=True)
    failed_marker = sample_out_dir / ".FAILED"

    _wait_until(lambda: failed_marker.exists())

    assert not (sample_out_dir / ".DONE").exists()
    report = _read_json(sample_out_dir / "failure.json")
    assert report["reason"] == "finalize_exception"
    assert report["details"]["error"] == "finalize exploded"


def test_reload_validation_uses_persisted_logical_frame_count_layout(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample"
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path)
    (sample_out_dir / "windows.jsonl").write_text(
        json.dumps(
            {
                "task_id": "demo::sample_w0_r0",
                "dispatch_id": "d1",
                "window_id": 0,
                "repeat_index": 0,
                "logical_frame_count": 4,
                "vlm_json": {"transitions": [99], "instructions": ["Add potatoes", "Stir the pot"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))
    monkeypatch.setattr(
        app_module,
        "build_windows",
        lambda *_args, **_kwargs: [Window(window_id=0, start_frame=0, end_frame=15, frame_ids=[0, 5, 10, 15])],
    )
    monkeypatch.setattr(
        app_module,
        "build_segments_via_cuts",
        lambda *_args, **_kwargs: {
            "segments": [{"seg_id": 0, "start_frame": 0, "end_frame": 15, "instruction": "Add potatoes"}]
        },
    )
    monkeypatch.setattr(
        app_module,
        "export_sample_outputs",
        lambda **_kwargs: {"export_enabled": False, "export_attempted": False},
    )
    monkeypatch.setattr(app_module, "FrameExtractor", _NoopFrameExtractor)

    _, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=True)
    failed_marker = sample_out_dir / ".FAILED"
    done_marker = sample_out_dir / ".DONE"
    failure_report = sample_out_dir / "failure.json"

    _wait_until(lambda: failed_marker.exists() or done_marker.exists())
    _wait_until(lambda: failure_report.exists())

    assert failed_marker.exists()
    assert not done_marker.exists()
    report = _read_json(failure_report)
    assert report["reason"] == "window_boundary_failed"
    assert report["details"]["errors"] == {"0": "invalid_vlm_json"}


def test_postprocess_empty_result_marks_sample_failed_without_done(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / 'demo' / 'testrun' / 'samples' / 'sample'
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path, llm_merge={"enabled": True})
    _seed_completed_window_result(sample_out_dir)

    _install_basic_finalize_mocks(monkeypatch)
    monkeypatch.setattr(
        app_module,
        'run_llm_stage2_pass',
        lambda _sid, _segments, _config, **_kwargs: _stage2_result([], subtitle_items=[]),
    )
    monkeypatch.setattr(
        app_module,
        'export_sample_outputs',
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError('export should not run after postprocess emptied segments')),
    )

    _, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=True, llm_merge={"enabled": True})
    failed_marker = sample_out_dir / '.FAILED'
    done_marker = sample_out_dir / '.DONE'
    failure_report = sample_out_dir / 'failure.json'

    _wait_until(lambda: failed_marker.exists() or done_marker.exists())
    _wait_until(lambda: failure_report.exists())

    assert failed_marker.exists()
    assert not done_marker.exists()
    report = _read_json(failure_report)
    assert report['reason'] == 'finalize_empty_segments'
    assert report['details']['phase'] == 'postprocess'


def test_export_consumes_stage2_subtitles_when_stage2_enabled(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample"
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path, llm_merge={"enabled": True}, export={"enabled": True})
    _seed_completed_window_result(sample_out_dir)

    export_inputs: list[list[dict]] = []
    stage2_calls: list[str] = []

    _install_basic_finalize_mocks(monkeypatch)

    def fake_stage2(_sid, segments, _config, *, target_language="en", backend=None):
        del backend
        stage2_calls.append(str(target_language))
        return _stage2_result(
            segments,
            subtitle_items=[{"seg_id": 0, "subtitle": "加土豆"}],
            subtitle_diagnostics={
                "llm_subtitle_requested_language": "zh",
                "llm_subtitle_language": "zh",
                "llm_subtitle_output_language": "zh",
                "llm_subtitle_attempted": True,
                "llm_subtitle_applied": True,
                "llm_subtitle_reason": "applied",
            },
        )

    monkeypatch.setattr(app_module, "run_llm_stage2_pass", fake_stage2)
    monkeypatch.setattr(
        app_module,
        "export_sample_outputs",
        lambda **kwargs: (
            export_inputs.append([dict(segment) for segment in kwargs["segments"]]),
            {
                "export_enabled": True,
                "export_attempted": True,
                "export_mode": "annotated",
                "export_reason": "applied",
            },
        )[1],
    )

    _, _, sample_out_dir = _make_dataset_app(
        tmp_path,
        with_mp4=True,
        llm_merge={"enabled": True},
        export={"enabled": True, "subtitles": {"enabled": True, "language": "zh"}},
    )
    done_marker = sample_out_dir / ".DONE"
    failed_marker = sample_out_dir / ".FAILED"

    _wait_until(lambda: done_marker.exists() or failed_marker.exists())

    assert done_marker.exists()
    assert not failed_marker.exists()
    assert stage2_calls == ["zh"]
    assert export_inputs
    assert export_inputs[0][0]["export_subtitle"] == "加土豆"

    payload = _read_json(sample_out_dir / "segments.json")
    assert payload["segments"][0]["export_subtitle"] == "加土豆"
    sample_runtime = _read_json(sample_out_dir / "sample_runtime.json")
    assert sample_runtime["export"]["status"] == "applied"

def test_step_a_invalid_artifact_fails_sample_before_dispatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))
    monkeypatch.setattr(app_module, "build_windows", lambda *_args, **_kwargs: _single_window())
    monkeypatch.setattr(app_module, "FrameExtractor", _InvalidArtifactFrameExtractor)

    app, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=True)
    failed_marker = sample_out_dir / ".FAILED"

    _wait_until(lambda: failed_marker.exists())

    assert app.state.job_queue == []
    assert not (sample_out_dir / ".DONE").exists()
    report = _read_json(sample_out_dir / "failure.json")
    assert report["reason"] == "artifact_extraction_failed"
    assert report["details"]["phase"] == "step_a"
    assert report["details"]["issues"][0]["reason"] == "image_decode_failed"


def test_boundary_refinement_terminal_failure_blocks_done(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / 'demo' / 'testrun' / 'samples' / 'sample'
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path, windowing={'enable_boundary_refinement': True})
    _seed_completed_window_result(sample_out_dir)
    _write_jsonl(
        sample_out_dir / 'boundary_refinements.jsonl',
        [
            {
                'task_id': 'demo::sample_b0',
                'dispatch_id': 'd7',
                'boundary_id': 0,
                'coarse_boundary_frame': 8,
                'frame_ids': [6, 7, 8, 9],
                'vlm_json': {},
                'terminal_error': 'boundary_retry_exhausted',
            }
        ],
    )

    _install_basic_finalize_mocks(monkeypatch)
    monkeypatch.setattr(
        app_module,
        'build_boundary_refinement_windows',
        lambda *_args, **_kwargs: [
            BoundaryRefinementWindow(boundary_id=0, coarse_boundary_frame=8, start_frame=6, end_frame=9, frame_ids=[6, 7, 8, 9])
        ],
    )
    monkeypatch.setattr(app_module, 'apply_boundary_refinement_results', lambda segments, *_args, **_kwargs: segments)
    monkeypatch.setattr(app_module, 'export_sample_outputs', lambda **_kwargs: {'export_enabled': False, 'export_attempted': False})

    _, _, sample_out_dir = _make_dataset_app(
        tmp_path,
        with_mp4=True,
        windowing={'enable_boundary_refinement': True},
    )
    failed_marker = sample_out_dir / '.FAILED'
    done_marker = sample_out_dir / '.DONE'

    _wait_until(lambda: failed_marker.exists() or done_marker.exists())

    assert failed_marker.exists()
    assert not done_marker.exists()
    report = _read_json(sample_out_dir / 'failure.json')
    assert report['reason'] == 'boundary_refinement_failed'
    assert report['details']['errors'] == {'0': 'boundary_retry_exhausted'}


def test_deferred_label_invalid_artifact_fails_sample_before_dispatch(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / "demo" / "testrun" / "samples" / "sample"
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path, windowing={"segment_labeling_mode": "deferred"})
    _seed_completed_window_result(sample_out_dir)

    _install_basic_finalize_mocks(monkeypatch)
    monkeypatch.setattr(app_module, "FrameExtractor", _InvalidArtifactFrameExtractor)

    _, _, sample_out_dir = _make_dataset_app(
        tmp_path,
        with_mp4=True,
        windowing={"segment_labeling_mode": "deferred"},
    )
    failed_marker = sample_out_dir / ".FAILED"
    done_marker = sample_out_dir / ".DONE"

    _wait_until(lambda: failed_marker.exists() or done_marker.exists())

    assert failed_marker.exists()
    assert not done_marker.exists()
    report = _read_json(sample_out_dir / "failure.json")
    assert report["reason"] == "artifact_preparation_failed"
    assert report["details"]["phase"] == "segment_label_dispatch"
    assert report["details"]["issues"][0]["reason"] == "image_decode_failed"


def test_deferred_label_terminal_failure_blocks_done(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / 'demo' / 'testrun' / 'samples' / 'sample'
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path, windowing={'segment_labeling_mode': 'deferred'})
    _seed_completed_window_result(sample_out_dir)
    _write_jsonl(
        sample_out_dir / 'segment_labels.jsonl',
        [
            {
                'task_id': 'demo::sample_seg0',
                'dispatch_id': 'd8',
                'segment_id': 0,
                'vlm_json': {},
                'terminal_error': 'label_retry_exhausted',
            }
        ],
    )

    _install_basic_finalize_mocks(monkeypatch)
    monkeypatch.setattr(app_module, 'apply_deferred_segment_labels', lambda segments, _labels: segments)
    monkeypatch.setattr(app_module, 'export_sample_outputs', lambda **_kwargs: {'export_enabled': False, 'export_attempted': False})

    _, _, sample_out_dir = _make_dataset_app(
        tmp_path,
        with_mp4=True,
        windowing={'segment_labeling_mode': 'deferred'},
    )
    failed_marker = sample_out_dir / '.FAILED'
    done_marker = sample_out_dir / '.DONE'

    _wait_until(lambda: failed_marker.exists() or done_marker.exists())

    assert failed_marker.exists()
    assert not done_marker.exists()
    report = _read_json(sample_out_dir / 'failure.json')
    assert report['reason'] == 'segment_label_failed'
    assert report['details']['errors'] == {'0': 'label_retry_exhausted'}


def test_failed_sample_clears_queued_jobs_for_same_sample(tmp_path, monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(app_module, 'read_video_info', lambda _mp4: (30.0, 16))

    def raise_step_a(*_args, **_kwargs):
        entered.set()
        assert release.wait(2.0)
        raise RuntimeError('step a exploded')

    monkeypatch.setattr(app_module, 'build_windows', raise_step_a)

    app, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=True)
    _wait_until(entered.is_set)
    app.state.job_queue.extend(
        [
            {'task_id': 'demo::sample_seg0', 'meta': {'subset': 'demo', 'sample_id': 'sample', 'job_type': 'segment_label'}},
            {'task_id': 'demo::other_seg0', 'meta': {'subset': 'demo', 'sample_id': 'other', 'job_type': 'segment_label'}},
        ]
    )
    release.set()

    failed_marker = sample_out_dir / '.FAILED'
    _wait_until(lambda: failed_marker.exists())

    remaining_task_ids = [job['task_id'] for job in app.state.job_queue]
    assert remaining_task_ids == ['demo::other_seg0']


def test_pre_finalize_terminal_window_failure_clears_same_sample_queue_and_inflight(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / 'demo' / 'testrun' / 'samples' / 'sample'
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path)
    _write_jsonl(
        sample_out_dir / 'windows.jsonl',
        [
            {
                'task_id': 'demo::sample_w0_r0',
                'dispatch_id': 'd1',
                'window_id': 0,
                'repeat_index': 0,
                'logical_frame_count': 4,
                'vlm_json': {},
                'terminal_error': 'empty_retry_exhausted',
            }
        ],
    )

    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(app_module, 'read_video_info', lambda _mp4: (30.0, 16))

    def paused_build_windows(*_args, **_kwargs):
        entered.set()
        assert release.wait(2.0)
        return _single_window()

    monkeypatch.setattr(app_module, 'build_windows', paused_build_windows)

    app, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=True)
    _wait_until(entered.is_set)
    app.state.job_queue.extend(
        [
            {'task_id': 'demo::sample_seg0', 'meta': {'subset': 'demo', 'sample_id': 'sample', 'job_type': 'segment_label'}},
            {'task_id': 'demo::other_seg0', 'meta': {'subset': 'demo', 'sample_id': 'other', 'job_type': 'segment_label'}},
        ]
    )
    app.state.inflight['demo::sample_inflight'] = {
        'job': {'task_id': 'demo::sample_inflight', 'meta': {'subset': 'demo', 'sample_id': 'sample', 'job_type': 'segment_label'}},
        'ts': time.time(),
        'dispatch_id': 'dx',
    }
    app.state.inflight['demo::other_inflight'] = {
        'job': {'task_id': 'demo::other_inflight', 'meta': {'subset': 'demo', 'sample_id': 'other', 'job_type': 'segment_label'}},
        'ts': time.time(),
        'dispatch_id': 'dy',
    }
    release.set()

    failed_marker = sample_out_dir / '.FAILED'
    _wait_until(lambda: failed_marker.exists())

    assert [job['task_id'] for job in app.state.job_queue] == ['demo::other_seg0']
    assert sorted(app.state.inflight) == ['demo::other_inflight']


def test_finalize_terminal_window_failure_clears_same_sample_queue_and_inflight(tmp_path, monkeypatch) -> None:
    sample_out_dir = Path(tmp_path) / 'demo' / 'testrun' / 'samples' / 'sample'
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _seed_dataset_run_manifest(tmp_path)
    _seed_completed_window_result(sample_out_dir)

    entered = threading.Event()
    release = threading.Event()
    call_count = {'count': 0}
    windows_path = sample_out_dir / 'windows.jsonl'

    monkeypatch.setattr(app_module, 'read_video_info', lambda _mp4: (30.0, 16))

    def build_windows_with_finalize_pause(*_args, **_kwargs):
        call_count['count'] += 1
        if call_count['count'] == 2:
            entered.set()
            assert release.wait(2.0)
        return _single_window()

    monkeypatch.setattr(app_module, 'build_windows', build_windows_with_finalize_pause)
    monkeypatch.setattr(app_module, 'FrameExtractor', _NoopFrameExtractor)

    app, _, sample_out_dir = _make_dataset_app(tmp_path, with_mp4=True)
    _wait_until(entered.is_set)
    _append_jsonl_record(
        windows_path,
        {
            'task_id': 'demo::sample_w0_r1',
            'dispatch_id': 'd2',
            'window_id': 0,
            'repeat_index': 1,
            'logical_frame_count': 4,
            'vlm_json': {},
            'terminal_error': 'timeout_retry_exhausted',
        },
    )
    app.state.job_queue.extend(
        [
            {'task_id': 'demo::sample_seg0', 'meta': {'subset': 'demo', 'sample_id': 'sample', 'job_type': 'segment_label'}},
            {'task_id': 'demo::other_seg0', 'meta': {'subset': 'demo', 'sample_id': 'other', 'job_type': 'segment_label'}},
        ]
    )
    app.state.inflight['demo::sample_inflight'] = {
        'job': {'task_id': 'demo::sample_inflight', 'meta': {'subset': 'demo', 'sample_id': 'sample', 'job_type': 'segment_label'}},
        'ts': time.time(),
        'dispatch_id': 'dx',
    }
    app.state.inflight['demo::other_inflight'] = {
        'job': {'task_id': 'demo::other_inflight', 'meta': {'subset': 'demo', 'sample_id': 'other', 'job_type': 'segment_label'}},
        'ts': time.time(),
        'dispatch_id': 'dy',
    }
    release.set()

    failed_marker = sample_out_dir / '.FAILED'
    _wait_until(lambda: failed_marker.exists())

    assert [job['task_id'] for job in app.state.job_queue] == ['demo::other_seg0']
    assert sorted(app.state.inflight) == ['demo::other_inflight']
