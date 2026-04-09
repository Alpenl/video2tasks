import json
import time
from pathlib import Path

import video2tasks.server.app as app_module
from video2tasks.config import Config
from video2tasks.server.app import create_app
from video2tasks.server.windowing import Window


class _NoopFrameExtractor:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_many_b64_with_artifacts(self, *_args, **_kwargs):
        return [], None


def _wait_until(predicate, *, timeout: float = 3.0, interval: float = 0.02) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    assert predicate()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _sample_out_dir(tmp_path: Path) -> Path:
    return tmp_path / "demo" / "testrun" / "samples" / "sample"


def _build_contract_app(tmp_path: Path):
    data_root = tmp_path / "data"
    sample_dir = data_root / "demo" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "Frame_demo.mp4").write_bytes(b"synthetic-video")

    config = Config(
        datasets=[{"root": str(data_root), "subset": "demo"}],
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        worker={"backend": "dummy"},
        server={"auto_exit_after_all_done": False},
        llm_merge={"enabled": False},
        export={"enabled": False},
    )
    return create_app(config)


def test_pipeline_contracts_run_and_sample_artifacts_for_done_terminal(monkeypatch, tmp_path: Path) -> None:
    app = _build_contract_app(tmp_path)
    sample_out_dir = _sample_out_dir(tmp_path)
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        sample_out_dir / "windows.jsonl",
        [
            {
                "task_id": "demo::sample_w0_r0",
                "dispatch_id": "d1",
                "window_id": 0,
                "repeat_index": 0,
                "logical_frame_count": 4,
                "vlm_json": {
                    "transitions": [1],
                    "instructions": ["Add potatoes", "Stir the pot"],
                },
            }
        ],
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
            "sample_id": "sample",
            "nframes": 16,
            "segments": [{"seg_id": 0, "start_frame": 0, "end_frame": 15, "instruction": "Add potatoes"}],
        },
    )
    monkeypatch.setattr(app_module, "FrameExtractor", _NoopFrameExtractor)

    app.state.runtime.start()
    try:
        done_marker = sample_out_dir / ".DONE"
        failed_marker = sample_out_dir / ".FAILED"
        _wait_until(lambda: done_marker.exists() or failed_marker.exists())

        assert done_marker.exists()
        assert not failed_marker.exists()

        segments = _read_json(sample_out_dir / "segments.json")
        assert segments["sample_id"] == "sample"
        assert segments["segments"][0]["instruction"] == "Add potatoes"
        assert "diagnostics" not in segments

        sample_runtime = _read_json(sample_out_dir / "sample_runtime.json")
        assert sample_runtime["terminal_state"] == "done"
        assert sample_runtime["stages"]["required"] == ["stage1_segments"]
        assert sample_runtime["stages"]["completed"] == ["stage1_segments"]
        assert sample_runtime["failure"] is None

        run_dir = tmp_path / "demo" / "testrun"
        run_manifest = _read_json(run_dir / "run_manifest.json")
        assert run_manifest["subset"] == "demo"
        assert run_manifest["run_id"] == "testrun"
        assert run_manifest["required_stages"] == ["stage1_segments"]

        run_summary = _read_json(run_dir / "run_summary.json")
        assert run_summary["sample_counts"] == {"total": 1, "done": 1, "failed": 0, "pending": 0}
        assert run_summary["failure_reasons"] == {}
    finally:
        app.state.runtime.stop()
        app.state.runtime.join(timeout=1.0)


def test_pipeline_contracts_failed_terminal_closure_persists_runtime_and_summary(monkeypatch, tmp_path: Path) -> None:
    app = _build_contract_app(tmp_path)
    sample_out_dir = _sample_out_dir(tmp_path)
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    (sample_out_dir / "segments.json").write_text(
        json.dumps({"segments": [{"seg_id": 0}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))
    monkeypatch.setattr(
        app_module,
        "build_windows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("step a exploded")),
    )

    app.state.runtime.start()
    try:
        failed_marker = sample_out_dir / ".FAILED"
        _wait_until(lambda: failed_marker.exists())

        assert failed_marker.exists()
        assert not (sample_out_dir / ".DONE").exists()
        assert not (sample_out_dir / "segments.json").exists()

        failure_report = _read_json(sample_out_dir / "failure.json")
        assert failure_report["reason"] == "step_a_exception"
        assert failure_report["details"]["error"] == "step a exploded"

        sample_runtime = _read_json(sample_out_dir / "sample_runtime.json")
        assert sample_runtime["terminal_state"] == "failed"
        assert sample_runtime["failure"]["reason"] == "step_a_exception"

        run_summary_path = tmp_path / "demo" / "testrun" / "run_summary.json"
        _wait_until(lambda: _read_json(run_summary_path)["sample_counts"]["failed"] == 1)
        run_summary = _read_json(run_summary_path)
        assert run_summary["sample_counts"] == {"total": 1, "done": 0, "failed": 1, "pending": 0}
        assert run_summary["failure_reasons"] == {"step_a_exception": 1}
    finally:
        app.state.runtime.stop()
        app.state.runtime.join(timeout=1.0)
