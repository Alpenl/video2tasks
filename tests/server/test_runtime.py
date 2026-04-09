import threading
import time
from pathlib import Path

import video2tasks.server.app as app_module
from video2tasks.config import Config
from video2tasks.server.app import create_app, run_server
from video2tasks.server.runtime import ThreadRuntime
from video2tasks.server.windowing import Window


def _make_dataset_config(tmp_path: Path) -> Config:
    subset = "demo"
    data_root = tmp_path / "data"
    sample_dir = data_root / subset / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "Frame_demo.mp4").write_bytes(b"not-a-real-video")
    return Config(
        datasets=[{"root": str(data_root), "subset": subset}],
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={"auto_exit_after_all_done": False},
    )


def test_create_app_has_no_hidden_runtime_side_effects(tmp_path, monkeypatch) -> None:
    entered = threading.Event()

    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))

    def mark_build_windows(*_args, **_kwargs):
        entered.set()
        return [Window(window_id=0, start_frame=0, end_frame=15, frame_ids=[0, 5, 10, 15])]

    monkeypatch.setattr(app_module, "build_windows", mark_build_windows)

    app = create_app(_make_dataset_config(tmp_path))

    assert hasattr(app.state, "runtime")
    assert not entered.wait(0.2)
    assert app.state.job_queue == []
    sample_out_dir = tmp_path / "demo" / "testrun" / "samples" / "sample"
    assert not (sample_out_dir / ".DONE").exists()
    assert not (sample_out_dir / ".FAILED").exists()


def test_app_runtime_start_runs_producer_explicitly(tmp_path, monkeypatch) -> None:
    entered = threading.Event()

    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))

    def mark_build_windows(*_args, **_kwargs):
        entered.set()
        return [Window(window_id=0, start_frame=0, end_frame=15, frame_ids=[0, 5, 10, 15])]

    monkeypatch.setattr(app_module, "build_windows", mark_build_windows)

    app = create_app(_make_dataset_config(tmp_path))

    assert not entered.wait(0.2)

    runtime = app.state.runtime
    runtime.start()
    assert entered.wait(1.0)

    runtime.stop()
    runtime.join(timeout=2.0)
    assert not runtime.is_alive()


def test_app_runtime_uses_app_module_monkeypatches_applied_after_create_app(tmp_path, monkeypatch) -> None:
    config = _make_dataset_config(tmp_path)
    app = create_app(config)
    entered = threading.Event()

    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))

    def mark_build_windows(*_args, **_kwargs):
        entered.set()
        return [Window(window_id=0, start_frame=0, end_frame=15, frame_ids=[0, 5, 10, 15])]

    monkeypatch.setattr(app_module, "build_windows", mark_build_windows)

    runtime = app.state.runtime
    runtime.start()
    assert entered.wait(1.0)

    runtime.stop()
    runtime.join(timeout=2.0)
    assert not runtime.is_alive()


def test_thread_runtime_supports_explicit_stop_and_join() -> None:
    started = threading.Event()
    stopped = threading.Event()

    def loop(stop_event):
        started.set()
        while not stop_event.is_set():
            time.sleep(0.01)
        stopped.set()

    runtime = ThreadRuntime(name="demo-runtime", target=loop)

    runtime.start()
    assert started.wait(1.0)

    runtime.stop()
    runtime.join(timeout=1.0)

    assert stopped.is_set()
    assert not runtime.is_alive()


def test_run_server_explicitly_starts_and_stops_runtime(tmp_path, monkeypatch) -> None:
    app = create_app(_make_dataset_config(tmp_path))
    runtime = app.state.runtime
    observed = {}

    monkeypatch.setattr(app_module, "create_app", lambda _config: app)

    def fake_uvicorn_run(passed_app, **kwargs):
        observed["app"] = passed_app
        observed["is_alive_during_run"] = runtime.is_alive()
        observed["kwargs"] = kwargs

    monkeypatch.setattr(app_module.uvicorn, "run", fake_uvicorn_run)

    run_server(_make_dataset_config(tmp_path))

    assert observed["app"] is app
    assert observed["is_alive_during_run"] is True
    assert not runtime.is_alive()
