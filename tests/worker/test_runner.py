import requests

from video2tasks.config import Config
from video2tasks.worker.runner import run_worker


class DummyBackend:
    name = "dummy"

    def __init__(self) -> None:
        self.cleaned = False

    def warmup(self) -> None:
        return None

    def infer(self, images, prompt):
        return {
            "thought": "single task",
            "transitions": [],
            "instructions": ["Do the task"],
        }

    def cleanup(self) -> None:
        self.cleaned = True


class DummyResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_worker_exits_cleanly_when_server_disappears_after_prior_connection(
    monkeypatch,
    capsys,
) -> None:
    backend = DummyBackend()
    get_attempts = {"count": 0}

    def fake_get(url, timeout=None):
        get_attempts["count"] += 1
        if get_attempts["count"] == 1:
            return DummyResponse(
                200,
                {
                    "data": {
                        "task_id": "demo::sample_roboturk_tower_w0",
                        "images": [],
                        "meta": {"subset": "demo", "sample_id": "sample_roboturk_tower"},
                    }
                },
            )
        raise requests.exceptions.ConnectionError("server stopped")

    def fake_post(url, json=None, timeout=None):
        return DummyResponse(200, {"status": "ok"})

    monkeypatch.setattr("video2tasks.worker.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("video2tasks.worker.runner.requests.get", fake_get)
    monkeypatch.setattr("video2tasks.worker.runner.requests.post", fake_post)
    monkeypatch.setattr("video2tasks.worker.runner.time.sleep", lambda *_args, **_kwargs: None)

    cfg = Config(worker={"backend": "dummy", "server_url": "http://127.0.0.1:8099"})

    run_worker(cfg)

    out = capsys.readouterr().out
    assert "[Done] demo::sample_roboturk_tower_w0 (0logical/0img) -> Cuts: []" in out
    assert "[Worker] Failed to connect after 30 retries. Exiting." not in out
    assert "[Worker] Server became unavailable after prior connection. Exiting." in out
    assert backend.cleaned is True


def test_worker_uses_short_retry_budget_after_prior_connection(
    monkeypatch,
    capsys,
) -> None:
    backend = DummyBackend()
    get_attempts = {"count": 0}

    def fake_get(url, timeout=None):
        get_attempts["count"] += 1
        if get_attempts["count"] == 1:
            return DummyResponse(
                200,
                {
                    "data": {
                        "task_id": "demo::sample_roboturk_tower_w0",
                        "images": [],
                        "meta": {"subset": "demo", "sample_id": "sample_roboturk_tower"},
                    }
                },
            )
        raise requests.exceptions.ConnectionError("server stopped")

    def fake_post(url, json=None, timeout=None):
        return DummyResponse(200, {"status": "ok"})

    monkeypatch.setattr("video2tasks.worker.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("video2tasks.worker.runner.requests.get", fake_get)
    monkeypatch.setattr("video2tasks.worker.runner.requests.post", fake_post)
    monkeypatch.setattr("video2tasks.worker.runner.time.sleep", lambda *_args, **_kwargs: None)

    cfg = Config(worker={"backend": "dummy", "server_url": "http://127.0.0.1:8099"})

    run_worker(cfg)

    capsys.readouterr()
    assert get_attempts["count"] == 4


def test_worker_uses_center_scan_prompt_when_configured(
    monkeypatch,
    capsys,
) -> None:
    backend = DummyBackend()
    seen = {}
    get_attempts = {"count": 0}

    def fake_get(url, timeout=None):
        get_attempts["count"] += 1
        if get_attempts["count"] == 1:
            return DummyResponse(
                200,
                {
                    "data": {
                        "task_id": "demo::sample_roboturk_tower_w0",
                        "images": [],
                        "meta": {"subset": "demo", "sample_id": "sample_roboturk_tower"},
                    }
                },
            )
        raise requests.exceptions.ConnectionError("server stopped")

    def fake_post(url, json=None, timeout=None):
        return DummyResponse(200, {"status": "ok"})

    def fake_prompt_switch_detection(
        n_images,
        mode="freeform",
        contact_sheet_rows=0,
        contact_sheet_cols=0,
        sheet_count=0,
    ):
        seen["n_images"] = n_images
        seen["mode"] = mode
        seen["contact_sheet_rows"] = contact_sheet_rows
        seen["contact_sheet_cols"] = contact_sheet_cols
        seen["sheet_count"] = sheet_count
        return "prompt"

    monkeypatch.setattr("video2tasks.worker.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("video2tasks.worker.runner.prompt_switch_detection", fake_prompt_switch_detection)
    monkeypatch.setattr("video2tasks.worker.runner.requests.get", fake_get)
    monkeypatch.setattr("video2tasks.worker.runner.requests.post", fake_post)
    monkeypatch.setattr("video2tasks.worker.runner.time.sleep", lambda *_args, **_kwargs: None)

    cfg = Config(
        worker={"backend": "dummy", "server_url": "http://127.0.0.1:8099"},
        windowing={"boundary_prompt_mode": "center_scan"},
    )

    run_worker(cfg)

    capsys.readouterr()
    assert seen == {
        "n_images": 0,
        "mode": "center_scan",
        "contact_sheet_rows": 0,
        "contact_sheet_cols": 0,
        "sheet_count": 0,
    }


def test_worker_uses_logical_frame_count_for_contact_sheet_prompt(
    monkeypatch,
    capsys,
) -> None:
    backend = DummyBackend()
    seen = {}
    get_attempts = {"count": 0}

    def fake_get(url, timeout=None):
        get_attempts["count"] += 1
        if get_attempts["count"] == 1:
            return DummyResponse(
                200,
                {
                    "data": {
                        "task_id": "demo::sample_roboturk_tower_w0",
                        "images": ["", ""],
                        "meta": {
                            "subset": "demo",
                            "sample_id": "sample_roboturk_tower",
                            "frame_ids": list(range(16)),
                            "contact_sheet_rows": 4,
                            "contact_sheet_cols": 4,
                        },
                    }
                },
            )
        raise requests.exceptions.ConnectionError("server stopped")

    def fake_post(url, json=None, timeout=None):
        return DummyResponse(200, {"status": "ok"})

    def fake_prompt_switch_detection(
        n_images,
        mode="freeform",
        contact_sheet_rows=0,
        contact_sheet_cols=0,
        sheet_count=0,
    ):
        seen["n_images"] = n_images
        seen["mode"] = mode
        seen["contact_sheet_rows"] = contact_sheet_rows
        seen["contact_sheet_cols"] = contact_sheet_cols
        seen["sheet_count"] = sheet_count
        return "prompt"

    monkeypatch.setattr("video2tasks.worker.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("video2tasks.worker.runner.prompt_switch_detection", fake_prompt_switch_detection)
    monkeypatch.setattr("video2tasks.worker.runner.requests.get", fake_get)
    monkeypatch.setattr("video2tasks.worker.runner.requests.post", fake_post)
    monkeypatch.setattr("video2tasks.worker.runner.time.sleep", lambda *_args, **_kwargs: None)

    cfg = Config(worker={"backend": "dummy", "server_url": "http://127.0.0.1:8099"})

    run_worker(cfg)

    capsys.readouterr()
    assert seen == {
        "n_images": 16,
        "mode": "freeform",
        "contact_sheet_rows": 4,
        "contact_sheet_cols": 4,
        "sheet_count": 2,
    }


def test_worker_uses_segment_instruction_prompt_for_segment_label_job(
    monkeypatch,
    capsys,
) -> None:
    backend = DummyBackend()
    seen = {"switch_calls": 0, "segment_calls": 0}
    get_attempts = {"count": 0}

    def fake_get(url, timeout=None):
        get_attempts["count"] += 1
        if get_attempts["count"] == 1:
            return DummyResponse(
                200,
                {
                    "data": {
                        "task_id": "demo::sample_roboturk_tower_seg0",
                        "images": [],
                        "meta": {
                            "subset": "demo",
                            "sample_id": "sample_roboturk_tower",
                            "job_type": "segment_label",
                            "segment_id": 0,
                        },
                    }
                },
            )
        raise requests.exceptions.ConnectionError("server stopped")

    def fake_post(url, json=None, timeout=None):
        return DummyResponse(200, {"status": "ok"})

    def fake_prompt_switch_detection(
        n_images,
        mode="freeform",
        contact_sheet_rows=0,
        contact_sheet_cols=0,
        sheet_count=0,
    ):
        seen["switch_calls"] += 1
        return "switch-prompt"

    def fake_prompt_segment_instruction(
        n_images,
        contact_sheet_rows=0,
        contact_sheet_cols=0,
        sheet_count=0,
    ):
        seen["segment_calls"] += 1
        seen["n_images"] = n_images
        seen["contact_sheet_rows"] = contact_sheet_rows
        seen["contact_sheet_cols"] = contact_sheet_cols
        seen["sheet_count"] = sheet_count
        return "segment-prompt"

    monkeypatch.setattr("video2tasks.worker.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("video2tasks.worker.runner.prompt_switch_detection", fake_prompt_switch_detection)
    monkeypatch.setattr("video2tasks.worker.runner.prompt_segment_instruction", fake_prompt_segment_instruction)
    monkeypatch.setattr("video2tasks.worker.runner.requests.get", fake_get)
    monkeypatch.setattr("video2tasks.worker.runner.requests.post", fake_post)
    monkeypatch.setattr("video2tasks.worker.runner.time.sleep", lambda *_args, **_kwargs: None)

    cfg = Config(worker={"backend": "dummy", "server_url": "http://127.0.0.1:8099"})

    run_worker(cfg)

    capsys.readouterr()
    assert seen == {
        "switch_calls": 0,
        "segment_calls": 1,
        "n_images": 0,
        "contact_sheet_rows": 0,
        "contact_sheet_cols": 0,
        "sheet_count": 0,
    }


def test_worker_uses_boundary_refinement_prompt_for_boundary_refinement_job(
    monkeypatch,
    capsys,
) -> None:
    backend = DummyBackend()
    seen = {"switch_calls": 0, "segment_calls": 0, "refinement_calls": 0}
    get_attempts = {"count": 0}

    def fake_get(url, timeout=None):
        get_attempts["count"] += 1
        if get_attempts["count"] == 1:
            return DummyResponse(
                200,
                {
                    "data": {
                        "task_id": "demo::sample_roboturk_tower_b0",
                        "images": [],
                        "meta": {
                            "subset": "demo",
                            "sample_id": "sample_roboturk_tower",
                            "job_type": "boundary_refinement",
                            "boundary_id": 0,
                        },
                    }
                },
            )
        raise requests.exceptions.ConnectionError("server stopped")

    def fake_post(url, json=None, timeout=None):
        return DummyResponse(200, {"status": "ok"})

    def fake_prompt_switch_detection(
        n_images,
        mode="freeform",
        contact_sheet_rows=0,
        contact_sheet_cols=0,
        sheet_count=0,
    ):
        seen["switch_calls"] += 1
        return "switch-prompt"

    def fake_prompt_segment_instruction(
        n_images,
        contact_sheet_rows=0,
        contact_sheet_cols=0,
        sheet_count=0,
    ):
        seen["segment_calls"] += 1
        return "segment-prompt"

    def fake_prompt_boundary_refinement(
        n_images,
        contact_sheet_rows=0,
        contact_sheet_cols=0,
        sheet_count=0,
    ):
        seen["refinement_calls"] += 1
        seen["n_images"] = n_images
        seen["contact_sheet_rows"] = contact_sheet_rows
        seen["contact_sheet_cols"] = contact_sheet_cols
        seen["sheet_count"] = sheet_count
        return "boundary-refinement-prompt"

    monkeypatch.setattr("video2tasks.worker.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("video2tasks.worker.runner.prompt_switch_detection", fake_prompt_switch_detection)
    monkeypatch.setattr("video2tasks.worker.runner.prompt_segment_instruction", fake_prompt_segment_instruction)
    monkeypatch.setattr("video2tasks.worker.runner.prompt_boundary_refinement", fake_prompt_boundary_refinement)
    monkeypatch.setattr("video2tasks.worker.runner.requests.get", fake_get)
    monkeypatch.setattr("video2tasks.worker.runner.requests.post", fake_post)
    monkeypatch.setattr("video2tasks.worker.runner.time.sleep", lambda *_args, **_kwargs: None)

    cfg = Config(worker={"backend": "dummy", "server_url": "http://127.0.0.1:8099"})

    run_worker(cfg)

    capsys.readouterr()
    assert seen == {
        "switch_calls": 0,
        "segment_calls": 0,
        "refinement_calls": 1,
        "n_images": 0,
        "contact_sheet_rows": 0,
        "contact_sheet_cols": 0,
        "sheet_count": 0,
    }
