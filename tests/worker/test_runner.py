import base64
import requests
from io import BytesIO

from PIL import Image

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
                        "images": [
                            base64.b64encode(_png_bytes()).decode("utf-8"),
                            base64.b64encode(_png_bytes()).decode("utf-8"),
                        ],
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


class CountingBackend(DummyBackend):
    def __init__(self) -> None:
        super().__init__()
        self.infer_calls = 0
        self.last_image_count = None

    def infer(self, images, prompt):
        self.infer_calls += 1
        self.last_image_count = len(images)
        return super().infer(images, prompt)


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_worker_loads_images_from_paths_and_submits_dispatch_id(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    backend = CountingBackend()
    image_path = tmp_path / "sheet.png"
    image_path.write_bytes(_png_bytes())
    posted = {}
    get_attempts = {"count": 0}

    def fake_get(url, timeout=None):
        get_attempts["count"] += 1
        if get_attempts["count"] == 1:
            return DummyResponse(
                200,
                {
                    "data": {
                        "task_id": "demo::sample_roboturk_tower_w0",
                        "dispatch_id": "d7",
                        "image_paths": [str(image_path)],
                        "meta": {
                            "subset": "demo",
                            "sample_id": "sample_roboturk_tower",
                            "logical_frame_count": 16,
                            "contact_sheet_rows": 4,
                            "contact_sheet_cols": 4,
                        },
                    }
                },
            )
        raise requests.exceptions.ConnectionError("server stopped")

    def fake_post(url, json=None, timeout=None):
        posted.update(json or {})
        return DummyResponse(200, {"status": "received"})

    monkeypatch.setattr("video2tasks.worker.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("video2tasks.worker.runner.requests.get", fake_get)
    monkeypatch.setattr("video2tasks.worker.runner.requests.post", fake_post)
    monkeypatch.setattr("video2tasks.worker.runner.time.sleep", lambda *_args, **_kwargs: None)

    cfg = Config(worker={"backend": "dummy", "server_url": "http://127.0.0.1:8099"})

    run_worker(cfg)

    capsys.readouterr()
    assert backend.infer_calls == 1
    assert backend.last_image_count == 1
    assert posted["dispatch_id"] == "d7"


def test_worker_does_not_infer_when_inline_image_decode_fails(
    monkeypatch,
    capsys,
) -> None:
    backend = CountingBackend()
    get_attempts = {"count": 0}

    def fake_get(url, timeout=None):
        get_attempts["count"] += 1
        if get_attempts["count"] == 1:
            return DummyResponse(
                200,
                {
                    "data": {
                        "task_id": "demo::sample_roboturk_tower_w0",
                        "dispatch_id": "d1",
                        "images": ["not-valid-base64"],
                        "meta": {
                            "subset": "demo",
                            "sample_id": "sample_roboturk_tower",
                        },
                    }
                },
            )
        raise requests.exceptions.ConnectionError("server stopped")

    def fake_post(url, json=None, timeout=None):
        raise AssertionError("submit_result should not be called when image decode fails")

    monkeypatch.setattr("video2tasks.worker.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("video2tasks.worker.runner.requests.get", fake_get)
    monkeypatch.setattr("video2tasks.worker.runner.requests.post", fake_post)
    monkeypatch.setattr("video2tasks.worker.runner.time.sleep", lambda *_args, **_kwargs: None)

    cfg = Config(worker={"backend": "dummy", "server_url": "http://127.0.0.1:8099"})

    run_worker(cfg)

    out = capsys.readouterr().out
    assert backend.infer_calls == 0
    assert "failed to decode inline images at indices [0]" in out


def test_worker_retries_submit_until_server_ack(
    monkeypatch,
    capsys,
) -> None:
    backend = CountingBackend()
    get_attempts = {"count": 0}
    post_attempts = {"count": 0}

    def fake_get(url, timeout=None):
        get_attempts["count"] += 1
        if get_attempts["count"] == 1:
            return DummyResponse(
                200,
                {
                    "data": {
                        "task_id": "demo::sample_roboturk_tower_w0",
                        "dispatch_id": "d3",
                        "images": [],
                        "meta": {
                            "subset": "demo",
                            "sample_id": "sample_roboturk_tower",
                        },
                    }
                },
            )
        raise requests.exceptions.ConnectionError("server stopped")

    def fake_post(url, json=None, timeout=None):
        post_attempts["count"] += 1
        if post_attempts["count"] == 1:
            raise requests.exceptions.ConnectionError("temporary network issue")
        if post_attempts["count"] == 2:
            return DummyResponse(500, {"status": "error"})
        return DummyResponse(200, {"status": "received"})

    monkeypatch.setattr("video2tasks.worker.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("video2tasks.worker.runner.requests.get", fake_get)
    monkeypatch.setattr("video2tasks.worker.runner.requests.post", fake_post)
    monkeypatch.setattr("video2tasks.worker.runner.time.sleep", lambda *_args, **_kwargs: None)

    cfg = Config(worker={"backend": "dummy", "server_url": "http://127.0.0.1:8099"})

    run_worker(cfg)

    out = capsys.readouterr().out
    assert backend.infer_calls == 1
    assert post_attempts["count"] == 3
    assert "[Warn] Submit failed for demo::sample_roboturk_tower_w0" in out


class RawPayloadBackend(DummyBackend):
    name = "gemini"

    def __init__(self) -> None:
        super().__init__()
        self.seen_images = None

    def infer(self, images, prompt):
        self.seen_images = images
        return super().infer(images, prompt)


def test_worker_passes_raw_image_payloads_to_gemini_backend(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    backend = RawPayloadBackend()
    image_path = tmp_path / "sheet.png"
    image_bytes = _png_bytes()
    image_path.write_bytes(image_bytes)
    get_attempts = {"count": 0}

    def fake_get(url, timeout=None):
        get_attempts["count"] += 1
        if get_attempts["count"] == 1:
            return DummyResponse(
                200,
                {
                    "data": {
                        "task_id": "demo::sample_roboturk_tower_w0",
                        "dispatch_id": "d9",
                        "image_paths": [str(image_path)],
                        "meta": {
                            "subset": "demo",
                            "sample_id": "sample_roboturk_tower",
                            "logical_frame_count": 16,
                            "contact_sheet_rows": 4,
                            "contact_sheet_cols": 4,
                        },
                    }
                },
            )
        raise requests.exceptions.ConnectionError("server stopped")

    def fake_post(url, json=None, timeout=None):
        return DummyResponse(200, {"status": "received"})

    monkeypatch.setattr("video2tasks.worker.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("video2tasks.worker.runner.requests.get", fake_get)
    monkeypatch.setattr("video2tasks.worker.runner.requests.post", fake_post)
    monkeypatch.setattr("video2tasks.worker.runner.time.sleep", lambda *_args, **_kwargs: None)

    cfg = Config(worker={"backend": "gemini", "server_url": "http://127.0.0.1:8099", "gemini": {"api_key": "x"}})

    run_worker(cfg)

    capsys.readouterr()
    assert isinstance(backend.seen_images, list)
    assert backend.seen_images[0]["mime_type"] == "image/png"
    assert backend.seen_images[0]["raw_bytes"] == image_bytes


class InvalidPayloadBackend(DummyBackend):
    def __init__(self) -> None:
        super().__init__()
        self.infer_calls = 0

    def infer(self, images, prompt):
        self.infer_calls += 1
        return {
            "thought": "bad payload",
            "transitions": [1],
            "instructions": ["Only one instruction"],
        }


def test_worker_retries_invalid_structured_payload_and_submits_empty_json(
    monkeypatch,
    capsys,
) -> None:
    backend = InvalidPayloadBackend()
    submitted = {}
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
                        "meta": {
                            "subset": "demo",
                            "sample_id": "sample_roboturk_tower",
                            "logical_frame_count": 4,
                        },
                    }
                },
            )
        raise requests.exceptions.ConnectionError("server stopped")

    def fake_post(url, json=None, timeout=None):
        submitted["payload"] = json
        return DummyResponse(200, {"status": "ok"})

    monkeypatch.setattr("video2tasks.worker.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("video2tasks.worker.runner.requests.get", fake_get)
    monkeypatch.setattr("video2tasks.worker.runner.requests.post", fake_post)
    monkeypatch.setattr("video2tasks.worker.runner.time.sleep", lambda *_args, **_kwargs: None)

    cfg = Config(worker={"backend": "dummy", "server_url": "http://127.0.0.1:8099"})

    run_worker(cfg)

    out = capsys.readouterr().out
    assert backend.infer_calls == 4
    assert submitted["payload"]["vlm_json"] == {}
    assert "Empty or invalid VLM JSON" in out
