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
    assert "[Done] demo::sample_roboturk_tower_w0 (0f) -> Cuts: []" in out
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
