import base64
import json
import types

import numpy as np
import requests

from video2tasks.vlm.factory import create_backend
import video2tasks.vlm.gemini_api as gemini_api_module


class DummyResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_gemini_backend_posts_images_and_parses_structured_response(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json_module.dumps(
                                        {
                                            "thought": "Switch at frame 2",
                                            "transitions": [2],
                                            "instructions": ["Task A", "Task B"],
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
        )

    json_module = json
    monkeypatch.setattr("video2tasks.vlm.gemini_api.requests.post", fake_post)

    backend = create_backend(
        "gemini",
        api_key="gem-test",
        model="gemini-3-flash-preview",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        timeout_sec=15.0,
        max_output_tokens=256,
        jpeg_quality=70,
    )

    result = backend.infer(
        [
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8, 3), 255, dtype=np.uint8),
        ],
        "Detect switches",
    )

    assert result == {
        "thought": "Switch at frame 2",
        "transitions": [2],
        "instructions": ["Task A", "Task B"],
    }
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3-flash-preview:generateContent"
    )
    assert captured["headers"]["x-goog-api-key"] == "gem-test"
    assert captured["timeout"] == 15.0
    assert captured["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert captured["json"]["generationConfig"]["maxOutputTokens"] == 256

    parts = captured["json"]["contents"][0]["parts"]
    assert parts[0] == {"text": "Detect switches"}
    assert len(parts) == 3
    assert parts[1]["inlineData"]["mimeType"] == "image/jpeg"
    assert parts[1]["inlineData"]["data"]


def test_gemini_backend_returns_empty_dict_on_unparseable_response(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None):
        return DummyResponse(
            200,
            {"candidates": [{"content": {"parts": [{"text": "not-json"}]}}]},
        )

    monkeypatch.setattr("video2tasks.vlm.gemini_api.requests.post", fake_post)

    backend = create_backend(
        "gemini",
        api_key="gem-test",
        model="gemini-3-flash-preview",
    )

    result = backend.infer([np.zeros((8, 8, 3), dtype=np.uint8)], "Detect switches")

    assert result == {}


def test_gemini_backend_supports_openai_compatible_chat_endpoint(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps(
                                {
                                    "thought": "No switch",
                                    "transitions": [],
                                    "instructions": ["Task A"],
                                }
                            )
                        }
                    }
                ]
            },
        )

    json_module = json
    monkeypatch.setattr("video2tasks.vlm.gemini_api.requests.post", fake_post)

    backend = create_backend(
        "gemini",
        api_key="gem-test",
        model="gemini-3-flash-preview",
        base_url="https://api.laozhang.ai/v1",
        api_mode="openai_compatible",
        timeout_sec=10.0,
        max_output_tokens=123,
        jpeg_quality=65,
    )

    result = backend.infer([np.zeros((8, 8, 3), dtype=np.uint8)], "Detect switches")

    assert result == {
        "thought": "No switch",
        "transitions": [],
        "instructions": ["Task A"],
    }
    assert captured["url"] == "https://api.laozhang.ai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer gem-test"
    assert captured["json"]["model"] == "gemini-3-flash-preview"
    assert captured["json"]["max_tokens"] == 123
    assert captured["json"]["messages"][0]["role"] == "system"
    assert "JSON" in captured["json"]["messages"][0]["content"]
    user_content = captured["json"]["messages"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "Detect switches"}
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_gemini_backend_normalizes_bare_openai_compatible_base_url(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        return DummyResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps(
                                {
                                    "thought": "No switch",
                                    "transitions": [],
                                    "instructions": ["Task A"],
                                }
                            )
                        }
                    }
                ]
            },
        )

    json_module = json
    monkeypatch.setattr("video2tasks.vlm.gemini_api.requests.post", fake_post)

    backend = create_backend(
        "gemini",
        api_key="gem-test",
        model="gemini-3.1-pro-preview",
        base_url="https://api.duckcoding.ai",
        api_mode="openai_compatible",
        timeout_sec=10.0,
        max_output_tokens=123,
    )

    backend.infer([np.zeros((8, 8, 3), dtype=np.uint8)], "Detect switches")

    assert captured["url"] == "https://api.duckcoding.ai/v1/chat/completions"


def test_gemini_backend_normalizes_bare_native_base_url(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        return DummyResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json_module.dumps(
                                        {
                                            "thought": "Switch at frame 2",
                                            "transitions": [2],
                                            "instructions": ["Task A", "Task B"],
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
        )

    json_module = json
    monkeypatch.setattr("video2tasks.vlm.gemini_api.requests.post", fake_post)

    backend = create_backend(
        "gemini",
        api_key="gem-test",
        model="gemini-3-flash-preview",
        base_url="https://generativelanguage.googleapis.com",
        api_mode="native",
        timeout_sec=15.0,
        max_output_tokens=256,
    )

    backend.infer([np.zeros((8, 8, 3), dtype=np.uint8)], "Detect switches")

    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3-flash-preview:generateContent"
    )


def test_gemini_openai_compatible_falls_back_to_reasoning_content(monkeypatch) -> None:
    json_module = json

    def fake_post(url, json=None, headers=None, timeout=None):
        return DummyResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": json_module.dumps(
                                {
                                    "thought": "Recovered from reasoning",
                                    "transitions": [3],
                                    "instructions": ["Task A", "Task B"],
                                }
                            ),
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("video2tasks.vlm.gemini_api.requests.post", fake_post)

    backend = create_backend(
        "gemini",
        api_key="gem-test",
        model="gemini-3.1-pro-preview",
        base_url="https://api.duckcoding.ai/v1",
        api_mode="openai_compatible",
        timeout_sec=10.0,
        max_output_tokens=256,
    )

    result = backend.infer([np.zeros((8, 8, 3), dtype=np.uint8)], "Detect switches")

    assert result == {
        "thought": "Recovered from reasoning",
        "transitions": [3],
        "instructions": ["Task A", "Task B"],
    }


def test_gemini_backend_wraps_single_instruction_string(monkeypatch) -> None:
    json_module = json

    def fake_post(url, json=None, headers=None, timeout=None):
        return DummyResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps(
                                {
                                    "thought": "One coarse task",
                                    "transitions": [],
                                    "instructions": "Task A",
                                }
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("video2tasks.vlm.gemini_api.requests.post", fake_post)

    backend = create_backend(
        "gemini",
        api_key="gem-test",
        model="gemini-3-flash-preview",
        base_url="https://api.laozhang.ai/v1",
        api_mode="openai_compatible",
        timeout_sec=10.0,
        max_output_tokens=2048,
    )

    result = backend.infer([np.zeros((8, 8, 3), dtype=np.uint8)], "Detect switches")

    assert result == {
        "thought": "One coarse task",
        "transitions": [],
        "instructions": ["Task A"],
    }


def test_gemini_backend_retries_openai_compatible_after_request_exception(monkeypatch) -> None:
    calls = {"count": 0}
    json_module = json

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.SSLError("EOF during TLS read")
        return DummyResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps(
                                {
                                    "thought": "Recovered on retry",
                                    "transitions": [],
                                    "instructions": ["Task A"],
                                }
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("video2tasks.vlm.gemini_api.requests.post", fake_post)
    monkeypatch.setattr(
        gemini_api_module,
        "time",
        types.SimpleNamespace(sleep=lambda _sec: None),
        raising=False,
    )

    backend = create_backend(
        "gemini",
        api_key="gem-test",
        model="gemini-3-flash-preview",
        base_url="https://www.duckcoding.ai/v1",
        api_mode="openai_compatible",
        timeout_sec=10.0,
        max_output_tokens=2048,
    )

    result = backend.infer([np.zeros((8, 8, 3), dtype=np.uint8)], "Detect switches")

    assert result == {
        "thought": "Recovered on retry",
        "transitions": [],
        "instructions": ["Task A"],
    }
    assert calls["count"] == 2


def test_gemini_backend_retries_openai_compatible_after_empty_payload(monkeypatch) -> None:
    calls = {"count": 0}
    json_module = json

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return DummyResponse(200, {"choices": [{"message": {"content": "```json\n{\n"}}]})
        return DummyResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps(
                                {
                                    "thought": "Recovered after empty payload",
                                    "transitions": [3],
                                    "instructions": ["Task A", "Task B"],
                                }
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("video2tasks.vlm.gemini_api.requests.post", fake_post)
    monkeypatch.setattr(
        gemini_api_module,
        "time",
        types.SimpleNamespace(sleep=lambda _sec: None),
        raising=False,
    )

    backend = create_backend(
        "gemini",
        api_key="gem-test",
        model="gemini-3-flash-preview",
        base_url="https://www.duckcoding.ai/v1",
        api_mode="openai_compatible",
        timeout_sec=10.0,
        max_output_tokens=2048,
    )

    result = backend.infer([np.zeros((8, 8, 3), dtype=np.uint8)], "Detect switches")

    assert result == {
        "thought": "Recovered after empty payload",
        "transitions": [3],
        "instructions": ["Task A", "Task B"],
    }
    assert calls["count"] == 2


def test_gemini_backend_falls_back_to_curl_after_repeated_empty_payload(monkeypatch) -> None:
    json_module = json
    request_calls = {"count": 0}
    curl_calls = {"count": 0}

    def fake_post_json(url, headers, payload, timeout_sec):
        request_calls["count"] += 1
        return 200, json_module.dumps({"choices": [{"message": {"content": ""}}]})

    def fake_post_json_via_curl(url, headers, payload, timeout_sec):
        curl_calls["count"] += 1
        return 200, json_module.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps(
                                {
                                    "thought": "Recovered via curl fallback",
                                    "transitions": [4],
                                    "instructions": ["Task A", "Task B"],
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(gemini_api_module, "_post_json", fake_post_json)
    monkeypatch.setattr(gemini_api_module, "_post_json_via_curl", fake_post_json_via_curl)
    monkeypatch.setattr(
        gemini_api_module,
        "time",
        types.SimpleNamespace(sleep=lambda _sec: None),
        raising=False,
    )

    backend = create_backend(
        "gemini",
        api_key="gem-test",
        model="gemini-3.1-pro-preview",
        base_url="https://api.duckcoding.ai",
        api_mode="openai_compatible",
        timeout_sec=10.0,
        max_output_tokens=2048,
    )

    result = backend.infer([np.zeros((8, 8, 3), dtype=np.uint8)], "Detect switches")

    assert result == {
        "thought": "Recovered via curl fallback",
        "transitions": [4],
        "instructions": ["Task A", "Task B"],
    }
    assert request_calls["count"] == 5
    assert curl_calls["count"] == 1


def test_gemini_backend_uses_raw_png_payload_for_native_requests(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return DummyResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json_module.dumps(
                                        {
                                            "thought": "Switch at frame 1",
                                            "transitions": [1],
                                            "instructions": ["Task A", "Task B"],
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
        )

    json_module = json
    monkeypatch.setattr("video2tasks.vlm.gemini_api.requests.post", fake_post)

    backend = create_backend(
        "gemini",
        api_key="gem-test",
        model="gemini-3-flash-preview",
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )

    raw_png = b"\x89PNG\r\n\x1a\nraw-png"
    result = backend.infer(
        [{"raw_bytes": raw_png, "mime_type": "image/png"}],
        "Detect switches",
    )

    assert result["transitions"] == [1]
    parts = captured["json"]["contents"][0]["parts"]
    assert parts[1]["inlineData"]["mimeType"] == "image/png"
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == raw_png


def test_gemini_backend_uses_raw_png_payload_for_openai_compatible_requests(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return DummyResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps(
                                {
                                    "thought": "No switch",
                                    "transitions": [],
                                    "instructions": ["Task A"],
                                }
                            )
                        }
                    }
                ]
            },
        )

    json_module = json
    monkeypatch.setattr("video2tasks.vlm.gemini_api.requests.post", fake_post)

    backend = create_backend(
        "gemini",
        api_key="gem-test",
        model="gemini-3.1-pro-preview",
        base_url="https://api.duckcoding.ai",
        api_mode="openai_compatible",
    )

    raw_png = b"\x89PNG\r\n\x1a\nraw-png"
    result = backend.infer(
        [{"raw_bytes": raw_png, "mime_type": "image/png"}],
        "Detect switches",
    )

    assert result["instructions"] == ["Task A"]
    user_content = captured["json"]["messages"][1]["content"]
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    encoded = user_content[1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == raw_png
