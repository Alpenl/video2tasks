import json

import numpy as np

from video2tasks.vlm.factory import create_backend


class DummyResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_openai_backend_posts_images_and_parses_structured_response(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse(
            200,
            {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": json_module.dumps(
                                    {
                                        "thought": "Switch at frame 1",
                                        "transitions": [1],
                                        "instructions": ["Task A", "Task B"],
                                    }
                                ),
                            }
                        ]
                    }
                ]
            },
        )

    json_module = json
    monkeypatch.setattr("video2tasks.vlm.openai_api.requests.post", fake_post)

    backend = create_backend(
        "openai",
        api_key="sk-test",
        model="gpt-5.2",
        base_url="https://api.openai.com/v1",
        timeout_sec=12.0,
        reasoning_effort="low",
        max_output_tokens=300,
        jpeg_quality=75,
    )

    result = backend.infer(
        [
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8, 3), 255, dtype=np.uint8),
        ],
        "Detect switches",
    )

    assert result == {
        "thought": "Switch at frame 1",
        "transitions": [1],
        "instructions": ["Task A", "Task B"],
    }
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["timeout"] == 12.0
    assert captured["json"]["model"] == "gpt-5.2"
    assert captured["json"]["reasoning"] == {"effort": "low"}
    assert captured["json"]["max_output_tokens"] == 300
    assert captured["json"]["text"]["format"]["type"] == "json_schema"

    content = captured["json"]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Detect switches"}
    assert len(content) == 3
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")


def test_openai_backend_returns_empty_dict_on_unparseable_response(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None):
        return DummyResponse(
            200,
            {"output": [{"content": [{"type": "output_text", "text": "not-json"}]}]},
        )

    monkeypatch.setattr("video2tasks.vlm.openai_api.requests.post", fake_post)

    backend = create_backend(
        "openai",
        api_key="sk-test",
        model="gpt-5.2",
    )

    result = backend.infer([np.zeros((8, 8, 3), dtype=np.uint8)], "Detect switches")

    assert result == {}
