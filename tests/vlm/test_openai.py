import json

import pytest

import numpy as np

from video2tasks.vlm.factory import create_backend


class DummyResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class DummyStreamResponse:
    def __init__(self, status_code: int, chunks):
        self.status_code = status_code
        self._chunks = chunks

    def iter_lines(self, decode_unicode=False):
        for chunk in self._chunks:
            if decode_unicode:
                yield chunk
            else:
                yield chunk.encode("utf-8")


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


def test_openai_backend_infer_text_json_posts_text_only_schema(monkeypatch) -> None:
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
                                        "thought": "Merge adjacent fragments.",
                                        "merged_ranges": [
                                            {"start_seg_id": 0, "end_seg_id": 1},
                                            {"start_seg_id": 2, "end_seg_id": 2},
                                        ],
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
    )

    result = backend.infer_text_json(
        "Merge adjacent fragments",
        schema_name="segment_merge_result",
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "thought": {"type": "string"},
                "merged_ranges": {"type": "array"},
            },
            "required": ["thought", "merged_ranges"],
        },
        max_output_tokens=640,
        reasoning_effort="high",
    )

    assert result == {
        "thought": "Merge adjacent fragments.",
        "merged_ranges": [
            {"start_seg_id": 0, "end_seg_id": 1},
            {"start_seg_id": 2, "end_seg_id": 2},
        ],
    }
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["json"]["input"][0]["content"] == [{"type": "input_text", "text": "Merge adjacent fragments"}]
    assert captured["json"]["text"]["format"]["name"] == "segment_merge_result"
    assert captured["json"]["max_output_tokens"] == 640
    assert captured["json"]["reasoning"] == {"effort": "high"}


def test_openai_backend_infer_text_json_falls_back_to_chat_completions(monkeypatch) -> None:
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if url.endswith("/responses"):
            return DummyResponse(200, {})
        if url.endswith("/chat/completions"):
            return DummyResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_module.dumps(
                                    {
                                        "thought": "Merge adjacent fragments.",
                                        "merged_ranges": [
                                            {"start_seg_id": 0, "end_seg_id": 1},
                                            {"start_seg_id": 2, "end_seg_id": 2},
                                        ],
                                    }
                                )
                            }
                        }
                    ]
                },
            )
        raise AssertionError(f"unexpected url: {url}")

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
    )

    result = backend.infer_text_json(
        "Merge adjacent fragments",
        schema_name="segment_merge_result",
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "thought": {"type": "string"},
                "merged_ranges": {"type": "array"},
            },
            "required": ["thought", "merged_ranges"],
        },
        max_output_tokens=640,
        reasoning_effort="high",
    )

    assert result == {
        "thought": "Merge adjacent fragments.",
        "merged_ranges": [
            {"start_seg_id": 0, "end_seg_id": 1},
            {"start_seg_id": 2, "end_seg_id": 2},
        ],
    }
    assert [call["url"] for call in calls] == [
        "https://api.openai.com/v1/responses",
        "https://api.openai.com/v1/chat/completions",
    ]
    assert calls[1]["json"]["messages"][0]["content"] == [{"type": "text", "text": "Merge adjacent fragments"}]
    assert calls[1]["json"]["response_format"]["json_schema"]["name"] == "segment_merge_result"


def test_openai_backend_infer_text_json_prefers_stream_first_for_proxy_base_url(monkeypatch) -> None:
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None, stream=False):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout, "stream": stream})
        if url.endswith("/chat/completions") and stream is True:
            return DummyStreamResponse(
                200,
                [
                    "data: " + json_module.dumps({"choices": [{"delta": {"content": '{\"thought\":\"Merge adjacent fragments.\",\"merged_ranges\":[{'}, "finish_reason": None}]}),
                    "data: " + json_module.dumps({"choices": [{"delta": {"content": '\"start_seg_id\":0,\"end_seg_id\":1},{\"start_seg_id\":2,\"end_seg_id\":2}]}'}, "finish_reason": "stop"}]}),
                    "data: [DONE]",
                ],
            )
        raise AssertionError(f"unexpected url: {url}")

    json_module = json
    monkeypatch.setattr("video2tasks.vlm.openai_api.requests.post", fake_post)

    backend = create_backend(
        "openai",
        api_key="sk-test",
        model="gpt-5.2",
        base_url="https://sub.alpen-y.top/v1",
        timeout_sec=12.0,
        reasoning_effort="low",
        max_output_tokens=300,
    )

    result = backend.infer_text_json(
        "Merge adjacent fragments",
        schema_name="segment_merge_result",
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "thought": {"type": "string"},
                "merged_ranges": {"type": "array"},
            },
            "required": ["thought", "merged_ranges"],
        },
        max_output_tokens=640,
        reasoning_effort="high",
    )

    assert result == {
        "thought": "Merge adjacent fragments.",
        "merged_ranges": [
            {"start_seg_id": 0, "end_seg_id": 1},
            {"start_seg_id": 2, "end_seg_id": 2},
        ],
    }
    assert [call["url"] for call in calls] == [
        "https://sub.alpen-y.top/v1/chat/completions",
    ]
    assert calls[0]["stream"] is True
    assert calls[0]["json"]["stream"] is True
    assert calls[0]["json"]["messages"][0]["content"] == [{"type": "text", "text": "Merge adjacent fragments"}]


def test_openai_backend_infer_text_json_can_raise_on_http_error(monkeypatch) -> None:
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(url)
        return DummyResponse(401, {"error": "unauthorized"})

    monkeypatch.setattr("video2tasks.vlm.openai_api.requests.post", fake_post)

    backend = create_backend(
        "openai",
        api_key="sk-test",
        model="gpt-5.2",
        base_url="https://api.openai.com/v1",
    )

    with pytest.raises(RuntimeError, match="openai_text_json_http_error"):
        backend.infer_text_json(
            "Merge adjacent fragments",
            schema_name="segment_merge_result",
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "thought": {"type": "string"},
                    "merged_ranges": {"type": "array"},
                },
                "required": ["thought", "merged_ranges"],
            },
            raise_on_http_error=True,
        )

    assert calls == [
        "https://api.openai.com/v1/responses",
        "https://api.openai.com/v1/chat/completions",
    ]


def test_openai_backend_infer_falls_back_to_chat_completions_for_images(monkeypatch) -> None:
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if url.endswith("/responses"):
            return DummyResponse(500, {"error": "server error"})
        if url.endswith("/chat/completions"):
            return DummyResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_module.dumps(
                                    {
                                        "thought": "Switch at frame 1",
                                        "transitions": [1],
                                        "instructions": ["Task A", "Task B"],
                                    }
                                )
                            }
                        }
                    ]
                },
            )
        raise AssertionError(f"unexpected url: {url}")

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

    result = backend.infer([np.zeros((8, 8, 3), dtype=np.uint8)], "Detect switches")

    assert result == {
        "thought": "Switch at frame 1",
        "transitions": [1],
        "instructions": ["Task A", "Task B"],
    }
    assert [call["url"] for call in calls] == [
        "https://api.openai.com/v1/responses",
        "https://api.openai.com/v1/chat/completions",
    ]
    fallback_content = calls[1]["json"]["messages"][0]["content"]
    assert fallback_content[0] == {"type": "text", "text": "Detect switches"}
    assert fallback_content[1]["type"] == "image_url"
    assert fallback_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_backend_rejects_mismatched_instruction_count(monkeypatch) -> None:
    json_module = json

    def fake_post(url, json=None, headers=None, timeout=None):
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
                                        "thought": "Bad payload",
                                        "transitions": [1],
                                        "instructions": ["Only one instruction", "", "Extra"],
                                    }
                                ),
                            }
                        ]
                    }
                ]
            },
        )

    monkeypatch.setattr("video2tasks.vlm.openai_api.requests.post", fake_post)

    backend = create_backend(
        "openai",
        api_key="sk-test",
        model="gpt-5.2",
    )

    result = backend.infer([np.zeros((8, 8, 3), dtype=np.uint8)], "Detect switches")

    assert result == {}
