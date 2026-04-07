import json

import requests

from video2tasks.vlm.factory import create_backend


class DummyResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class DummyStreamResponse:
    def __init__(self, status_code, chunks):
        self.status_code = status_code
        self._chunks = chunks

    def iter_lines(self, decode_unicode=False):
        for chunk in self._chunks:
            if decode_unicode:
                yield chunk
            else:
                yield chunk.encode("utf-8")


def _merge_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "thought": {"type": "string"},
            "merged_ranges": {"type": "array"},
        },
        "required": ["thought", "merged_ranges"],
    }


def test_openai_backend_records_body_shape_mismatch_then_chat_success_diagnostics(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None):
        if url.endswith("/responses"):
            return DummyResponse(
                200,
                {
                    "output": [
                        {
                            "content": [
                                {"type": "output_text", "refusal": "no structured text"},
                            ]
                        }
                    ]
                },
            )
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

    backend = create_backend("openai", api_key="sk-test", model="gpt-5.2")

    result = backend.infer_text_json(
        "Merge adjacent fragments",
        schema_name="segment_merge_result",
        schema=_merge_schema(),
    )

    diagnostics = backend.last_text_json_diagnostics

    assert result["merged_ranges"][0] == {"start_seg_id": 0, "end_seg_id": 1}
    assert diagnostics["parsed_endpoint"] == "chat_completions"
    assert diagnostics["final_failure_reason"] is None
    assert diagnostics["responses"]["called"] is True
    assert diagnostics["responses"]["request_succeeded"] is True
    assert diagnostics["responses"]["http_status_code"] == 200
    assert diagnostics["responses"]["json_received"] is True
    assert diagnostics["responses"]["top_level_keys"] == ["output"]
    assert diagnostics["responses"]["structured_payload_found"] is False
    assert diagnostics["responses"]["failure_reason"] == "body_shape_mismatch"
    assert diagnostics["chat_completions"]["called"] is True
    assert diagnostics["chat_completions"]["http_status_code"] == 200
    assert diagnostics["chat_completions"]["json_received"] is True
    assert diagnostics["chat_completions"]["top_level_keys"] == ["choices"]
    assert diagnostics["chat_completions"]["structured_payload_found"] is True


def test_openai_backend_records_parse_failure_and_http_error_diagnostics(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None):
        if url.endswith("/responses"):
            return DummyResponse(
                200,
                {
                    "output_text": "not-json",
                    "output": [],
                },
            )
        if url.endswith("/chat/completions"):
            return DummyResponse(503, {"error": "server error"})
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("video2tasks.vlm.openai_api.requests.post", fake_post)

    backend = create_backend("openai", api_key="sk-test", model="gpt-5.2")

    result = backend.infer_text_json(
        "Merge adjacent fragments",
        schema_name="segment_merge_result",
        schema=_merge_schema(),
    )

    diagnostics = backend.last_text_json_diagnostics

    assert result == {}
    assert diagnostics["parsed_endpoint"] is None
    assert diagnostics["final_failure_reason"] == "chat_completions_http_error"
    assert diagnostics["responses"]["failure_reason"] == "parse_failure"
    assert diagnostics["chat_completions"]["failure_reason"] == "http_error"
    assert diagnostics["chat_completions"]["http_status_code"] == 503
    assert diagnostics["chat_completions"]["json_received"] is True


def test_openai_backend_records_request_exception_before_chat_success(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None):
        if url.endswith("/responses"):
            raise requests.ConnectionError("responses down")
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

    backend = create_backend("openai", api_key="sk-test", model="gpt-5.2")

    result = backend.infer_text_json(
        "Merge adjacent fragments",
        schema_name="segment_merge_result",
        schema=_merge_schema(),
    )

    diagnostics = backend.last_text_json_diagnostics

    assert result["merged_ranges"][1] == {"start_seg_id": 2, "end_seg_id": 2}
    assert diagnostics["parsed_endpoint"] == "chat_completions"
    assert diagnostics["responses"]["request_succeeded"] is False
    assert diagnostics["responses"]["http_status_code"] is None
    assert diagnostics["responses"]["json_received"] is False
    assert diagnostics["responses"]["failure_reason"] == "request_exception"
    assert diagnostics["responses"]["exception_type"] == "ConnectionError"
    assert diagnostics["chat_completions"]["structured_payload_found"] is True


def test_openai_backend_records_content_missing_when_usage_exists_without_text(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None, stream=False):
        if url.endswith("/responses"):
            return DummyResponse(
                200,
                {
                    "output": [],
                    "usage": {"output_tokens": 26, "total_tokens": 73},
                },
            )
        if url.endswith("/chat/completions") and stream is True:
            return DummyStreamResponse(200, ["data: [DONE]"])
        if url.endswith("/chat/completions"):
            return DummyResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {"role": "assistant"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 26, "total_tokens": 73},
                },
            )
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("video2tasks.vlm.openai_api.requests.post", fake_post)

    backend = create_backend("openai", api_key="sk-test", model="gpt-5.2")

    result = backend.infer_text_json(
        "Merge adjacent fragments",
        schema_name="segment_merge_result",
        schema=_merge_schema(),
    )

    diagnostics = backend.last_text_json_diagnostics

    assert result == {}
    assert diagnostics["final_failure_reason"] == "chat_completions_stream_content_missing"
    assert diagnostics["responses"]["failure_reason"] == "content_missing"
    assert diagnostics["responses"]["usage_output_tokens"] == 26
    assert diagnostics["chat_completions"]["failure_reason"] == "content_missing"
    assert diagnostics["chat_completions"]["usage_output_tokens"] == 26
    assert diagnostics["chat_completions"]["finish_reasons"] == ["stop"]
    assert diagnostics["chat_completions_stream"]["called"] is True
    assert diagnostics["chat_completions_stream"]["failure_reason"] == "content_missing"


def test_openai_backend_stream_fallback_recovers_when_non_stream_body_is_missing(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None, stream=False):
        if url.endswith("/responses"):
            return DummyResponse(
                200,
                {
                    "output": [],
                    "usage": {"output_tokens": 26, "total_tokens": 73},
                },
            )
        if url.endswith("/chat/completions") and stream is True:
            return DummyStreamResponse(
                200,
                [
                    "data: " + json_module.dumps({"choices": [{"delta": {"content": '{"thought":"ok","merged_ranges":['}, "finish_reason": None}]}),
                    "data: " + json_module.dumps({"choices": [{"delta": {"content": '{"start_seg_id":0,"end_seg_id":1},{"start_seg_id":2,"end_seg_id":2}]'}, "finish_reason": None}]}),
                    "data: " + json_module.dumps({"choices": [{"delta": {"content": "}"}, "finish_reason": "stop"}]}),
                    "data: [DONE]",
                ],
            )
        if url.endswith("/chat/completions"):
            return DummyResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {"role": "assistant"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 26, "total_tokens": 73},
                },
            )
        raise AssertionError(f"unexpected url: {url}")

    json_module = json
    monkeypatch.setattr("video2tasks.vlm.openai_api.requests.post", fake_post)

    backend = create_backend("openai", api_key="sk-test", model="gpt-5.2")

    result = backend.infer_text_json(
        "Merge adjacent fragments",
        schema_name="segment_merge_result",
        schema=_merge_schema(),
    )

    diagnostics = backend.last_text_json_diagnostics

    assert result == {
        "thought": "ok",
        "merged_ranges": [
            {"start_seg_id": 0, "end_seg_id": 1},
            {"start_seg_id": 2, "end_seg_id": 2},
        ],
    }
    assert diagnostics["parsed_endpoint"] == "chat_completions_stream"
    assert diagnostics["final_failure_reason"] is None
    assert diagnostics["chat_completions"]["failure_reason"] == "content_missing"
    assert diagnostics["chat_completions_stream"]["called"] is True
    assert diagnostics["chat_completions_stream"]["structured_payload_found"] is True
    assert diagnostics["chat_completions_stream"]["chunk_count"] == 3
    assert diagnostics["chat_completions_stream"]["finish_reasons"] == ["stop"]
