from typing import Any, Dict, List, Optional, Tuple
import base64
import json
import os
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from .base import VLMBackend, normalize_task_window_result


def _encode_jpeg_data_url(img_bgr: np.ndarray, quality: int = 85) -> str:
    if img_bgr is None:
        return ""

    ok, buf = cv2.imencode(
        ".jpg",
        img_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))],
    )
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _extract_json_candidate(text: str) -> Tuple[Optional[Any], bool]:
    if not text:
        return None, False

    clean = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean), True
    except json.JSONDecodeError:
        pass

    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(clean[start : end + 1]), True
        except json.JSONDecodeError:
            pass

    return None, False


def _extract_json(text: str) -> Dict[str, Any]:
    parsed, parse_succeeded = _extract_json_candidate(text)
    if parse_succeeded and isinstance(parsed, dict):
        return parsed
    return {}


def _body_shape_summary(data: Any) -> str:
    if data is None:
        return "null"
    if isinstance(data, dict):
        return "dict"
    if isinstance(data, list):
        return "list"
    if isinstance(data, str):
        return "str"
    if isinstance(data, bool):
        return "bool"
    if isinstance(data, (int, float)):
        return "number"
    return type(data).__name__


def _top_level_keys(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    return sorted(str(key) for key in data.keys())[:12]


def _usage_output_tokens(data: Any) -> Optional[int]:
    if not isinstance(data, dict):
        return None

    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None

    for key in ("output_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _chat_finish_reasons(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []

    finish_reasons: List[str] = []
    for choice in data.get("choices", []):
        if not isinstance(choice, dict):
            continue
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            finish_reasons.append(finish_reason)
    return finish_reasons[:8]


def _new_endpoint_diagnostics(*, called: bool) -> Dict[str, Any]:
    return {
        "called": bool(called),
        "request_succeeded": False,
        "http_status_code": None,
        "json_received": False,
        "body_shape": "unknown" if called else "not_called",
        "top_level_keys": [],
        "structured_payload_found": False,
        "failure_reason": None,
        "usage_output_tokens": None,
        "finish_reasons": [],
        "chunk_count": 0,
        "content_chars": 0,
    }


def _extract_response_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    payload, _ = _extract_response_payload_with_reason(data)
    return payload


def _extract_response_payload_with_reason(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    if not isinstance(data, dict):
        return {}, "body_shape_mismatch"

    output_parsed = data.get("output_parsed")
    if isinstance(output_parsed, dict):
        if output_parsed:
            return output_parsed, None
        return {}, "structured_payload_empty"

    text_candidates: List[str] = []
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        text_candidates.append(output_text)

    text_value = data.get("text")
    if isinstance(text_value, str):
        text_candidates.append(text_value)

    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                text_candidates.append(text)

    saw_text_candidate = False
    for candidate in text_candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        saw_text_candidate = True
        parsed, parse_succeeded = _extract_json_candidate(candidate)
        if not parse_succeeded:
            continue
        if isinstance(parsed, dict):
            if parsed:
                return parsed, None
            return {}, "structured_payload_empty"
        return {}, "body_shape_mismatch"

    if saw_text_candidate:
        return {}, "parse_failure"
    if (_usage_output_tokens(data) or 0) > 0:
        return {}, "content_missing"
    return {}, "body_shape_mismatch"


def _extract_chat_completions_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    payload, _ = _extract_chat_completions_payload_with_reason(data)
    return payload


def _extract_chat_completions_payload_with_reason(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    if not isinstance(data, dict):
        return {}, "body_shape_mismatch"

    saw_text_candidate = False
    for choice in data.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue

        parsed = message.get("parsed")
        if isinstance(parsed, dict):
            if parsed:
                return parsed, None
            return {}, "structured_payload_empty"

        text_candidates: List[str] = []
        content = message.get("content")
        if isinstance(content, str):
            text_candidates.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    text_candidates.append(text)

        for candidate in text_candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            saw_text_candidate = True
            parsed_candidate, parse_succeeded = _extract_json_candidate(candidate)
            if not parse_succeeded:
                continue
            if isinstance(parsed_candidate, dict):
                if parsed_candidate:
                    return parsed_candidate, None
                return {}, "structured_payload_empty"
            return {}, "body_shape_mismatch"

    if saw_text_candidate:
        return {}, "parse_failure"
    if (_usage_output_tokens(data) or 0) > 0:
        return {}, "content_missing"
    return {}, "body_shape_mismatch"


class OpenAIBackend(VLMBackend):
    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-5.2",
        base_url: str = "https://api.openai.com/v1",
        timeout_sec: float = 60.0,
        organization: str = "",
        project: str = "",
        reasoning_effort: str = "low",
        max_output_tokens: int = 512,
        jpeg_quality: int = 85,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set worker.openai.api_key or OPENAI_API_KEY.")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.organization = organization
        self.project = project
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = int(max_output_tokens)
        self.jpeg_quality = int(jpeg_quality)
        self.last_text_json_diagnostics: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "openai"

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project
        return headers

    def _prefer_stream_first_for_text_json(self) -> bool:
        parsed = urlparse(self.base_url)
        host = (parsed.netloc or "").lower()
        return host not in {"api.openai.com"}

    def _post_chat_completions_stream_json(
        self,
        content: List[Dict[str, Any]],
        *,
        max_output_tokens: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        diagnostics = _new_endpoint_diagnostics(called=True)
        diagnostics["body_shape"] = "stream"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(max_output_tokens or self.max_output_tokens),
            "stream": True,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._build_headers(),
                timeout=self.timeout_sec,
                stream=True,
            )
        except requests.RequestException as exc:
            diagnostics["failure_reason"] = "request_exception"
            diagnostics["exception_type"] = type(exc).__name__
            diagnostics["error"] = str(exc).strip() or type(exc).__name__
            return {}, diagnostics

        diagnostics["request_succeeded"] = True
        diagnostics["http_status_code"] = int(response.status_code)
        if response.status_code != 200:
            diagnostics["failure_reason"] = "http_error"
            return {}, diagnostics

        content_parts: List[str] = []
        json_error: Optional[str] = None
        finish_reasons: List[str] = []

        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue

            line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else str(raw_line)
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError as exc:
                json_error = str(exc).strip() or type(exc).__name__
                continue

            diagnostics["json_received"] = True
            diagnostics["top_level_keys"] = _top_level_keys(event)

            for choice in event.get("choices", []):
                if not isinstance(choice, dict):
                    continue

                finish_reason = choice.get("finish_reason")
                if isinstance(finish_reason, str) and finish_reason:
                    finish_reasons.append(finish_reason)

                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue

                delta_content = delta.get("content")
                if isinstance(delta_content, str) and delta_content:
                    content_parts.append(delta_content)
                    diagnostics["chunk_count"] += 1
                    diagnostics["content_chars"] += len(delta_content)
                    continue

                if isinstance(delta_content, list):
                    for item in delta_content:
                        if not isinstance(item, dict):
                            continue
                        text = item.get("text")
                        if isinstance(text, str) and text:
                            content_parts.append(text)
                            diagnostics["chunk_count"] += 1
                            diagnostics["content_chars"] += len(text)

        diagnostics["finish_reasons"] = finish_reasons[:8]

        stream_text = "".join(content_parts)
        if stream_text.strip():
            parsed, parse_succeeded = _extract_json_candidate(stream_text)
            if parse_succeeded and isinstance(parsed, dict):
                if parsed:
                    diagnostics["structured_payload_found"] = True
                    diagnostics["failure_reason"] = None
                    return parsed, diagnostics
                diagnostics["failure_reason"] = "structured_payload_empty"
                return {}, diagnostics
            if parse_succeeded:
                diagnostics["failure_reason"] = "body_shape_mismatch"
                return {}, diagnostics
            diagnostics["failure_reason"] = "parse_failure"
            return {}, diagnostics

        if json_error:
            diagnostics["failure_reason"] = "json_decode_error"
            diagnostics["error"] = json_error
            diagnostics["body_shape"] = "invalid_json"
            return {}, diagnostics

        diagnostics["failure_reason"] = "content_missing"
        return {}, diagnostics

    def _post_structured_json(
        self,
        content: List[Dict[str, Any]],
        *,
        schema_name: str,
        schema: Dict[str, Any],
        max_output_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        raise_on_http_error: bool = False,
        prefer_stream_first: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": reasoning_effort or self.reasoning_effort},
            "max_output_tokens": int(max_output_tokens or self.max_output_tokens),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }

        diagnostics: Dict[str, Any] = {
            "responses": _new_endpoint_diagnostics(called=True),
            "chat_completions": _new_endpoint_diagnostics(called=False),
            "chat_completions_stream": _new_endpoint_diagnostics(called=False),
            "parsed_endpoint": None,
            "final_failure_reason": None,
        }

        def finalize(payload_result: Dict[str, Any], *, final_failure_reason: Optional[str] = None) -> Dict[str, Any]:
            diagnostics["final_failure_reason"] = final_failure_reason
            self.last_text_json_diagnostics = diagnostics
            return payload_result

        def build_http_error_message() -> str:
            responses_diag = diagnostics["responses"]
            chat_diag = diagnostics["chat_completions"]
            stream_diag = diagnostics["chat_completions_stream"]
            return (
                "openai_text_json_http_error:"
                f"responses_status={responses_diag.get('http_status_code')};"
                f"responses_error={responses_diag.get('error', '')};"
                f"chat_status={chat_diag.get('http_status_code')};"
                f"chat_error={chat_diag.get('error', '')};"
                f"chat_stream_status={stream_diag.get('http_status_code')};"
                f"chat_stream_error={stream_diag.get('error', '')};"
                f"final_failure_reason={diagnostics.get('final_failure_reason') or ''}"
            )

        chat_content: List[Dict[str, Any]] = []
        for item in content:
            item_type = str(item.get("type", ""))
            if item_type == "input_text":
                chat_content.append({"type": "text", "text": str(item.get("text", ""))})
            elif item_type == "input_image":
                image_url = str(item.get("image_url", ""))
                if image_url:
                    chat_content.append({"type": "image_url", "image_url": {"url": image_url}})

        if prefer_stream_first:
            stream_parsed, stream_diagnostics = self._post_chat_completions_stream_json(
                chat_content,
                max_output_tokens=max_output_tokens,
            )
            diagnostics["chat_completions_stream"] = stream_diagnostics
            if stream_parsed:
                diagnostics["parsed_endpoint"] = "chat_completions_stream"
                return finalize(stream_parsed)

        try:
            response = requests.post(
                f"{self.base_url}/responses",
                json=payload,
                headers=self._build_headers(),
                timeout=self.timeout_sec,
            )
        except requests.RequestException as exc:
            diagnostics["responses"]["failure_reason"] = "request_exception"
            diagnostics["responses"]["exception_type"] = type(exc).__name__
            diagnostics["responses"]["error"] = str(exc).strip() or type(exc).__name__
            print(f"[OpenAI] /responses request failed: {type(exc).__name__}")
            response = None

        if response is not None:
            diagnostics["responses"]["request_succeeded"] = True
            diagnostics["responses"]["http_status_code"] = int(response.status_code)
            try:
                data = response.json()
                diagnostics["responses"]["json_received"] = True
                diagnostics["responses"]["body_shape"] = _body_shape_summary(data)
                diagnostics["responses"]["top_level_keys"] = _top_level_keys(data)
                diagnostics["responses"]["usage_output_tokens"] = _usage_output_tokens(data)
            except json.JSONDecodeError as exc:
                data = None
                diagnostics["responses"]["failure_reason"] = "json_decode_error"
                diagnostics["responses"]["body_shape"] = "invalid_json"
                diagnostics["responses"]["error"] = str(exc).strip() or type(exc).__name__

            if response.status_code == 200 and diagnostics["responses"]["json_received"]:
                parsed, failure_reason = _extract_response_payload_with_reason(data)
                if parsed:
                    diagnostics["responses"]["structured_payload_found"] = True
                    diagnostics["responses"]["failure_reason"] = None
                    diagnostics["parsed_endpoint"] = "responses"
                    return finalize(parsed)
                diagnostics["responses"]["failure_reason"] = failure_reason
                print("[OpenAI] /responses returned empty structured payload; trying /chat/completions fallback")
            elif response.status_code != 200:
                diagnostics["responses"]["failure_reason"] = "http_error"
                print(f"[OpenAI] Error: status={response.status_code}")

        fallback_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": chat_content}],
            "max_tokens": int(max_output_tokens or self.max_output_tokens),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }

        diagnostics["chat_completions"]["called"] = True
        diagnostics["chat_completions"]["body_shape"] = "unknown"
        try:
            fallback_response = requests.post(
                f"{self.base_url}/chat/completions",
                json=fallback_payload,
                headers=self._build_headers(),
                timeout=self.timeout_sec,
            )
        except requests.RequestException as exc:
            diagnostics["chat_completions"]["failure_reason"] = "request_exception"
            diagnostics["chat_completions"]["exception_type"] = type(exc).__name__
            diagnostics["chat_completions"]["error"] = str(exc).strip() or type(exc).__name__
            print(f"[OpenAI] /chat/completions fallback failed: {type(exc).__name__}")
            diagnostics["final_failure_reason"] = "chat_completions_request_exception"
            if raise_on_http_error:
                finalize({}, final_failure_reason="chat_completions_request_exception")
                raise RuntimeError(build_http_error_message())
            return finalize({}, final_failure_reason="chat_completions_request_exception")

        diagnostics["chat_completions"]["request_succeeded"] = True
        diagnostics["chat_completions"]["http_status_code"] = int(fallback_response.status_code)
        try:
            fallback_data = fallback_response.json()
            diagnostics["chat_completions"]["json_received"] = True
            diagnostics["chat_completions"]["body_shape"] = _body_shape_summary(fallback_data)
            diagnostics["chat_completions"]["top_level_keys"] = _top_level_keys(fallback_data)
            diagnostics["chat_completions"]["usage_output_tokens"] = _usage_output_tokens(fallback_data)
            diagnostics["chat_completions"]["finish_reasons"] = _chat_finish_reasons(fallback_data)
        except json.JSONDecodeError as exc:
            fallback_data = None
            diagnostics["chat_completions"]["failure_reason"] = "json_decode_error"
            diagnostics["chat_completions"]["body_shape"] = "invalid_json"
            diagnostics["chat_completions"]["error"] = str(exc).strip() or type(exc).__name__

        if fallback_response.status_code != 200:
            diagnostics["chat_completions"]["failure_reason"] = "http_error"
            diagnostics["final_failure_reason"] = "chat_completions_http_error"
            print(f"[OpenAI] Fallback error: status={fallback_response.status_code}")
            if raise_on_http_error:
                finalize({}, final_failure_reason="chat_completions_http_error")
                raise RuntimeError(build_http_error_message())
            return finalize({}, final_failure_reason="chat_completions_http_error")

        if diagnostics["chat_completions"]["json_received"]:
            parsed, failure_reason = _extract_chat_completions_payload_with_reason(fallback_data)
            if parsed:
                diagnostics["chat_completions"]["structured_payload_found"] = True
                diagnostics["chat_completions"]["failure_reason"] = None
                diagnostics["parsed_endpoint"] = "chat_completions"
                return finalize(parsed)
            diagnostics["chat_completions"]["failure_reason"] = failure_reason

            if failure_reason == "content_missing":
                stream_parsed, stream_diagnostics = self._post_chat_completions_stream_json(
                    chat_content,
                    max_output_tokens=max_output_tokens,
                )
                diagnostics["chat_completions_stream"] = stream_diagnostics
                if stream_parsed:
                    diagnostics["parsed_endpoint"] = "chat_completions_stream"
                    return finalize(stream_parsed)

                stream_failure_reason = str(stream_diagnostics.get("failure_reason") or "empty_response")
                final_failure_reason = f"chat_completions_stream_{stream_failure_reason}"
                if raise_on_http_error and stream_failure_reason in {"http_error", "request_exception"}:
                    finalize({}, final_failure_reason=final_failure_reason)
                    raise RuntimeError(build_http_error_message())
                return finalize({}, final_failure_reason=final_failure_reason)

        final_failure_reason = str(diagnostics["chat_completions"].get("failure_reason") or "empty_response")
        return finalize({}, final_failure_reason=f"chat_completions_{final_failure_reason}")

    def infer_text_json(
        self,
        prompt: str,
        *,
        schema_name: str,
        schema: Dict[str, Any],
        max_output_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        raise_on_http_error: bool = False,
    ) -> Dict[str, Any]:
        return self._post_structured_json(
            [{"type": "input_text", "text": prompt}],
            schema_name=schema_name,
            schema=schema,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            raise_on_http_error=raise_on_http_error,
            prefer_stream_first=self._prefer_stream_first_for_text_json(),
        )

    def infer(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for image in images:
            image_url = _encode_jpeg_data_url(image, self.jpeg_quality)
            if image_url:
                content.append({"type": "input_image", "image_url": image_url})

        payload = self._post_structured_json(
            content,
            schema_name="task_window_result",
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "thought": {"type": "string"},
                    "transitions": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "instructions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["thought", "transitions", "instructions"],
            },
        )
        return normalize_task_window_result(payload)
