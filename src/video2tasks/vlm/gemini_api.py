from typing import Any, Dict, List
import base64
import json
import os
import time
import subprocess
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import requests

from .base import VLMBackend


def _encode_jpeg_b64(img_bgr: np.ndarray, quality: int = 85) -> str:
    if img_bgr is None:
        return ""

    ok, buf = cv2.imencode(
        ".jpg",
        img_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))],
    )
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _encode_jpeg_data_url(img_bgr: np.ndarray, quality: int = 85) -> str:
    image_b64 = _encode_jpeg_b64(img_bgr, quality)
    if not image_b64:
        return ""
    return f"data:image/jpeg;base64,{image_b64}"


def _extract_raw_image_payload(image: Any) -> tuple[str, bytes]:
    if not isinstance(image, dict):
        return "", b""

    raw_bytes = image.get("raw_bytes")
    mime_type = str(image.get("mime_type", "")).strip()
    if not isinstance(raw_bytes, (bytes, bytearray)) or not raw_bytes:
        return "", b""
    if not mime_type.startswith("image/"):
        return "", b""
    return mime_type, bytes(raw_bytes)


def _encode_raw_image_data_url(image: Any) -> str:
    mime_type, raw_bytes = _extract_raw_image_payload(image)
    if not raw_bytes:
        return ""
    return f"data:{mime_type};base64,{base64.b64encode(raw_bytes).decode('utf-8')}"


def _normalize_base_url(base_url: str, api_mode: str) -> str:
    cleaned = (base_url or "").strip().rstrip("/")
    if not cleaned:
        return cleaned

    parts = urlsplit(cleaned)
    path = parts.path.rstrip("/")
    if not path:
        path = "/v1" if api_mode == "openai_compatible" else "/v1beta"

    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment)).rstrip("/")


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    clean = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(clean[start : end + 1])
        except json.JSONDecodeError:
            return {}

    return {}


def _normalize_vlm_json(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict) or "instructions" not in data:
        return {}

    transitions: List[int] = []
    for item in data.get("transitions", []):
        try:
            transitions.append(int(item))
        except (TypeError, ValueError):
            return {}

    instructions = data.get("instructions", [])
    if isinstance(instructions, str):
        instructions = [instructions]
    if not isinstance(instructions, list) or any(not isinstance(v, str) for v in instructions):
        return {}

    thought = data.get("thought", "")
    if thought is None:
        thought = ""
    if not isinstance(thought, str):
        thought = str(thought)

    return {
        "thought": thought,
        "transitions": transitions,
        "instructions": instructions,
    }


def _extract_response_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    text_candidates: List[str] = []
    for candidate in data.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []):
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                text_candidates.append(text)

    for candidate in text_candidates:
        parsed = _extract_json(candidate)
        if parsed:
            return parsed

    return {}


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_sec: float) -> tuple[int, str]:
    retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
    max_attempts = 4
    last_status_code = 0
    last_response_text = ""

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout_sec,
            )
            response_text = getattr(response, "text", None)
            if not isinstance(response_text, str):
                try:
                    response_text = json.dumps(response.json(), ensure_ascii=False)
                except Exception:
                    response_text = ""

            last_status_code = response.status_code
            last_response_text = response_text
            if response.status_code == 200 or response.status_code not in retryable_statuses:
                return response.status_code, response_text

            if attempt < max_attempts:
                delay_s = float(min(2 * attempt, 8))
                print(
                    f"[Gemini] Retryable status={response.status_code}; "
                    f"sleeping {delay_s:.1f}s before retry {attempt + 1}/{max_attempts}"
                )
                time.sleep(delay_s)
                continue
            return response.status_code, response_text
        except requests.RequestException as exc:
            if attempt < max_attempts:
                delay_s = float(min(2 * attempt, 8))
                print(
                    f"[Gemini] requests failed (attempt {attempt}/{max_attempts}), "
                    f"retrying in {delay_s:.1f}s: {exc}"
                )
                time.sleep(delay_s)
                continue
            print(f"[Gemini] requests failed, falling back to curl: {exc}")
            return _post_json_via_curl(url, headers, payload, timeout_sec)

    return last_status_code, last_response_text


def _post_json_via_curl(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_sec: float) -> tuple[int, str]:
    cmd = [
        "curl",
        "-sS",
        "--http1.1",
        "-X",
        "POST",
        url,
        "--connect-timeout",
        str(min(float(timeout_sec), 10.0)),
        "--max-time",
        str(float(timeout_sec)),
        "--data-binary",
        "@-",
        "-w",
        "\n%{http_code}",
    ]
    for key, value in headers.items():
        cmd.extend(["-H", f"{key}: {value}"])

    proc = subprocess.run(
        cmd,
        input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        if stderr:
            print(f"[Gemini] curl fallback failed: {stderr}")
        return 0, ""

    stdout = proc.stdout.decode("utf-8", errors="replace")
    if "\n" not in stdout:
        return 0, stdout
    body, status_text = stdout.rsplit("\n", 1)
    try:
        status_code = int(status_text.strip())
    except ValueError:
        return 0, stdout
    return status_code, body


def _parse_structured_response_text(response_text: str, extractor) -> Dict[str, Any]:
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        data = {}
    return _normalize_vlm_json(extractor(data))


def _collect_openai_text_candidates(value: Any, candidates: List[str]) -> None:
    if isinstance(value, str):
        candidates.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _collect_openai_text_candidates(item, candidates)
        return
    if not isinstance(value, dict):
        return

    text_value = value.get("text")
    if isinstance(text_value, dict):
        _collect_openai_text_candidates(text_value.get("value"), candidates)
    else:
        _collect_openai_text_candidates(text_value, candidates)

    for key in ("content", "reasoning_content", "arguments"):
        _collect_openai_text_candidates(value.get(key), candidates)


def _extract_openai_compatible_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    for choice in data.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        if not isinstance(message, dict):
            continue

        text_candidates: List[str] = []
        _collect_openai_text_candidates(message, text_candidates)

        for candidate in text_candidates:
            parsed = _extract_json(candidate)
            if parsed:
                return parsed

    return {}


class GeminiBackend(VLMBackend):
    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-3-flash-preview",
        api_mode: str = "native",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_sec: float = 60.0,
        max_output_tokens: int = 512,
        jpeg_quality: int = 85,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set worker.gemini.api_key or GEMINI_API_KEY.")

        self.model = model
        self.api_mode = api_mode
        self.base_url = _normalize_base_url(base_url, api_mode)
        self.timeout_sec = float(timeout_sec)
        self.max_output_tokens = int(max_output_tokens)
        self.jpeg_quality = int(jpeg_quality)

    @property
    def name(self) -> str:
        return "gemini"

    def infer(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        if self.api_mode == "openai_compatible":
            return self._infer_openai_compatible(images, prompt)
        return self._infer_native(images, prompt)


    def _request_with_payload_retries(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        extractor,
    ) -> Dict[str, Any]:
        max_payload_attempts = 5
        for attempt in range(1, max_payload_attempts + 1):
            status_code, response_text = _post_json(url, headers, payload, self.timeout_sec)
            if status_code != 200:
                print(f"[Gemini] Error: status={status_code}")
                return {}

            parsed = _parse_structured_response_text(response_text, extractor)
            if parsed:
                return parsed

            if attempt < max_payload_attempts:
                delay_s = float(min(2 * attempt, 8))
                print(
                    f"[Gemini] Empty structured payload; sleeping {delay_s:.1f}s "
                    f"before retry {attempt + 1}/{max_payload_attempts}"
                )
                time.sleep(delay_s)

        max_curl_payload_attempts = 2
        print("[Gemini] Falling back to curl after repeated empty structured payload")
        for attempt in range(1, max_curl_payload_attempts + 1):
            status_code, response_text = _post_json_via_curl(url, headers, payload, self.timeout_sec)
            if status_code != 200:
                print(f"[Gemini] curl fallback error: status={status_code}")
                return {}

            parsed = _parse_structured_response_text(response_text, extractor)
            if parsed:
                return parsed

            if attempt < max_curl_payload_attempts:
                delay_s = float(min(2 * attempt, 8))
                print(
                    f"[Gemini] curl fallback still empty; sleeping {delay_s:.1f}s "
                    f"before retry {attempt + 1}/{max_curl_payload_attempts}"
                )
                time.sleep(delay_s)

        return {}

    def _infer_native(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = [{"text": prompt}]
        for image in images:
            raw_mime_type, raw_bytes = _extract_raw_image_payload(image)
            if raw_bytes:
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": raw_mime_type,
                            "data": base64.b64encode(raw_bytes).decode("utf-8"),
                        }
                    }
                )
                continue

            image_b64 = _encode_jpeg_b64(image, self.jpeg_quality)
            if image_b64:
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": image_b64,
                        }
                    }
                )

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": {
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
                "maxOutputTokens": self.max_output_tokens,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
            "Connection": "close",
        }

        return self._request_with_payload_retries(
            f"{self.base_url}/models/{self.model}:generateContent",
            headers,
            payload,
            _extract_response_payload,
        )

    def _infer_openai_compatible(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            image_url = _encode_raw_image_data_url(image)
            if not image_url:
                image_url = _encode_jpeg_data_url(image, self.jpeg_quality)
            if image_url:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    }
                )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return JSON only with keys thought, transitions, and "
                        "instructions. transitions must be integer frame indexes. "
                        "instructions must be an array of strings. "
                        "The thought field must be one short sentence under 20 words. "
                        "Do not use markdown fences."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "max_tokens": self.max_output_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        }

        return self._request_with_payload_retries(
            f"{self.base_url}/chat/completions",
            headers,
            payload,
            _extract_openai_compatible_payload,
        )
