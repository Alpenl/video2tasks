from typing import Any, Dict, List
import base64
import json
import os

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


def _extract_openai_compatible_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    for choice in data.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parsed = _extract_json(content)
            if parsed:
                return parsed
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parsed = _extract_json(text)
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
        self.base_url = base_url.rstrip("/")
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

    def _infer_native(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = [{"text": prompt}]
        for image in images:
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
        }

        response = requests.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            json=payload,
            headers=headers,
            timeout=self.timeout_sec,
        )
        if response.status_code != 200:
            print(f"[Gemini] Error: status={response.status_code}")
            return {}

        try:
            data = response.json()
        except json.JSONDecodeError:
            return {}

        return _normalize_vlm_json(_extract_response_payload(data))

    def _infer_openai_compatible(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
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
                        "instructions must be an array of strings."
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
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_sec,
        )
        if response.status_code != 200:
            print(f"[Gemini] Error: status={response.status_code}")
            return {}

        try:
            data = response.json()
        except json.JSONDecodeError:
            return {}

        return _normalize_vlm_json(_extract_openai_compatible_payload(data))
