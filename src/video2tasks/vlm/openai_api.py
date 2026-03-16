from typing import Any, Dict, List
import base64
import json
import os

import cv2
import numpy as np
import requests

from .base import VLMBackend


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

    output_parsed = data.get("output_parsed")
    if isinstance(output_parsed, dict):
        return output_parsed

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

    for candidate in text_candidates:
        parsed = _extract_json(candidate)
        if parsed:
            return parsed

    return {}


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

    @property
    def name(self) -> str:
        return "openai"

    def infer(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for image in images:
            image_url = _encode_jpeg_data_url(image, self.jpeg_quality)
            if image_url:
                content.append({"type": "input_image", "image_url": image_url})

        payload = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "task_window_result",
                    "strict": True,
                    "schema": {
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
                }
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project

        response = requests.post(
            f"{self.base_url}/responses",
            json=payload,
            headers=headers,
            timeout=self.timeout_sec,
        )
        if response.status_code != 200:
            print(f"[OpenAI] Error: status={response.status_code}")
            return {}

        try:
            data = response.json()
        except json.JSONDecodeError:
            return {}

        return _normalize_vlm_json(_extract_response_payload(data))
