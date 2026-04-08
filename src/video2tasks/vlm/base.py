"""VLM Backend interface and structured output validation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

VLMRawImagePayload = Dict[str, Any]
VLMInferImage = Union[np.ndarray, VLMRawImagePayload]


@dataclass
class LoadedTransportImage:
    raw_bytes: bytes
    mime_type: str
    bgr: Optional[np.ndarray] = None


class VLMBackend(ABC):
    """Abstract base class for VLM backends."""

    @abstractmethod
    def infer(self, images: List[VLMInferImage], prompt: str) -> Dict[str, Any]:
        """
        Run inference on a list of images.

        Args:
            images: List of decoded arrays or backend-specific raw image payloads
            prompt: The prompt text

        Returns:
            Dictionary with keys:
                - thought: str, reasoning process
                - transitions: List[int], frame indices where task switches occur
                - instructions: List[str], task descriptions for each segment
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""
        pass

    def warmup(self) -> None:
        """Optional warmup routine. Called before main loop."""
        pass

    def cleanup(self) -> None:
        """Optional cleanup routine. Called on shutdown."""
        pass

    def uses_raw_transport_images(self) -> bool:
        """Whether this backend consumes raw bytes + mime metadata directly."""
        return backend_uses_raw_transport_images(self)

    def prepare_images(self, image_records: List[LoadedTransportImage]) -> List[VLMInferImage]:
        """Convert transport-loaded images into the backend's inference input shape."""
        return prepare_backend_images(self, image_records)


def _backend_overrides_method(backend: Any, method_name: str, base_method: Any) -> bool:
    backend_method = getattr(type(backend), method_name, None)
    return callable(backend_method) and backend_method is not base_method


def backend_uses_raw_transport_images(backend: Any) -> bool:
    if _backend_overrides_method(backend, "uses_raw_transport_images", VLMBackend.uses_raw_transport_images):
        return bool(backend.uses_raw_transport_images())
    return str(getattr(backend, "name", "")).strip() == "gemini"


def prepare_backend_images(backend: Any, image_records: List[LoadedTransportImage]) -> List[VLMInferImage]:
    if _backend_overrides_method(backend, "prepare_images", VLMBackend.prepare_images):
        return backend.prepare_images(image_records)

    if backend_uses_raw_transport_images(backend):
        return [
            {
                "raw_bytes": record.raw_bytes,
                "mime_type": record.mime_type,
            }
            for record in image_records
        ]

    prepared = [record.bgr for record in image_records if record.bgr is not None]
    if len(prepared) != len(image_records):
        raise ValueError("decoded image arrays are required for this backend")
    return prepared


def normalize_task_window_result(
    data: Any,
    *,
    max_transition_index: Optional[int] = None,
    allowed_transition_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Validate and normalize the structured window result schema.

    The pipeline depends on this schema being semantically usable, not merely JSON-shaped.
    """
    if not isinstance(data, dict):
        return {}

    raw_instructions = data.get("instructions")
    if isinstance(raw_instructions, str):
        raw_instructions = [raw_instructions]
    if not isinstance(raw_instructions, list):
        return {}

    instructions: List[str] = []
    for item in raw_instructions:
        if not isinstance(item, str):
            return {}
        text = item.strip()
        if not text:
            return {}
        instructions.append(text)

    if not instructions:
        return {}

    raw_transitions = data.get("transitions", [])
    if raw_transitions is None:
        raw_transitions = []
    if not isinstance(raw_transitions, list):
        return {}

    allowed_transition_set = None
    if allowed_transition_indices is not None:
        allowed_transition_set = {int(item) for item in allowed_transition_indices}

    transitions: List[int] = []
    previous = -1
    for item in raw_transitions:
        try:
            transition = int(item)
        except (TypeError, ValueError):
            return {}
        if transition <= 0:
            return {}
        if max_transition_index is not None and transition > max_transition_index:
            return {}
        if allowed_transition_set is not None and transition not in allowed_transition_set:
            return {}
        if transition <= previous:
            return {}
        transitions.append(transition)
        previous = transition

    if len(instructions) != len(transitions) + 1:
        return {}

    thought = data.get("thought", "")
    if thought is None:
        thought = ""
    elif not isinstance(thought, str):
        thought = str(thought)

    return {
        "thought": thought,
        "transitions": transitions,
        "instructions": instructions,
    }
