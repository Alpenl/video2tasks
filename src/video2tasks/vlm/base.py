"""VLM Backend interface and structured output validation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class VLMBackend(ABC):
    """Abstract base class for VLM backends."""

    @abstractmethod
    def infer(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        """
        Run inference on a list of images.

        Args:
            images: List of images as numpy arrays (BGR format)
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
