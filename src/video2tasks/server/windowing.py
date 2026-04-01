"""Video windowing and frame extraction utilities."""

import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import Counter
import numpy as np
import cv2
import base64


@dataclass
class Window:
    """Video window definition."""
    window_id: int
    start_frame: int
    end_frame: int
    frame_ids: List[int]


def read_video_info(mp4_path: str) -> Tuple[float, int]:
    """Read video FPS and frame count."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps is None or fps != fps or abs(fps) < 1e-6:
        fps = 30.0
    
    return float(fps), max(0, nframes)


def build_windows(
    fps: float,
    nframes: int,
    window_sec: float = 16.0,
    step_sec: float = 8.0,
    frames_per_window: int = 16
) -> List[Window]:
    """Build video windows with frame sampling."""
    if fps < 1e-6:
        fps = 30.0
    
    win_len = max(1, int(round(window_sec * fps)))
    step = max(1, int(round(step_sec * fps)))
    windows: List[Window] = []
    
    def get_frames(s: int, e: int, num: int) -> List[int]:
        idx = np.linspace(s, e, num=num).astype(int)
        return np.clip(idx, 0, nframes - 1).tolist()
    
    s = 0
    wid = 0
    while s < nframes:
        e = min(nframes - 1, s + win_len - 1)
        if (e - s < win_len // 2) and wid > 0:
            break
        windows.append(Window(wid, s, e, get_frames(s, e, frames_per_window)))
        wid += 1
        s += step
    
    return windows


def encode_image_720p_png(
    img_bgr: np.ndarray,
    target_w: int = 720,
    target_h: int = 480,
    compression: int = 0
) -> str:
    """Encode image to base64 PNG, resizing if needed."""
    if img_bgr is None:
        return ""
    
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return ""
    
    if (w != target_w) or (h != target_h):
        img_bgr = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    ok, buf = cv2.imencode(
        ".png",
        img_bgr,
        [int(cv2.IMWRITE_PNG_COMPRESSION), int(np.clip(compression, 0, 9))]
    )
    
    return base64.b64encode(buf).decode("utf-8") if ok else ""


class FrameExtractor:
    """Extract frames from video file."""
    
    def __init__(self, mp4_path: str):
        self.cap = cv2.VideoCapture(mp4_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {mp4_path}")
    
    def close(self) -> None:
        """Release video capture."""
        if self.cap.isOpened():
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def get_many_b64(
        self,
        frame_ids: List[int],
        target_w: int = 720,
        target_h: int = 480,
        compression: int = 0
    ) -> List[str]:
        """Extract multiple frames as base64 PNGs."""
        sorted_indices = sorted(list(set(frame_ids)))
        frame_map: dict = {}
        
        for fid in sorted_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, bgr = self.cap.read()
            frame_map[fid] = encode_image_720p_png(
                bgr, target_w, target_h, compression
            ) if (ok and bgr is not None) else ""
        
        return [frame_map.get(fid, "") for fid in frame_ids]


_DESTINATION_SPLIT_RE = re.compile(
    r"\b(?:to|onto|into|over|inside|within|toward|towards|from|in|on)\b"
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "again", "all", "around", "at", "away", "body", "by",
    "central", "different", "for", "it", "its", "left", "of", "off", "out",
    "right", "same", "smallest", "the", "their", "them", "this", "those",
    "through", "to", "up", "using", "with",
}
_STRONG_ACTION_TOKENS = {
    "pick", "place", "stack", "lift", "move", "carry", "separate", "connect",
    "plug", "retrieve", "open", "close", "insert", "remove", "transfer",
    "pour", "fold", "nest",
}
_GENERIC_ACTION_TOKENS = {
    "adjust", "position", "reposition", "manipulate", "handle", "interact",
    "hold", "tilt", "push", "stabilize", "align", "support",
}
_PREP_ACTION_TOKENS = {"prepare", "begin", "start", "reach", "approach", "align", "hover"}
_ACTION_FILLERS = {
    "grasp", "release", "put", "set", "make", "keep", "continue", "moving",
    "placing", "picking", "lifting", "holding",
}
_ROBOT_MOTION_TOKENS = {"gripper", "robot", "arm", "workspace", "area", "work"}
_FILLER_TOKENS = {"wait", "explain", "describe", "narrate", "instruction", "gesture", "gesturing"}
_COOKING_CONTAINER_TOKENS = {"bowl", "pot", "pan", "processor", "salad"}
_PREP_INGREDIENT_ACTION_TOKENS = {
    "chop", "slice", "dice", "mince", "grate", "tear", "peel", "cut",
    "crush", "smash", "slit", "trim", "roll", "flatten",
}
_RECIPE_ACTION_TOKENS = {
    "add", "season", "stir", "mix", "whisk", "pour", "cook", "saute", "grind",
    "toss", "fry", "boil",
    "fill", "fold", "roll",
}
_MIXING_ACTION_TOKENS = {"season", "stir", "mix", "whisk", "toss"}
_ACTION_FAMILY_SUPPORT_THRESHOLD = 1.2


def _singularize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("ls"):
        return token[:-1]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _instruction_tokens(text: str) -> List[str]:
    return [_singularize_token(tok) for tok in _TOKEN_RE.findall(text.lower())]


def _instruction_action_head_tokens(text: str, limit: int = 2) -> set[str]:
    tokens = [
        tok for tok in _instruction_tokens(text.replace("/", " "))
        if tok not in {"a", "an", "the"}
    ]
    return set(tokens[:limit])


def _primary_object_tokens(text: str) -> set[str]:
    head = _DESTINATION_SPLIT_RE.split(text.lower(), maxsplit=1)[0]
    tokens = []
    for token in _instruction_tokens(head.replace("/", " ")):
        if (
            token in _STOPWORDS
            or token in _STRONG_ACTION_TOKENS
            or token in _GENERIC_ACTION_TOKENS
            or token in _PREP_ACTION_TOKENS
            or token in _ACTION_FILLERS
            or token in _ROBOT_MOTION_TOKENS
            or token == "new"
        ):
            continue
        tokens.append(token)
    return set(tokens)


def _instruction_specificity(text: str) -> int:
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    action_tokens = _instruction_action_head_tokens(text)
    if tokens & _FILLER_TOKENS:
        return -3
    strong = len(action_tokens & _STRONG_ACTION_TOKENS)
    mixing = len(action_tokens & _MIXING_ACTION_TOKENS)
    generic = len(action_tokens & _GENERIC_ACTION_TOKENS)
    prep_ingredient = len(action_tokens & _PREP_INGREDIENT_ACTION_TOKENS)
    prep = len(action_tokens & _PREP_ACTION_TOKENS)
    if prep:
        return -2
    return strong * 2 + mixing * 2 - generic - prep_ingredient


def _instruction_is_generic(text: str) -> bool:
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    action_tokens = _instruction_action_head_tokens(text)
    if tokens & _FILLER_TOKENS:
        return True
    if action_tokens & _PREP_ACTION_TOKENS:
        return True
    strong = len(action_tokens & _STRONG_ACTION_TOKENS)
    generic = len(action_tokens & _GENERIC_ACTION_TOKENS)
    return generic >= max(1, strong)


def _instruction_is_prep_like(text: str) -> bool:
    return bool(_instruction_action_head_tokens(text) & _PREP_ACTION_TOKENS)


def _instruction_is_bridge_motion(text: str) -> bool:
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    object_tokens = _primary_object_tokens(text)
    if object_tokens:
        return False
    if tokens & _ROBOT_MOTION_TOKENS:
        return True
    return ("reposition" in tokens or "move" in tokens) and "gripper" in tokens


def _instruction_is_filler_segment(text: str) -> bool:
    lower = text.lower().strip()
    if any(lower.startswith(prefix) for prefix in ("wait", "explain", "describe", "narrate")):
        return True
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    return bool(tokens & _FILLER_TOKENS)


def _instruction_is_preparatory_segment(text: str) -> bool:
    lower = text.lower().strip()
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    completion_tokens = {
        "place", "stack", "insert", "remove", "retrieve", "connect", "plug",
        "fold", "pour", "nest",
    }
    if lower.startswith("reach "):
        return True
    if "align the gripper" in lower or "align gripper" in lower:
        return True
    if "position the gripper" in lower or "position gripper" in lower:
        return True
    if "reposition the gripper" in lower:
        return True
    if "prepare" in tokens and not (tokens & completion_tokens):
        return True
    if (tokens & {"align", "hover", "reach", "approach"}) and not (tokens & completion_tokens):
        return True
    return False


def _token_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(len(left | right))


def _destination_tokens(text: str) -> set[str]:
    parts = _DESTINATION_SPLIT_RE.split(text.lower(), maxsplit=1)
    if len(parts) < 2:
        return set()

    tokens = []
    for token in _instruction_tokens(parts[1].replace("/", " ")):
        if (
            token in _STOPWORDS
            or token in _STRONG_ACTION_TOKENS
            or token in _GENERIC_ACTION_TOKENS
            or token in _PREP_ACTION_TOKENS
            or token in _ACTION_FILLERS
            or token in _ROBOT_MOTION_TOKENS
            or token in _FILLER_TOKENS
            or token == "new"
        ):
            continue
        tokens.append(token)
    return set(tokens)


def _ingredient_tokens(text: str) -> set[str]:
    tokens = _primary_object_tokens(text) - _COOKING_CONTAINER_TOKENS
    tokens -= _PREP_INGREDIENT_ACTION_TOKENS
    tokens -= _RECIPE_ACTION_TOKENS
    tokens -= {"ingredient", "mixture", "content", "topping", "top"}
    if "scallion" in tokens:
        tokens |= {"green", "onion"}
    if "green" in tokens and "onion" in tokens:
        tokens.add("scallion")
    return tokens


def _instruction_has_recipe_action(text: str) -> bool:
    return bool(_instruction_action_head_tokens(text) & _RECIPE_ACTION_TOKENS)


def _instruction_has_prep_ingredient_action(text: str) -> bool:
    return bool(_instruction_action_head_tokens(text) & _PREP_INGREDIENT_ACTION_TOKENS)


def _instruction_has_mixing_action(text: str) -> bool:
    return bool(_instruction_action_head_tokens(text) & _MIXING_ACTION_TOKENS)


def _instruction_mentions_wrapper(text: str) -> bool:
    lower = text.lower()
    if "spring roll" in lower or "egg roll" in lower:
        return True
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    return bool(tokens & {"wrap", "wrapper"})


def _is_wrapper_fill_to_roll_pair(left_instruction: str, right_instruction: str) -> bool:
    left_tokens = set(_instruction_tokens(left_instruction.replace("/", " ")))
    right_tokens = set(_instruction_tokens(right_instruction.replace("/", " ")))
    left_head_tokens = _instruction_action_head_tokens(left_instruction)
    right_head_tokens = _instruction_action_head_tokens(right_instruction)
    left_wrapper_or_dumpling_assembly = (
        _instruction_mentions_wrapper(left_instruction)
        or (
            bool(left_tokens & {"dumpling", "pierogi", "pierogy"})
            and bool(left_tokens & {"assemble", "fill", "fold"})
        )
    )
    return (
        left_wrapper_or_dumpling_assembly
        and _instruction_mentions_wrapper(right_instruction)
        and bool((left_head_tokens & {"place", "fill", "spread", "assemble", "fold"}) or (left_tokens & {"assemble"}))
        and bool((right_head_tokens & {"roll", "fold", "tuck"}) or (right_tokens & {"pleat"}))
    )


def _action_tokens(text: str) -> set[str]:
    tokens = _instruction_action_head_tokens(text)
    return tokens & (
        _RECIPE_ACTION_TOKENS
        | _PREP_INGREDIENT_ACTION_TOKENS
        | _MIXING_ACTION_TOKENS
        | _STRONG_ACTION_TOKENS
    )


def _action_families(text: str) -> set[str]:
    tokens = _action_tokens(text)
    families = set()
    if tokens & {"add", "pour", "fill"}:
        families.add("add")
    if tokens & {"season", "stir", "mix", "whisk", "toss"}:
        families.add("mix")
    if tokens & {"cook", "saute", "fry", "boil"}:
        families.add("cook")
    if tokens & _PREP_INGREDIENT_ACTION_TOKENS:
        families.add("prep")
    if tokens & {"place", "transfer", "remove", "move"}:
        families.add("transfer")
    return families


def _segment_duration_sec(segment: dict, fps: float) -> float:
    if fps < 1e-6:
        fps = 30.0
    return max(0.0, (segment["end_frame"] - segment["start_frame"]) / fps)


def _boundary_support_between(left: dict, right: dict) -> Tuple[float, bool]:
    supports = []
    has_support = False
    for segment, key in ((left, "boundary_support_after"), (right, "boundary_support_before")):
        if key in segment:
            has_support = True
            try:
                supports.append(max(0.0, float(segment.get(key, 0.0))))
            except (TypeError, ValueError):
                supports.append(0.0)
    return (max(supports) if supports else 0.0), has_support


def _choose_instruction(left: dict, right: dict, fps: float) -> str:
    if _is_wrapper_fill_to_roll_pair(left["instruction"], right["instruction"]):
        return right["instruction"]

    shared_ingredient_tokens = _ingredient_tokens(left["instruction"]) & _ingredient_tokens(right["instruction"])
    if (
        shared_ingredient_tokens
        and _instruction_has_prep_ingredient_action(left["instruction"])
        and _instruction_has_prep_ingredient_action(right["instruction"])
        and _action_tokens(left["instruction"]) != _action_tokens(right["instruction"])
    ):
        return right["instruction"]

    left_specificity = _instruction_specificity(left["instruction"])
    right_specificity = _instruction_specificity(right["instruction"])
    if right_specificity > left_specificity:
        return right["instruction"]
    if left_specificity > right_specificity:
        return left["instruction"]

    left_duration = _segment_duration_sec(left, fps)
    right_duration = _segment_duration_sec(right, fps)
    if right_duration >= left_duration * 0.75:
        return right["instruction"]
    return left["instruction"]


def _should_merge_segments(left: dict, right: dict, fps: float, boundary_support_threshold: float = 0.9) -> bool:
    left_tokens = _primary_object_tokens(left["instruction"])
    right_tokens = _primary_object_tokens(right["instruction"])
    similarity = _token_similarity(left_tokens, right_tokens)
    left_dest_tokens = _destination_tokens(left["instruction"])
    right_dest_tokens = _destination_tokens(right["instruction"])
    shared_dest_tokens = left_dest_tokens & right_dest_tokens
    left_ingredient_tokens = _ingredient_tokens(left["instruction"])
    right_ingredient_tokens = _ingredient_tokens(right["instruction"])
    shared_ingredient_tokens = left_ingredient_tokens & right_ingredient_tokens
    shared_action_tokens = _action_tokens(left["instruction"]) & _action_tokens(right["instruction"])
    shared_action_families = _action_families(left["instruction"]) & _action_families(right["instruction"])
    boundary_support, has_boundary_support = _boundary_support_between(left, right)
    strong_boundary = has_boundary_support and boundary_support_threshold > 0.0 and boundary_support >= boundary_support_threshold

    left_recipe = _instruction_has_recipe_action(left["instruction"])
    right_recipe = _instruction_has_recipe_action(right["instruction"])
    left_prep_ingredient = _instruction_has_prep_ingredient_action(left["instruction"])
    right_prep_ingredient = _instruction_has_prep_ingredient_action(right["instruction"])
    left_mixing = _instruction_has_mixing_action(left["instruction"])
    right_mixing = _instruction_has_mixing_action(right["instruction"])
    left_generic = _instruction_is_generic(left["instruction"])
    right_generic = _instruction_is_generic(right["instruction"])
    left_prep = _instruction_is_prep_like(left["instruction"])
    right_prep = _instruction_is_prep_like(right["instruction"])
    left_filler = _instruction_is_filler_segment(left["instruction"])
    right_filler = _instruction_is_filler_segment(right["instruction"])
    same_ingredient_prep_to_transfer = bool(shared_ingredient_tokens) and (
        (left_prep_ingredient and right_recipe) or (right_prep_ingredient and left_recipe)
    )
    same_ingredient_same_action_prep = (
        bool(shared_ingredient_tokens)
        and bool(shared_action_tokens)
        and left_prep_ingredient
        and right_prep_ingredient
    )
    distinct_same_ingredient_prep_steps = (
        bool(shared_ingredient_tokens)
        and left_prep_ingredient
        and right_prep_ingredient
        and not shared_action_tokens
    )
    distinct_prepped_ingredient_steps = (
        bool(shared_dest_tokens & _COOKING_CONTAINER_TOKENS)
        and left_recipe
        and right_recipe
        and not (left_mixing or right_mixing)
        and bool(left_ingredient_tokens)
        and bool(right_ingredient_tokens)
        and not shared_ingredient_tokens
        and (left_prep_ingredient or right_prep_ingredient)
    )
    same_container_same_family_recipe_steps = (
        bool(shared_dest_tokens & _COOKING_CONTAINER_TOKENS)
        and left_recipe
        and right_recipe
        and bool(shared_action_families & {"add", "mix"})
    )
    heated_distinct_add_sequence = (
        bool(shared_dest_tokens & {"pot", "pan"})
        and shared_action_families == {"add"}
        and bool(left_ingredient_tokens)
        and bool(right_ingredient_tokens)
        and not shared_ingredient_tokens
    )
    wrapper_fill_to_roll = _is_wrapper_fill_to_roll_pair(left["instruction"], right["instruction"])

    if similarity < 0.34:
        # Adjacent cooking sub-steps often swap ingredients while staying in the same bowl/pot.
        if not (
            left_prep
            or right_prep
            or left_filler
            or right_filler
            or (
                shared_dest_tokens & _COOKING_CONTAINER_TOKENS
                and (left_recipe or right_recipe)
                and (shared_ingredient_tokens or shared_action_tokens)
            )
            or same_ingredient_prep_to_transfer
            or same_ingredient_same_action_prep
            or (distinct_same_ingredient_prep_steps and has_boundary_support and not strong_boundary)
            or (
                same_container_same_family_recipe_steps
                and not heated_distinct_add_sequence
                and has_boundary_support
                and boundary_support < _ACTION_FAMILY_SUPPORT_THRESHOLD
            )
            or wrapper_fill_to_roll
        ):
            return False

    left_duration = _segment_duration_sec(left, fps)
    right_duration = _segment_duration_sec(right, fps)

    if distinct_prepped_ingredient_steps:
        return False
    if distinct_same_ingredient_prep_steps:
        if not has_boundary_support:
            return False
        if strong_boundary:
            return False
        if min(left_duration, right_duration) <= 8.0:
            return True
        return False

    if left_generic and right_generic and not (left_prep or right_prep or left_filler or right_filler):
        return True
    if left_generic and left_duration <= 4.5 and not (left_prep or left_filler):
        return True
    if right_generic and right_duration <= 4.5 and not (right_prep or right_filler):
        return True
    if wrapper_fill_to_roll:
        return True
    if shared_dest_tokens & _COOKING_CONTAINER_TOKENS and (left_recipe or right_recipe):
        if shared_ingredient_tokens and min(left_duration, right_duration) <= 8.0:
            return True
        if shared_action_tokens and max(left_duration, right_duration) <= 8.0:
            return True
        if (left_mixing or right_mixing) and shared_ingredient_tokens and max(left_duration, right_duration) <= 20.0:
            return True
    if same_ingredient_prep_to_transfer:
        if min(left_duration, right_duration) <= 8.0:
            return True
    if same_ingredient_same_action_prep:
        if min(left_duration, right_duration) <= 8.0:
            return True
    if (
        same_container_same_family_recipe_steps
        and not heated_distinct_add_sequence
        and has_boundary_support
        and boundary_support < _ACTION_FAMILY_SUPPORT_THRESHOLD
    ):
        if min(left_duration, right_duration) <= 16.0:
            return True
    if similarity >= 0.6 and min(left_duration, right_duration) <= 6.5:
        return True
    return False


def merge_task_level_segments(
    segments: List[dict],
    fps: float,
    boundary_support_threshold: float = 0.9,
) -> List[dict]:
    """Merge over-segmented adjacent spans into task-level segments."""
    if not segments:
        return []

    merged: List[dict] = []

    for segment in segments:
        current = dict(segment)
        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        if _should_merge_segments(previous, current, fps, boundary_support_threshold=boundary_support_threshold):
            chosen_instruction = _choose_instruction(previous, current, fps)
            previous["end_frame"] = current["end_frame"]
            previous["instruction"] = chosen_instruction
            previous["confidence"] = max(
                float(previous.get("confidence", 0.0)),
                float(current.get("confidence", 0.0)),
            )
            previous["boundary_support_after"] = current.get(
                "boundary_support_after",
                previous.get("boundary_support_after", 0.0),
            )
        else:
            merged.append(current)

    return cleanup_auxiliary_segments(merged, fps)


def cleanup_auxiliary_segments(segments: List[dict], fps: float) -> List[dict]:
    """Absorb bridge/prep filler spans without semantic task merging."""
    if not segments:
        return []

    bridge_cleaned: List[dict] = []
    pending_bridge: Optional[dict] = None
    for segment in segments:
        current = dict(segment)
        if _instruction_is_bridge_motion(current["instruction"]) or _instruction_is_filler_segment(current["instruction"]):
            if pending_bridge is None:
                pending_bridge = current
            else:
                pending_bridge["end_frame"] = current["end_frame"]
                pending_bridge["confidence"] = max(
                    float(pending_bridge.get("confidence", 0.0)),
                    float(current.get("confidence", 0.0)),
                )
            continue

        if pending_bridge is not None:
            current["start_frame"] = pending_bridge["start_frame"]
            current["confidence"] = max(
                float(current.get("confidence", 0.0)),
                float(pending_bridge.get("confidence", 0.0)),
            )
            pending_bridge = None

        bridge_cleaned.append(current)

    if pending_bridge is not None and bridge_cleaned:
        bridge_cleaned[-1]["end_frame"] = pending_bridge["end_frame"]
        bridge_cleaned[-1]["confidence"] = max(
            float(bridge_cleaned[-1].get("confidence", 0.0)),
            float(pending_bridge.get("confidence", 0.0)),
        )

    prep_cleaned: List[dict] = []
    pending_prep: Optional[dict] = None
    for segment in (bridge_cleaned or merged):
        current = dict(segment)
        if _instruction_is_preparatory_segment(current["instruction"]) and _segment_duration_sec(current, fps) <= 5.0:
            if pending_prep is None:
                pending_prep = current
            else:
                pending_prep["end_frame"] = current["end_frame"]
                pending_prep["confidence"] = max(
                    float(pending_prep.get("confidence", 0.0)),
                    float(current.get("confidence", 0.0)),
                )
            continue

        if pending_prep is not None:
            current["start_frame"] = pending_prep["start_frame"]
            current["confidence"] = max(
                float(current.get("confidence", 0.0)),
                float(pending_prep.get("confidence", 0.0)),
            )
            pending_prep = None

        prep_cleaned.append(current)

    if pending_prep is not None and prep_cleaned:
        if _segment_duration_sec(pending_prep, fps) > 3.0:
            prep_cleaned[-1]["end_frame"] = pending_prep["end_frame"]
            prep_cleaned[-1]["confidence"] = max(
                float(prep_cleaned[-1].get("confidence", 0.0)),
                float(pending_prep.get("confidence", 0.0)),
            )

    final_segments = prep_cleaned or bridge_cleaned or segments

    for idx, segment in enumerate(final_segments):
        segment["seg_id"] = idx

    return final_segments


def _segment_overlap_frames(left: dict, right: dict) -> int:
    return max(
        0,
        min(int(left["end_frame"]), int(right["end_frame"])) - max(int(left["start_frame"]), int(right["start_frame"]))
    )


def _contributors_for_segment(segment: dict, source_segments: List[dict]) -> List[dict]:
    contributors = []
    for candidate in source_segments:
        overlap = _segment_overlap_frames(segment, candidate)
        if overlap > 0:
            enriched = dict(candidate)
            enriched["_overlap_frames"] = overlap
            contributors.append(enriched)
    return contributors


def _is_specific_instruction(text: str) -> bool:
    return not (
        _instruction_is_generic(text)
        or _instruction_is_prep_like(text)
        or _instruction_is_filler_segment(text)
    )


def _merged_segment_looks_overcollapsed(
    segment: dict,
    source_segments: List[dict],
    fps: float,
    median_source_duration_sec: float,
) -> bool:
    contributors = _contributors_for_segment(segment, source_segments)
    if len(contributors) < 3:
        return False

    distinct_specific = {
        item["instruction"].strip().lower()
        for item in contributors
        if _is_specific_instruction(item["instruction"])
    }
    if len(distinct_specific) < 2:
        return False

    duration_sec = _segment_duration_sec(segment, fps)
    return duration_sec >= max(10.0, median_source_duration_sec * 1.8)


def _should_fallback_to_light_cleanup(
    light_segments: List[dict],
    merged_segments: List[dict],
    fps: float,
    min_segments: int,
    collapse_ratio: float,
) -> bool:
    if len(light_segments) < min_segments:
        return False
    if not merged_segments:
        return False
    if len(merged_segments) >= max(1, int(np.ceil(len(light_segments) * collapse_ratio))):
        return False

    source_durations = [_segment_duration_sec(segment, fps) for segment in light_segments]
    median_source_duration_sec = float(np.median(source_durations)) if source_durations else 0.0
    return any(
        _merged_segment_looks_overcollapsed(segment, light_segments, fps, median_source_duration_sec)
        for segment in merged_segments
    )


def _score_instruction_candidate(segment: dict, candidate: dict, current_instruction: str) -> float:
    overlap = float(candidate.get("_overlap_frames", 0))
    text = candidate["instruction"]
    score = overlap
    score += float(_instruction_specificity(text)) * 24.0
    if text == current_instruction:
        score += 18.0
    if _instruction_is_generic(text):
        score -= 12.0
    if _instruction_is_prep_like(text):
        score -= 16.0
    if _instruction_is_filler_segment(text):
        score -= 24.0
    return score


def refine_segment_instructions(final_segments: List[dict], source_segments: List[dict]) -> List[dict]:
    """Refine final labels from contributing pre-merge segment labels."""
    refined: List[dict] = []
    for segment in final_segments:
        current = dict(segment)
        contributors = _contributors_for_segment(current, source_segments)
        if not contributors:
            refined.append(current)
            continue

        best = max(
            contributors,
            key=lambda candidate: _score_instruction_candidate(current, candidate, current["instruction"]),
        )
        current_is_weak = not _is_specific_instruction(current["instruction"])
        best_is_specific = _is_specific_instruction(best["instruction"])
        overlap_ratio = (
            float(best["_overlap_frames"]) / max(1.0, float(current["end_frame"] - current["start_frame"]))
        )

        if best_is_specific and (current_is_weak or overlap_ratio >= 0.45):
            current["instruction"] = best["instruction"]

        refined.append(current)

    for idx, segment in enumerate(refined):
        segment["seg_id"] = idx
    return refined


def _dominant_instruction_from_candidates(candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None

    counts = Counter(candidates)
    return max(
        counts.items(),
        key=lambda item: (
            item[1],
            _instruction_specificity(item[0]),
            len(_ingredient_tokens(item[0])),
            len(_action_families(item[0])),
        ),
    )[0]


def _instruction_phase_signature(text: str) -> tuple:
    action_families = tuple(sorted(_action_families(text)))
    if not action_families:
        action_families = tuple(sorted(_action_tokens(text)))

    dest_tokens = tuple(sorted(_destination_tokens(text) & _COOKING_CONTAINER_TOKENS))
    focus_tokens = list(sorted(_ingredient_tokens(text)))
    if not focus_tokens:
        focus_tokens = list(sorted(_primary_object_tokens(text) - _COOKING_CONTAINER_TOKENS))

    return (
        action_families,
        dest_tokens[:1],
        tuple(focus_tokens[:2]),
    )


def _should_split_on_instruction_drift(left_instruction: str, right_instruction: str) -> bool:
    left_actions = _action_families(left_instruction)
    right_actions = _action_families(right_instruction)

    shared_containers = (
        _destination_tokens(left_instruction)
        & _destination_tokens(right_instruction)
        & _COOKING_CONTAINER_TOKENS
    )
    if not shared_containers:
        return False

    if shared_containers & {"bowl", "processor", "salad"}:
        if left_actions <= {"add", "mix"} and right_actions <= {"add", "mix"}:
            return False

    if left_actions != right_actions and (left_actions or right_actions):
        return True

    left_ingredients = _ingredient_tokens(left_instruction)
    right_ingredients = _ingredient_tokens(right_instruction)
    if (
        left_actions == right_actions == {"add"}
        and left_ingredients
        and right_ingredients
        and not (left_ingredients & right_ingredients)
    ):
        return True

    return False


def _instruction_runs_for_segment(
    segment: dict,
    instruction_timeline: List[List[str]],
    fps: float,
    bin_sec: float = 2.0,
) -> List[dict]:
    bin_frames = max(1, int(round(max(0.5, bin_sec) * max(fps, 1.0))))
    runs: List[dict] = []

    for start in range(int(segment["start_frame"]), int(segment["end_frame"]), bin_frames):
        end = min(int(segment["end_frame"]), start + bin_frames)
        candidates: List[str] = []
        for fid in range(start, end):
            if 0 <= fid < len(instruction_timeline):
                candidates.extend(instruction_timeline[fid])

        instruction = _dominant_instruction_from_candidates(candidates)
        if not instruction:
            continue

        signature = _instruction_phase_signature(instruction)
        if runs and runs[-1]["signature"] == signature:
            runs[-1]["end_frame"] = end
            runs[-1]["candidates"].append(instruction)
            runs[-1]["instruction"] = _dominant_instruction_from_candidates(runs[-1]["candidates"]) or runs[-1]["instruction"]
            continue

        runs.append({
            "start_frame": start,
            "end_frame": end,
            "instruction": instruction,
            "signature": signature,
            "candidates": [instruction],
        })

    return runs


def _split_segment_on_instruction_drift(
    segment: dict,
    instruction_timeline: List[List[str]],
    fps: float,
    min_segment_frames: int,
    min_phase_frames: int,
    depth: int,
) -> List[dict]:
    current = dict(segment)
    if depth <= 0 or (int(current["end_frame"]) - int(current["start_frame"])) < min_segment_frames:
        return [current]

    runs = _instruction_runs_for_segment(current, instruction_timeline, fps)
    if len(runs) < 2:
        return [current]

    candidates = []
    for idx in range(len(runs) - 1):
        left_run = runs[idx]
        right_run = runs[idx + 1]
        left_duration = int(left_run["end_frame"]) - int(left_run["start_frame"])
        right_duration = int(right_run["end_frame"]) - int(right_run["start_frame"])
        if left_duration < min_phase_frames or right_duration < min_phase_frames:
            continue
        if not _should_split_on_instruction_drift(left_run["instruction"], right_run["instruction"]):
            continue

        boundary = int((int(left_run["end_frame"]) + int(right_run["start_frame"])) / 2)
        if boundary - int(current["start_frame"]) < min_phase_frames:
            continue
        if int(current["end_frame"]) - boundary < min_phase_frames:
            continue

        score = float(min(left_duration, right_duration))
        if _action_families(left_run["instruction"]) != _action_families(right_run["instruction"]):
            score += float(min_phase_frames)
        if _is_specific_instruction(left_run["instruction"]):
            score += 0.25 * min_phase_frames
        if _is_specific_instruction(right_run["instruction"]):
            score += 0.25 * min_phase_frames

        candidates.append((score, boundary, left_run, right_run))

    if not candidates:
        return [current]

    _, boundary, left_run, right_run = max(candidates, key=lambda item: item[0])
    left_segment = dict(current)
    left_segment["end_frame"] = boundary
    left_segment["instruction"] = left_run["instruction"]
    left_segment["boundary_support_after"] = 0.0

    right_segment = dict(current)
    right_segment["start_frame"] = boundary
    right_segment["instruction"] = right_run["instruction"]
    right_segment["boundary_support_before"] = 0.0

    return (
        _split_segment_on_instruction_drift(
            left_segment,
            instruction_timeline,
            fps,
            min_segment_frames,
            min_phase_frames,
            depth - 1,
        )
        + _split_segment_on_instruction_drift(
            right_segment,
            instruction_timeline,
            fps,
            min_segment_frames,
            min_phase_frames,
            depth - 1,
        )
    )


def split_long_raw_segments_on_instruction_drift(
    raw_segments: List[dict],
    instruction_timeline: List[List[str]],
    fps: float,
    min_segment_sec: float = 30.0,
    min_phase_sec: float = 5.0,
) -> List[dict]:
    if not raw_segments:
        return []

    if fps < 1e-6:
        fps = 30.0

    min_segment_frames = max(1, int(round(min_segment_sec * fps)))
    min_phase_frames = max(1, int(round(min_phase_sec * fps)))
    split_segments: List[dict] = []

    for segment in raw_segments:
        split_segments.extend(
            _split_segment_on_instruction_drift(
                segment,
                instruction_timeline,
                fps,
                min_segment_frames,
                min_phase_frames,
                depth=3,
            )
        )

    for idx, segment in enumerate(split_segments):
        segment["seg_id"] = idx

    return split_segments


def build_segments_via_cuts(
    sample_id: str,
    windows: List[Window],
    by_wid: dict,
    fps: float,
    nframes: int,
    frames_per_window: int = 16,
    adaptive_merge_guard: bool = True,
    adaptive_merge_min_segments: int = 8,
    adaptive_merge_collapse_ratio: float = 0.6,
    boundary_support_threshold: float = 0.9,
    refine_final_instructions: bool = True,
) -> dict:
    """Build final segments from window results."""
    if nframes == 0:
        return {}
    
    if fps < 1e-6:
        fps = 30.0
    
    raw_cuts = []
    instruction_timeline = [[] for _ in range(nframes)]
    center_weights = np.hanning(frames_per_window + 2)[1:-1]
    
    for wid, w in enumerate(windows):
        rec = by_wid.get(wid)
        if not rec:
            continue
        
        vlm = rec.get("vlm_json", {})
        transitions = vlm.get("transitions", [])
        instructions = vlm.get("instructions", [])
        f_ids = w.frame_ids
        cur_len = len(f_ids)
        
        if cur_len == 0:
            continue
        
        # Collect cut points
        for t_idx in transitions:
            try:
                idx = int(t_idx)
                if 0 <= idx < cur_len:
                    global_fid = f_ids[idx]
                    if cur_len == frames_per_window:
                        w_val = center_weights[idx]
                    else:
                        w_val = 1.0 if min(idx, cur_len - 1 - idx) > 2 else 0.5
                    raw_cuts.append((global_fid, float(w_val)))
            except (ValueError, IndexError):
                pass
        
        # Collect instructions
        try:
            boundaries = [0] + [int(t) for t in transitions if 0 <= int(t) < cur_len] + [cur_len]
            boundaries = sorted(list(set(boundaries)))
            
            for i in range(len(boundaries) - 1):
                if i < len(instructions):
                    inst = str(instructions[i]).strip()
                    if inst and inst.lower() != "unknown":
                        s_local, e_local = boundaries[i], boundaries[i + 1]
                        for k in range(s_local, e_local):
                            if k < cur_len:
                                global_fid = f_ids[k]
                                if global_fid < nframes:
                                    instruction_timeline[global_fid].append(inst)
        except (ValueError, IndexError):
            pass
    
    # Cluster cuts
    final_cut_points = [0]
    cut_support_by_point: dict[int, float] = {}
    
    if raw_cuts:
        raw_cuts.sort(key=lambda x: x[0])
        cluster_gap = max(1.0, 2.5 * fps)
        cur_frames = []
        cur_weights = []
        
        for fid, w in raw_cuts:
            if not cur_frames:
                cur_frames.append(fid)
                cur_weights.append(w)
                continue
            
            if (fid - cur_frames[-1]) < cluster_gap:
                cur_frames.append(fid)
                cur_weights.append(w)
            else:
                cluster_support = float(sum(cur_weights))
                if cur_weights and sum(cur_weights) > 1e-9:
                    avg = np.average(cur_frames, weights=cur_weights)
                    point = int(avg)
                else:
                    point = int(np.mean(cur_frames))
                final_cut_points.append(point)
                cut_support_by_point[point] = max(cut_support_by_point.get(point, 0.0), cluster_support)
                cur_frames = [fid]
                cur_weights = [w]
        
        if cur_frames:
            cluster_support = float(sum(cur_weights))
            if cur_weights and sum(cur_weights) > 1e-9:
                avg = np.average(cur_frames, weights=cur_weights)
                point = int(avg)
            else:
                point = int(np.mean(cur_frames))
            final_cut_points.append(point)
            cut_support_by_point[point] = max(cut_support_by_point.get(point, 0.0), cluster_support)
    
    final_cut_points.append(nframes)
    final_cut_points = sorted(list(set(final_cut_points)))
    
    # Build segments
    raw_segments = []
    seg_id = 0
    
    for i in range(len(final_cut_points) - 1):
        s, e = int(final_cut_points[i]), int(final_cut_points[i + 1])
        min_frames = max(1, int(0.8 * fps))
        
        if (e - s) < min_frames:
            continue
        
        margin = int((e - s) * 0.2) if e > s else 0
        mid_s, mid_e = s + margin, e - margin
        
        candidates = []
        for f in range(mid_s, mid_e + 1):
            if f < nframes:
                candidates.extend(instruction_timeline[f])
        
        if not candidates:
            for f in range(s, e):
                if f < nframes:
                    candidates.extend(instruction_timeline[f])
        
        if candidates:
            best_inst = Counter(candidates).most_common(1)[0][0]
            raw_segments.append({
                "seg_id": seg_id,
                "start_frame": s,
                "end_frame": e,
                "instruction": best_inst,
                "confidence": 1.0,
                "boundary_support_before": float(cut_support_by_point.get(s, 0.0)),
                "boundary_support_after": float(cut_support_by_point.get(e, 0.0)),
            })
            seg_id += 1

    raw_segments = split_long_raw_segments_on_instruction_drift(
        raw_segments,
        instruction_timeline,
        fps,
    )
    light_segments = cleanup_auxiliary_segments(raw_segments, fps)
    merged_segments = merge_task_level_segments(
        raw_segments,
        fps,
        boundary_support_threshold=boundary_support_threshold,
    )
    use_light_fallback = adaptive_merge_guard and _should_fallback_to_light_cleanup(
        light_segments,
        merged_segments,
        fps,
        min_segments=adaptive_merge_min_segments,
        collapse_ratio=adaptive_merge_collapse_ratio,
    )
    final_output = light_segments if use_light_fallback else merged_segments

    if refine_final_instructions:
        final_output = refine_segment_instructions(final_output, light_segments)
    
    return {
        "sample_id": sample_id,
        "nframes": nframes,
        "segments": final_output,
        "diagnostics": {
            "light_segment_count": len(light_segments),
            "merged_segment_count": len(merged_segments),
            "selected_segment_count": len(final_output),
            "selection_policy": "light_cleanup_fallback" if use_light_fallback else "semantic_merge",
        },
    }
