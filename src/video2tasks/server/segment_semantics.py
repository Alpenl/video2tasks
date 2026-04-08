"""Public segment semantics shared by Stage 1 and Stage 2 code paths."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


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
    "sprinkle", "toss", "fry", "boil", "toast", "fill", "fold", "roll",
}
_MIXING_ACTION_TOKENS = {"season", "stir", "mix", "whisk", "toss"}
_ACTION_FAMILY_SUPPORT_THRESHOLD = 1.2
_SAUCE_BASE_TOKENS = {"sauce", "tomato", "onion", "liquid", "mixture"}
_GENERIC_PHASE_NOUN_TOKENS = {"mixture", "content", "contents", "sauce", "base"}
_GENERIC_REFERENCE_TOKENS = {"component", "food", "ingredient", "ingredients"}
_DISH_FOCUS_TOKENS = {"salad"}
_FOCUS_NOISE_TOKENS = {
    "adding", "additional", "combined", "finish", "mash", "season", "simmering",
    "smooth", "sprinkle", "stir", "together", "use", "well", "tong",
}
_STEP_ORDER_TOKENS = {
    "another", "final", "first", "fourth", "last", "next", "second",
    "seventh", "sixth", "third", "fifth", "eighth", "ninth", "tenth",
}
_GENERIC_ADDITIVE_TOKENS = {
    "butter", "liquid", "oil", "pepper", "salt", "seasoning", "spice",
    "spices", "sugar", "water",
}
_SEASONING_FOCUS_TOKENS = _GENERIC_ADDITIVE_TOKENS | {"five", "powder"}
_UTENSIL_TOKENS = {"fork", "knife", "ladle", "masher", "processor", "spoon", "tong", "tongs"}
_DESTINATION_FOCUS_SKIP_TOKENS = (
    _GENERIC_ADDITIVE_TOKENS
    | _UTENSIL_TOKENS
    | _GENERIC_PHASE_NOUN_TOKENS
    | _FOCUS_NOISE_TOKENS
    | {"dish", "plate", "side", "top"}
)


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


def _instruction_sequence_markers(text: str) -> set[str]:
    return set(_instruction_tokens(text.replace("/", " "))) & _STEP_ORDER_TOKENS


def has_distinct_sequence_markers(left_instruction: str, right_instruction: str) -> bool:
    left_markers = _instruction_sequence_markers(left_instruction)
    right_markers = _instruction_sequence_markers(right_instruction)
    if not left_markers and not right_markers:
        return False
    return left_markers != right_markers


def _instruction_action_head_tokens(text: str, limit: int = 2) -> set[str]:
    tokens = _instruction_tokens(text.replace("/", " "))
    return set(tokens[:limit])


def _destination_food_tokens(text: str) -> set[str]:
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
            or token in _COOKING_CONTAINER_TOKENS
            or token in _DESTINATION_FOCUS_SKIP_TOKENS
            or token == "new"
        ):
            continue
        tokens.append(token)
    return set(tokens)


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


def _destination_focus_tokens(text: str) -> set[str]:
    tokens = _destination_tokens(text)
    tokens -= (_COOKING_CONTAINER_TOKENS - _DISH_FOCUS_TOKENS)
    tokens -= _STRONG_ACTION_TOKENS
    tokens -= _GENERIC_ACTION_TOKENS
    tokens -= _PREP_ACTION_TOKENS
    tokens -= _PREP_INGREDIENT_ACTION_TOKENS
    tokens -= _RECIPE_ACTION_TOKENS
    tokens -= _GENERIC_PHASE_NOUN_TOKENS
    tokens -= _GENERIC_REFERENCE_TOKENS
    tokens -= _FOCUS_NOISE_TOKENS
    tokens -= _UTENSIL_TOKENS
    return tokens - _SEASONING_FOCUS_TOKENS


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


def _instruction_is_filler_segment(text: str) -> bool:
    lower = text.lower().strip()
    if any(lower.startswith(prefix) for prefix in ("wait", "explain", "describe", "narrate")):
        return True
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    return bool(tokens & _FILLER_TOKENS)


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


def _instruction_has_generic_phase_noun(text: str) -> bool:
    return bool(set(_instruction_tokens(text.replace("/", " "))) & _GENERIC_PHASE_NOUN_TOKENS)


def _subject_focus_tokens(text: str) -> set[str]:
    tokens = _primary_object_tokens(text)
    tokens -= _STRONG_ACTION_TOKENS
    tokens -= _GENERIC_ACTION_TOKENS
    tokens -= _PREP_ACTION_TOKENS
    tokens -= _PREP_INGREDIENT_ACTION_TOKENS
    tokens -= _RECIPE_ACTION_TOKENS
    tokens -= _GENERIC_PHASE_NOUN_TOKENS
    tokens -= _GENERIC_REFERENCE_TOKENS
    tokens -= _FOCUS_NOISE_TOKENS
    tokens -= _UTENSIL_TOKENS
    return tokens - _SEASONING_FOCUS_TOKENS


def _generic_reference_subject_tokens(text: str) -> set[str]:
    tokens = _primary_object_tokens(text)
    tokens -= _STRONG_ACTION_TOKENS
    tokens -= _GENERIC_ACTION_TOKENS
    tokens -= _PREP_ACTION_TOKENS
    tokens -= _PREP_INGREDIENT_ACTION_TOKENS
    tokens -= _RECIPE_ACTION_TOKENS
    tokens -= _GENERIC_PHASE_NOUN_TOKENS
    tokens -= _FOCUS_NOISE_TOKENS
    tokens -= _UTENSIL_TOKENS
    tokens -= _SEASONING_FOCUS_TOKENS
    return tokens & _GENERIC_REFERENCE_TOKENS


def _instruction_focus_tokens(text: str) -> set[str]:
    primary_tokens = (
        _ingredient_tokens(text)
        | (_primary_object_tokens(text) - _COOKING_CONTAINER_TOKENS)
    )
    primary_tokens -= _STRONG_ACTION_TOKENS
    primary_tokens -= _GENERIC_ACTION_TOKENS
    primary_tokens -= _PREP_ACTION_TOKENS
    primary_tokens -= _PREP_INGREDIENT_ACTION_TOKENS
    primary_tokens -= _RECIPE_ACTION_TOKENS
    primary_tokens -= _GENERIC_PHASE_NOUN_TOKENS
    primary_tokens -= _GENERIC_REFERENCE_TOKENS
    primary_tokens -= _FOCUS_NOISE_TOKENS
    primary_tokens -= _UTENSIL_TOKENS

    subject_tokens = _subject_focus_tokens(text)
    destination_tokens = _destination_food_tokens(text)
    destination_focus_tokens = _destination_focus_tokens(text)
    if destination_tokens and primary_tokens <= _GENERIC_ADDITIVE_TOKENS:
        return destination_tokens

    focus_tokens = (primary_tokens - _GENERIC_ADDITIVE_TOKENS) | destination_tokens
    if destination_focus_tokens and (not focus_tokens or focus_tokens <= _SEASONING_FOCUS_TOKENS):
        return (focus_tokens - _SEASONING_FOCUS_TOKENS) | destination_focus_tokens
    if subject_tokens and (not focus_tokens or focus_tokens <= _SEASONING_FOCUS_TOKENS):
        return (focus_tokens - _SEASONING_FOCUS_TOKENS) | subject_tokens

    return focus_tokens


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
    if tokens & {"add", "pour", "fill", "sprinkle"}:
        families.add("add")
    if tokens & {"season", "stir", "mix", "whisk", "toss"}:
        families.add("mix")
    if tokens & {"cook", "saute", "fry", "boil", "toast"}:
        families.add("cook")
    if tokens & _PREP_INGREDIENT_ACTION_TOKENS:
        families.add("prep")
    if tokens & {"place", "transfer", "remove", "move"}:
        families.add("transfer")
    return families


def boundary_support_between(left: dict, right: dict) -> Tuple[float, bool]:
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


def _score_instruction_candidate(segment: dict, candidate: dict, current_instruction: str) -> float:
    del segment
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


def should_split_on_instruction_drift(
    left_instruction: str,
    right_instruction: str,
    fallback_containers: Optional[set[str]] = None,
) -> bool:
    del fallback_containers

    left_actions = _action_families(left_instruction)
    right_actions = _action_families(right_instruction)
    if left_actions != right_actions and (left_actions or right_actions):
        return True

    left_focus = _instruction_focus_tokens(left_instruction)
    right_focus = _instruction_focus_tokens(right_instruction)
    if left_focus and right_focus and not (left_focus & right_focus):
        return True

    left_primary = _primary_object_tokens(left_instruction) - _COOKING_CONTAINER_TOKENS
    right_primary = _primary_object_tokens(right_instruction) - _COOKING_CONTAINER_TOKENS
    if left_primary and right_primary and not (left_primary & right_primary):
        return True

    return False


__all__ = [
    "boundary_support_between",
    "has_distinct_sequence_markers",
    "refine_segment_instructions",
    "should_split_on_instruction_drift",
]
