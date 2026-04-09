"""Optional text-only LLM pass for merging obviously over-segmented adjacent segments."""

from __future__ import annotations

from collections import Counter
import copy
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from ..config import LLMMergeConfig
from ..prompt import prompt_segment_hierarchy, prompt_segment_merge, prompt_segment_subtitles
from ..vlm.openai_api import OpenAIBackend
from .segment_semantics import (
    boundary_support_between,
    has_distinct_sequence_markers,
    refine_segment_instructions,
    should_split_on_instruction_drift,
)


_MERGE_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "thought": {"type": "string"},
        "merged_ranges": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_seg_id": {"type": "integer"},
                    "end_seg_id": {"type": "integer"},
                },
                "required": ["start_seg_id", "end_seg_id"],
            },
        },
    },
    "required": ["thought", "merged_ranges"],
}


_SUBTITLE_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "thought": {"type": "string"},
        "subtitles": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "seg_id": {"type": "integer"},
                    "subtitle": {"type": "string"},
                },
                "required": ["seg_id", "subtitle"],
            },
        },
    },
    "required": ["thought", "subtitles"],
}


_COARSE_SEQUENCE_MARKER_IGNORED_TOKENS = {
    "a",
    "an",
    "the",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "next",
    "another",
    "final",
    "last",
    "initial",
    "1st",
    "2nd",
    "3rd",
    "4th",
    "5th",
    "6th",
    "7th",
    "8th",
    "9th",
    "10th",
    "11th",
    "12th",
}

_PROMPT_BOUNDARY_HINT_SUPPORT_THRESHOLD = 0.45
_PROMPT_BOUNDARY_HINT_MAX_COUNT = 24
_SUMMARY_LEVEL_NAMES = ("coarse", "medium", "fine")

# Canonical language codes used by Stage 2 subtitle localization.
# Source instructions are always treated as English (en); localization targets only subtitles.
_LANGUAGE_CODE_ALIASES = {
    "zh": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "cn": "zh",
    "中文": "zh",
    "chinese": "zh",
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "english": "en",
}

def _normalize_language_code(language: str) -> str:
    normalized = str(language).strip().lower()
    return _LANGUAGE_CODE_ALIASES.get(normalized, normalized)


def _seed_instruction(segments: List[dict]) -> str:
    unique_instructions: List[str] = []
    seen: set[str] = set()
    for segment in segments:
        text = str(segment.get("instruction", "")).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_instructions.append(text)

    if not unique_instructions:
        return "Unknown task step"
    if len(unique_instructions) == 1:
        return unique_instructions[0]
    return max(unique_instructions, key=len)


def validate_merged_ranges(
    payload: Dict[str, Any],
    segment_count: int,
    min_output_ratio: float,
    *,
    segment_ids: Optional[List[int]] = None,
) -> Tuple[Optional[List[Tuple[int, int]]], str]:
    if not isinstance(payload, dict):
        return None, "invalid_payload"

    raw_ranges = payload.get("merged_ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        return None, "invalid_merged_ranges"

    expected_ids = list(segment_ids) if isinstance(segment_ids, list) else list(range(int(segment_count)))
    if len(expected_ids) != int(segment_count):
        expected_ids = list(range(int(segment_count)))

    # If seg_id values are not unique, fall back to positional indices.
    if len(set(expected_ids)) != len(expected_ids):
        expected_ids = list(range(int(segment_count)))

    id_to_index = {int(seg_id): index for index, seg_id in enumerate(expected_ids)}

    def _parse_with(token_to_index) -> Tuple[Optional[List[Tuple[int, int]]], str]:
        normalized: List[Tuple[int, int]] = []
        expected_start_index = 0

        for item in raw_ranges:
            if not isinstance(item, dict):
                return None, "invalid_merged_ranges"
            try:
                start_token = int(item.get("start_seg_id"))
                end_token = int(item.get("end_seg_id"))
            except (TypeError, ValueError):
                return None, "invalid_merged_ranges"

            start_index = token_to_index(start_token)
            end_index = token_to_index(end_token)
            if start_index is None or end_index is None:
                return None, "invalid_partition"

            if start_index != expected_start_index:
                return None, "invalid_partition"
            if start_index < 0 or end_index < start_index or end_index >= int(segment_count):
                return None, "invalid_partition"

            normalized.append((int(start_index), int(end_index)))
            expected_start_index = int(end_index) + 1

        if expected_start_index != int(segment_count):
            return None, "incomplete_partition"

        min_output_segments = max(1, int(math.ceil(float(segment_count) * float(min_output_ratio))))
        if len(normalized) < min_output_segments:
            return None, "collapsed_too_aggressively"

        return normalized, "ok"

    def token_to_index_seg_id_first(token: int) -> Optional[int]:
        token = int(token)
        if token in id_to_index:
            return id_to_index[token]
        if 0 <= token < int(segment_count):
            return token
        return None

    def token_to_index_index_first(token: int) -> Optional[int]:
        token = int(token)
        if 0 <= token < int(segment_count):
            return token
        return id_to_index.get(token)

    # Prefer interpreting tokens as seg_id when possible (fixes common 1-based seg_id collisions).
    parsed, reason = _parse_with(token_to_index_seg_id_first)
    if parsed:
        return parsed, reason

    parsed, reason2 = _parse_with(token_to_index_index_first)
    if parsed:
        return parsed, reason2

    return None, reason


def validate_merged_partition(
    payload: Dict[str, Any],
    segment_count: int,
    *,
    segment_ids: Optional[List[int]] = None,
) -> Tuple[Optional[List[Tuple[int, int]]], str]:
    return validate_merged_ranges(
        payload,
        segment_count=segment_count,
        min_output_ratio=1.0 / float(max(1, segment_count)),
        segment_ids=segment_ids,
    )


def merged_range_count_below_ratio(
    merged_ranges: List[Tuple[int, int]],
    segment_count: int,
    min_output_ratio: float,
) -> bool:
    min_output_segments = max(1, int(math.ceil(float(segment_count) * float(min_output_ratio))))
    return len(merged_ranges) < min_output_segments


def merge_segments_by_ranges(segments: List[dict], merged_ranges: List[Tuple[int, int]]) -> List[dict]:
    merged_segments: List[dict] = []
    for out_seg_id, (start_seg_id, end_seg_id) in enumerate(merged_ranges):
        group = segments[start_seg_id : end_seg_id + 1]
        if not group:
            return []

        total_frames = 0
        weighted_confidence = 0.0
        for segment in group:
            duration_frames = max(1, int(segment["end_frame"]) - int(segment["start_frame"]))
            total_frames += duration_frames
            weighted_confidence += float(segment.get("confidence", 1.0)) * float(duration_frames)

        merged_segments.append(
            {
                "seg_id": out_seg_id,
                "start_frame": int(group[0]["start_frame"]),
                "end_frame": int(group[-1]["end_frame"]),
                "instruction": _seed_instruction(group),
                "confidence": (weighted_confidence / float(total_frames)) if total_frames > 0 else 1.0,
                "boundary_support_before": float(group[0].get("boundary_support_before", 0.0)),
                "boundary_support_after": float(group[-1].get("boundary_support_after", 0.0)),
            }
        )

    return refine_segment_instructions(merged_segments, segments)


def _merge_guard_reasons(left: dict, right: dict, merge_config: LLMMergeConfig) -> List[str]:
    reasons: List[str] = []
    boundary_support, has_boundary_support = boundary_support_between(left, right)

    if (
        merge_config.protect_supported_boundaries
        and has_boundary_support
        and boundary_support >= float(merge_config.protected_boundary_support_threshold)
    ):
        reasons.append("boundary_support")

    if (
        merge_config.protect_distinct_sequence_markers
        and has_distinct_sequence_markers(
            str(left.get("instruction", "")),
            str(right.get("instruction", "")),
        )
    ):
        reasons.append("sequence_markers")

    if (
        merge_config.protect_instruction_drift
        and should_split_on_instruction_drift(
            str(left.get("instruction", "")),
            str(right.get("instruction", "")),
        )
    ):
        reasons.append("instruction_drift")

    return reasons


def _segment_duration_frames(segment: dict) -> int:
    return max(1, int(segment["end_frame"]) - int(segment["start_frame"]))


def _coarse_sequence_marker_round_key(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(text).strip().lower())
    filtered_tokens = [token for token in tokens if token not in _COARSE_SEQUENCE_MARKER_IGNORED_TOKENS]
    return " ".join(filtered_tokens)


def _is_coarse_sequence_marker_round_only(left_instruction: str, right_instruction: str) -> bool:
    left_text = str(left_instruction).strip().lower()
    right_text = str(right_instruction).strip().lower()
    if not left_text or not right_text or left_text == right_text:
        return False

    left_key = _coarse_sequence_marker_round_key(left_text)
    right_key = _coarse_sequence_marker_round_key(right_text)
    return bool(left_key) and left_key == right_key


def _blocked_boundary_info(
    segments: List[dict],
    boundary_after_seg_id: int,
    merge_config: LLMMergeConfig,
) -> Optional[Dict[str, Any]]:
    left = segments[boundary_after_seg_id]
    right = segments[boundary_after_seg_id + 1]
    left_instruction = str(left.get("instruction", "")).strip()
    right_instruction = str(right.get("instruction", "")).strip()
    reasons = _merge_guard_reasons(left, right, merge_config)
    if not reasons:
        return None

    boundary_support, has_boundary_support = boundary_support_between(left, right)
    info = {
        "boundary_after_seg_id": int(boundary_after_seg_id),
        "left_seg_id": int(left.get("seg_id", boundary_after_seg_id)),
        "right_seg_id": int(right.get("seg_id", boundary_after_seg_id + 1)),
        "left_instruction": left_instruction,
        "right_instruction": right_instruction,
        "reasons": reasons,
        "boundary_support": float(boundary_support),
        "has_boundary_support": bool(has_boundary_support),
        "left_duration_frames": _segment_duration_frames(left),
        "right_duration_frames": _segment_duration_frames(right),
    }
    if "sequence_markers" in reasons and _is_coarse_sequence_marker_round_only(left_instruction, right_instruction):
        info["sequence_marker_round_only"] = True
    return info


def _coarse_guard_anchor_score(boundary_info: Dict[str, Any]) -> float:
    reasons = set(str(reason) for reason in boundary_info.get("reasons", []))
    sequence_marker_round_only = bool(boundary_info.get("sequence_marker_round_only", False))
    score = float(boundary_info.get("boundary_support", 0.0))
    if "instruction_drift" in reasons:
        score += 0.5
    if "sequence_markers" in reasons and not sequence_marker_round_only:
        score += 0.5
    return score


def _coarse_guard_anchor_sort_key(boundary_info: Dict[str, Any]) -> Tuple[float, int, float, int, int, int]:
    reasons = set(str(reason) for reason in boundary_info.get("reasons", []))
    sequence_marker_round_only = bool(boundary_info.get("sequence_marker_round_only", False))
    left_duration_frames = int(boundary_info.get("left_duration_frames", 1))
    right_duration_frames = int(boundary_info.get("right_duration_frames", 1))
    non_support_reason_count = sum(
        1
        for reason in reasons
        if reason != "boundary_support" and not (reason == "sequence_markers" and sequence_marker_round_only)
    )
    return (
        float(boundary_info.get("coarse_anchor_score", _coarse_guard_anchor_score(boundary_info))),
        non_support_reason_count,
        float(boundary_info.get("boundary_support", 0.0)),
        min(left_duration_frames, right_duration_frames),
        left_duration_frames + right_duration_frames,
        -int(boundary_info.get("boundary_after_seg_id", 0)),
    )


def _selected_blocked_boundaries_for_range(
    range_blockers: List[Dict[str, Any]],
    merge_config: LLMMergeConfig,
    start_seg_id: int,
    end_seg_id: int,
) -> List[Dict[str, Any]]:
    if str(merge_config.granularity) != "coarse":
        return sorted(range_blockers, key=lambda item: int(item["boundary_after_seg_id"]))

    support_anchor_limit = max(0, int(merge_config.coarse_max_supported_anchors_per_range))
    if support_anchor_limit <= 0 or not range_blockers:
        return []

    range_segment_count = max(1, int(end_seg_id) - int(start_seg_id) + 1)
    min_spacing = max(1, int(merge_config.coarse_anchor_min_spacing_segments))
    min_side_segments = max(1, int(merge_config.coarse_anchor_min_side_segments))
    min_score = max(0.0, float(merge_config.coarse_anchor_min_score))
    effective_anchor_limit = support_anchor_limit
    long_range_threshold = max(10, int(min_spacing * 4))
    if range_segment_count >= long_range_threshold:
        effective_anchor_limit = max(
            support_anchor_limit,
            int(math.ceil(float(max(1, range_segment_count - 1)) / float(max(1, min_spacing * 2)))),
        )
    ranked_blockers: List[Dict[str, Any]] = []
    for boundary_info in range_blockers:
        scored_boundary_info = dict(boundary_info)
        scored_boundary_info["coarse_anchor_score"] = _coarse_guard_anchor_score(scored_boundary_info)
        ranked_blockers.append(scored_boundary_info)

    use_balanced_anchor_sort = range_segment_count >= long_range_threshold

    def coarse_range_sort_key(boundary_info: Dict[str, Any]) -> Tuple[int, float, int, float, int, int, int]:
        boundary_after_seg_id = int(boundary_info["boundary_after_seg_id"])
        left_segment_count = boundary_after_seg_id - int(start_seg_id) + 1
        right_segment_count = int(end_seg_id) - boundary_after_seg_id
        if not use_balanced_anchor_sort:
            return (0, *_coarse_guard_anchor_sort_key(boundary_info))
        return (
            min(left_segment_count, right_segment_count),
            *_coarse_guard_anchor_sort_key(boundary_info),
        )

    # Coarse mode treats guard signals as anchor-ranking hints, not hard split-all constraints.
    selected: List[Dict[str, Any]] = []
    selected_boundary_ids: List[int] = []
    for boundary_info in sorted(ranked_blockers, key=coarse_range_sort_key, reverse=True):
        boundary_after_seg_id = int(boundary_info["boundary_after_seg_id"])
        left_segment_count = boundary_after_seg_id - int(start_seg_id) + 1
        right_segment_count = int(end_seg_id) - boundary_after_seg_id
        if float(boundary_info.get("coarse_anchor_score", 0.0)) < min_score:
            continue
        if left_segment_count < min_side_segments or right_segment_count < min_side_segments:
            continue
        if any(abs(boundary_after_seg_id - existing) < min_spacing for existing in selected_boundary_ids):
            continue
        selected.append(boundary_info)
        selected_boundary_ids.append(boundary_after_seg_id)
        if len(selected) >= effective_anchor_limit:
            break

    return sorted(selected, key=lambda item: int(item["boundary_after_seg_id"]))


def _normalized_instruction_key(segment: dict) -> str:
    text = str(segment.get("instruction", "")).strip().lower()
    return " ".join(text.split())


def _build_prompt_boundary_hints(
    segments: List[dict],
    merge_config: LLMMergeConfig,
) -> List[Dict[str, Any]]:
    ranked_hints: List[Dict[str, Any]] = []
    for boundary_after_seg_id in range(max(0, len(segments) - 1)):
        boundary_info = _blocked_boundary_info(segments, boundary_after_seg_id, merge_config)
        if boundary_info is None:
            left = segments[boundary_after_seg_id]
            right = segments[boundary_after_seg_id + 1]
            left_instruction = str(left.get("instruction", "")).strip()
            right_instruction = str(right.get("instruction", "")).strip()
            boundary_support, has_boundary_support = boundary_support_between(left, right)
            sequence_markers = has_distinct_sequence_markers(left_instruction, right_instruction)
            instruction_drift = should_split_on_instruction_drift(left_instruction, right_instruction)
            if not (
                (has_boundary_support and boundary_support >= _PROMPT_BOUNDARY_HINT_SUPPORT_THRESHOLD)
                or sequence_markers
                or instruction_drift
            ):
                continue
            boundary_info = {
                "boundary_after_seg_id": int(boundary_after_seg_id),
                "left_seg_id": int(left.get("seg_id", boundary_after_seg_id)),
                "right_seg_id": int(right.get("seg_id", boundary_after_seg_id + 1)),
                "left_instruction": left_instruction,
                "right_instruction": right_instruction,
                "reasons": [
                    *( ["boundary_support"] if has_boundary_support and boundary_support >= _PROMPT_BOUNDARY_HINT_SUPPORT_THRESHOLD else [] ),
                    *( ["sequence_markers"] if sequence_markers else [] ),
                    *( ["instruction_drift"] if instruction_drift else [] ),
                ],
                "boundary_support": float(boundary_support),
                "has_boundary_support": bool(has_boundary_support),
                "left_duration_frames": _segment_duration_frames(left),
                "right_duration_frames": _segment_duration_frames(right),
            }

        ranked_hint = dict(boundary_info)
        ranked_hint["coarse_anchor_score"] = _coarse_guard_anchor_score(ranked_hint)
        ranked_hints.append(ranked_hint)

    if not ranked_hints:
        return []

    max_hint_count = max(8, min(_PROMPT_BOUNDARY_HINT_MAX_COUNT, int(math.ceil(max(1, len(segments) - 1) / 4.0))))
    min_spacing = max(1, int(merge_config.coarse_anchor_min_spacing_segments))
    selected_hints: List[Dict[str, Any]] = []
    selected_boundary_ids: List[int] = []

    for hint in sorted(ranked_hints, key=_coarse_guard_anchor_sort_key, reverse=True):
        boundary_after_seg_id = int(hint["boundary_after_seg_id"])
        if any(abs(boundary_after_seg_id - existing) < min_spacing for existing in selected_boundary_ids):
            continue
        selected_hints.append(
            {
                "boundary_after_seg_id": boundary_after_seg_id,
                "boundary_frame": int(segments[boundary_after_seg_id].get("end_frame", 0)),
                "boundary_support": float(hint.get("boundary_support", 0.0)),
                "has_boundary_support": bool(hint.get("has_boundary_support", False)),
                "sequence_markers": "sequence_markers" in set(str(reason) for reason in hint.get("reasons", [])),
                "instruction_drift": "instruction_drift" in set(str(reason) for reason in hint.get("reasons", [])),
                "left_instruction": str(hint.get("left_instruction", "")).strip(),
                "right_instruction": str(hint.get("right_instruction", "")).strip(),
            }
        )
        selected_boundary_ids.append(boundary_after_seg_id)
        if len(selected_hints) >= max_hint_count:
            break

    return sorted(selected_hints, key=lambda item: int(item["boundary_after_seg_id"]))


def sanitize_merged_ranges(
    segments: List[dict],
    merged_ranges: List[Tuple[int, int]],
    merge_config: LLMMergeConfig,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    sanitized: List[Tuple[int, int]] = []
    blocked_boundaries: List[Dict[str, Any]] = []

    for start_seg_id, end_seg_id in merged_ranges:
        range_blockers: List[Dict[str, Any]] = []
        for boundary_after_seg_id in range(int(start_seg_id), int(end_seg_id)):
            boundary_info = _blocked_boundary_info(segments, boundary_after_seg_id, merge_config)
            if boundary_info:
                range_blockers.append(boundary_info)

        selected_blockers = _selected_blocked_boundaries_for_range(
            range_blockers,
            merge_config,
            start_seg_id=int(start_seg_id),
            end_seg_id=int(end_seg_id),
        )
        blocked_boundaries.extend(selected_blockers)

        current_start = int(start_seg_id)
        for boundary_info in selected_blockers:
            boundary_after_seg_id = int(boundary_info["boundary_after_seg_id"])
            sanitized.append((current_start, int(boundary_after_seg_id)))
            current_start = int(boundary_after_seg_id) + 1

        sanitized.append((current_start, int(end_seg_id)))

    return sanitized, blocked_boundaries


def preserve_duplicate_tail_anchors(
    segments: List[dict],
    merged_ranges: List[Tuple[int, int]],
    merge_config: LLMMergeConfig,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    if str(merge_config.granularity) == "coarse":
        return merged_ranges, []

    if not merge_config.protect_duplicate_tail_anchor:
        return merged_ranges, []

    preserved_ranges: List[Tuple[int, int]] = []
    preserved_tail_anchors: List[Dict[str, Any]] = []

    for start_seg_id, end_seg_id in merged_ranges:
        start_seg_id = int(start_seg_id)
        end_seg_id = int(end_seg_id)
        if end_seg_id <= start_seg_id or (end_seg_id + 1) >= len(segments):
            preserved_ranges.append((start_seg_id, end_seg_id))
            continue

        tail_prev = segments[end_seg_id - 1]
        tail = segments[end_seg_id]
        next_external = segments[end_seg_id + 1]
        tail_duration_frames = max(1, int(tail["end_frame"]) - int(tail["start_frame"]))
        next_boundary_reasons = _merge_guard_reasons(tail, next_external, merge_config)
        has_semantic_external_shift = any(
            reason in {"sequence_markers", "instruction_drift"}
            for reason in next_boundary_reasons
        )

        if (
            tail_duration_frames >= int(merge_config.duplicate_tail_anchor_min_frames)
            and _normalized_instruction_key(tail_prev) == _normalized_instruction_key(tail)
            and has_semantic_external_shift
        ):
            preserved_ranges.append((start_seg_id, end_seg_id - 1))
            preserved_ranges.append((end_seg_id, end_seg_id))
            preserved_tail_anchors.append(
                {
                    "boundary_after_seg_id": end_seg_id - 1,
                    "left_seg_id": int(tail_prev.get("seg_id", end_seg_id - 1)),
                    "right_seg_id": int(tail.get("seg_id", end_seg_id)),
                    "instruction": str(tail.get("instruction", "")).strip(),
                    "tail_duration_frames": tail_duration_frames,
                    "next_boundary_reasons": next_boundary_reasons,
                }
            )
            continue

        preserved_ranges.append((start_seg_id, end_seg_id))

    return preserved_ranges, preserved_tail_anchors


def _effective_min_output_ratio(merge_config: LLMMergeConfig) -> float:
    if str(merge_config.granularity) == "coarse":
        return float(merge_config.coarse_min_output_ratio)
    return float(merge_config.min_output_ratio)


def _request_merge_payload(
    resolved_backend: Any,
    prompt: str,
    merge_config: LLMMergeConfig,
) -> Tuple[Dict[str, Any], str, str, int, List[Dict[str, Any]]]:
    return _request_structured_payload(
        resolved_backend,
        prompt,
        merge_config,
        schema_name="segment_merge_result",
        schema=_MERGE_RESULT_SCHEMA,
    )


def _request_structured_payload(
    resolved_backend: Any,
    prompt: str,
    merge_config: LLMMergeConfig,
    *,
    schema_name: str,
    schema: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, str, int, List[Dict[str, Any]]]:
    payload: Dict[str, Any] = {}
    max_attempts = max(1, int(merge_config.max_attempts))
    last_reason = "empty_response"
    last_error = ""
    request_attempt_count = 0
    adapter_diagnostics_attempts: List[Dict[str, Any]] = []

    def clone_adapter_diagnostics() -> Optional[Dict[str, Any]]:
        raw = getattr(resolved_backend, "last_text_json_diagnostics", None)
        if not isinstance(raw, dict):
            return None
        return copy.deepcopy(raw)

    def derive_reason_from_adapter_diagnostics(adapter_diagnostics: Dict[str, Any], default: str) -> str:
        final_failure_reason = str(adapter_diagnostics.get("final_failure_reason", "")).strip()
        if final_failure_reason:
            return final_failure_reason
        return default

    def derive_error_from_adapter_diagnostics(adapter_diagnostics: Dict[str, Any]) -> str:
        for endpoint_name in ("chat_completions", "responses"):
            endpoint_diagnostics = adapter_diagnostics.get(endpoint_name)
            if not isinstance(endpoint_diagnostics, dict):
                continue
            error = str(endpoint_diagnostics.get("error", "")).strip()
            if error:
                return error
            exception_type = str(endpoint_diagnostics.get("exception_type", "")).strip()
            if exception_type:
                return exception_type
        return ""

    for attempt_index in range(1, max_attempts + 1):
        request_attempt_count = attempt_index
        try:
            payload = resolved_backend.infer_text_json(
                prompt,
                schema_name=schema_name,
                schema=schema,
                max_output_tokens=int(merge_config.max_output_tokens),
                reasoning_effort=str(merge_config.reasoning_effort),
                raise_on_http_error=True,
            )
        except Exception as exc:
            adapter_diagnostics = clone_adapter_diagnostics()
            if adapter_diagnostics:
                adapter_diagnostics["request_attempt_index"] = attempt_index
                adapter_diagnostics_attempts.append(adapter_diagnostics)
                last_reason = derive_reason_from_adapter_diagnostics(
                    adapter_diagnostics,
                    default=f"request_failed:{type(exc).__name__}",
                )
                last_error = derive_error_from_adapter_diagnostics(adapter_diagnostics) or str(exc).strip() or type(exc).__name__
            else:
                last_reason = f"request_failed:{type(exc).__name__}"
                last_error = str(exc).strip() or type(exc).__name__
            payload = {}
            continue

        adapter_diagnostics = clone_adapter_diagnostics()
        if adapter_diagnostics:
            adapter_diagnostics["request_attempt_index"] = attempt_index
            adapter_diagnostics_attempts.append(adapter_diagnostics)

        if payload:
            return payload, "ok", "", request_attempt_count, adapter_diagnostics_attempts

        if adapter_diagnostics:
            last_reason = derive_reason_from_adapter_diagnostics(adapter_diagnostics, default="empty_response")
            last_error = derive_error_from_adapter_diagnostics(adapter_diagnostics)
        else:
            last_reason = "empty_response"
            last_error = ""

    return payload, last_reason, last_error, request_attempt_count, adapter_diagnostics_attempts


def _enabled_summary_level_names(summary_levels: List[int]) -> List[str]:
    enabled_levels: List[str] = []
    for index, level_name in enumerate(_SUMMARY_LEVEL_NAMES):
        if index < len(summary_levels) and int(summary_levels[index]) == 1:
            enabled_levels.append(level_name)
    return enabled_levels


def _single_attempt_config(merge_config: LLMMergeConfig) -> LLMMergeConfig:
    if int(merge_config.max_attempts) <= 1:
        return merge_config
    return merge_config.model_copy(update={"max_attempts": 1})


def _attach_export_subtitles(segments: List[dict], subtitles: List[str]) -> List[dict]:
    localized_segments: List[dict] = []
    for index, segment in enumerate(segments):
        localized_segment = dict(segment)
        subtitle = subtitles[index] if index < len(subtitles) else str(segment.get("instruction", "")).strip()
        localized_segment["export_subtitle"] = str(subtitle).strip() or str(segment.get("instruction", "")).strip() or "Unknown task step"
        localized_segments.append(localized_segment)
    return localized_segments


def _source_instruction_subtitles(segments: List[dict]) -> List[str]:
    return [
        str(segment.get("instruction", "")).strip() or "Unknown task step"
        for segment in segments
    ]


def _segment_id_sequence(segments: List[dict]) -> List[int]:
    """Return the seg_id tokens that prompts/payloads are expected to reference.

    If seg_id values are missing, non-integer, or duplicated, fall back to positional indices.
    """

    ids: List[int] = []
    for index, segment in enumerate(segments):
        try:
            ids.append(int(segment.get("seg_id", index)))
        except (TypeError, ValueError):
            ids.append(int(index))

    if len(set(ids)) != len(ids):
        return list(range(len(segments)))
    return ids


def validate_subtitle_payload(
    payload: Dict[str, Any],
    segment_count: int,
    *,
    segment_ids: Optional[List[int]] = None,
) -> Tuple[Optional[List[str]], str]:
    if not isinstance(payload, dict):
        return None, "invalid_payload"

    raw_subtitles = payload.get("subtitles")
    if not isinstance(raw_subtitles, list) or len(raw_subtitles) != int(segment_count):
        return None, "invalid_subtitles"

    expected_ids = list(segment_ids) if isinstance(segment_ids, list) else list(range(int(segment_count)))
    if len(expected_ids) != int(segment_count):
        expected_ids = list(range(int(segment_count)))

    if len(set(expected_ids)) != len(expected_ids):
        expected_ids = list(range(int(segment_count)))

    id_to_index = {int(seg_id): index for index, seg_id in enumerate(expected_ids)}

    def _build_with(token_to_index) -> Optional[List[str]]:
        by_index: Dict[int, str] = {}
        for item in raw_subtitles:
            if not isinstance(item, dict):
                return None
            try:
                token = int(item.get("seg_id"))
            except (TypeError, ValueError):
                return None
            subtitle = str(item.get("subtitle", "")).strip()
            if not subtitle:
                return None

            index = token_to_index(token)
            if index is None or not (0 <= int(index) < int(segment_count)):
                return None
            if int(index) in by_index:
                return None
            by_index[int(index)] = subtitle

        if len(by_index) != int(segment_count):
            return None
        return [by_index[i] for i in range(int(segment_count))]

    def token_to_index_seg_id_first(token: int) -> Optional[int]:
        token = int(token)
        if token in id_to_index:
            return id_to_index[token]
        if 0 <= token < int(segment_count):
            return token
        return None

    def token_to_index_index_first(token: int) -> Optional[int]:
        token = int(token)
        if 0 <= token < int(segment_count):
            return token
        return id_to_index.get(token)

    subtitles = _build_with(token_to_index_seg_id_first)
    if subtitles is not None:
        return subtitles, "ok"

    subtitles = _build_with(token_to_index_index_first)
    if subtitles is not None:
        return subtitles, "ok"

    return None, "invalid_subtitles"


def _summary_range_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "start_seg_id": {"type": "integer"},
            "end_seg_id": {"type": "integer"},
            "summary": {"type": "string"},
        },
        "required": ["start_seg_id", "end_seg_id", "summary"],
    }


def build_summary_result_schema(enabled_level_names: List[str]) -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "thought": {"type": "string"},
    }
    required = ["thought"]
    range_schema = _summary_range_schema()
    for level_name in enabled_level_names:
        properties[level_name] = {
            "type": "array",
            "items": range_schema,
        }
        required.append(level_name)

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _validate_summary_level_partition(
    raw_items: Any,
    *,
    level_name: str,
    segment_count: int,
    segment_ids: List[int],
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    if not isinstance(raw_items, list) or not raw_items:
        return None, f"invalid_{level_name}_ranges"

    expected_ids = list(segment_ids)
    if len(expected_ids) != int(segment_count) or len(set(expected_ids)) != len(expected_ids):
        expected_ids = list(range(int(segment_count)))

    id_to_index = {int(seg_id): index for index, seg_id in enumerate(expected_ids)}

    def _parse_with(token_to_index) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        normalized: List[Dict[str, Any]] = []
        expected_start_index = 0

        for item in raw_items:
            if not isinstance(item, dict):
                return None, f"invalid_{level_name}_ranges"
            try:
                start_token = int(item.get("start_seg_id"))
                end_token = int(item.get("end_seg_id"))
            except (TypeError, ValueError):
                return None, f"invalid_{level_name}_ranges"

            summary = str(item.get("summary", "")).strip()
            if not summary:
                return None, f"invalid_{level_name}_summary"

            start_index = token_to_index(start_token)
            end_index = token_to_index(end_token)
            if start_index is None or end_index is None:
                return None, f"invalid_{level_name}_partition"

            if start_index != expected_start_index:
                return None, f"invalid_{level_name}_partition"
            if start_index < 0 or end_index < start_index or end_index >= int(segment_count):
                return None, f"invalid_{level_name}_partition"

            normalized.append(
                {
                    "start_seg_id": int(start_index),
                    "end_seg_id": int(end_index),
                    "summary": summary,
                }
            )
            expected_start_index = int(end_index) + 1

        if expected_start_index != int(segment_count):
            return None, f"incomplete_{level_name}_partition"

        return normalized, "ok"

    def token_to_index_seg_id_first(token: int) -> Optional[int]:
        token = int(token)
        if token in id_to_index:
            return id_to_index[token]
        if 0 <= token < int(segment_count):
            return token
        return None

    def token_to_index_index_first(token: int) -> Optional[int]:
        token = int(token)
        if 0 <= token < int(segment_count):
            return token
        return id_to_index.get(token)

    parsed, reason = _parse_with(token_to_index_seg_id_first)
    if parsed:
        return parsed, reason

    parsed, reason2 = _parse_with(token_to_index_index_first)
    if parsed:
        return parsed, reason2

    return None, reason


def _ranges_are_nested(
    outer_ranges: List[Dict[str, Any]],
    inner_ranges: List[Dict[str, Any]],
) -> bool:
    outer_index = 0
    for inner_item in inner_ranges:
        inner_start = int(inner_item["start_seg_id"])
        inner_end = int(inner_item["end_seg_id"])
        while outer_index < len(outer_ranges) and int(outer_ranges[outer_index]["end_seg_id"]) < inner_start:
            outer_index += 1
        if outer_index >= len(outer_ranges):
            return False
        outer_item = outer_ranges[outer_index]
        if inner_start < int(outer_item["start_seg_id"]) or inner_end > int(outer_item["end_seg_id"]):
            return False
    return True


def validate_summary_partitions(
    payload: Dict[str, Any],
    segment_count: int,
    enabled_level_names: List[str],
    *,
    segment_ids: Optional[List[int]] = None,
) -> Tuple[Optional[Dict[str, List[Dict[str, Any]]]], str]:
    if not isinstance(payload, dict):
        return None, "invalid_payload"

    expected_ids = list(segment_ids) if isinstance(segment_ids, list) else list(range(int(segment_count)))

    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for level_name in enabled_level_names:
        partition, reason = _validate_summary_level_partition(
            payload.get(level_name),
            level_name=level_name,
            segment_count=segment_count,
            segment_ids=expected_ids,
        )
        if not partition:
            return None, reason
        normalized[level_name] = partition

    for outer_level_name, inner_level_name in zip(enabled_level_names, enabled_level_names[1:]):
        if not _ranges_are_nested(normalized[outer_level_name], normalized[inner_level_name]):
            return None, f"cross_level_nesting_violation:{outer_level_name}->{inner_level_name}"

    return normalized, "ok"


def _identity_summary_partitions(
    segments: List[dict],
    enabled_level_names: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    finest_partition = [
        {
            "start_seg_id": int(index),
            "end_seg_id": int(index),
            "summary": str(segment.get("instruction", "")).strip() or "Unknown task step",
        }
        for index, segment in enumerate(segments)
    ]
    return {
        level_name: [dict(item) for item in finest_partition]
        for level_name in enabled_level_names
    }


def build_task_hierarchy(
    segments: List[dict],
    partitions: Dict[str, List[Dict[str, Any]]],
    enabled_levels: List[int],
) -> Dict[str, Any]:
    enabled_level_names = [
        level_name for level_name in _SUMMARY_LEVEL_NAMES if level_name in partitions
    ]
    nodes_by_level: Dict[str, List[Dict[str, Any]]] = {}

    for level_name in enabled_level_names:
        level_nodes: List[Dict[str, Any]] = []
        for item in partitions[level_name]:
            start_seg_id = int(item["start_seg_id"])
            end_seg_id = int(item["end_seg_id"])
            level_nodes.append(
                {
                    "level": level_name,
                    "summary": str(item["summary"]).strip(),
                    "start_seg_id": start_seg_id,
                    "end_seg_id": end_seg_id,
                    "start_frame": int(segments[start_seg_id].get("start_frame", 0)),
                    "end_frame": int(segments[end_seg_id].get("end_frame", 0)),
                    "children": [],
                }
            )
        nodes_by_level[level_name] = level_nodes

    for parent_level_name, child_level_name in zip(enabled_level_names, enabled_level_names[1:]):
        child_nodes = nodes_by_level[child_level_name]
        child_index = 0
        for parent_node in nodes_by_level[parent_level_name]:
            parent_end_seg_id = int(parent_node["end_seg_id"])
            while child_index < len(child_nodes) and int(child_nodes[child_index]["start_seg_id"]) < int(parent_node["start_seg_id"]):
                child_index += 1

            children: List[Dict[str, Any]] = []
            while child_index < len(child_nodes) and int(child_nodes[child_index]["end_seg_id"]) <= parent_end_seg_id:
                children.append(child_nodes[child_index])
                child_index += 1
            parent_node["children"] = children

    root_level = enabled_level_names[0]
    return {
        "enabled_levels": [int(value) for value in enabled_levels],
        "enabled_level_names": enabled_level_names,
        "root_level": root_level,
        "roots": nodes_by_level[root_level],
    }


def _boundary_after_seg_ids_from_ranges(merged_ranges: List[Tuple[int, int]]) -> List[int]:
    return [int(end_seg_id) for _, end_seg_id in merged_ranges[:-1]]


def _ranges_from_boundary_after_seg_ids(boundary_after_seg_ids: List[int], segment_count: int) -> List[Tuple[int, int]]:
    if segment_count <= 0:
        return []

    normalized = sorted(
        {
            int(boundary_after_seg_id)
            for boundary_after_seg_id in boundary_after_seg_ids
            if 0 <= int(boundary_after_seg_id) < (segment_count - 1)
        }
    )
    ranges: List[Tuple[int, int]] = []
    current_start = 0
    for boundary_after_seg_id in normalized:
        ranges.append((current_start, int(boundary_after_seg_id)))
        current_start = int(boundary_after_seg_id) + 1
    ranges.append((current_start, segment_count - 1))
    return ranges


def _build_merge_candidate(
    input_segments: List[dict],
    payload: Dict[str, Any],
    merge_config: LLMMergeConfig,
    effective_min_output_ratio: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    input_count = len(input_segments)
    thought = str(payload.get("thought", "")).strip()

    merged_ranges, validation_reason = validate_merged_partition(
        payload,
        segment_count=input_count,
        segment_ids=_segment_id_sequence(input_segments),
    )
    if not merged_ranges:
        return None, validation_reason

    requested_ranges = [
        {"start_seg_id": int(start_seg_id), "end_seg_id": int(end_seg_id)}
        for start_seg_id, end_seg_id in merged_ranges
    ]

    merged_ranges, blocked_boundaries = sanitize_merged_ranges(
        input_segments,
        merged_ranges,
        merge_config,
    )
    merged_ranges, preserved_tail_anchors = preserve_duplicate_tail_anchors(
        input_segments,
        merged_ranges,
        merge_config,
    )

    if merged_range_count_below_ratio(
        merged_ranges,
        segment_count=input_count,
        min_output_ratio=effective_min_output_ratio,
    ):
        return None, "collapsed_too_aggressively"

    return {
        "thought": thought,
        "requested_ranges": requested_ranges,
        "ranges": list(merged_ranges),
        "blocked_boundaries": list(blocked_boundaries),
        "preserved_tail_anchors": list(preserved_tail_anchors),
        "boundary_after_seg_ids": _boundary_after_seg_ids_from_ranges(list(merged_ranges)),
        "output_segment_count": len(merged_ranges),
    }, "ok"


def _select_coarse_candidate(candidates: List[Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]], int]:
    quorum = (len(candidates) // 2) + 1
    boundary_counts = Counter(
        boundary_after_seg_id
        for candidate in candidates
        for boundary_after_seg_id in candidate.get("boundary_after_seg_ids", [])
    )
    output_segment_counts = sorted(int(candidate.get("output_segment_count", 0)) for candidate in candidates)
    median_output_segment_count = output_segment_counts[len(output_segment_counts) // 2] if output_segment_counts else 0

    scored_candidates: List[Dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        boundary_ids = [int(boundary_after_seg_id) for boundary_after_seg_id in candidate.get("boundary_after_seg_ids", [])]
        supported_boundary_count = sum(
            1 for boundary_after_seg_id in boundary_ids if boundary_counts[boundary_after_seg_id] >= quorum
        )
        boundary_count = len(boundary_ids)
        output_segment_count = int(candidate.get("output_segment_count", boundary_count + 1))
        unsupported_boundary_count = boundary_count - supported_boundary_count
        supported_boundary_ratio = (
            float(supported_boundary_count) / float(boundary_count)
            if boundary_count > 0
            else 1.0
        )
        scored_candidates.append(
            {
                "candidate_index": candidate_index,
                "output_segment_count": output_segment_count,
                "boundary_count": boundary_count,
                "supported_boundary_count": supported_boundary_count,
                "unsupported_boundary_count": unsupported_boundary_count,
                "supported_boundary_ratio": supported_boundary_ratio,
                "output_count_distance_from_median": abs(output_segment_count - median_output_segment_count),
            }
        )

    max_supported_boundary_count = max(
        int(item["supported_boundary_count"]) for item in scored_candidates
    )
    filtered_candidates = [
        item
        for item in scored_candidates
        if int(item["supported_boundary_count"]) >= (max_supported_boundary_count - 1)
    ]

    selected = max(
        filtered_candidates,
        key=lambda item: (
            int(item["supported_boundary_count"]),
            -int(item["output_count_distance_from_median"]),
            -int(item["unsupported_boundary_count"]),
            float(item["supported_boundary_ratio"]),
            -int(item["candidate_index"]),
        ),
    )
    return int(selected["candidate_index"]), scored_candidates, quorum


def _build_boundary_vote_stats(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    boundary_counts = Counter(
        boundary_after_seg_id
        for candidate in candidates
        for boundary_after_seg_id in candidate.get("boundary_after_seg_ids", [])
    )
    return [
        {
            "boundary_after_seg_id": int(boundary_after_seg_id),
            "vote_count": int(vote_count),
        }
        for boundary_after_seg_id, vote_count in sorted(boundary_counts.items())
    ]


def _build_coarse_consensus_ranges(
    candidates: List[Dict[str, Any]],
    input_count: int,
    merge_config: LLMMergeConfig,
    effective_min_output_ratio: float,
) -> Optional[Tuple[List[Tuple[int, int]], List[int]]]:
    if len(candidates) <= 1:
        return None

    boundary_counts = Counter(
        boundary_after_seg_id
        for candidate in candidates
        for boundary_after_seg_id in candidate.get("boundary_after_seg_ids", [])
    )
    vote_threshold = float(merge_config.boundary_vote_threshold)
    selected_boundary_after_seg_ids = [
        int(boundary_after_seg_id)
        for boundary_after_seg_id, vote_count in sorted(boundary_counts.items())
        if float(vote_count) > (float(len(candidates)) * vote_threshold)
    ]
    if not selected_boundary_after_seg_ids:
        return None

    consensus_ranges = _ranges_from_boundary_after_seg_ids(selected_boundary_after_seg_ids, input_count)
    if merged_range_count_below_ratio(
        consensus_ranges,
        segment_count=input_count,
        min_output_ratio=effective_min_output_ratio,
    ):
        return None
    return consensus_ranges, selected_boundary_after_seg_ids


def run_llm_merge_pass(
    sample_id: str,
    segments: List[dict],
    merge_config: LLMMergeConfig,
    backend: Any = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    del sample_id

    is_coarse = str(merge_config.granularity) == "coarse"
    input_segments = [dict(segment) for segment in segments if isinstance(segment, dict)]
    input_count = len(input_segments)
    diagnostics: Dict[str, Any] = {
        "llm_merge_enabled": bool(merge_config.enabled),
        "llm_merge_attempted": False,
        "llm_merge_applied": False,
        "llm_merge_reason": "disabled",
        "llm_merge_model": str(merge_config.model),
        "llm_merge_granularity": str(merge_config.granularity),
        "llm_merge_input_segment_count": input_count,
        "llm_merge_output_segment_count": input_count,
    }
    diagnostics["llm_merge_max_attempts"] = int(merge_config.max_attempts)
    diagnostics["llm_merge_repeat_count"] = int(merge_config.repeat_count)
    if is_coarse:
        diagnostics["llm_merge_coarse_max_supported_anchors_per_range"] = int(
            merge_config.coarse_max_supported_anchors_per_range
        )
        diagnostics["llm_merge_coarse_anchor_min_spacing_segments"] = int(
            merge_config.coarse_anchor_min_spacing_segments
        )
        diagnostics["llm_merge_coarse_anchor_min_side_segments"] = int(
            merge_config.coarse_anchor_min_side_segments
        )
        diagnostics["llm_merge_coarse_anchor_min_score"] = float(
            merge_config.coarse_anchor_min_score
        )
        diagnostics["llm_merge_boundary_vote_threshold"] = float(
            merge_config.boundary_vote_threshold
        )

    if not merge_config.enabled:
        return input_segments, diagnostics
    if input_count == 0:
        diagnostics["llm_merge_reason"] = "empty_input"
        return input_segments, diagnostics
    if input_count < int(merge_config.min_input_segments):
        diagnostics["llm_merge_reason"] = "below_min_input_segments"
        return input_segments, diagnostics
    if input_count > int(merge_config.max_input_segments):
        diagnostics["llm_merge_reason"] = "above_max_input_segments"
        return input_segments, diagnostics
    if merge_config.backend != "openai":
        diagnostics["llm_merge_reason"] = "unsupported_backend"
        return input_segments, diagnostics

    try:
        resolved_backend = backend or OpenAIBackend(
            api_key=merge_config.api_key,
            model=merge_config.model,
            base_url=merge_config.base_url,
            timeout_sec=merge_config.timeout_sec,
            organization=merge_config.organization,
            project=merge_config.project,
            reasoning_effort=merge_config.reasoning_effort,
            max_output_tokens=merge_config.max_output_tokens,
        )
    except Exception as exc:
        diagnostics["llm_merge_reason"] = f"backend_init_failed:{type(exc).__name__}"
        return input_segments, diagnostics

    diagnostics["llm_merge_attempted"] = True
    effective_min_output_ratio = _effective_min_output_ratio(merge_config)
    diagnostics["llm_merge_effective_min_output_ratio"] = effective_min_output_ratio
    prompt_boundary_hints: List[Dict[str, Any]] = []
    if is_coarse:
        if (
            merge_config.protect_supported_boundaries
            or merge_config.protect_distinct_sequence_markers
            or merge_config.protect_instruction_drift
        ):
            prompt_boundary_hints = _build_prompt_boundary_hints(input_segments, merge_config)
        diagnostics["llm_merge_prompt_boundary_hint_count"] = len(prompt_boundary_hints)
        if prompt_boundary_hints:
            diagnostics["llm_merge_prompt_boundary_hints"] = prompt_boundary_hints

    prompt = prompt_segment_merge(
        input_segments,
        granularity=str(merge_config.granularity),
        boundary_hints=prompt_boundary_hints,
    )
    valid_candidates: List[Dict[str, Any]] = []
    total_request_attempt_count = 0
    sample_request_attempt_counts: List[int] = []
    sample_reasons: List[str] = []
    adapter_diagnostics_attempts: List[Dict[str, Any]] = []
    last_reason = "empty_response"
    last_error = ""

    sample_count = int(merge_config.repeat_count) if is_coarse else 1
    for sample_index in range(max(1, sample_count)):
        payload, request_reason, request_error, request_attempt_count, request_adapter_diagnostics = _request_merge_payload(
            resolved_backend,
            prompt,
            merge_config,
        )
        for adapter_diagnostics in request_adapter_diagnostics:
            adapter_diagnostics["sample_index"] = int(sample_index) + 1
            adapter_diagnostics_attempts.append(adapter_diagnostics)
        total_request_attempt_count += int(request_attempt_count)
        sample_request_attempt_counts.append(int(request_attempt_count))

        if not payload:
            sample_reasons.append(request_reason)
            last_reason = request_reason
            if request_error:
                last_error = request_error
            if not is_coarse:
                break
            continue

        candidate, candidate_reason = _build_merge_candidate(
            input_segments,
            payload,
            merge_config,
            effective_min_output_ratio,
        )
        sample_reasons.append(candidate_reason)
        last_reason = candidate_reason
        last_error = ""
        if candidate:
            valid_candidates.append(candidate)
            if not is_coarse:
                break

    diagnostics["llm_merge_request_attempt_count"] = total_request_attempt_count
    diagnostics["llm_merge_sample_request_attempt_counts"] = sample_request_attempt_counts
    diagnostics["llm_merge_sample_reasons"] = sample_reasons
    if adapter_diagnostics_attempts:
        diagnostics["llm_merge_adapter_diagnostics_attempts"] = adapter_diagnostics_attempts
        diagnostics["llm_merge_adapter_diagnostics"] = copy.deepcopy(adapter_diagnostics_attempts[-1])

    if not valid_candidates:
        diagnostics["llm_merge_reason"] = last_reason
        if last_error:
            diagnostics["llm_merge_error"] = last_error
        return input_segments, diagnostics

    diagnostics["llm_merge_valid_candidate_count"] = len(valid_candidates)
    diagnostics["llm_merge_successful_sample_count"] = len(valid_candidates)
    diagnostics["llm_merge_candidate_output_segment_counts"] = [
        int(candidate.get("output_segment_count", input_count)) for candidate in valid_candidates
    ]
    if is_coarse:
        diagnostics["llm_merge_boundary_vote_counts"] = _build_boundary_vote_stats(valid_candidates)

    selected_candidate = valid_candidates[0]
    diagnostics["llm_merge_candidate_selection_mode"] = "first_valid_candidate"
    if is_coarse and len(valid_candidates) > 1:
        consensus_result = _build_coarse_consensus_ranges(
            valid_candidates,
            input_count,
            merge_config,
            effective_min_output_ratio,
        )
        if consensus_result is not None:
            consensus_ranges, consensus_boundary_after_seg_ids = consensus_result
            diagnostics["llm_merge_candidate_selection_mode"] = "coarse_boundary_consensus"
            diagnostics["llm_merge_consensus_boundaries"] = [
                int(boundary_after_seg_id) for boundary_after_seg_id in consensus_boundary_after_seg_ids
            ]
            selected_candidate = {
                "thought": valid_candidates[0].get("thought", ""),
                "requested_ranges": valid_candidates[0].get("requested_ranges", []),
                "ranges": consensus_ranges,
                "blocked_boundaries": [],
                "preserved_tail_anchors": [],
                "output_segment_count": len(consensus_ranges),
            }
        else:
            diagnostics["llm_merge_consensus_boundaries"] = []
        selected_candidate_index, candidate_vote_stats, candidate_consensus_quorum = _select_coarse_candidate(
            valid_candidates
        )
        if diagnostics["llm_merge_candidate_selection_mode"] != "coarse_boundary_consensus":
            diagnostics["llm_merge_candidate_selection_mode"] = "coarse_consensus_candidate"
            diagnostics["llm_merge_selected_candidate_index"] = selected_candidate_index
            diagnostics["llm_merge_candidate_consensus_quorum"] = candidate_consensus_quorum
            diagnostics["llm_merge_candidate_vote_stats"] = candidate_vote_stats
            selected_candidate = valid_candidates[selected_candidate_index]

    thought = str(selected_candidate.get("thought", "")).strip()
    if thought:
        diagnostics["llm_merge_thought"] = thought

    requested_ranges = list(selected_candidate.get("requested_ranges", []))
    diagnostics["llm_merge_requested_ranges"] = requested_ranges

    merged_ranges = list(selected_candidate.get("ranges", []))
    blocked_boundaries = list(selected_candidate.get("blocked_boundaries", []))
    preserved_tail_anchors = list(selected_candidate.get("preserved_tail_anchors", []))
    diagnostics["llm_merge_ranges"] = [
        {"start_seg_id": int(start_seg_id), "end_seg_id": int(end_seg_id)}
        for start_seg_id, end_seg_id in merged_ranges
    ]
    diagnostics["llm_merge_ranges_sanitized"] = diagnostics["llm_merge_ranges"] != requested_ranges
    if blocked_boundaries:
        diagnostics["llm_merge_blocked_boundaries"] = blocked_boundaries
    if preserved_tail_anchors:
        diagnostics["llm_merge_preserved_tail_anchors"] = preserved_tail_anchors
    diagnostics["llm_merge_output_segment_count"] = len(merged_ranges)

    if len(merged_ranges) == input_count:
        diagnostics["llm_merge_reason"] = (
            "guard_blocked_all_merges"
            if (blocked_boundaries or preserved_tail_anchors)
            else "no_merges_selected"
        )
        return input_segments, diagnostics

    merged_segments = merge_segments_by_ranges(input_segments, merged_ranges)
    if not merged_segments:
        diagnostics["llm_merge_reason"] = "merge_reconstruction_failed"
        diagnostics["llm_merge_output_segment_count"] = input_count
        return input_segments, diagnostics

    diagnostics["llm_merge_applied"] = True
    diagnostics["llm_merge_reason"] = "applied"
    diagnostics["llm_merge_output_segment_count"] = len(merged_segments)
    return merged_segments, diagnostics


def run_llm_summary_pass(
    sample_id: str,
    segments: List[dict],
    merge_config: LLMMergeConfig,
    backend: Any = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    del sample_id

    input_segments = [dict(segment) for segment in segments if isinstance(segment, dict)]
    enabled_levels = [int(value) for value in merge_config.summary_levels]
    enabled_level_names = _enabled_summary_level_names(enabled_levels)
    input_count = len(input_segments)
    diagnostics: Dict[str, Any] = {
        "llm_summary_enabled": bool(merge_config.enabled and enabled_level_names),
        "llm_summary_attempted": False,
        "llm_summary_applied": False,
        "llm_summary_fallback_used": False,
        "llm_summary_reason": "disabled",
        "llm_summary_model": str(merge_config.model),
        "llm_summary_input_segment_count": input_count,
        "llm_summary_levels": enabled_levels,
        "llm_summary_levels_named": {
            "coarse": enabled_levels[0],
            "medium": enabled_levels[1],
            "fine": enabled_levels[2],
        },
        "llm_summary_enabled_level_names": enabled_level_names,
    }

    if not merge_config.enabled:
        return None, diagnostics
    if not enabled_level_names:
        diagnostics["llm_summary_reason"] = "no_summary_levels_enabled"
        return None, diagnostics
    if input_count == 0:
        diagnostics["llm_summary_reason"] = "empty_input"
        return None, diagnostics
    if merge_config.backend != "openai":
        diagnostics["llm_summary_reason"] = "unsupported_backend"
        return None, diagnostics

    fallback_partitions = _identity_summary_partitions(input_segments, enabled_level_names)

    def _apply_summary_output(
        partitions: Dict[str, List[Dict[str, Any]]],
        *,
        applied: bool,
        reason: str,
        fallback_used: bool,
        thought: str = "",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        hierarchy = build_task_hierarchy(input_segments, partitions, enabled_levels)
        diagnostics["llm_summary_applied"] = applied
        diagnostics["llm_summary_fallback_used"] = fallback_used
        diagnostics["llm_summary_reason"] = reason
        diagnostics["llm_summary_output_level_counts"] = {
            level_name: len(partitions[level_name])
            for level_name in hierarchy.get("enabled_level_names", [])
        }
        diagnostics["llm_summary_root_level"] = str(hierarchy.get("root_level", ""))
        diagnostics["llm_summary_root_count"] = len(hierarchy.get("roots", []))
        if thought:
            diagnostics["llm_summary_thought"] = thought
        if fallback_used:
            diagnostics["llm_summary_fallback_reason"] = reason
        return hierarchy, diagnostics

    try:
        resolved_backend = backend or OpenAIBackend(
            api_key=merge_config.api_key,
            model=merge_config.model,
            base_url=merge_config.base_url,
            timeout_sec=merge_config.timeout_sec,
            organization=merge_config.organization,
            project=merge_config.project,
            reasoning_effort=merge_config.reasoning_effort,
            max_output_tokens=merge_config.max_output_tokens,
        )
    except Exception as exc:
        diagnostics["llm_summary_backend_error"] = str(exc).strip() or type(exc).__name__
        return _apply_summary_output(
            fallback_partitions,
            applied=False,
            reason=f"backend_init_failed:{type(exc).__name__}",
            fallback_used=True,
        )

    diagnostics["llm_summary_attempted"] = True
    prompt = prompt_segment_hierarchy(input_segments, enabled_level_names)
    schema = build_summary_result_schema(enabled_level_names)
    summary_request_config = _single_attempt_config(merge_config)
    payload, request_reason, request_error, request_attempt_count, adapter_diagnostics_attempts = _request_structured_payload(
        resolved_backend,
        prompt,
        summary_request_config,
        schema_name="segment_hierarchy_result",
        schema=schema,
    )
    diagnostics["llm_summary_request_attempt_count"] = int(request_attempt_count)
    if adapter_diagnostics_attempts:
        diagnostics["llm_summary_adapter_diagnostics_attempts"] = adapter_diagnostics_attempts
        diagnostics["llm_summary_adapter_diagnostics"] = copy.deepcopy(adapter_diagnostics_attempts[-1])

    if not payload:
        if request_error:
            diagnostics["llm_summary_error"] = request_error
        return _apply_summary_output(
            fallback_partitions,
            applied=False,
            reason=request_reason,
            fallback_used=True,
        )

    thought = str(payload.get("thought", "")).strip()
    partitions, validation_reason = validate_summary_partitions(
        payload,
        segment_count=input_count,
        enabled_level_names=enabled_level_names,
        segment_ids=_segment_id_sequence(input_segments),
    )
    if not partitions:
        return _apply_summary_output(
            fallback_partitions,
            applied=False,
            reason=validation_reason,
            fallback_used=True,
            thought=thought,
        )

    return _apply_summary_output(
        partitions,
        applied=True,
        reason="applied",
        fallback_used=False,
        thought=thought,
    )


def attach_stage2_subtitles_to_segments(
    segments: List[dict],
    subtitle_items: List[Dict[str, Any]],
) -> List[dict]:
    """Attach canonical Stage 2 subtitle items onto segments for app/export consumers.

    The Stage 2 envelope keeps subtitle localization separate from segment rows. The
    app still persists `segments.json` as the formal result-layer artifact, so it
    materializes `subtitles.items[]` onto `segment["export_subtitle"]` here.

    Invalid or incomplete subtitle items fall back to source instructions.
    """

    input_segments = [dict(segment) for segment in segments if isinstance(segment, dict)]
    source_subtitles = _source_instruction_subtitles(input_segments)
    payload = {
        "thought": "",
        "subtitles": [dict(item) for item in subtitle_items if isinstance(item, dict)],
    }
    subtitles, _reason = validate_subtitle_payload(
        payload,
        len(input_segments),
        segment_ids=_segment_id_sequence(input_segments),
    )
    if not subtitles:
        subtitles = source_subtitles
    return _attach_export_subtitles(input_segments, subtitles)


def run_export_subtitle_localization_pass(
    sample_id: str,
    segments: List[dict],
    merge_config: LLMMergeConfig,
    target_language: str,
    backend: Any = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    del sample_id

    input_segments = [dict(segment) for segment in segments if isinstance(segment, dict)]
    input_count = len(input_segments)
    requested_language = _normalize_language_code(target_language)
    output_language = "en"
    diagnostics: Dict[str, Any] = {
        "export_subtitle_requested_language": requested_language,
        # Actual output language for segment["export_subtitle"].
        "export_subtitle_language": output_language,
        "export_subtitle_output_language": output_language,
        "export_subtitle_attempted": False,
        "export_subtitle_applied": False,
        "export_subtitle_fallback_used": False,
        "export_subtitle_reason": "disabled",
        "export_subtitle_model": str(merge_config.model),
        "export_subtitle_segment_count": input_count,
    }

    source_subtitles = _source_instruction_subtitles(input_segments)
    if not merge_config.enabled:
        diagnostics["export_subtitle_reason"] = "disabled"
        return _attach_export_subtitles(input_segments, source_subtitles), diagnostics

    if input_count == 0:
        diagnostics["export_subtitle_reason"] = "empty_input"
        return input_segments, diagnostics

    if requested_language == "en":
        diagnostics["export_subtitle_reason"] = "source_instruction_reused"
        return _attach_export_subtitles(input_segments, source_subtitles), diagnostics

    if requested_language != "zh":
        diagnostics["export_subtitle_reason"] = "unsupported_language"
        diagnostics["export_subtitle_fallback_used"] = True
        return _attach_export_subtitles(input_segments, source_subtitles), diagnostics

    if merge_config.backend != "openai":
        diagnostics["export_subtitle_reason"] = "unsupported_backend"
        diagnostics["export_subtitle_fallback_used"] = True
        return _attach_export_subtitles(input_segments, source_subtitles), diagnostics

    try:
        resolved_backend = backend or OpenAIBackend(
            api_key=merge_config.api_key,
            model=merge_config.model,
            base_url=merge_config.base_url,
            timeout_sec=merge_config.timeout_sec,
            organization=merge_config.organization,
            project=merge_config.project,
            reasoning_effort=merge_config.reasoning_effort,
            max_output_tokens=merge_config.max_output_tokens,
        )
    except Exception as exc:
        diagnostics["export_subtitle_reason"] = f"backend_init_failed:{type(exc).__name__}"
        diagnostics["export_subtitle_error"] = str(exc).strip() or type(exc).__name__
        diagnostics["export_subtitle_fallback_used"] = True
        return _attach_export_subtitles(input_segments, source_subtitles), diagnostics

    diagnostics["export_subtitle_attempted"] = True
    prompt = prompt_segment_subtitles(input_segments, requested_language)
    subtitle_request_config = _single_attempt_config(merge_config)
    payload, request_reason, request_error, request_attempt_count, adapter_diagnostics_attempts = _request_structured_payload(
        resolved_backend,
        prompt,
        subtitle_request_config,
        schema_name="segment_subtitle_result",
        schema=_SUBTITLE_RESULT_SCHEMA,
    )
    diagnostics["export_subtitle_request_attempt_count"] = int(request_attempt_count)
    if adapter_diagnostics_attempts:
        diagnostics["export_subtitle_adapter_diagnostics_attempts"] = adapter_diagnostics_attempts
        diagnostics["export_subtitle_adapter_diagnostics"] = copy.deepcopy(adapter_diagnostics_attempts[-1])

    if not payload:
        diagnostics["export_subtitle_reason"] = request_reason
        diagnostics["export_subtitle_fallback_used"] = True
        if request_error:
            diagnostics["export_subtitle_error"] = request_error
        return _attach_export_subtitles(input_segments, source_subtitles), diagnostics

    thought = str(payload.get("thought", "")).strip()
    subtitles, validation_reason = validate_subtitle_payload(
        payload,
        input_count,
        segment_ids=_segment_id_sequence(input_segments),
    )
    if not subtitles:
        diagnostics["export_subtitle_reason"] = validation_reason
        diagnostics["export_subtitle_fallback_used"] = True
        if thought:
            diagnostics["export_subtitle_thought"] = thought
        return _attach_export_subtitles(input_segments, source_subtitles), diagnostics

    diagnostics["export_subtitle_applied"] = True
    output_language = "zh"
    diagnostics["export_subtitle_language"] = output_language
    diagnostics["export_subtitle_output_language"] = output_language
    diagnostics["export_subtitle_reason"] = "applied"
    if thought:
        diagnostics["export_subtitle_thought"] = thought
    return _attach_export_subtitles(input_segments, subtitles), diagnostics


def run_llm_postprocess_pass(
    sample_id: str,
    segments: List[dict],
    merge_config: LLMMergeConfig,
    backend: Any = None,
) -> Tuple[List[dict], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Legacy Stage 2 API kept as a compatibility facade.

    App-side orchestration should use `run_llm_stage2_pass(...)` as the canonical
    Stage 2 contract. This wrapper remains for module consumers that still expect
    the older merge+summary tuple shape.

    Returns:
      - cleaned_segments: merge output (or original segments on merge failure)
      - task_hierarchy: optional summary hierarchy
      - diagnostics: merged diagnostics from merge + summary

    Notes:
      - Summary is independent from merge: merge failures do not suppress summary.
      - Subtitle localization is intentionally not part of this legacy return shape.
    """

    cleaned_segments, merge_diagnostics = run_llm_merge_pass(
        sample_id,
        segments,
        merge_config,
        backend=backend,
    )

    task_hierarchy, summary_diagnostics = run_llm_summary_pass(
        sample_id,
        cleaned_segments,
        merge_config,
        backend=backend,
    )

    diagnostics = dict(merge_diagnostics)
    diagnostics.update(summary_diagnostics)
    return cleaned_segments, task_hierarchy, diagnostics


def run_llm_subtitle_localization_pass(
    sample_id: str,
    segments: List[dict],
    merge_config: LLMMergeConfig,
    target_language: str,
    backend: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Stage 2 subtitle localization pass.

    This is a Stage 2 artifact producer (independent of exporter). It returns a
    stable, persistable list of subtitle items aligned to the provided segments.

    Contract:
      - `items[i]["seg_id"] == i` and `items[i]["subtitle"]` is non-empty.
      - `source instruction` remains English; `language=en` reuses instructions.
    """

    del sample_id
    input_segments = [dict(segment) for segment in segments if isinstance(segment, dict)]
    input_count = len(input_segments)
    requested_language = _normalize_language_code(target_language)
    output_language = "en"
    diagnostics: Dict[str, Any] = {
        "llm_subtitle_requested_language": requested_language,
        # Actual output language for items[].subtitle. Source instructions are always English.
        "llm_subtitle_language": output_language,
        "llm_subtitle_output_language": output_language,
        "llm_subtitle_attempted": False,
        "llm_subtitle_applied": False,
        "llm_subtitle_fallback_used": False,
        "llm_subtitle_reason": "disabled",
        "llm_subtitle_model": str(merge_config.model),
        "llm_subtitle_segment_count": input_count,
    }

    source_subtitles = _source_instruction_subtitles(input_segments)

    def _items_from(subtitles: List[str]) -> List[Dict[str, Any]]:
        return [
            {"seg_id": int(index), "subtitle": str(subtitle).strip() or "Unknown task step"}
            for index, subtitle in enumerate(subtitles)
        ]

    if not merge_config.enabled:
        diagnostics["llm_subtitle_reason"] = "disabled"
        return _items_from(source_subtitles), diagnostics

    if input_count == 0:
        diagnostics["llm_subtitle_reason"] = "empty_input"
        return [], diagnostics

    if requested_language == "en":
        diagnostics["llm_subtitle_reason"] = "source_instruction_reused"
        return _items_from(source_subtitles), diagnostics

    if requested_language != "zh":
        diagnostics["llm_subtitle_reason"] = "unsupported_language"
        diagnostics["llm_subtitle_fallback_used"] = True
        return _items_from(source_subtitles), diagnostics

    if merge_config.backend != "openai":
        diagnostics["llm_subtitle_reason"] = "unsupported_backend"
        diagnostics["llm_subtitle_fallback_used"] = True
        return _items_from(source_subtitles), diagnostics

    try:
        resolved_backend = backend or OpenAIBackend(
            api_key=merge_config.api_key,
            model=merge_config.model,
            base_url=merge_config.base_url,
            timeout_sec=merge_config.timeout_sec,
            organization=merge_config.organization,
            project=merge_config.project,
            reasoning_effort=merge_config.reasoning_effort,
            max_output_tokens=merge_config.max_output_tokens,
        )
    except Exception as exc:
        diagnostics["llm_subtitle_reason"] = f"backend_init_failed:{type(exc).__name__}"
        diagnostics["llm_subtitle_error"] = str(exc).strip() or type(exc).__name__
        diagnostics["llm_subtitle_fallback_used"] = True
        return _items_from(source_subtitles), diagnostics

    diagnostics["llm_subtitle_attempted"] = True
    prompt = prompt_segment_subtitles(input_segments, requested_language)
    subtitle_request_config = _single_attempt_config(merge_config)
    payload, request_reason, request_error, request_attempt_count, adapter_diagnostics_attempts = _request_structured_payload(
        resolved_backend,
        prompt,
        subtitle_request_config,
        schema_name="segment_subtitle_result",
        schema=_SUBTITLE_RESULT_SCHEMA,
    )
    diagnostics["llm_subtitle_request_attempt_count"] = int(request_attempt_count)
    if adapter_diagnostics_attempts:
        diagnostics["llm_subtitle_adapter_diagnostics_attempts"] = adapter_diagnostics_attempts
        diagnostics["llm_subtitle_adapter_diagnostics"] = copy.deepcopy(adapter_diagnostics_attempts[-1])

    if not payload:
        diagnostics["llm_subtitle_reason"] = request_reason
        diagnostics["llm_subtitle_fallback_used"] = True
        if request_error:
            diagnostics["llm_subtitle_error"] = request_error
        return _items_from(source_subtitles), diagnostics

    thought = str(payload.get("thought", "")).strip()
    subtitles, validation_reason = validate_subtitle_payload(
        payload,
        input_count,
        segment_ids=_segment_id_sequence(input_segments),
    )
    if not subtitles:
        diagnostics["llm_subtitle_reason"] = validation_reason
        diagnostics["llm_subtitle_fallback_used"] = True
        if thought:
            diagnostics["llm_subtitle_thought"] = thought
        return _items_from(source_subtitles), diagnostics

    diagnostics["llm_subtitle_applied"] = True
    output_language = "zh"
    diagnostics["llm_subtitle_language"] = output_language
    diagnostics["llm_subtitle_output_language"] = output_language
    diagnostics["llm_subtitle_reason"] = "applied"
    if thought:
        diagnostics["llm_subtitle_thought"] = thought
    return _items_from(subtitles), diagnostics


def run_llm_stage2_pass(
    sample_id: str,
    segments: List[dict],
    merge_config: LLMMergeConfig,
    *,
    target_language: str = "en",
    backend: Any = None,
) -> Dict[str, Any]:
    """Stable Stage 2 artifact envelope.

    This is the module-side contract intended for app integration: it cleanly
    separates merge / summary / subtitle artifacts with independent diagnostics.

    Output is JSON-serializable by construction.
    """

    cleaned_segments, merge_diagnostics = run_llm_merge_pass(
        sample_id,
        segments,
        merge_config,
        backend=backend,
    )

    task_hierarchy, summary_diagnostics = run_llm_summary_pass(
        sample_id,
        cleaned_segments,
        merge_config,
        backend=backend,
    )

    subtitle_items, subtitle_diagnostics = run_llm_subtitle_localization_pass(
        sample_id,
        cleaned_segments,
        merge_config,
        target_language=target_language,
        backend=backend,
    )

    requested_language = _normalize_language_code(target_language)
    output_language = str(subtitle_diagnostics.get("llm_subtitle_language", "en")).strip().lower()
    if output_language not in ("en", "zh"):
        output_language = "en"
    return {
        "stage": "stage2",
        "version": 2,
        "merge": {
            "applied": bool(merge_diagnostics.get("llm_merge_applied", False)),
            "segments": cleaned_segments,
            "diagnostics": merge_diagnostics,
        },
        "summary": {
            "applied": bool(summary_diagnostics.get("llm_summary_applied", False)),
            "hierarchy": task_hierarchy,
            "diagnostics": summary_diagnostics,
        },
        "subtitles": {
            # Requested language is what the caller asked for (canonicalized).
            "requested_language": requested_language,
            "target_language": requested_language,
            # language/output_language always describe items[].subtitle.
            "language": output_language,
            "output_language": output_language,
            "source_instruction_language": "en",
            "applied": bool(subtitle_diagnostics.get("llm_subtitle_applied", False)),
            "items": subtitle_items,
            "diagnostics": subtitle_diagnostics,
        },
    }
