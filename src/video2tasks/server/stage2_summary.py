"""Stage 2 summary schemas, validation, and hierarchy building."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from ..config import LLMMergeConfig
from ..prompt import prompt_segment_hierarchy
from ..vlm.openai_api import OpenAIBackend
from .stage2_merge import _request_structured_payload, _segment_id_sequence, _single_attempt_config
_SUMMARY_LEVEL_NAMES = ("coarse", "medium", "fine")


def _enabled_summary_level_names(summary_levels: List[int]) -> List[str]:
    enabled_levels: List[str] = []
    for index, level_name in enumerate(_SUMMARY_LEVEL_NAMES):
        if index < len(summary_levels) and int(summary_levels[index]) == 1:
            enabled_levels.append(level_name)
    return enabled_levels


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
