"""Stage 2 subtitle localization, attachment, and envelope assembly."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from ..config import LLMMergeConfig
from ..prompt import prompt_segment_subtitles
from ..vlm.openai_api import OpenAIBackend
from .stage2_merge import _request_structured_payload, _single_attempt_config, run_llm_merge_pass
from .stage2_summary import run_llm_summary_pass
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
