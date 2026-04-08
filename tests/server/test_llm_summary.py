import json
import time
from pathlib import Path

import video2tasks.server.app as app_module
from video2tasks.config import Config, LLMMergeConfig
from video2tasks.server.app import create_app
from video2tasks.server.windowing import Window
from video2tasks.server.llm_merge import (
    run_export_subtitle_localization_pass,
    run_llm_postprocess_pass,
    run_llm_stage2_pass,
    run_llm_subtitle_localization_pass,
    run_llm_summary_pass,
    validate_summary_partitions,
)


class SequenceBackend:
    def __init__(self, responses, diagnostics_sequence=None):
        self.responses = list(responses)
        self.diagnostics_sequence = list(diagnostics_sequence or [])
        self.calls = []
        self.model = "gpt-5.2"
        self.last_text_json_diagnostics = {}

    def infer_text_json(self, prompt, *, schema_name, schema, max_output_tokens=None, reasoning_effort=None, raise_on_http_error=False):
        if self.diagnostics_sequence:
            self.last_text_json_diagnostics = dict(self.diagnostics_sequence.pop(0))
        else:
            self.last_text_json_diagnostics = {}
        self.calls.append(
            {
                "prompt": prompt,
                "schema_name": schema_name,
                "schema": schema,
                "max_output_tokens": max_output_tokens,
                "reasoning_effort": reasoning_effort,
                "raise_on_http_error": raise_on_http_error,
            }
        )
        if not self.responses:
            return {}
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _segments():
    return [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "Pick up the first item",
            "confidence": 0.9,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.2,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 20,
            "instruction": "Place the first item into the target area",
            "confidence": 0.95,
            "boundary_support_before": 0.2,
            "boundary_support_after": 0.4,
        },
        {
            "seg_id": 2,
            "start_frame": 20,
            "end_frame": 30,
            "instruction": "Pick up the second item",
            "confidence": 0.92,
            "boundary_support_before": 0.4,
            "boundary_support_after": 0.5,
        },
        {
            "seg_id": 3,
            "start_frame": 30,
            "end_frame": 40,
            "instruction": "Place the second item into the target area",
            "confidence": 0.96,
            "boundary_support_before": 0.5,
            "boundary_support_after": 0.0,
        },
    ]


def test_validate_summary_partitions_rejects_cross_level_ranges() -> None:
    partitions, reason = validate_summary_partitions(
        {
            "thought": "Bad nesting.",
            "coarse": [
                {"start_seg_id": 0, "end_seg_id": 1, "summary": "First major task"},
                {"start_seg_id": 2, "end_seg_id": 3, "summary": "Second major task"},
            ],
            "fine": [
                {"start_seg_id": 0, "end_seg_id": 2, "summary": "Crossing fine summary"},
                {"start_seg_id": 3, "end_seg_id": 3, "summary": "Last step"},
            ],
        },
        segment_count=4,
        enabled_level_names=["coarse", "fine"],
    )

    assert partitions is None
    assert reason == "cross_level_nesting_violation:coarse->fine"


def test_run_llm_summary_pass_builds_nested_task_hierarchy() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "The cleaned segments summarize cleanly across three levels.",
                "coarse": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Handle the first item"},
                    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Handle the second item"},
                ],
                "medium": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Load the first item"},
                    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Load the second item"},
                ],
                "fine": [
                    {"start_seg_id": 0, "end_seg_id": 0, "summary": "Pick up the first item"},
                    {"start_seg_id": 1, "end_seg_id": 1, "summary": "Place the first item into the target area"},
                    {"start_seg_id": 2, "end_seg_id": 2, "summary": "Pick up the second item"},
                    {"start_seg_id": 3, "end_seg_id": 3, "summary": "Place the second item into the target area"},
                ],
            }
        ]
    )
    cfg = LLMMergeConfig(enabled=True, summary_levels=[1, 1, 1])

    hierarchy, diagnostics = run_llm_summary_pass(
        "demo_sample",
        _segments(),
        cfg,
        backend=backend,
    )

    assert hierarchy is not None
    assert diagnostics["llm_summary_applied"] is True
    assert diagnostics["llm_summary_fallback_used"] is False
    assert diagnostics["llm_summary_reason"] == "applied"
    assert diagnostics["llm_summary_root_level"] == "coarse"
    assert diagnostics["llm_summary_output_level_counts"] == {"coarse": 2, "medium": 2, "fine": 4}
    assert hierarchy["enabled_levels"] == [1, 1, 1]
    assert diagnostics["llm_summary_levels_named"] == {"coarse": 1, "medium": 1, "fine": 1}
    assert hierarchy["enabled_level_names"] == ["coarse", "medium", "fine"]
    assert hierarchy["roots"][0]["summary"] == "Handle the first item"
    assert hierarchy["roots"][0]["children"][0]["summary"] == "Load the first item"
    assert hierarchy["roots"][0]["children"][0]["children"][0]["summary"] == "Pick up the first item"
    assert backend.calls[0]["schema_name"] == "segment_hierarchy_result"


def test_run_llm_summary_pass_exposes_adapter_endpoint_diagnostics() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "The cleaned segments summarize cleanly across three levels.",
                "coarse": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Handle the first item"},
                    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Handle the second item"},
                ],
                "medium": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Load the first item"},
                    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Load the second item"},
                ],
                "fine": [
                    {"start_seg_id": 0, "end_seg_id": 0, "summary": "Pick up the first item"},
                    {"start_seg_id": 1, "end_seg_id": 1, "summary": "Place the first item into the target area"},
                    {"start_seg_id": 2, "end_seg_id": 2, "summary": "Pick up the second item"},
                    {"start_seg_id": 3, "end_seg_id": 3, "summary": "Place the second item into the target area"},
                ],
            }
        ],
        diagnostics_sequence=[
            {
                "parsed_endpoint": "chat_completions",
                "final_failure_reason": None,
                "responses": {
                    "called": True,
                    "request_succeeded": True,
                    "http_status_code": 200,
                    "json_received": True,
                    "body_shape": "dict",
                    "top_level_keys": ["output"],
                    "structured_payload_found": False,
                    "failure_reason": "body_shape_mismatch",
                },
                "chat_completions": {
                    "called": True,
                    "request_succeeded": True,
                    "http_status_code": 200,
                    "json_received": True,
                    "body_shape": "dict",
                    "top_level_keys": ["choices"],
                    "structured_payload_found": True,
                    "failure_reason": None,
                },
            }
        ],
    )
    cfg = LLMMergeConfig(enabled=True, summary_levels=[1, 1, 1])

    _, diagnostics = run_llm_summary_pass(
        "demo_sample",
        _segments(),
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_summary_adapter_diagnostics"]["parsed_endpoint"] == "chat_completions"
    assert diagnostics["llm_summary_adapter_diagnostics"]["responses"]["failure_reason"] == "body_shape_mismatch"
    assert diagnostics["llm_summary_adapter_diagnostics"]["chat_completions"]["structured_payload_found"] is True
    assert diagnostics["llm_summary_adapter_diagnostics_attempts"][0]["request_attempt_index"] == 1


def test_run_llm_summary_pass_falls_back_to_identity_hierarchy_on_invalid_partition() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "This partition crosses levels.",
                "coarse": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "First major task"},
                    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Second major task"},
                ],
                "fine": [
                    {"start_seg_id": 0, "end_seg_id": 2, "summary": "Crossing fine task"},
                    {"start_seg_id": 3, "end_seg_id": 3, "summary": "Last step"},
                ],
            }
        ]
    )
    cfg = LLMMergeConfig(enabled=True, summary_levels=[1, 0, 1])

    hierarchy, diagnostics = run_llm_summary_pass(
        "demo_sample",
        _segments(),
        cfg,
        backend=backend,
    )

    assert hierarchy is not None
    assert diagnostics["llm_summary_applied"] is False
    assert diagnostics["llm_summary_fallback_used"] is True
    assert diagnostics["llm_summary_fallback_reason"] == "cross_level_nesting_violation:coarse->fine"
    assert diagnostics["llm_summary_output_level_counts"] == {"coarse": 4, "fine": 4}
    assert diagnostics["llm_summary_levels_named"] == {"coarse": 1, "medium": 0, "fine": 1}
    assert hierarchy["roots"][0]["summary"] == "Pick up the first item"
    assert hierarchy["root_level"] == "coarse"


def test_run_llm_postprocess_pass_runs_merge_then_summary() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "Merge the first two fragments.",
                "merged_ranges": [
                    {"start_seg_id": 0, "end_seg_id": 1},
                    {"start_seg_id": 2, "end_seg_id": 2},
                    {"start_seg_id": 3, "end_seg_id": 3},
                ],
            },
            {
                "thought": "Summarize the cleaned segments at two levels.",
                "medium": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Handle the first loaded item"},
                    {"start_seg_id": 2, "end_seg_id": 2, "summary": "Handle the remaining item"},
                ],
                "fine": [
                    {"start_seg_id": 0, "end_seg_id": 0, "summary": "Place the first item into the target area"},
                    {"start_seg_id": 1, "end_seg_id": 1, "summary": "Pick up the second item"},
                    {"start_seg_id": 2, "end_seg_id": 2, "summary": "Place the second item into the target area"},
                ],
            },
        ]
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=4,
        summary_levels=[0, 1, 1],
        protect_supported_boundaries=False,
        protect_distinct_sequence_markers=False,
        protect_instruction_drift=False,
        protect_duplicate_tail_anchor=False,
    )

    cleaned_segments, task_hierarchy, diagnostics = run_llm_postprocess_pass(
        "demo_sample",
        _segments(),
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_summary_applied"] is True
    assert len(cleaned_segments) == 3
    assert cleaned_segments[0]["start_frame"] == 0
    assert cleaned_segments[0]["end_frame"] == 20
    assert task_hierarchy is not None
    assert task_hierarchy["enabled_level_names"] == ["medium", "fine"]
    assert task_hierarchy["root_level"] == "medium"
    assert task_hierarchy["roots"][0]["children"][0]["summary"] == "Place the first item into the target area"
    assert [call["schema_name"] for call in backend.calls] == ["segment_merge_result", "segment_hierarchy_result"]


def test_run_llm_postprocess_pass_runs_summary_even_when_merge_request_fails() -> None:
    backend = SequenceBackend(
        [
            RuntimeError("merge backend timeout"),
            RuntimeError("merge backend timeout"),
            RuntimeError("merge backend timeout"),
            {
                "thought": "Summarize even though merge failed; input segments are still usable.",
                "coarse": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Handle the first item"},
                    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Handle the second item"},
                ],
                "medium": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Load the first item"},
                    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Load the second item"},
                ],
                "fine": [
                    {"start_seg_id": 0, "end_seg_id": 0, "summary": "Pick up the first item"},
                    {"start_seg_id": 1, "end_seg_id": 1, "summary": "Place the first item into the target area"},
                    {"start_seg_id": 2, "end_seg_id": 2, "summary": "Pick up the second item"},
                    {"start_seg_id": 3, "end_seg_id": 3, "summary": "Place the second item into the target area"},
                ],
            },
        ]
    )
    cfg = LLMMergeConfig(enabled=True, min_input_segments=4, summary_levels=[1, 1, 1])

    cleaned_segments, task_hierarchy, diagnostics = run_llm_postprocess_pass(
        "demo_sample",
        _segments(),
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is False
    assert diagnostics["llm_merge_reason"] == "request_failed:RuntimeError"
    assert diagnostics["llm_summary_attempted"] is True
    assert diagnostics["llm_summary_applied"] is True
    assert diagnostics["llm_summary_fallback_used"] is False
    assert diagnostics["llm_summary_reason"] == "applied"
    assert task_hierarchy is not None
    assert task_hierarchy["enabled_level_names"] == ["coarse", "medium", "fine"]
    assert len(task_hierarchy["roots"]) == 2
    assert len(cleaned_segments) == 4
    assert [call["schema_name"] for call in backend.calls] == [
        "segment_merge_result",
        "segment_merge_result",
        "segment_merge_result",
        "segment_hierarchy_result",
    ]


def test_run_llm_summary_pass_uses_single_remote_attempt_before_fallback() -> None:
    backend = SequenceBackend([RuntimeError("summary timeout")])
    cfg = LLMMergeConfig(enabled=True, summary_levels=[1, 1, 1], max_attempts=3)

    hierarchy, diagnostics = run_llm_summary_pass(
        "demo_sample",
        _segments(),
        cfg,
        backend=backend,
    )

    assert hierarchy is not None
    assert diagnostics["llm_summary_attempted"] is True
    assert diagnostics["llm_summary_applied"] is False
    assert diagnostics["llm_summary_fallback_used"] is True
    assert diagnostics["llm_summary_reason"] == "request_failed:RuntimeError"
    assert diagnostics["llm_summary_request_attempt_count"] == 1
    assert len(backend.calls) == 1


def test_run_export_subtitle_localization_pass_localizes_to_chinese() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "The finalized instructions can be localized one by one.",
                "subtitles": [
                    {"seg_id": 0, "subtitle": "拿起第一个物体"},
                    {"seg_id": 1, "subtitle": "将第一个物体放入目标区域"},
                    {"seg_id": 2, "subtitle": "拿起第二个物体"},
                    {"seg_id": 3, "subtitle": "将第二个物体放入目标区域"},
                ],
            }
        ]
    )
    cfg = LLMMergeConfig(enabled=True)

    localized_segments, diagnostics = run_export_subtitle_localization_pass(
        "demo_sample",
        _segments(),
        cfg,
        target_language="zh",
        backend=backend,
    )

    assert diagnostics["export_subtitle_attempted"] is True
    assert diagnostics["export_subtitle_applied"] is True
    assert diagnostics["export_subtitle_reason"] == "applied"
    assert localized_segments[0]["instruction"] == "Pick up the first item"
    assert localized_segments[0]["export_subtitle"] == "拿起第一个物体"
    assert localized_segments[3]["export_subtitle"] == "将第二个物体放入目标区域"
    assert backend.calls[0]["schema_name"] == "segment_subtitle_result"


def test_run_export_subtitle_localization_pass_exposes_adapter_endpoint_diagnostics() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "The finalized instructions can be localized one by one.",
                "subtitles": [
                    {"seg_id": 0, "subtitle": "拿起第一个物体"},
                    {"seg_id": 1, "subtitle": "将第一个物体放入目标区域"},
                    {"seg_id": 2, "subtitle": "拿起第二个物体"},
                    {"seg_id": 3, "subtitle": "将第二个物体放入目标区域"},
                ],
            }
        ],
        diagnostics_sequence=[
            {
                "parsed_endpoint": "responses",
                "final_failure_reason": None,
                "responses": {
                    "called": True,
                    "request_succeeded": True,
                    "http_status_code": 200,
                    "json_received": True,
                    "body_shape": "dict",
                    "top_level_keys": ["output", "output_text"],
                    "structured_payload_found": True,
                    "failure_reason": None,
                },
                "chat_completions": {
                    "called": False,
                    "request_succeeded": False,
                    "http_status_code": None,
                    "json_received": False,
                    "body_shape": "not_called",
                    "top_level_keys": [],
                    "structured_payload_found": False,
                    "failure_reason": None,
                },
            }
        ],
    )
    cfg = LLMMergeConfig(enabled=True)

    _, diagnostics = run_export_subtitle_localization_pass(
        "demo_sample",
        _segments(),
        cfg,
        target_language="zh",
        backend=backend,
    )

    assert diagnostics["export_subtitle_adapter_diagnostics"]["parsed_endpoint"] == "responses"
    assert diagnostics["export_subtitle_adapter_diagnostics"]["responses"]["structured_payload_found"] is True
    assert diagnostics["export_subtitle_adapter_diagnostics"]["chat_completions"]["called"] is False
    assert diagnostics["export_subtitle_adapter_diagnostics_attempts"][0]["request_attempt_index"] == 1


def test_run_export_subtitle_localization_pass_reuses_source_instruction_for_english() -> None:
    backend = SequenceBackend([])
    cfg = LLMMergeConfig(enabled=True)

    localized_segments, diagnostics = run_export_subtitle_localization_pass(
        "demo_sample",
        _segments(),
        cfg,
        target_language="en",
        backend=backend,
    )

    assert diagnostics["export_subtitle_attempted"] is False
    assert diagnostics["export_subtitle_applied"] is False
    assert diagnostics["export_subtitle_reason"] == "source_instruction_reused"
    assert localized_segments[0]["export_subtitle"] == "Pick up the first item"
    assert backend.calls == []



def test_run_export_subtitle_localization_pass_short_circuits_when_merge_disabled() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "Should not be called when Stage 2 is disabled.",
                "subtitles": [
                    {"seg_id": 0, "subtitle": "拿起第一个物体"},
                    {"seg_id": 1, "subtitle": "将第一个物体放入目标区域"},
                    {"seg_id": 2, "subtitle": "拿起第二个物体"},
                    {"seg_id": 3, "subtitle": "将第二个物体放入目标区域"},
                ],
            }
        ]
    )
    cfg = LLMMergeConfig(enabled=False)

    localized_segments, diagnostics = run_export_subtitle_localization_pass(
        "demo_sample",
        _segments(),
        cfg,
        target_language="zh",
        backend=backend,
    )

    assert diagnostics["export_subtitle_attempted"] is False
    assert diagnostics["export_subtitle_applied"] is False
    assert diagnostics["export_subtitle_reason"] == "disabled"
    assert localized_segments[0]["export_subtitle"] == "Pick up the first item"
    assert backend.calls == []


def test_run_llm_subtitle_localization_pass_short_circuits_when_stage2_disabled() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "Should not be called when Stage 2 is disabled.",
                "subtitles": [
                    {"seg_id": 0, "subtitle": "拿起第一个物体"},
                    {"seg_id": 1, "subtitle": "将第一个物体放入目标区域"},
                    {"seg_id": 2, "subtitle": "拿起第二个物体"},
                    {"seg_id": 3, "subtitle": "将第二个物体放入目标区域"},
                ],
            }
        ]
    )
    cfg = LLMMergeConfig(enabled=False)

    items, diagnostics = run_llm_subtitle_localization_pass(
        "demo_sample",
        _segments(),
        cfg,
        target_language="zh",
        backend=backend,
    )

    assert diagnostics["llm_subtitle_attempted"] is False
    assert diagnostics["llm_subtitle_applied"] is False
    assert diagnostics["llm_subtitle_reason"] == "disabled"
    assert items[0]["seg_id"] == 0
    assert items[0]["subtitle"] == "Pick up the first item"
    assert backend.calls == []


def test_run_llm_subtitle_helpers_agree_on_disabled_empty_input_reason() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "Should not be called on empty input.",
                "subtitles": [
                    {"seg_id": 0, "subtitle": "不应被调用"},
                ],
            }
        ]
    )
    cfg = LLMMergeConfig(enabled=False)

    export_segments, export_diagnostics = run_export_subtitle_localization_pass(
        "demo_sample",
        [],
        cfg,
        target_language="zh",
        backend=backend,
    )

    llm_items, llm_diagnostics = run_llm_subtitle_localization_pass(
        "demo_sample",
        [],
        cfg,
        target_language="zh",
        backend=backend,
    )

    assert export_segments == []
    assert export_diagnostics["export_subtitle_reason"] == "disabled"

    assert llm_items == []
    assert llm_diagnostics["llm_subtitle_reason"] == "disabled"

    assert backend.calls == []


def test_run_llm_subtitle_localization_pass_localizes_to_chinese() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "The finalized instructions can be localized one by one.",
                "subtitles": [
                    {"seg_id": 0, "subtitle": "拿起第一个物体"},
                    {"seg_id": 1, "subtitle": "将第一个物体放入目标区域"},
                    {"seg_id": 2, "subtitle": "拿起第二个物体"},
                    {"seg_id": 3, "subtitle": "将第二个物体放入目标区域"},
                ],
            }
        ]
    )
    cfg = LLMMergeConfig(enabled=True)

    items, diagnostics = run_llm_subtitle_localization_pass(
        "demo_sample",
        _segments(),
        cfg,
        target_language="zh",
        backend=backend,
    )

    assert diagnostics["llm_subtitle_attempted"] is True
    assert diagnostics["llm_subtitle_applied"] is True
    assert diagnostics["llm_subtitle_reason"] == "applied"
    assert items[0] == {"seg_id": 0, "subtitle": "拿起第一个物体"}
    assert backend.calls[0]["schema_name"] == "segment_subtitle_result"


def test_run_export_subtitle_localization_pass_uses_single_remote_attempt_before_fallback() -> None:
    backend = SequenceBackend([RuntimeError("subtitle timeout")])
    cfg = LLMMergeConfig(enabled=True, max_attempts=3)

    localized_segments, diagnostics = run_export_subtitle_localization_pass(
        "demo_sample",
        _segments(),
        cfg,
        target_language="zh",
        backend=backend,
    )

    assert diagnostics["export_subtitle_attempted"] is True
    assert diagnostics["export_subtitle_applied"] is False
    assert diagnostics["export_subtitle_fallback_used"] is True
    assert diagnostics["export_subtitle_reason"] == "request_failed:RuntimeError"
    assert diagnostics["export_subtitle_request_attempt_count"] == 1
    assert localized_segments[0]["export_subtitle"] == "Pick up the first item"
    assert len(backend.calls) == 1


def test_run_llm_stage2_pass_persists_localized_subtitles_as_formal_stage2_artifact() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "No merge needed.",
                "merged_ranges": [
                    {"start_seg_id": 0, "end_seg_id": 0},
                    {"start_seg_id": 1, "end_seg_id": 1},
                    {"start_seg_id": 2, "end_seg_id": 2},
                    {"start_seg_id": 3, "end_seg_id": 3},
                ],
            },
            {
                "thought": "The segments summarize cleanly across three levels.",
                "coarse": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Handle the first item"},
                    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Handle the second item"},
                ],
                "medium": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Load the first item"},
                    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Load the second item"},
                ],
                "fine": [
                    {"start_seg_id": 0, "end_seg_id": 0, "summary": "Pick up the first item"},
                    {"start_seg_id": 1, "end_seg_id": 1, "summary": "Place the first item into the target area"},
                    {"start_seg_id": 2, "end_seg_id": 2, "summary": "Pick up the second item"},
                    {"start_seg_id": 3, "end_seg_id": 3, "summary": "Place the second item into the target area"},
                ],
            },
            {
                "thought": "The finalized instructions can be localized one by one.",
                "subtitles": [
                    {"seg_id": 0, "subtitle": "拿起第一个物体"},
                    {"seg_id": 1, "subtitle": "将第一个物体放入目标区域"},
                    {"seg_id": 2, "subtitle": "拿起第二个物体"},
                    {"seg_id": 3, "subtitle": "将第二个物体放入目标区域"},
                ],
            },
        ]
    )
    cfg = LLMMergeConfig(enabled=True, min_input_segments=4, summary_levels=[1, 1, 1], max_attempts=1)

    result = run_llm_stage2_pass(
        "demo_sample",
        _segments(),
        cfg,
        target_language="zh",
        backend=backend,
    )

    assert result["stage"] == "stage2"
    assert result["version"] == 2
    assert result["merge"]["segments"][0]["instruction"] == "Pick up the first item"
    assert result["merge"]["diagnostics"]["llm_merge_attempted"] is True
    assert result["summary"]["applied"] is True
    assert result["summary"]["hierarchy"]["roots"][0]["summary"] == "Handle the first item"
    assert result["subtitles"]["requested_language"] == "zh"
    assert result["subtitles"]["language"] == "zh"
    assert result["subtitles"]["output_language"] == "zh"
    assert result["subtitles"]["applied"] is True
    assert result["subtitles"]["items"][0]["subtitle"] == "拿起第一个物体"
    assert [call["schema_name"] for call in backend.calls] == [
        "segment_merge_result",
        "segment_hierarchy_result",
        "segment_subtitle_result",
    ]


def test_run_llm_stage2_pass_runs_summary_and_subtitles_when_merge_fails() -> None:
    backend = SequenceBackend(
        [
            RuntimeError("merge backend timeout"),
            RuntimeError("merge backend timeout"),
            RuntimeError("merge backend timeout"),
            {
                "thought": "Summarize even though merge failed; input segments are still usable.",
                "coarse": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Handle the first item"},
                    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Handle the second item"},
                ],
                "medium": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Load the first item"},
                    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Load the second item"},
                ],
                "fine": [
                    {"start_seg_id": 0, "end_seg_id": 0, "summary": "Pick up the first item"},
                    {"start_seg_id": 1, "end_seg_id": 1, "summary": "Place the first item into the target area"},
                    {"start_seg_id": 2, "end_seg_id": 2, "summary": "Pick up the second item"},
                    {"start_seg_id": 3, "end_seg_id": 3, "summary": "Place the second item into the target area"},
                ],
            },
            {
                "thought": "The finalized instructions can be localized one by one.",
                "subtitles": [
                    {"seg_id": 0, "subtitle": "拿起第一个物体"},
                    {"seg_id": 1, "subtitle": "将第一个物体放入目标区域"},
                    {"seg_id": 2, "subtitle": "拿起第二个物体"},
                    {"seg_id": 3, "subtitle": "将第二个物体放入目标区域"},
                ],
            },
        ]
    )
    cfg = LLMMergeConfig(enabled=True, min_input_segments=4, summary_levels=[1, 1, 1], max_attempts=3)

    result = run_llm_stage2_pass(
        "demo_sample",
        _segments(),
        cfg,
        target_language="zh",
        backend=backend,
    )

    assert result["merge"]["applied"] is False
    assert result["merge"]["diagnostics"]["llm_merge_reason"] == "request_failed:RuntimeError"

    assert result["summary"]["applied"] is True
    assert result["summary"]["hierarchy"]["roots"][0]["summary"] == "Handle the first item"
    assert result["summary"]["diagnostics"]["llm_summary_attempted"] is True

    assert result["subtitles"]["requested_language"] == "zh"
    assert result["subtitles"]["language"] == "zh"
    assert result["subtitles"]["output_language"] == "zh"
    assert result["subtitles"]["applied"] is True
    assert result["subtitles"]["items"][0] == {"seg_id": 0, "subtitle": "拿起第一个物体"}
    assert result["subtitles"]["diagnostics"]["llm_subtitle_attempted"] is True

    assert [call["schema_name"] for call in backend.calls] == [
        "segment_merge_result",
        "segment_merge_result",
        "segment_merge_result",
        "segment_hierarchy_result",
        "segment_subtitle_result",
    ]


def test_stage2_language_contract_canonicalizes_aliases_and_keeps_source_instruction_english() -> None:
    """Lock Stage 2 language semantics:

    - target_language aliases like zh-CN should be treated as zh for subtitle localization.
    - `language` refers to subtitle output language, not the instruction language.
    - source instructions remain unchanged (English contract).
    """

    backend = SequenceBackend(
        [
            {
                "thought": "No merge needed.",
                "merged_ranges": [
                    {"start_seg_id": 0, "end_seg_id": 0},
                    {"start_seg_id": 1, "end_seg_id": 1},
                    {"start_seg_id": 2, "end_seg_id": 2},
                    {"start_seg_id": 3, "end_seg_id": 3},
                ],
            },
            {
                "thought": "The segments summarize cleanly across three levels.",
                "coarse": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Handle the first item"},
                    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Handle the second item"},
                ],
                "medium": [
                    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Load the first item"},
                    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Load the second item"},
                ],
                "fine": [
                    {"start_seg_id": 0, "end_seg_id": 0, "summary": "Pick up the first item"},
                    {"start_seg_id": 1, "end_seg_id": 1, "summary": "Place the first item into the target area"},
                    {"start_seg_id": 2, "end_seg_id": 2, "summary": "Pick up the second item"},
                    {"start_seg_id": 3, "end_seg_id": 3, "summary": "Place the second item into the target area"},
                ],
            },
            {
                "thought": "The finalized instructions can be localized one by one.",
                "subtitles": [
                    {"seg_id": 0, "subtitle": "拿起第一个物体"},
                    {"seg_id": 1, "subtitle": "将第一个物体放入目标区域"},
                    {"seg_id": 2, "subtitle": "拿起第二个物体"},
                    {"seg_id": 3, "subtitle": "将第二个物体放入目标区域"},
                ],
            },
        ]
    )
    cfg = LLMMergeConfig(enabled=True, min_input_segments=4, summary_levels=[1, 1, 1], max_attempts=1)

    result = run_llm_stage2_pass(
        "demo_sample",
        _segments(),
        cfg,
        target_language="zh-CN",
        backend=backend,
    )

    assert result["merge"]["segments"][0]["instruction"] == "Pick up the first item"
    assert result["subtitles"]["requested_language"] == "zh"
    assert result["subtitles"]["language"] == "zh"
    assert result["subtitles"]["output_language"] == "zh"
    assert result["subtitles"]["target_language"] == "zh"
    assert result["subtitles"]["source_instruction_language"] == "en"
    assert result["subtitles"]["items"][0]["subtitle"] == "拿起第一个物体"
    assert [call["schema_name"] for call in backend.calls] == [
        "segment_merge_result",
        "segment_hierarchy_result",
        "segment_subtitle_result",
    ]


def test_llm_subtitle_localization_pass_treats_en_us_as_english_reuse() -> None:
    backend = SequenceBackend([])
    cfg = LLMMergeConfig(enabled=True)

    items, diagnostics = run_llm_subtitle_localization_pass(
        "demo_sample",
        _segments(),
        cfg,
        target_language="en-US",
        backend=backend,
    )

    assert diagnostics["llm_subtitle_requested_language"] == "en"
    assert diagnostics["llm_subtitle_language"] == "en"
    assert diagnostics["llm_subtitle_output_language"] == "en"
    assert diagnostics["llm_subtitle_reason"] == "source_instruction_reused"
    assert items[0]["subtitle"] == "Pick up the first item"
    assert backend.calls == []


def test_stage2_subtitle_language_never_mismatches_output_items() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "Should never be called when Stage 2 is disabled.",
                "subtitles": [
                    {"seg_id": 0, "subtitle": "不应被调用"},
                ],
            }
        ]
    )
    cfg = LLMMergeConfig(enabled=False)

    result = run_llm_stage2_pass(
        "demo_sample",
        _segments(),
        cfg,
        target_language="zh",
        backend=backend,
    )

    assert result["subtitles"]["requested_language"] == "zh"
    # Actual output remains English because Stage 2 is disabled.
    assert result["subtitles"]["language"] == "en"
    assert result["subtitles"]["output_language"] == "en"
    assert result["subtitles"]["items"][0]["subtitle"] == "Pick up the first item"
    assert backend.calls == []


def test_stage2_subtitles_fallback_marks_output_language_en() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "No merge needed.",
                "merged_ranges": [
                    {"start_seg_id": 0, "end_seg_id": 0},
                    {"start_seg_id": 1, "end_seg_id": 1},
                    {"start_seg_id": 2, "end_seg_id": 2},
                    {"start_seg_id": 3, "end_seg_id": 3},
                ],
            },
            {
                "thought": "Summarize at all levels.",
                "coarse": [
                    {"start_seg_id": 0, "end_seg_id": 3, "summary": "Handle both items"},
                ],
                "medium": [
                    {"start_seg_id": 0, "end_seg_id": 3, "summary": "Process the items"},
                ],
                "fine": [
                    {"start_seg_id": 0, "end_seg_id": 0, "summary": "Pick up the first item"},
                    {"start_seg_id": 1, "end_seg_id": 1, "summary": "Place the first item into the target area"},
                    {"start_seg_id": 2, "end_seg_id": 2, "summary": "Pick up the second item"},
                    {"start_seg_id": 3, "end_seg_id": 3, "summary": "Place the second item into the target area"},
                ],
            },
            RuntimeError("subtitle timeout"),
        ]
    )
    cfg = LLMMergeConfig(enabled=True, min_input_segments=4, summary_levels=[1, 1, 1], max_attempts=3)

    result = run_llm_stage2_pass(
        "demo_sample",
        _segments(),
        cfg,
        target_language="zh",
        backend=backend,
    )

    assert result["subtitles"]["requested_language"] == "zh"
    assert result["subtitles"]["language"] == "en"
    assert result["subtitles"]["output_language"] == "en"
    assert result["subtitles"]["applied"] is False
    assert result["subtitles"]["items"][0]["subtitle"] == "Pick up the first item"


def _segments_with_offset_seg_ids() -> list[dict]:
    base = _segments()
    offset = []
    for index, segment in enumerate(base):
        updated = dict(segment)
        updated["seg_id"] = 10 + index
        offset.append(updated)
    return offset


def test_run_llm_summary_pass_fallback_identity_hierarchy_handles_non_zero_based_seg_id() -> None:
    backend = SequenceBackend([])
    cfg = LLMMergeConfig(enabled=True, summary_levels=[1, 0, 0], max_attempts=1)

    hierarchy, diagnostics = run_llm_summary_pass(
        "demo_sample",
        _segments_with_offset_seg_ids(),
        cfg,
        backend=backend,
    )

    assert hierarchy is not None
    assert diagnostics["llm_summary_attempted"] is True
    assert diagnostics["llm_summary_applied"] is False
    assert diagnostics["llm_summary_fallback_used"] is True
    assert diagnostics["llm_summary_reason"] == "empty_response"
    assert hierarchy["root_level"] == "coarse"
    assert len(hierarchy["roots"]) == 4
    assert hierarchy["roots"][0]["start_frame"] == 0


def test_run_llm_summary_pass_accepts_payload_using_input_seg_id_tokens() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "Use input seg_id tokens rather than 0-based indices.",
                "coarse": [
                    {"start_seg_id": 10, "end_seg_id": 11, "summary": "Handle the first item"},
                    {"start_seg_id": 12, "end_seg_id": 13, "summary": "Handle the second item"},
                ],
                "fine": [
                    {"start_seg_id": 10, "end_seg_id": 10, "summary": "Pick up the first item"},
                    {"start_seg_id": 11, "end_seg_id": 11, "summary": "Place the first item into the target area"},
                    {"start_seg_id": 12, "end_seg_id": 12, "summary": "Pick up the second item"},
                    {"start_seg_id": 13, "end_seg_id": 13, "summary": "Place the second item into the target area"},
                ],
            }
        ]
    )
    cfg = LLMMergeConfig(enabled=True, summary_levels=[1, 0, 1], max_attempts=1)

    hierarchy, diagnostics = run_llm_summary_pass(
        "demo_sample",
        _segments_with_offset_seg_ids(),
        cfg,
        backend=backend,
    )

    assert hierarchy is not None
    assert diagnostics["llm_summary_applied"] is True
    assert hierarchy["root_level"] == "coarse"
    assert len(hierarchy["roots"]) == 2
    assert hierarchy["roots"][0]["summary"] == "Handle the first item"


def test_run_llm_subtitle_localization_pass_accepts_payload_using_input_seg_id_tokens() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "Return subtitles keyed by input seg_id tokens.",
                "subtitles": [
                    {"seg_id": 13, "subtitle": "将第二个物体放入目标区域"},
                    {"seg_id": 12, "subtitle": "拿起第二个物体"},
                    {"seg_id": 11, "subtitle": "将第一个物体放入目标区域"},
                    {"seg_id": 10, "subtitle": "拿起第一个物体"},
                ],
            }
        ]
    )
    cfg = LLMMergeConfig(enabled=True, max_attempts=1)

    items, diagnostics = run_llm_subtitle_localization_pass(
        "demo_sample",
        _segments_with_offset_seg_ids(),
        cfg,
        target_language="zh",
        backend=backend,
    )

    assert diagnostics["llm_subtitle_applied"] is True
    assert diagnostics["llm_subtitle_requested_language"] == "zh"
    assert diagnostics["llm_subtitle_language"] == "zh"
    assert items[0] == {"seg_id": 0, "subtitle": "拿起第一个物体"}


def _segments_with_one_based_seg_ids() -> list[dict]:
    base = _segments()
    one_based = []
    for index, segment in enumerate(base):
        updated = dict(segment)
        updated["seg_id"] = 1 + index
        one_based.append(updated)
    return one_based


def test_run_llm_summary_pass_accepts_payload_using_one_based_seg_id_tokens() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "Use 1-based seg_id tokens.",
                "coarse": [
                    {"start_seg_id": 1, "end_seg_id": 2, "summary": "Handle the first item"},
                    {"start_seg_id": 3, "end_seg_id": 4, "summary": "Handle the second item"},
                ],
                "fine": [
                    {"start_seg_id": 1, "end_seg_id": 1, "summary": "Pick up the first item"},
                    {"start_seg_id": 2, "end_seg_id": 2, "summary": "Place the first item into the target area"},
                    {"start_seg_id": 3, "end_seg_id": 3, "summary": "Pick up the second item"},
                    {"start_seg_id": 4, "end_seg_id": 4, "summary": "Place the second item into the target area"},
                ],
            }
        ]
    )
    cfg = LLMMergeConfig(enabled=True, summary_levels=[1, 0, 1], max_attempts=1)

    hierarchy, diagnostics = run_llm_summary_pass(
        "demo_sample",
        _segments_with_one_based_seg_ids(),
        cfg,
        backend=backend,
    )

    assert hierarchy is not None
    assert diagnostics["llm_summary_applied"] is True
    assert len(hierarchy["roots"]) == 2
    assert hierarchy["roots"][0]["summary"] == "Handle the first item"


def test_run_llm_subtitle_localization_pass_accepts_positional_payload_for_non_zero_based_seg_id() -> None:
    backend = SequenceBackend(
        [
            {
                "thought": "Return subtitles keyed by positional indices.",
                "subtitles": [
                    {"seg_id": 3, "subtitle": "将第二个物体放入目标区域"},
                    {"seg_id": 2, "subtitle": "拿起第二个物体"},
                    {"seg_id": 1, "subtitle": "将第一个物体放入目标区域"},
                    {"seg_id": 0, "subtitle": "拿起第一个物体"},
                ],
            }
        ]
    )
    cfg = LLMMergeConfig(enabled=True, max_attempts=1)

    items, diagnostics = run_llm_subtitle_localization_pass(
        "demo_sample",
        _segments_with_offset_seg_ids(),
        cfg,
        target_language="zh",
        backend=backend,
    )

    assert diagnostics["llm_subtitle_applied"] is True
    assert diagnostics["llm_subtitle_language"] == "zh"
    assert items[0] == {"seg_id": 0, "subtitle": "拿起第一个物体"}



class _NoopFrameExtractorForAppStage2:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_many_b64_with_artifacts(self, *_args, **_kwargs):
        return [], None


def _wait_until(predicate, *, timeout: float = 3.0, interval: float = 0.02) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    assert predicate()


def _make_stage2_writeback_app(tmp_path: Path, monkeypatch, *, subtitle_pass):
    subset = "demo"
    sample_id = "sample"

    data_root = tmp_path / "data"
    sample_dir = data_root / subset / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "Frame_demo.mp4").write_bytes(b"not-a-real-video")

    sample_out_dir = tmp_path / subset / "testrun" / "samples" / sample_id
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    (sample_out_dir / "windows.jsonl").write_text(
        json.dumps(
            {
                "task_id": "demo::sample_w0_r0",
                "dispatch_id": "d1",
                "window_id": 0,
                "repeat_index": 0,
                "logical_frame_count": 4,
                "vlm_json": {"transitions": [1], "instructions": ["Pick up bowl", "Place bowl"]},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "read_video_info", lambda _mp4: (30.0, 16))
    monkeypatch.setattr(
        app_module,
        "build_windows",
        lambda *_args, **_kwargs: [Window(window_id=0, start_frame=0, end_frame=15, frame_ids=[0, 5, 10, 15])],
    )
    monkeypatch.setattr(
        app_module,
        "build_segments_via_cuts",
        lambda *_args, **_kwargs: {
            "segments": [
                {
                    "seg_id": 0,
                    "start_frame": 0,
                    "end_frame": 15,
                    "instruction": "Pick up bowl",
                }
            ]
        },
    )
    monkeypatch.setattr(
        app_module,
        "run_llm_postprocess_pass",
        lambda _sid, segments, _config: (
            segments,
            {
                "enabled_levels": [1, 0, 0],
                "enabled_level_names": ["coarse"],
                "root_level": "coarse",
                "roots": [
                    {
                        "level": "coarse",
                        "start_seg_id": 0,
                        "end_seg_id": 0,
                        "summary": "Pick up bowl",
                        "children": [],
                    }
                ],
            },
            {"llm_merge_applied": False, "llm_summary_applied": True},
        ),
    )
    monkeypatch.setattr(app_module, "run_export_subtitle_localization_pass", subtitle_pass)
    monkeypatch.setattr(
        app_module,
        "export_sample_outputs",
        lambda **_kwargs: {"export_enabled": False, "export_attempted": False},
    )
    monkeypatch.setattr(app_module, "FrameExtractor", _NoopFrameExtractorForAppStage2)

    config = Config(
        datasets=[{"root": str(data_root), "subset": subset}],
        run={"base_dir": str(tmp_path), "run_id": "testrun", "force_resume": True},
        server={"auto_exit_after_all_done": False},
        export={"enabled": False, "subtitles": {"enabled": False, "language": "zh"}},
    )
    app = create_app(config)
    app.state.runtime.start()
    return sample_out_dir


def test_app_finalize_writes_stage2_localized_subtitles_back_to_segments_json(tmp_path, monkeypatch) -> None:
    subtitle_calls: list[str] = []

    def fake_subtitle_pass(_sid, segments, _config, target_language):
        subtitle_calls.append(str(target_language))
        localized = [dict(segment, export_subtitle="拿起碗") for segment in segments]
        return localized, {
            "export_subtitle_requested_language": "zh",
            "export_subtitle_language": "zh",
            "export_subtitle_output_language": "zh",
            "export_subtitle_attempted": True,
            "export_subtitle_applied": True,
            "export_subtitle_fallback_used": False,
            "export_subtitle_reason": "applied",
            "export_subtitle_segment_count": len(localized),
        }

    sample_out_dir = _make_stage2_writeback_app(tmp_path, monkeypatch, subtitle_pass=fake_subtitle_pass)

    done_marker = sample_out_dir / ".DONE"
    failed_marker = sample_out_dir / ".FAILED"
    _wait_until(lambda: done_marker.exists() or failed_marker.exists())

    assert done_marker.exists()
    assert not failed_marker.exists()
    assert subtitle_calls == ["zh"]

    result = json.loads((sample_out_dir / "segments.json").read_text(encoding="utf-8"))
    assert result["segments"][0]["instruction"] == "Pick up bowl"
    assert result["segments"][0]["export_subtitle"] == "拿起碗"
    assert result["task_hierarchy"]["roots"][0]["summary"] == "Pick up bowl"


def test_app_finalize_writes_stage2_subtitle_fallback_back_to_segments_json(tmp_path, monkeypatch) -> None:
    subtitle_calls: list[str] = []

    def fake_subtitle_pass(_sid, segments, _config, target_language):
        subtitle_calls.append(str(target_language))
        fallback = [dict(segment, export_subtitle=str(segment.get("instruction", "")).strip()) for segment in segments]
        return fallback, {
            "export_subtitle_requested_language": "zh",
            "export_subtitle_language": "en",
            "export_subtitle_output_language": "en",
            "export_subtitle_attempted": True,
            "export_subtitle_applied": False,
            "export_subtitle_fallback_used": True,
            "export_subtitle_reason": "request_failed:RuntimeError",
            "export_subtitle_segment_count": len(fallback),
        }

    sample_out_dir = _make_stage2_writeback_app(tmp_path, monkeypatch, subtitle_pass=fake_subtitle_pass)

    done_marker = sample_out_dir / ".DONE"
    failed_marker = sample_out_dir / ".FAILED"
    _wait_until(lambda: done_marker.exists() or failed_marker.exists())

    assert done_marker.exists()
    assert not failed_marker.exists()
    assert subtitle_calls == ["zh"]

    result = json.loads((sample_out_dir / "segments.json").read_text(encoding="utf-8"))
    assert result["segments"][0]["instruction"] == "Pick up bowl"
    assert result["segments"][0]["export_subtitle"] == "Pick up bowl"
    assert result["diagnostics"]["export_subtitle_reason"] == "request_failed:RuntimeError"
