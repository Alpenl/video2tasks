from video2tasks.config import LLMMergeConfig
from video2tasks.server.llm_merge import (
    run_export_subtitle_localization_pass,
    run_llm_postprocess_pass,
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


def test_run_llm_postprocess_pass_skips_live_summary_after_merge_request_failure() -> None:
    backend = SequenceBackend(
        [
            RuntimeError("merge backend timeout"),
            RuntimeError("merge backend timeout"),
            RuntimeError("merge backend timeout"),
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
    assert diagnostics["llm_summary_attempted"] is False
    assert diagnostics["llm_summary_applied"] is False
    assert diagnostics["llm_summary_fallback_used"] is True
    assert diagnostics["llm_summary_reason"] == "skipped_after_merge_failure:request_failed:RuntimeError"
    assert task_hierarchy is not None
    assert task_hierarchy["enabled_level_names"] == ["coarse", "medium", "fine"]
    assert len(task_hierarchy["roots"]) == 4
    assert len(cleaned_segments) == 4
    assert [call["schema_name"] for call in backend.calls] == [
        "segment_merge_result",
        "segment_merge_result",
        "segment_merge_result",
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
