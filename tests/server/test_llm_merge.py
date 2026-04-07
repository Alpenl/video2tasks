from video2tasks.config import LLMMergeConfig
from video2tasks.server.llm_merge import run_llm_merge_pass, validate_merged_partition, validate_merged_ranges


class FakeMergeBackend:
    def __init__(self, payload, diagnostics=None):
        self.payload = payload
        self.diagnostics = diagnostics or {}
        self.calls = []
        self.model = "gpt-5.2"
        self.last_text_json_diagnostics = {}

    def infer_text_json(self, prompt, *, schema_name, schema, max_output_tokens=None, reasoning_effort=None, raise_on_http_error=False):
        self.last_text_json_diagnostics = dict(self.diagnostics)
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
        return self.payload


class RaisingMergeBackend:
    def __init__(self):
        self.calls = 0

    def infer_text_json(self, prompt, *, schema_name, schema, max_output_tokens=None, reasoning_effort=None, raise_on_http_error=False):
        self.calls += 1
        raise RuntimeError("synthetic backend failure")


class SequenceMergeBackend:
    def __init__(self, responses, diagnostics_sequence=None):
        self.responses = list(responses)
        self.diagnostics_sequence = list(diagnostics_sequence or [])
        self.calls = 0
        self.last_text_json_diagnostics = {}

    def infer_text_json(self, prompt, *, schema_name, schema, max_output_tokens=None, reasoning_effort=None, raise_on_http_error=False):
        self.calls += 1
        if self.diagnostics_sequence:
            self.last_text_json_diagnostics = dict(self.diagnostics_sequence.pop(0))
        else:
            self.last_text_json_diagnostics = {}
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
            "end_frame": 12,
            "instruction": "Continue the task",
            "confidence": 0.8,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.2,
        },
        {
            "seg_id": 1,
            "start_frame": 12,
            "end_frame": 24,
            "instruction": "Place the flat item into the target region",
            "confidence": 0.95,
            "boundary_support_before": 0.2,
            "boundary_support_after": 0.4,
        },
        {
            "seg_id": 2,
            "start_frame": 24,
            "end_frame": 36,
            "instruction": "Remove the hand from the target region",
            "confidence": 0.9,
            "boundary_support_before": 0.4,
            "boundary_support_after": 0.7,
        },
        {
            "seg_id": 3,
            "start_frame": 36,
            "end_frame": 48,
            "instruction": "Pick up the next item",
            "confidence": 0.9,
            "boundary_support_before": 0.7,
            "boundary_support_after": 0.8,
        },
    ]


def test_validate_merged_ranges_rejects_gaps() -> None:
    ranges, reason = validate_merged_ranges(
        {
            "thought": "Bad partition.",
            "merged_ranges": [
                {"start_seg_id": 0, "end_seg_id": 1},
                {"start_seg_id": 3, "end_seg_id": 3},
            ],
        },
        segment_count=4,
        min_output_ratio=0.35,
    )

    assert ranges is None
    assert reason == "invalid_partition"


def test_validate_merged_partition_allows_temporarily_aggressive_coarse_partition() -> None:
    ranges, reason = validate_merged_partition(
        {
            "thought": "Collapse first, recover anchors later.",
            "merged_ranges": [
                {"start_seg_id": 0, "end_seg_id": 3},
            ],
        },
        segment_count=4,
    )

    assert ranges == [(0, 3)]
    assert reason == "ok"


def test_run_llm_merge_pass_merges_adjacent_ranges_and_refines_label() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 12,
            "instruction": "Place the flat item",
            "confidence": 0.8,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.2,
        },
        {
            "seg_id": 1,
            "start_frame": 12,
            "end_frame": 24,
            "instruction": "Place the flat item into the target region",
            "confidence": 0.95,
            "boundary_support_before": 0.2,
            "boundary_support_after": 0.4,
        },
        {
            "seg_id": 2,
            "start_frame": 24,
            "end_frame": 36,
            "instruction": "Remove the hand from the target region",
            "confidence": 0.9,
            "boundary_support_before": 0.4,
            "boundary_support_after": 0.7,
        },
        {
            "seg_id": 3,
            "start_frame": 36,
            "end_frame": 48,
            "instruction": "Pick up the next item",
            "confidence": 0.9,
            "boundary_support_before": 0.7,
            "boundary_support_after": 0.8,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "The first two short fragments are one placement step.",
            "merged_ranges": [
                {"start_seg_id": 0, "end_seg_id": 1},
                {"start_seg_id": 2, "end_seg_id": 2},
                {"start_seg_id": 3, "end_seg_id": 3},
            ],
        }
    )
    cfg = LLMMergeConfig(enabled=True, min_input_segments=4, min_output_ratio=0.35)

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_attempted"] is True
    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_reason"] == "applied"
    assert diagnostics["llm_merge_granularity"] == "guarded"
    assert diagnostics["llm_merge_effective_min_output_ratio"] == 0.35
    assert diagnostics["llm_merge_input_segment_count"] == 4
    assert diagnostics["llm_merge_output_segment_count"] == 3
    assert merged_segments[0]["seg_id"] == 0
    assert merged_segments[0]["start_frame"] == 0
    assert merged_segments[0]["end_frame"] == 24
    assert merged_segments[0]["instruction"] == "Place the flat item into the target region"
    assert merged_segments[1]["seg_id"] == 1
    assert merged_segments[1]["start_frame"] == 24
    assert merged_segments[1]["end_frame"] == 36
    assert backend.calls[0]["schema_name"] == "segment_merge_result"
    assert "same workspace" in backend.calls[0]["prompt"].lower()


def test_run_llm_merge_pass_uses_coarse_prompt_and_ratio() -> None:
    source_segments = _segments()
    backend = FakeMergeBackend(
        {
            "thought": "Coarsen the over-segmented list into broader steps.",
            "merged_ranges": [
                {"start_seg_id": 0, "end_seg_id": 1},
                {"start_seg_id": 2, "end_seg_id": 3},
            ],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=4,
        granularity="coarse",
        min_output_ratio=0.5,
        coarse_min_output_ratio=0.25,
        protect_supported_boundaries=False,
        protect_distinct_sequence_markers=False,
        protect_instruction_drift=False,
        protect_duplicate_tail_anchor=False,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_granularity"] == "coarse"
    assert diagnostics["llm_merge_effective_min_output_ratio"] == 0.25
    assert diagnostics["llm_merge_prompt_boundary_hint_count"] == 0
    assert diagnostics["llm_merge_coarse_max_supported_anchors_per_range"] == 1
    assert diagnostics["llm_merge_coarse_anchor_min_spacing_segments"] == 3
    assert diagnostics["llm_merge_coarse_anchor_min_score"] == 1.03
    assert len(merged_segments) == 2
    assert "coarse task-level steps" in backend.calls[0]["prompt"].lower()
    assert "objective boundary hints" not in backend.calls[0]["prompt"].lower()


def test_run_llm_merge_pass_coarse_uses_prompt_boundary_hints_when_available() -> None:
    source_segments = _segments()
    backend = FakeMergeBackend(
        {
            "thought": "Merge into broader steps while keeping major anchor transitions.",
            "merged_ranges": [
                {"start_seg_id": 0, "end_seg_id": 1},
                {"start_seg_id": 2, "end_seg_id": 3},
            ],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=4,
        granularity="coarse",
        min_output_ratio=0.5,
        coarse_min_output_ratio=0.25,
        protect_supported_boundaries=True,
        protect_distinct_sequence_markers=True,
        protect_instruction_drift=True,
        protect_duplicate_tail_anchor=False,
    )

    _, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_prompt_boundary_hint_count"] > 0
    assert "objective boundary hints" in backend.calls[0]["prompt"].lower()
    assert "after seg_id=" in backend.calls[0]["prompt"].lower()


def test_run_llm_merge_pass_exposes_adapter_endpoint_diagnostics() -> None:
    backend = FakeMergeBackend(
        {
            "thought": "The first two short fragments are one placement step.",
            "merged_ranges": [
                {"start_seg_id": 0, "end_seg_id": 1},
                {"start_seg_id": 2, "end_seg_id": 2},
                {"start_seg_id": 3, "end_seg_id": 3},
            ],
        },
        diagnostics={
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
        },
    )
    cfg = LLMMergeConfig(enabled=True, min_input_segments=4)

    _, diagnostics = run_llm_merge_pass(
        "demo_sample",
        _segments(),
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_adapter_diagnostics"]["parsed_endpoint"] == "chat_completions"
    assert diagnostics["llm_merge_adapter_diagnostics"]["responses"]["failure_reason"] == "body_shape_mismatch"
    assert diagnostics["llm_merge_adapter_diagnostics"]["chat_completions"]["structured_payload_found"] is True
    assert diagnostics["llm_merge_adapter_diagnostics_attempts"][0]["request_attempt_index"] == 1


def test_run_llm_merge_pass_coarse_keeps_only_top_support_anchor_per_range() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.91,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 24,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.91,
            "boundary_support_after": 0.97,
        },
        {
            "seg_id": 2,
            "start_frame": 24,
            "end_frame": 40,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.97,
            "boundary_support_after": 0.95,
        },
        {
            "seg_id": 3,
            "start_frame": 40,
            "end_frame": 52,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.95,
            "boundary_support_after": 0.93,
        },
        {
            "seg_id": 4,
            "start_frame": 52,
            "end_frame": 64,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.93,
            "boundary_support_after": 0.0,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "Coarsen the fragments into broader steps.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 4}],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=5,
        granularity="coarse",
        min_output_ratio=0.2,
        coarse_min_output_ratio=0.2,
        coarse_max_supported_anchors_per_range=1,
        coarse_anchor_min_spacing_segments=3,
        coarse_anchor_min_score=0.0,
        protected_boundary_support_threshold=0.8,
        protect_distinct_sequence_markers=False,
        protect_instruction_drift=False,
        protect_duplicate_tail_anchor=False,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_ranges_sanitized"] is True
    assert diagnostics["llm_merge_ranges"] == [
        {"start_seg_id": 0, "end_seg_id": 1},
        {"start_seg_id": 2, "end_seg_id": 4},
    ]
    assert diagnostics["llm_merge_blocked_boundaries"] == [
        {
            "boundary_after_seg_id": 1,
            "left_seg_id": 1,
            "right_seg_id": 2,
            "left_instruction": "Transfer the item toward the target area",
            "right_instruction": "Transfer the item toward the target area",
            "reasons": ["boundary_support"],
            "boundary_support": 0.97,
            "has_boundary_support": True,
            "left_duration_frames": 14,
            "right_duration_frames": 16,
            "coarse_anchor_score": 0.97,
        }
    ]
    assert len(merged_segments) == 2
    assert merged_segments[0]["start_frame"] == 0
    assert merged_segments[0]["end_frame"] == 24
    assert merged_segments[1]["start_frame"] == 24
    assert merged_segments[1]["end_frame"] == 64


def test_run_llm_merge_pass_coarse_prefers_semantic_anchor_when_budget_is_limited() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "First placement into the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.92,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 20,
            "instruction": "Second transfer onto the raised rack",
            "confidence": 1.0,
            "boundary_support_before": 0.92,
            "boundary_support_after": 0.99,
        },
        {
            "seg_id": 2,
            "start_frame": 20,
            "end_frame": 32,
            "instruction": "Second transfer onto the raised rack",
            "confidence": 1.0,
            "boundary_support_before": 0.99,
            "boundary_support_after": 0.98,
        },
        {
            "seg_id": 3,
            "start_frame": 32,
            "end_frame": 44,
            "instruction": "Second transfer onto the raised rack",
            "confidence": 1.0,
            "boundary_support_before": 0.98,
            "boundary_support_after": 0.0,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "Coarsen into broad placement steps.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 3}],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=4,
        granularity="coarse",
        min_output_ratio=0.25,
        coarse_min_output_ratio=0.25,
        coarse_max_supported_anchors_per_range=1,
        coarse_anchor_min_spacing_segments=1,
        coarse_anchor_min_side_segments=1,
        coarse_anchor_min_score=0.0,
        protected_boundary_support_threshold=0.8,
        protect_distinct_sequence_markers=True,
        protect_instruction_drift=False,
        protect_duplicate_tail_anchor=False,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_ranges"] == [
        {"start_seg_id": 0, "end_seg_id": 0},
        {"start_seg_id": 1, "end_seg_id": 3},
    ]
    assert diagnostics["llm_merge_blocked_boundaries"][0]["reasons"] == ["boundary_support", "sequence_markers"]
    assert diagnostics["llm_merge_blocked_boundaries"][0]["coarse_anchor_score"] == 1.42
    assert len(merged_segments) == 2


def test_run_llm_merge_pass_coarse_respects_anchor_spacing() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.3,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 24,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.3,
            "boundary_support_after": 0.99,
        },
        {
            "seg_id": 2,
            "start_frame": 24,
            "end_frame": 38,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.99,
            "boundary_support_after": 0.98,
        },
        {
            "seg_id": 3,
            "start_frame": 38,
            "end_frame": 52,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.98,
            "boundary_support_after": 0.2,
        },
        {
            "seg_id": 4,
            "start_frame": 52,
            "end_frame": 66,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.2,
            "boundary_support_after": 0.97,
        },
        {
            "seg_id": 5,
            "start_frame": 66,
            "end_frame": 80,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.97,
            "boundary_support_after": 0.0,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "Keep only a few coarse anchors.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 5}],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=6,
        granularity="coarse",
        min_output_ratio=0.1,
        coarse_min_output_ratio=0.1,
        coarse_max_supported_anchors_per_range=2,
        coarse_anchor_min_spacing_segments=3,
        coarse_anchor_min_side_segments=1,
        coarse_anchor_min_score=0.0,
        protected_boundary_support_threshold=0.8,
        protect_distinct_sequence_markers=False,
        protect_instruction_drift=False,
        protect_duplicate_tail_anchor=False,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_ranges"] == [
        {"start_seg_id": 0, "end_seg_id": 1},
        {"start_seg_id": 2, "end_seg_id": 4},
        {"start_seg_id": 5, "end_seg_id": 5},
    ]
    assert [item["boundary_after_seg_id"] for item in diagnostics["llm_merge_blocked_boundaries"]] == [1, 4]
    assert len(merged_segments) == 3


def test_run_llm_merge_pass_coarse_balances_long_range_anchor_selection() -> None:
    source_segments = []
    frame = 0
    boundary_support_after = [0.1, 0.95, 0.1, 0.1, 0.1, 0.97, 0.1, 0.1, 0.96, 0.1, 0.1, 0.0]
    for seg_id, support_after in enumerate(boundary_support_after):
        source_segments.append(
            {
                "seg_id": seg_id,
                "start_frame": frame,
                "end_frame": frame + 10,
                "instruction": "Transfer the item toward the target area",
                "confidence": 1.0,
                "boundary_support_before": boundary_support_after[seg_id - 1] if seg_id > 0 else 0.0,
                "boundary_support_after": support_after,
            }
        )
        frame += 10

    backend = FakeMergeBackend(
        {
            "thought": "Use a few balanced internal anchors for the long range.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 11}],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=12,
        granularity="coarse",
        min_output_ratio=0.1,
        coarse_min_output_ratio=0.1,
        coarse_max_supported_anchors_per_range=1,
        coarse_anchor_min_spacing_segments=3,
        coarse_anchor_min_side_segments=1,
        coarse_anchor_min_score=0.0,
        protected_boundary_support_threshold=0.8,
        protect_distinct_sequence_markers=False,
        protect_instruction_drift=False,
        protect_duplicate_tail_anchor=False,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_ranges"] == [
        {"start_seg_id": 0, "end_seg_id": 5},
        {"start_seg_id": 6, "end_seg_id": 8},
        {"start_seg_id": 9, "end_seg_id": 11},
    ]
    assert [item["boundary_after_seg_id"] for item in diagnostics["llm_merge_blocked_boundaries"]] == [5, 8]
    assert len(merged_segments) == 3


def test_run_llm_merge_pass_coarse_skips_support_only_anchor_below_min_score() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.91,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 24,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.91,
            "boundary_support_after": 0.97,
        },
        {
            "seg_id": 2,
            "start_frame": 24,
            "end_frame": 40,
            "instruction": "Transfer the item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.97,
            "boundary_support_after": 0.0,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "Coarsen the fragments into one broader step.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 2}],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=3,
        granularity="coarse",
        min_output_ratio=0.33,
        coarse_min_output_ratio=0.33,
        coarse_max_supported_anchors_per_range=1,
        coarse_anchor_min_spacing_segments=3,
        coarse_anchor_min_score=1.05,
        protected_boundary_support_threshold=0.8,
        protect_distinct_sequence_markers=False,
        protect_instruction_drift=False,
        protect_duplicate_tail_anchor=False,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_ranges_sanitized"] is False
    assert diagnostics["llm_merge_ranges"] == [{"start_seg_id": 0, "end_seg_id": 2}]
    assert "llm_merge_blocked_boundaries" not in diagnostics
    assert len(merged_segments) == 1


def test_run_llm_merge_pass_coarse_keeps_instruction_drift_anchor_at_exact_min_score() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "Chop the greens on the cutting board",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.53,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 24,
            "instruction": "Gather the chopped greens with the knife",
            "confidence": 1.0,
            "boundary_support_before": 0.53,
            "boundary_support_after": 0.0,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "Keep the two broad placement rounds separate.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 1}],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=2,
        granularity="coarse",
        min_output_ratio=0.5,
        coarse_min_output_ratio=0.5,
        coarse_max_supported_anchors_per_range=1,
        coarse_anchor_min_spacing_segments=1,
        coarse_anchor_min_side_segments=1,
        coarse_anchor_min_score=1.03,
        protected_boundary_support_threshold=0.5,
        protect_distinct_sequence_markers=False,
        protect_instruction_drift=True,
        protect_duplicate_tail_anchor=False,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is False
    assert diagnostics["llm_merge_reason"] == "guard_blocked_all_merges"
    assert diagnostics["llm_merge_ranges_sanitized"] is True
    assert diagnostics["llm_merge_blocked_boundaries"][0]["coarse_anchor_score"] == 1.03
    assert merged_segments == source_segments


def test_run_llm_merge_pass_coarse_treats_ordinal_only_round_markers_as_support_only() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "First placement into the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.92,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 24,
            "instruction": "Second placement into the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.92,
            "boundary_support_after": 0.0,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "Coarsen the repeated placement fragments into one broad step.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 1}],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=2,
        granularity="coarse",
        min_output_ratio=0.5,
        coarse_min_output_ratio=0.5,
        coarse_max_supported_anchors_per_range=1,
        coarse_anchor_min_spacing_segments=1,
        coarse_anchor_min_score=1.03,
        protected_boundary_support_threshold=0.5,
        protect_distinct_sequence_markers=True,
        protect_instruction_drift=False,
        protect_duplicate_tail_anchor=False,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_ranges_sanitized"] is False
    assert diagnostics["llm_merge_ranges"] == [{"start_seg_id": 0, "end_seg_id": 1}]
    assert "llm_merge_blocked_boundaries" not in diagnostics
    assert len(merged_segments) == 1


def test_run_llm_merge_pass_coarse_with_zero_anchor_budget_keeps_requested_ranges() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "First placement into the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.92,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 20,
            "instruction": "Second placement into the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.92,
            "boundary_support_after": 0.0,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "Coarsen the two short placement steps.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 1}],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=2,
        granularity="coarse",
        min_output_ratio=0.5,
        coarse_min_output_ratio=0.5,
        coarse_max_supported_anchors_per_range=0,
        protected_boundary_support_threshold=0.8,
        protect_distinct_sequence_markers=True,
        protect_instruction_drift=True,
        protect_duplicate_tail_anchor=False,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_ranges_sanitized"] is False
    assert "llm_merge_blocked_boundaries" not in diagnostics
    assert len(merged_segments) == 1


def test_run_llm_merge_pass_coarse_ignores_duplicate_tail_anchor_preservation() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 23,
            "instruction": "Move the tool toward the target object",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.37,
        },
        {
            "seg_id": 1,
            "start_frame": 23,
            "end_frame": 64,
            "instruction": "Move the tool toward the target object",
            "confidence": 1.0,
            "boundary_support_before": 0.37,
            "boundary_support_after": 0.11,
        },
        {
            "seg_id": 2,
            "start_frame": 64,
            "end_frame": 135,
            "instruction": "Move the tool toward the target object",
            "confidence": 1.0,
            "boundary_support_before": 0.11,
            "boundary_support_after": 1.06,
        },
        {
            "seg_id": 3,
            "start_frame": 135,
            "end_frame": 233,
            "instruction": "First transfer of the item onto the destination",
            "confidence": 1.0,
            "boundary_support_before": 1.06,
            "boundary_support_after": 0.2,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "Coarsen the repeated motion fragments.",
            "merged_ranges": [
                {"start_seg_id": 0, "end_seg_id": 2},
                {"start_seg_id": 3, "end_seg_id": 3},
            ],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=4,
        granularity="coarse",
        coarse_min_output_ratio=0.25,
        protected_boundary_support_threshold=0.8,
        protect_duplicate_tail_anchor=True,
        protect_distinct_sequence_markers=True,
        protect_instruction_drift=True,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_ranges"] == [
        {"start_seg_id": 0, "end_seg_id": 2},
        {"start_seg_id": 3, "end_seg_id": 3},
    ]
    assert "llm_merge_preserved_tail_anchors" not in diagnostics
    assert len(merged_segments) == 2


def test_run_llm_merge_pass_falls_back_on_invalid_output() -> None:
    source_segments = _segments()
    backend = FakeMergeBackend(
        {
            "thought": "This collapses too much.",
            "merged_ranges": [
                {"start_seg_id": 0, "end_seg_id": 3},
            ],
        }
    )
    cfg = LLMMergeConfig(enabled=True, min_input_segments=4, min_output_ratio=0.75)

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert merged_segments == source_segments
    assert diagnostics["llm_merge_attempted"] is True
    assert diagnostics["llm_merge_applied"] is False
    assert diagnostics["llm_merge_reason"] == "guard_blocked_all_merges"
    assert diagnostics["llm_merge_ranges_sanitized"] is True


def test_run_llm_merge_pass_coarse_applies_after_anchor_restore_even_if_requested_ranges_are_very_coarse() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "Pour dark liquid into the pan",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.95,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 20,
            "instruction": "Heat the liquid in the pan",
            "confidence": 1.0,
            "boundary_support_before": 0.95,
            "boundary_support_after": 0.96,
        },
        {
            "seg_id": 2,
            "start_frame": 20,
            "end_frame": 30,
            "instruction": "Transfer the item toward the next destination",
            "confidence": 1.0,
            "boundary_support_before": 0.2,
            "boundary_support_after": 0.96,
        },
        {
            "seg_id": 3,
            "start_frame": 30,
            "end_frame": 40,
            "instruction": "Transfer the item toward the next destination",
            "confidence": 1.0,
            "boundary_support_before": 0.96,
            "boundary_support_after": 0.0,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "Collapse to very broad coarse phases first.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 3}],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=4,
        granularity="coarse",
        coarse_min_output_ratio=0.75,
        coarse_max_supported_anchors_per_range=2,
        coarse_anchor_min_spacing_segments=1,
        coarse_anchor_min_side_segments=1,
        coarse_anchor_min_score=0.0,
        protected_boundary_support_threshold=0.8,
        protect_distinct_sequence_markers=True,
        protect_instruction_drift=True,
        protect_duplicate_tail_anchor=False,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_reason"] == "applied"
    assert diagnostics["llm_merge_ranges"] == [
        {"start_seg_id": 0, "end_seg_id": 0},
        {"start_seg_id": 1, "end_seg_id": 1},
        {"start_seg_id": 2, "end_seg_id": 3},
    ]
    assert [item["boundary_after_seg_id"] for item in diagnostics["llm_merge_blocked_boundaries"]] == [0, 1]
    assert len(merged_segments) == 3


def test_run_llm_merge_pass_coarse_skips_anchor_when_requested_range_side_is_too_small() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "Pour dark liquid into the pan",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.95,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 20,
            "instruction": "Heat the liquid in the pan",
            "confidence": 1.0,
            "boundary_support_before": 0.95,
            "boundary_support_after": 0.96,
        },
        {
            "seg_id": 2,
            "start_frame": 20,
            "end_frame": 30,
            "instruction": "Transfer the item toward the next destination",
            "confidence": 1.0,
            "boundary_support_before": 0.96,
            "boundary_support_after": 0.97,
        },
        {
            "seg_id": 3,
            "start_frame": 30,
            "end_frame": 40,
            "instruction": "Transfer the item toward the next destination",
            "confidence": 1.0,
            "boundary_support_before": 0.97,
            "boundary_support_after": 0.0,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "Split only the materially supported internal boundary.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 3}],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=4,
        granularity="coarse",
        min_output_ratio=0.25,
        coarse_min_output_ratio=0.25,
        coarse_max_supported_anchors_per_range=2,
        coarse_anchor_min_spacing_segments=1,
        coarse_anchor_min_side_segments=2,
        protected_boundary_support_threshold=0.8,
        protect_distinct_sequence_markers=True,
        protect_instruction_drift=True,
        protect_duplicate_tail_anchor=False,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_ranges"] == [
        {"start_seg_id": 0, "end_seg_id": 1},
        {"start_seg_id": 2, "end_seg_id": 3},
    ]
    assert [item["boundary_after_seg_id"] for item in diagnostics["llm_merge_blocked_boundaries"]] == [1]
    assert len(merged_segments) == 2


def test_run_llm_merge_pass_skips_small_inputs() -> None:
    cfg = LLMMergeConfig(enabled=True, min_input_segments=10)

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        _segments(),
        cfg,
        backend=FakeMergeBackend({}),
    )

    assert merged_segments == _segments()
    assert diagnostics["llm_merge_attempted"] is False
    assert diagnostics["llm_merge_reason"] == "below_min_input_segments"


def test_run_llm_merge_pass_falls_back_on_backend_request_error() -> None:
    backend = RaisingMergeBackend()
    cfg = LLMMergeConfig(enabled=True, min_input_segments=4, max_attempts=3)

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        _segments(),
        cfg,
        backend=backend,
    )

    assert merged_segments == _segments()
    assert diagnostics["llm_merge_attempted"] is True
    assert diagnostics["llm_merge_applied"] is False
    assert diagnostics["llm_merge_reason"] == "request_failed:RuntimeError"
    assert diagnostics["llm_merge_error"] == "synthetic backend failure"
    assert diagnostics["llm_merge_request_attempt_count"] == 3
    assert backend.calls == 3


def test_run_llm_merge_pass_retries_empty_response_until_success() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 12,
            "instruction": "Place the flat item",
            "confidence": 0.8,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.2,
        },
        {
            "seg_id": 1,
            "start_frame": 12,
            "end_frame": 24,
            "instruction": "Place the flat item into the target region",
            "confidence": 0.95,
            "boundary_support_before": 0.2,
            "boundary_support_after": 0.4,
        },
        {
            "seg_id": 2,
            "start_frame": 24,
            "end_frame": 36,
            "instruction": "Remove the hand from the target region",
            "confidence": 0.9,
            "boundary_support_before": 0.4,
            "boundary_support_after": 0.7,
        },
        {
            "seg_id": 3,
            "start_frame": 36,
            "end_frame": 48,
            "instruction": "Pick up the next item",
            "confidence": 0.9,
            "boundary_support_before": 0.7,
            "boundary_support_after": 0.8,
        },
    ]
    backend = SequenceMergeBackend(
        [
            {},
            {
                "thought": "Merge the first placement fragments.",
                "merged_ranges": [
                    {"start_seg_id": 0, "end_seg_id": 1},
                    {"start_seg_id": 2, "end_seg_id": 2},
                    {"start_seg_id": 3, "end_seg_id": 3},
                ],
            },
        ]
    )
    cfg = LLMMergeConfig(enabled=True, min_input_segments=4, max_attempts=3)

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_reason"] == "applied"
    assert diagnostics["llm_merge_request_attempt_count"] == 2
    assert backend.calls == 2
    assert len(merged_segments) == 3


def test_run_llm_merge_pass_coarse_prefers_consensus_supported_candidate_across_attempts() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "Move the first item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.2,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 20,
            "instruction": "Move the second item toward the target area",
            "confidence": 1.0,
            "boundary_support_before": 0.2,
            "boundary_support_after": 0.2,
        },
        {
            "seg_id": 2,
            "start_frame": 20,
            "end_frame": 30,
            "instruction": "Place both items onto the target surface",
            "confidence": 1.0,
            "boundary_support_before": 0.2,
            "boundary_support_after": 0.2,
        },
        {
            "seg_id": 3,
            "start_frame": 30,
            "end_frame": 40,
            "instruction": "Align the placed items on the surface",
            "confidence": 1.0,
            "boundary_support_before": 0.2,
            "boundary_support_after": 0.2,
        },
        {
            "seg_id": 4,
            "start_frame": 40,
            "end_frame": 50,
            "instruction": "Cover the target surface with the lid",
            "confidence": 1.0,
            "boundary_support_before": 0.2,
            "boundary_support_after": 0.2,
        },
        {
            "seg_id": 5,
            "start_frame": 50,
            "end_frame": 60,
            "instruction": "Press the lid into its final position",
            "confidence": 1.0,
            "boundary_support_before": 0.2,
            "boundary_support_after": 0.0,
        },
    ]
    backend = SequenceMergeBackend(
        [
            {
                "thought": "Keep three broad steps.",
                "merged_ranges": [
                    {"start_seg_id": 0, "end_seg_id": 1},
                    {"start_seg_id": 2, "end_seg_id": 3},
                    {"start_seg_id": 4, "end_seg_id": 5},
                ],
            },
            {
                "thought": "Keep two broader steps.",
                "merged_ranges": [
                    {"start_seg_id": 0, "end_seg_id": 1},
                    {"start_seg_id": 2, "end_seg_id": 5},
                ],
            },
            {
                "thought": "Keep two broader steps again.",
                "merged_ranges": [
                    {"start_seg_id": 0, "end_seg_id": 1},
                    {"start_seg_id": 2, "end_seg_id": 5},
                ],
            },
        ]
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=6,
        granularity="coarse",
        max_attempts=1,
        repeat_count=3,
        boundary_vote_threshold=0.5,
        coarse_min_output_ratio=0.1,
        protect_supported_boundaries=False,
        protect_distinct_sequence_markers=False,
        protect_instruction_drift=False,
        protect_duplicate_tail_anchor=False,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_request_attempt_count"] == 3
    assert diagnostics["llm_merge_valid_candidate_count"] == 3
    assert diagnostics["llm_merge_successful_sample_count"] == 3
    assert diagnostics["llm_merge_candidate_selection_mode"] == "coarse_boundary_consensus"
    assert diagnostics["llm_merge_consensus_boundaries"] == [1]
    assert diagnostics["llm_merge_candidate_output_segment_counts"] == [3, 2, 2]
    assert diagnostics["llm_merge_ranges"] == [
        {"start_seg_id": 0, "end_seg_id": 1},
        {"start_seg_id": 2, "end_seg_id": 5},
    ]
    assert len(merged_segments) == 2


def test_run_llm_merge_pass_blocks_high_support_boundary_inside_requested_range() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "Pour dark sauce over the food in the pan",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.2,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 20,
            "instruction": "Pour dark sauce over the food in the pan",
            "confidence": 1.0,
            "boundary_support_before": 0.2,
            "boundary_support_after": 0.95,
        },
        {
            "seg_id": 2,
            "start_frame": 20,
            "end_frame": 30,
            "instruction": "Pour dark sauce over the food in the pan",
            "confidence": 1.0,
            "boundary_support_before": 0.95,
            "boundary_support_after": 0.1,
        },
        {
            "seg_id": 3,
            "start_frame": 30,
            "end_frame": 40,
            "instruction": "Pour dark sauce over the food in the pan",
            "confidence": 1.0,
            "boundary_support_before": 0.1,
            "boundary_support_after": 0.0,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "All four fragments are the same step.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 3}],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=4,
        min_output_ratio=0.25,
        protected_boundary_support_threshold=0.8,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_ranges_sanitized"] is True
    assert diagnostics["llm_merge_ranges"] == [
        {"start_seg_id": 0, "end_seg_id": 1},
        {"start_seg_id": 2, "end_seg_id": 3},
    ]
    assert diagnostics["llm_merge_blocked_boundaries"][0]["reasons"] == ["boundary_support"]
    assert len(merged_segments) == 2
    assert merged_segments[0]["start_frame"] == 0
    assert merged_segments[0]["end_frame"] == 20
    assert merged_segments[1]["start_frame"] == 20
    assert merged_segments[1]["end_frame"] == 40


def test_run_llm_merge_pass_blocks_distinct_sequence_markers() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "First pour of dark liquid into the pan",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.1,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 20,
            "instruction": "Second pour of dark liquid into the pan",
            "confidence": 1.0,
            "boundary_support_before": 0.1,
            "boundary_support_after": 0.0,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "These two pours look similar enough to merge.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 1}],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=2,
        protected_boundary_support_threshold=0.9,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert merged_segments == source_segments
    assert diagnostics["llm_merge_applied"] is False
    assert diagnostics["llm_merge_reason"] == "guard_blocked_all_merges"
    assert diagnostics["llm_merge_ranges_sanitized"] is True
    assert diagnostics["llm_merge_blocked_boundaries"][0]["reasons"] == ["sequence_markers"]


def test_run_llm_merge_pass_blocks_instruction_drift() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "Chop the greens on the cutting board",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.1,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 20,
            "instruction": "Gather the chopped greens with the knife",
            "confidence": 1.0,
            "boundary_support_before": 0.1,
            "boundary_support_after": 0.0,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "Both fragments are just handling the same greens.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 1}],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=2,
        protected_boundary_support_threshold=0.9,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert merged_segments == source_segments
    assert diagnostics["llm_merge_applied"] is False
    assert diagnostics["llm_merge_reason"] == "guard_blocked_all_merges"
    assert diagnostics["llm_merge_ranges_sanitized"] is True
    assert diagnostics["llm_merge_blocked_boundaries"][0]["reasons"] == ["instruction_drift"]


def test_run_llm_merge_pass_allows_low_support_micro_fragment_merge() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 10,
            "instruction": "Mash the peas and butter together in the pot",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.16,
        },
        {
            "seg_id": 1,
            "start_frame": 10,
            "end_frame": 20,
            "instruction": "Mash the peas and butter together in the pot",
            "confidence": 1.0,
            "boundary_support_before": 0.16,
            "boundary_support_after": 0.0,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "This is one micro-fragmented mashing step.",
            "merged_ranges": [{"start_seg_id": 0, "end_seg_id": 1}],
        }
    )
    cfg = LLMMergeConfig(enabled=True, min_input_segments=2)

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_reason"] == "applied"
    assert diagnostics["llm_merge_ranges_sanitized"] is False
    assert "llm_merge_blocked_boundaries" not in diagnostics
    assert len(merged_segments) == 1
    assert merged_segments[0]["instruction"] == "Mash the peas and butter together in the pot"


def test_run_llm_merge_pass_preserves_duplicate_tail_anchor_before_strong_next_boundary() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 23,
            "instruction": "Garnish the salad with additional red vegetable pieces",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.37,
        },
        {
            "seg_id": 1,
            "start_frame": 23,
            "end_frame": 64,
            "instruction": "Garnish the salad with additional red vegetable pieces",
            "confidence": 1.0,
            "boundary_support_before": 0.37,
            "boundary_support_after": 0.11,
        },
        {
            "seg_id": 2,
            "start_frame": 64,
            "end_frame": 135,
            "instruction": "Garnish the salad with additional red vegetable pieces",
            "confidence": 1.0,
            "boundary_support_before": 0.11,
            "boundary_support_after": 1.06,
        },
        {
            "seg_id": 3,
            "start_frame": 135,
            "end_frame": 233,
            "instruction": "First application of ground pepper onto the bowl of contents",
            "confidence": 1.0,
            "boundary_support_before": 1.06,
            "boundary_support_after": 0.2,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "The garnish fragments can be merged.",
            "merged_ranges": [
                {"start_seg_id": 0, "end_seg_id": 2},
                {"start_seg_id": 3, "end_seg_id": 3},
            ],
        }
    )
    cfg = LLMMergeConfig(enabled=True, min_input_segments=4, min_output_ratio=0.25)

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_ranges"] == [
        {"start_seg_id": 0, "end_seg_id": 1},
        {"start_seg_id": 2, "end_seg_id": 2},
        {"start_seg_id": 3, "end_seg_id": 3},
    ]
    assert diagnostics["llm_merge_ranges_sanitized"] is True
    assert diagnostics["llm_merge_preserved_tail_anchors"][0]["boundary_after_seg_id"] == 1
    assert merged_segments[0]["end_frame"] == 64
    assert merged_segments[1]["start_frame"] == 64


def test_run_llm_merge_pass_coarse_does_not_reintroduce_duplicate_tail_anchor_split() -> None:
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 23,
            "instruction": "Garnish the bowl with additional small pieces",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.37,
        },
        {
            "seg_id": 1,
            "start_frame": 23,
            "end_frame": 64,
            "instruction": "Garnish the bowl with additional small pieces",
            "confidence": 1.0,
            "boundary_support_before": 0.37,
            "boundary_support_after": 0.11,
        },
        {
            "seg_id": 2,
            "start_frame": 64,
            "end_frame": 135,
            "instruction": "Garnish the bowl with additional small pieces",
            "confidence": 1.0,
            "boundary_support_before": 0.11,
            "boundary_support_after": 1.06,
        },
        {
            "seg_id": 3,
            "start_frame": 135,
            "end_frame": 233,
            "instruction": "First application of ground seasoning onto the bowl contents",
            "confidence": 1.0,
            "boundary_support_before": 1.06,
            "boundary_support_after": 0.2,
        },
    ]
    backend = FakeMergeBackend(
        {
            "thought": "Merge the garnish fragments into one broad coarse step.",
            "merged_ranges": [
                {"start_seg_id": 0, "end_seg_id": 2},
                {"start_seg_id": 3, "end_seg_id": 3},
            ],
        }
    )
    cfg = LLMMergeConfig(
        enabled=True,
        min_input_segments=4,
        granularity="coarse",
        coarse_min_output_ratio=0.25,
        protect_supported_boundaries=False,
        protect_distinct_sequence_markers=True,
        protect_instruction_drift=True,
        protect_duplicate_tail_anchor=True,
    )

    merged_segments, diagnostics = run_llm_merge_pass(
        "demo_sample",
        source_segments,
        cfg,
        backend=backend,
    )

    assert diagnostics["llm_merge_applied"] is True
    assert diagnostics["llm_merge_ranges"] == [
        {"start_seg_id": 0, "end_seg_id": 2},
        {"start_seg_id": 3, "end_seg_id": 3},
    ]
    assert "llm_merge_preserved_tail_anchors" not in diagnostics
    assert len(merged_segments) == 2
    assert merged_segments[0]["start_frame"] == 0
    assert merged_segments[0]["end_frame"] == 135
