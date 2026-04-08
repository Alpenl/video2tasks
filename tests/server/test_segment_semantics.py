import video2tasks.server.windowing as windowing_module

from video2tasks.server.segment_semantics import (
    boundary_support_between,
    has_distinct_sequence_markers,
    refine_segment_instructions,
    should_split_on_instruction_drift,
)


def test_boundary_support_between_prefers_stronger_side() -> None:
    support, has_support = boundary_support_between(
        {"boundary_support_after": 0.25},
        {"boundary_support_before": 0.75},
    )

    assert support == 0.75
    assert has_support is True


def test_has_distinct_sequence_markers_detects_order_change() -> None:
    assert has_distinct_sequence_markers(
        "First add potatoes to the pot",
        "Then add potatoes to the pot",
    )


def test_should_split_on_instruction_drift_when_action_family_changes() -> None:
    assert should_split_on_instruction_drift(
        "Add potatoes to the pot",
        "Stir the pot",
    )


def test_refine_segment_instructions_promotes_more_specific_contributor() -> None:
    refined = refine_segment_instructions(
        [
            {
                "seg_id": 0,
                "start_frame": 0,
                "end_frame": 20,
                "instruction": "Prepare ingredients",
            }
        ],
        [
            {
                "seg_id": 0,
                "start_frame": 0,
                "end_frame": 20,
                "instruction": "Dice potatoes",
            }
        ],
    )

    assert refined[0]["instruction"] == "Dice potatoes"


def test_segment_semantics_boundary_support_does_not_forward_to_windowing(monkeypatch) -> None:
    monkeypatch.setattr(
        windowing_module,
        "_boundary_support_between",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("forwarded to windowing")),
    )

    support, has_support = boundary_support_between(
        {"boundary_support_after": 0.25},
        {"boundary_support_before": 0.75},
    )

    assert support == 0.75
    assert has_support is True


def test_segment_semantics_refine_instructions_does_not_forward_to_windowing(monkeypatch) -> None:
    monkeypatch.setattr(
        windowing_module,
        "refine_segment_instructions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("forwarded to windowing")),
    )

    refined = refine_segment_instructions(
        [
            {
                "seg_id": 0,
                "start_frame": 0,
                "end_frame": 20,
                "instruction": "Prepare ingredients",
            }
        ],
        [
            {
                "seg_id": 0,
                "start_frame": 0,
                "end_frame": 20,
                "instruction": "Dice potatoes",
            }
        ],
    )

    assert refined[0]["instruction"] == "Dice potatoes"
