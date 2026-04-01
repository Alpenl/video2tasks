from video2tasks.server.windowing import (
    merge_task_level_segments,
    cleanup_auxiliary_segments,
    refine_segment_instructions,
    _should_fallback_to_light_cleanup,
)


def test_merge_task_level_segments_merges_short_adjustments_around_same_bowl_task() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 111,
            "instruction": "Adjust the large light-green bowl (right)",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 111,
            "end_frame": 541,
            "instruction": "Grasp and move the large light-green bowl (lift and place it onto/over the small light-green bowl)",
            "confidence": 1.0,
        },
        {
            "seg_id": 2,
            "start_frame": 541,
            "end_frame": 621,
            "instruction": "Adjust/position the green bowl",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 621
    assert merged[0]["instruction"] == (
        "Grasp and move the large light-green bowl "
        "(lift and place it onto/over the small light-green bowl)"
    )


def test_merge_task_level_segments_keeps_distinct_object_tasks_separate() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 96,
            "instruction": "Adjust/position the orange plate",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 96,
            "end_frame": 512,
            "instruction": "Pick up the purple bowl and place/stack it onto the green bowl",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 2
    assert merged[0]["instruction"] == "Adjust/position the orange plate"
    assert merged[1]["instruction"] == "Pick up the purple bowl and place/stack it onto the green bowl"


def test_merge_task_level_segments_merges_prepare_stage_into_following_main_task() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 144,
            "instruction": "Grasp and manipulate (prepare to move/lift) the dark blue bowl",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 144,
            "end_frame": 912,
            "instruction": "Pick up the blue bowl and place it onto the stack",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 912
    assert merged[0]["instruction"] == "Pick up the blue bowl and place it onto the stack"


def test_merge_task_level_segments_merges_repeated_stack_adjustments() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 140,
            "instruction": "Push/tilt the central nested stack of bowls",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 140,
            "end_frame": 360,
            "instruction": "Adjust/reposition the nested bowl stack",
            "confidence": 1.0,
        },
        {
            "seg_id": 2,
            "start_frame": 360,
            "end_frame": 658,
            "instruction": "Adjust/stack the nested bowls",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 658
    assert merged[0]["instruction"] == "Adjust/stack the nested bowls"


def test_merge_task_level_segments_absorbs_bridge_motion_into_following_task() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 560,
            "instruction": "Pick up the large purple bowl and place it onto the large green bowl",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 560,
            "end_frame": 736,
            "instruction": "Reposition the gripper to a new work area",
            "confidence": 1.0,
        },
        {
            "seg_id": 2,
            "start_frame": 736,
            "end_frame": 1198,
            "instruction": "Pick up the red bowl from the stack and place it on the left side of the table",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 2
    assert merged[0]["instruction"] == "Pick up the large purple bowl and place it onto the large green bowl"
    assert merged[1]["start_frame"] == 560
    assert merged[1]["end_frame"] == 1198
    assert merged[1]["instruction"] == (
        "Pick up the red bowl from the stack and place it on the left side of the table"
    )


def test_merge_task_level_segments_absorbs_short_alignment_segment_into_following_task() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 808,
            "instruction": "Pick up the large light-green bowl and move it to the orange bowl area for placement",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 808,
            "end_frame": 947,
            "instruction": "Move to the orange plate and align the gripper above it",
            "confidence": 1.0,
        },
        {
            "seg_id": 2,
            "start_frame": 947,
            "end_frame": 1365,
            "instruction": "Pick up the purple bowl and place it on top of the green bowl",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 2
    assert merged[1]["start_frame"] == 808
    assert merged[1]["end_frame"] == 1365
    assert merged[1]["instruction"] == "Pick up the purple bowl and place it on top of the green bowl"


def test_merge_task_level_segments_absorbs_gripper_positioning_segment_into_following_task() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 833,
            "instruction": "Pick up the large light-green bowl and move it toward the center of the table",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 833,
            "end_frame": 978,
            "instruction": "Move to the orange plate and position the gripper over it",
            "confidence": 1.0,
        },
        {
            "seg_id": 2,
            "start_frame": 978,
            "end_frame": 1365,
            "instruction": "Pick up the purple bowl and place it on top of the green bowl",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 2
    assert merged[1]["start_frame"] == 833
    assert merged[1]["end_frame"] == 1365
    assert merged[1]["instruction"] == "Pick up the purple bowl and place it on top of the green bowl"


def test_merge_task_level_segments_drops_tiny_trailing_prep_segment() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 1525,
            "instruction": "Place the blue bowl onto the stack of bowls",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 1525,
            "end_frame": 1570,
            "instruction": "Reach to the orange bowl to pick it up",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 1525
    assert merged[0]["instruction"] == "Place the blue bowl onto the stack of bowls"


def test_merge_task_level_segments_keeps_distinct_bowl_recipe_actions_separate() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 150,
            "instruction": "Add mayonnaise to the red bowl",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 150,
            "end_frame": 330,
            "instruction": "Add chopped garlic to the bowl",
            "confidence": 1.0,
        },
        {
            "seg_id": 2,
            "start_frame": 330,
            "end_frame": 520,
            "instruction": "Season the mixture in the bowl",
            "confidence": 1.0,
        },
        {
            "seg_id": 3,
            "start_frame": 520,
            "end_frame": 940,
            "instruction": "Whisk the ingredients in the red bowl",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 3
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 330
    assert merged[0]["instruction"] == "Add chopped garlic to the bowl"
    assert merged[1]["start_frame"] == 330
    assert merged[1]["end_frame"] == 520
    assert merged[1]["instruction"] == "Season the mixture in the bowl"
    assert merged[2]["start_frame"] == 520
    assert merged[2]["end_frame"] == 940
    assert merged[2]["instruction"] == "Whisk the ingredients in the red bowl"


def test_merge_task_level_segments_merges_prepared_ingredient_into_following_transfer() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 180,
            "instruction": "Grate cheese onto the cutting board",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 180,
            "end_frame": 460,
            "instruction": "Add shredded cheese to the bowl",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 460
    assert merged[0]["instruction"] == "Add shredded cheese to the bowl"


def test_merge_task_level_segments_absorbs_explanatory_filler_into_following_task() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 420,
            "instruction": "Saute the onions in the pot",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 420,
            "end_frame": 560,
            "instruction": "Explain the cooking instructions",
            "confidence": 1.0,
        },
        {
            "seg_id": 2,
            "start_frame": 560,
            "end_frame": 980,
            "instruction": "Add chopped tomatoes to the pot and stir",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 2
    assert merged[1]["start_frame"] == 420
    assert merged[1]["end_frame"] == 980
    assert merged[1]["instruction"] == "Add chopped tomatoes to the pot and stir"


def test_merge_task_level_segments_keeps_distinct_salad_ingredient_tasks_separate() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 326,
            "instruction": "Slice cherry tomatoes and add them to the salad bowl",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 326,
            "end_frame": 502,
            "instruction": "Chop basil and add it to the salad bowl",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=25.0)

    assert len(merged) == 2
    assert merged[0]["instruction"] == "Slice cherry tomatoes and add them to the salad bowl"
    assert merged[1]["instruction"] == "Chop basil and add it to the salad bowl"


def test_merge_task_level_segments_keeps_distinct_pot_recipe_actions_separate() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 210,
            "instruction": "Add whole spices to the pot",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 210,
            "end_frame": 510,
            "instruction": "Stir the onions in the pot",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 2
    assert merged[0]["instruction"] == "Add whole spices to the pot"
    assert merged[1]["instruction"] == "Stir the onions in the pot"


def test_merge_task_level_segments_keeps_distinct_prep_actions_on_same_ingredient_separate() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 180,
            "instruction": "Cut the dough ball in half",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 180,
            "end_frame": 480,
            "instruction": "Roll out the dough with a rolling pin",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 2
    assert merged[0]["instruction"] == "Cut the dough ball in half"
    assert merged[1]["instruction"] == "Roll out the dough with a rolling pin"


def test_merge_task_level_segments_does_not_treat_object_name_as_action_token() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 240,
            "instruction": "Roll the spring roll",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 240,
            "end_frame": 480,
            "instruction": "Place the spring rolls on a baking sheet",
            "confidence": 1.0,
        },
    ]

    merged = merge_task_level_segments(segments, fps=30.0)

    assert len(merged) == 2
    assert merged[0]["instruction"] == "Roll the spring roll"
    assert merged[1]["instruction"] == "Place the spring rolls on a baking sheet"


def test_merge_task_level_segments_merges_same_ingredient_prep_steps_when_boundary_is_weak() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 326,
            "instruction": "Peel the ginger",
            "confidence": 1.0,
            "boundary_support_after": 0.75,
        },
        {
            "seg_id": 1,
            "start_frame": 326,
            "end_frame": 519,
            "instruction": "Grate the peeled ginger",
            "confidence": 1.0,
            "boundary_support_before": 0.75,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=25.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 519
    assert merged[0]["instruction"] == "Grate the peeled ginger"


def test_adaptive_merge_fallback_preserves_light_segments() -> None:
    fps = 30.0
    raw_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 120,
            "instruction": "Pick up the red bowl from the counter",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 120,
            "end_frame": 240,
            "instruction": "Place the red bowl onto the blue plate",
            "confidence": 1.0,
        },
        {
            "seg_id": 2,
            "start_frame": 240,
            "end_frame": 360,
            "instruction": "Stack the red bowl with the green bowl",
            "confidence": 1.0,
        },
    ]

    light_segments = cleanup_auxiliary_segments(raw_segments, fps)
    merged_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 360,
            "instruction": "Move the red bowl through the workspace",
            "confidence": 1.0,
        }
    ]

    fallback = _should_fallback_to_light_cleanup(
        light_segments,
        merged_segments,
        fps,
        min_segments=3,
        collapse_ratio=0.6,
    )

    assert fallback
    final_segments = light_segments if fallback else merged_segments
    assert final_segments == light_segments


def test_refine_segment_instructions_prefers_specific_contributors() -> None:
    final_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 240,
            "instruction": "Adjust the red bowl",
            "confidence": 1.0,
        }
    ]
    source_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 150,
            "instruction": "Pick up the red bowl from the counter",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 150,
            "end_frame": 210,
            "instruction": "Place the red bowl onto the blue plate",
            "confidence": 1.0,
        },
        {
            "seg_id": 2,
            "start_frame": 210,
            "end_frame": 240,
            "instruction": "Adjust the red bowl slightly",
            "confidence": 1.0,
        },
    ]

    refined = refine_segment_instructions(final_segments, source_segments)
    assert refined[0]["instruction"] == "Pick up the red bowl from the counter"
