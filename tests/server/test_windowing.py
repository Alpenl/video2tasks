from video2tasks.server.windowing import (
    FrameExtractor,
    Window,
    apply_boundary_refinement_results,
    apply_deferred_segment_labels,
    build_boundary_refinement_windows,
    build_segments_via_cuts,
    build_window_prompt_metadata,
    build_refinement_windows,
    chunk_frame_ids_for_contact_sheets,
    merge_task_level_segments,
    cleanup_auxiliary_segments,
    refine_segment_instructions,
    split_long_raw_segments_on_instruction_drift,
    _should_fallback_to_light_cleanup,
    _action_families,
    _instruction_action_head_tokens,
    _instruction_focus_tokens,
)


def test_chunk_frame_ids_for_contact_sheets_preserves_order() -> None:
    assert chunk_frame_ids_for_contact_sheets(list(range(10)), 4) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]


def test_frame_extractor_contact_sheets_fall_back_to_cv2_when_ffmpeg_returns_empty(
    monkeypatch,
) -> None:
    extractor = FrameExtractor.__new__(FrameExtractor)
    extractor.mp4_path = "demo.mp4"

    calls = []

    def fake_ffmpeg(self, group, group_start_index, target_w, target_h, rows, cols):
        calls.append(("ffmpeg", list(group), group_start_index, rows, cols))
        return "" if group_start_index == 0 else "ffmpeg-sheet"

    def fake_cv2(self, group, group_start_index, target_w, target_h, compression, rows, cols):
        calls.append(("cv2", list(group), group_start_index, rows, cols))
        return f"cv2-sheet-{group_start_index}"

    monkeypatch.setattr(FrameExtractor, "_build_contact_sheet_b64_via_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(FrameExtractor, "_build_contact_sheet_b64_via_cv2", fake_cv2)

    sheets = extractor.get_many_b64(
        [0, 1, 2, 3, 4],
        use_contact_sheets=True,
        contact_sheet_rows=2,
        contact_sheet_cols=2,
    )

    assert sheets == ["cv2-sheet-0", "ffmpeg-sheet"]
    assert calls == [
        ("ffmpeg", [0, 1, 2, 3], 0, 2, 2),
        ("cv2", [0, 1, 2, 3], 0, 2, 2),
        ("ffmpeg", [4], 4, 2, 2),
    ]




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


def test_cleanup_auxiliary_segments_attaches_medium_filler_backward_on_phase_shift() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 300,
            "instruction": "Saute the onions in the pot",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 300,
            "end_frame": 520,
            "instruction": "Explain the next cooking step",
            "confidence": 1.0,
        },
        {
            "seg_id": 2,
            "start_frame": 520,
            "end_frame": 900,
            "instruction": "Cut the tomato",
            "confidence": 1.0,
        },
    ]

    cleaned = cleanup_auxiliary_segments(segments, fps=25.0)

    assert len(cleaned) == 2
    assert cleaned[0]["start_frame"] == 0
    assert cleaned[0]["end_frame"] == 520
    assert cleaned[1]["start_frame"] == 520
    assert cleaned[1]["end_frame"] == 900


def test_cleanup_auxiliary_segments_absorbs_short_filler_segment() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 300,
            "instruction": "Saute the onions in the pot",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 300,
            "end_frame": 380,
            "instruction": "Explain the next cooking step",
            "confidence": 1.0,
        },
        {
            "seg_id": 2,
            "start_frame": 380,
            "end_frame": 900,
            "instruction": "Cut the tomato",
            "confidence": 1.0,
        },
    ]

    cleaned = cleanup_auxiliary_segments(segments, fps=25.0)

    assert len(cleaned) == 2
    assert cleaned[0]["end_frame"] == 300
    assert cleaned[1]["start_frame"] == 300
    assert cleaned[1]["end_frame"] == 900
    assert cleaned[1]["instruction"] == "Cut the tomato"


def test_cleanup_auxiliary_segments_keeps_very_long_filler_segment_as_boundary() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 300,
            "instruction": "Saute the onions in the pot",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 300,
            "end_frame": 700,
            "instruction": "Explain the next cooking step",
            "confidence": 1.0,
        },
        {
            "seg_id": 2,
            "start_frame": 700,
            "end_frame": 1100,
            "instruction": "Cut the tomato",
            "confidence": 1.0,
        },
    ]

    cleaned = cleanup_auxiliary_segments(segments, fps=25.0)

    assert len(cleaned) == 3
    assert cleaned[0]["end_frame"] == 300
    assert cleaned[1]["start_frame"] == 300
    assert cleaned[1]["end_frame"] == 700
    assert cleaned[2]["start_frame"] == 700
    assert cleaned[2]["end_frame"] == 1100


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


def test_merge_task_level_segments_merges_scallion_and_green_onion_prep_when_boundary_is_weak() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 193,
            "instruction": "Chop the scallions",
            "confidence": 1.0,
            "boundary_support_after": 0.75,
        },
        {
            "seg_id": 1,
            "start_frame": 193,
            "end_frame": 360,
            "instruction": "Trim the roots off the green onions",
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
    assert merged[0]["end_frame"] == 360
    assert merged[0]["instruction"] == "Trim the roots off the green onions"


def test_merge_task_level_segments_merges_roll_out_dough_steps_when_boundary_is_weak() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 180,
            "instruction": "Flatten and roll out the dough",
            "confidence": 1.0,
            "boundary_support_after": 0.75,
        },
        {
            "seg_id": 1,
            "start_frame": 180,
            "end_frame": 660,
            "instruction": "Roll out the dough with a rolling pin",
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
    assert merged[0]["end_frame"] == 660
    assert merged[0]["instruction"] == "Roll out the dough with a rolling pin"


def test_merge_task_level_segments_merges_dough_roll_continuation_despite_strong_boundary() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 180,
            "instruction": "Flatten and roll out the dough",
            "confidence": 1.0,
            "boundary_support_after": 0.97,
        },
        {
            "seg_id": 1,
            "start_frame": 180,
            "end_frame": 660,
            "instruction": "Roll out the dough with a rolling pin",
            "confidence": 1.0,
            "boundary_support_before": 0.97,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=25.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 660
    assert merged[0]["instruction"] == "Roll out the dough with a rolling pin"


def test_merge_task_level_segments_merges_same_container_add_steps_when_boundary_is_weak() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 390,
            "instruction": "Add broccoli slaw to the bowl",
            "confidence": 1.0,
            "boundary_support_after": 1.1,
        },
        {
            "seg_id": 1,
            "start_frame": 390,
            "end_frame": 780,
            "instruction": "Add chopped garlic to the bowl",
            "confidence": 1.0,
            "boundary_support_before": 1.1,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=25.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 780
    assert merged[0]["instruction"] == "Add chopped garlic to the bowl"


def test_merge_task_level_segments_keeps_distinct_pot_add_sequences_separate_even_with_weak_boundary() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 436,
            "instruction": "Pour water into the pot with chopped potatoes",
            "confidence": 1.0,
            "boundary_support_after": 0.75,
        },
        {
            "seg_id": 1,
            "start_frame": 436,
            "end_frame": 642,
            "instruction": "Pour peas into the pot",
            "confidence": 1.0,
            "boundary_support_before": 0.75,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=25.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 2
    assert merged[0]["instruction"] == "Pour water into the pot with chopped potatoes"
    assert merged[1]["instruction"] == "Pour peas into the pot"


def test_merge_task_level_segments_keeps_add_and_stir_separate_even_with_weak_boundary() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 390,
            "instruction": "Add chopped garlic to the bowl",
            "confidence": 1.0,
            "boundary_support_after": 0.4,
        },
        {
            "seg_id": 1,
            "start_frame": 390,
            "end_frame": 780,
            "instruction": "Stir the mixture in the bowl",
            "confidence": 1.0,
            "boundary_support_before": 0.4,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=25.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 2
    assert merged[0]["instruction"] == "Add chopped garlic to the bowl"
    assert merged[1]["instruction"] == "Stir the mixture in the bowl"


def test_merge_task_level_segments_keeps_generic_mix_phase_separate_from_new_explicit_ingredient_phase() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 720,
            "instruction": "Stir and mash the mixture in the pot until it is well combined",
            "confidence": 1.0,
            "boundary_support_after": 1.05,
        },
        {
            "seg_id": 1,
            "start_frame": 720,
            "end_frame": 1080,
            "instruction": "Stir the peas in the pot",
            "confidence": 1.0,
            "boundary_support_before": 1.05,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=30.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 2
    assert merged[0]["instruction"] == "Stir and mash the mixture in the pot until it is well combined"
    assert merged[1]["instruction"] == "Stir the peas in the pot"


def test_instruction_focus_tokens_prefers_specific_target_in_destination_clause() -> None:
    focus = _instruction_focus_tokens(
        "Sprinkle seasoning over the sliced potatoes in the pot"
    )

    assert "potatoe" in focus
    assert "seasoning" not in focus


def test_instruction_focus_tokens_prefers_dish_subject_over_seasoning_tokens() -> None:
    focus = _instruction_focus_tokens(
        "Season the salad with five-spice powder"
    )

    assert "salad" in focus
    assert "powder" not in focus


def test_instruction_focus_tokens_uses_destination_dish_when_subject_is_additive_only() -> None:
    focus = _instruction_focus_tokens(
        "Grind pepper into the salad bowl"
    )

    assert "salad" in focus
    assert "pepper" not in focus


def test_merge_task_level_segments_merges_same_target_food_when_seasoning_moves_to_destination_clause() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 480,
            "instruction": "Season the sliced potatoes in the pot",
            "confidence": 1.0,
            "boundary_support_after": 0.0,
        },
        {
            "seg_id": 1,
            "start_frame": 480,
            "end_frame": 918,
            "instruction": "Sprinkle seasoning over the sliced potatoes in the pot",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=30.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 918


def test_merge_task_level_segments_merges_short_same_pan_toast_continuation_with_shared_focus() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 300,
            "instruction": "Prepare croutons by toasting bread cubes with butter and seasoning in a pan",
            "confidence": 1.0,
            "boundary_support_after": 0.0,
        },
        {
            "seg_id": 1,
            "start_frame": 300,
            "end_frame": 420,
            "instruction": "Toast bread cubes in a pan with butter",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=30.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 420
    assert merged[0]["instruction"] == "Toast bread cubes in a pan with butter"


def test_merge_task_level_segments_merges_short_generic_placeholder_prep_segment_in_same_bowl_phase() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 240,
            "instruction": "Grate the green apple into the bowl",
            "confidence": 1.0,
            "boundary_support_after": 0.0,
        },
        {
            "seg_id": 1,
            "start_frame": 240,
            "end_frame": 330,
            "instruction": "Grate the ingredient into the bowl",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=30.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 330
    assert merged[0]["instruction"] == "Grate the green apple into the bowl"


def test_merge_task_level_segments_merges_short_additive_only_seasoning_into_same_bowl_mix_phase() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 150,
            "instruction": "Grind pepper into the salad bowl",
            "confidence": 1.0,
            "boundary_support_after": 0.0,
        },
        {
            "seg_id": 1,
            "start_frame": 150,
            "end_frame": 390,
            "instruction": "Toss the lettuce in the glass bowl",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=30.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 390
    assert merged[0]["instruction"] == "Toss the lettuce in the glass bowl"


def test_merge_task_level_segments_merges_additive_only_seasoning_into_same_bowl_mix_phase_for_medium_segments() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 335,
            "instruction": "Grind pepper into the salad bowl",
            "confidence": 1.0,
            "boundary_support_after": 0.0,
        },
        {
            "seg_id": 1,
            "start_frame": 335,
            "end_frame": 785,
            "instruction": "Toss the lettuce in the glass bowl",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=30.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 785
    assert merged[0]["instruction"] == "Toss the lettuce in the glass bowl"


def test_merge_task_level_segments_merges_pre_cook_pot_loading_run_but_keeps_following_cook_phase() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 205,
            "instruction": "Pour peas into the pot and season with salt",
            "confidence": 1.0,
            "boundary_support_after": 0.69,
        },
        {
            "seg_id": 1,
            "start_frame": 205,
            "end_frame": 318,
            "instruction": "Add mint leaves to the pot",
            "confidence": 1.0,
            "boundary_support_before": 0.69,
            "boundary_support_after": 0.96,
        },
        {
            "seg_id": 2,
            "start_frame": 318,
            "end_frame": 693,
            "instruction": "Sprinkle sugar into the pot of peas and herbs",
            "confidence": 1.0,
            "boundary_support_before": 0.96,
            "boundary_support_after": 0.92,
        },
        {
            "seg_id": 3,
            "start_frame": 693,
            "end_frame": 940,
            "instruction": "Cook the peas and herbs in the pot",
            "confidence": 1.0,
            "boundary_support_before": 0.92,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=30.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 2
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 693
    assert merged[1]["start_frame"] == 693
    assert merged[1]["end_frame"] == 940
    assert merged[1]["instruction"] == "Cook the peas and herbs in the pot"


def test_merge_task_level_segments_keeps_generic_spice_loading_steps_separate_in_pot() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 210,
            "instruction": "Add spices to the pot",
            "confidence": 1.0,
            "boundary_support_after": 0.75,
        },
        {
            "seg_id": 1,
            "start_frame": 210,
            "end_frame": 420,
            "instruction": "Add the diced ingredients to the pot",
            "confidence": 1.0,
            "boundary_support_before": 0.75,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=25.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 2
    assert merged[0]["instruction"] == "Add spices to the pot"
    assert merged[1]["instruction"] == "Add the diced ingredients to the pot"


def test_merge_task_level_segments_merges_same_pot_masher_continuation_but_keeps_next_phase_shift() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 332,
            "instruction": "Pour additional liquid into the pot",
            "confidence": 1.0,
            "boundary_support_after": 0.82,
        },
        {
            "seg_id": 1,
            "start_frame": 332,
            "end_frame": 992,
            "instruction": "Mash the potatoes in the pot with the potato masher",
            "confidence": 1.0,
            "boundary_support_before": 0.82,
            "boundary_support_after": 0.0,
        },
        {
            "seg_id": 2,
            "start_frame": 992,
            "end_frame": 1708,
            "instruction": "Stir and mash the mixture in the pot until it is well combined",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 1.06,
        },
        {
            "seg_id": 3,
            "start_frame": 1708,
            "end_frame": 2023,
            "instruction": "Stir the peas in the pot",
            "confidence": 1.0,
            "boundary_support_before": 1.06,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=30.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 2
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 1708
    assert merged[1]["start_frame"] == 1708
    assert merged[1]["instruction"] == "Stir the peas in the pot"


def test_merge_task_level_segments_merges_same_ingredient_finish_chain_with_butter_and_mash() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 315,
            "instruction": "Stir the peas in the pot",
            "confidence": 1.0,
            "boundary_support_after": 2.09,
        },
        {
            "seg_id": 1,
            "start_frame": 315,
            "end_frame": 540,
            "instruction": "Stir the peas and butter together with a spoon",
            "confidence": 1.0,
            "boundary_support_before": 2.09,
            "boundary_support_after": 1.06,
        },
        {
            "seg_id": 2,
            "start_frame": 540,
            "end_frame": 804,
            "instruction": "Mash the peas in the pot with the potato masher",
            "confidence": 1.0,
            "boundary_support_before": 1.06,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=30.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 804
    assert merged[0]["instruction"] == "Mash the peas in the pot with the potato masher"


def test_merge_task_level_segments_merges_plated_mashed_potato_finish_chain() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 144,
            "instruction": "Scoop and place a mound of mashed potatoes onto the plate",
            "confidence": 1.0,
            "boundary_support_after": 0.83,
        },
        {
            "seg_id": 1,
            "start_frame": 144,
            "end_frame": 386,
            "instruction": "Place the sausages on top of the mashed potatoes",
            "confidence": 1.0,
            "boundary_support_before": 0.83,
            "boundary_support_after": 0.47,
        },
        {
            "seg_id": 2,
            "start_frame": 386,
            "end_frame": 609,
            "instruction": "Finish adding sauce/onion topping over the sausages and mashed potatoes",
            "confidence": 1.0,
            "boundary_support_before": 0.47,
            "boundary_support_after": 1.04,
        },
        {
            "seg_id": 3,
            "start_frame": 609,
            "end_frame": 703,
            "instruction": "Place a green garnish (e.g., puree/herb) on top of the dish",
            "confidence": 1.0,
            "boundary_support_before": 1.04,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=30.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 703


def test_merge_task_level_segments_merges_wrapper_fill_then_roll_into_single_assembly_step() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 575,
            "instruction": "Place the filling onto the wrapper",
            "confidence": 1.0,
            "boundary_support_after": 1.4,
        },
        {
            "seg_id": 1,
            "start_frame": 575,
            "end_frame": 2037,
            "instruction": "Roll the spring roll",
            "confidence": 1.0,
            "boundary_support_before": 1.4,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=25.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 2037
    assert merged[0]["instruction"] == "Roll the spring roll"


def test_merge_task_level_segments_merges_wrapper_fill_then_pleat_into_single_assembly_step() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 784,
            "instruction": "Assemble the dumplings by filling and folding the wrappers",
            "confidence": 1.0,
            "boundary_support_after": 1.1,
        },
        {
            "seg_id": 1,
            "start_frame": 784,
            "end_frame": 1193,
            "instruction": "Pleat the edges of the dumpling wrapper",
            "confidence": 1.0,
            "boundary_support_before": 1.1,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=25.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 1193
    assert merged[0]["instruction"] == "Pleat the edges of the dumpling wrapper"


def test_merge_task_level_segments_merges_dumpling_assembly_then_pleat_into_single_step() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 784,
            "instruction": "Assemble the dumplings by filling, folding, and plating them",
            "confidence": 1.0,
            "boundary_support_after": 0.97,
        },
        {
            "seg_id": 1,
            "start_frame": 784,
            "end_frame": 1193,
            "instruction": "Pleat the edges of the dumpling wrapper",
            "confidence": 1.0,
            "boundary_support_before": 0.97,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=25.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 1
    assert merged[0]["start_frame"] == 0
    assert merged[0]["end_frame"] == 1193
    assert merged[0]["instruction"] == "Pleat the edges of the dumpling wrapper"


def test_merge_task_level_segments_keeps_wrapper_transfer_and_frying_separate() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 331,
            "instruction": "Place the spring rolls on a baking sheet",
            "confidence": 1.0,
            "boundary_support_after": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 331,
            "end_frame": 632,
            "instruction": "Deep fry the spring rolls in the wok",
            "confidence": 1.0,
            "boundary_support_before": 1.0,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=25.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 2
    assert merged[0]["instruction"] == "Place the spring rolls on a baking sheet"
    assert merged[1]["instruction"] == "Deep fry the spring rolls in the wok"


def test_merge_task_level_segments_keeps_long_same_action_pot_steps_separate() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 111,
            "instruction": "Add the sliced ingredients to the pot and stir",
            "confidence": 1.0,
            "boundary_support_after": 1.4,
        },
        {
            "seg_id": 1,
            "start_frame": 111,
            "end_frame": 750,
            "instruction": "Add spices and seasoning to the pot",
            "confidence": 1.0,
            "boundary_support_before": 1.4,
        },
    ]

    merged = merge_task_level_segments(
        segments,
        fps=25.0,
        boundary_support_threshold=0.9,
    )

    assert len(merged) == 2
    assert merged[0]["instruction"] == "Add the sliced ingredients to the pot and stir"
    assert merged[1]["instruction"] == "Add spices and seasoning to the pot"


def test_split_long_raw_segments_on_instruction_drift_splits_long_heated_add_sequence() -> None:
    raw_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 900,
            "instruction": "Add ingredients into the pot",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.0,
        }
    ]
    instruction_timeline = (
        [["Add ginger and garlic paste to the pot"] for _ in range(450)]
        + [["Add chopped tomatoes to the pot"] for _ in range(450)]
    )

    split_segments = split_long_raw_segments_on_instruction_drift(
        raw_segments,
        instruction_timeline,
        fps=25.0,
    )

    assert len(split_segments) == 2
    assert split_segments[0]["start_frame"] == 0
    assert split_segments[0]["end_frame"] == 450
    assert split_segments[0]["instruction"] == "Add ginger and garlic paste to the pot"
    assert split_segments[1]["start_frame"] == 450
    assert split_segments[1]["end_frame"] == 900
    assert split_segments[1]["instruction"] == "Add chopped tomatoes to the pot"


def test_split_long_raw_segments_on_instruction_drift_keeps_long_bowl_assembly_together() -> None:
    raw_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 900,
            "instruction": "Add ingredients into the bowl",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.0,
        }
    ]
    instruction_timeline = (
        [["Add chopped scallions to the bowl"] for _ in range(450)]
        + [["Add minced garlic to the bowl"] for _ in range(450)]
    )

    split_segments = split_long_raw_segments_on_instruction_drift(
        raw_segments,
        instruction_timeline,
        fps=25.0,
    )

    assert len(split_segments) == 1
    assert split_segments[0]["start_frame"] == 0
    assert split_segments[0]["end_frame"] == 900
    assert split_segments[0]["instruction"] == "Add ingredients into the bowl"


def test_split_long_raw_segments_on_instruction_drift_keeps_bowl_add_and_mix_assembly_together() -> None:
    raw_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 900,
            "instruction": "Prepare the salad in the bowl",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.0,
        }
    ]
    instruction_timeline = (
        [["Add minced garlic to the bowl"] for _ in range(450)]
        + [["Toss the salad in the bowl"] for _ in range(450)]
    )

    split_segments = split_long_raw_segments_on_instruction_drift(
        raw_segments,
        instruction_timeline,
        fps=25.0,
    )

    assert len(split_segments) == 1
    assert split_segments[0]["start_frame"] == 0
    assert split_segments[0]["end_frame"] == 900
    assert split_segments[0]["instruction"] == "Prepare the salad in the bowl"


def test_split_long_raw_segments_on_instruction_drift_splits_sauce_phase_from_sausage_return() -> None:
    raw_segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 800,
            "instruction": "Add liquid to the sautéed onions and tomatoes in the pan",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
            "boundary_support_after": 0.0,
        }
    ]
    instruction_timeline = (
        [["Add sauce to the onion and tomato mixture in the pan"] for _ in range(100)]
        + [["Cook the tomato sauce in the pan"] for _ in range(100)]
        + [["Add liquid to the sauce in the pan"] for _ in range(100)]
        + [["Cook the tomato sauce in the pan"] for _ in range(100)]
        + [["Add and arrange the sausages in the sauce"] for _ in range(400)]
    )

    split_segments = split_long_raw_segments_on_instruction_drift(
        raw_segments,
        instruction_timeline,
        fps=25.0,
    )

    assert len(split_segments) == 2
    assert split_segments[0]["start_frame"] == 0
    assert split_segments[0]["end_frame"] == 400
    assert split_segments[0]["instruction"] == "Cook the tomato sauce in the pan"
    assert split_segments[1]["start_frame"] == 400
    assert split_segments[1]["end_frame"] == 800
    assert split_segments[1]["instruction"] == "Add and arrange the sausages in the sauce"


def test_build_segments_via_cuts_prunes_single_window_cut_for_multi_probe_mode() -> None:
    windows = [
        Window(0, 0, 299, [0, 42, 85, 128, 170, 213, 256, 299]),
        Window(1, 150, 449, [150, 192, 235, 278, 320, 363, 406, 449]),
        Window(2, 300, 599, [300, 342, 385, 428, 470, 513, 556, 599]),
    ]
    by_wid = {
        0: {
            "window_id": 0,
            "vlm_json": {
                "transitions": [4],
                "instructions": ["Add spices into the pot", "Prepare the onions on the cutting board"],
            },
        },
        1: {
            "window_id": 1,
            "vlm_json": {
                "transitions": [5],
                "instructions": ["Add spices into the pot", "Prepare the onions on the cutting board"],
            },
        },
        2: {
            "window_id": 2,
            "vlm_json": {
                "transitions": [2],
                "instructions": ["Add spices into the pot", "Prepare the onions on the cutting board"],
            },
        },
    }

    result = build_segments_via_cuts(
        "demo",
        windows,
        by_wid,
        fps=25.0,
        nframes=600,
        frames_per_window=8,
        boundary_prompt_mode="multi_probe_scan",
        refine_final_instructions=False,
    )

    boundaries = [segment["end_frame"] for segment in result["segments"][:-1]]

    assert len(boundaries) == 1
    assert 340 <= boundaries[0] <= 400


def test_apply_deferred_segment_labels_overrides_final_instructions() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 100,
            "instruction": "Add ingredients into the pot",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 100,
            "end_frame": 200,
            "instruction": "Cut vegetables on the board",
            "confidence": 1.0,
        },
    ]
    label_results = {
        0: {"instructions": ["Add black cardamom to the pot"]},
        1: {"instructions": ["Chop the tomatoes on the cutting board"]},
    }

    labeled = apply_deferred_segment_labels(segments, label_results)

    assert labeled[0]["instruction"] == "Add black cardamom to the pot"
    assert labeled[1]["instruction"] == "Chop the tomatoes on the cutting board"


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


def test_build_window_prompt_metadata_includes_temporal_context() -> None:
    window = Window(
        window_id=3,
        start_frame=90,
        end_frame=389,
        frame_ids=[90, 180, 270, 360],
    )

    metadata = build_window_prompt_metadata(window, fps=29.97, nframes=1200)

    assert metadata == {
        "window_id": 3,
        "frame_ids": [90, 180, 270, 360],
        "fps": 29.97,
        "window_start_frame": 90,
        "window_end_frame": 389,
        "nframes": 1200,
    }


def test_build_refinement_windows_splits_ambiguous_multi_phase_window() -> None:
    base_window = Window(
        window_id=0,
        start_frame=0,
        end_frame=119,
        frame_ids=[0, 17, 34, 51, 68, 85, 102, 119],
    )
    by_wid = {
        0: {
            "window_id": 0,
            "vlm_json": {
                "transitions": [],
                "instructions": ["Add spices to the pot and stir the mixture"],
            },
        }
    }

    refinement_windows = build_refinement_windows(
        [base_window],
        by_wid,
        fps=10.0,
        nframes=240,
        frames_per_window=8,
    )

    assert [window.window_id for window in refinement_windows] == [1000000, 1000001, 1000002]
    assert [(window.start_frame, window.end_frame) for window in refinement_windows] == [
        (0, 59),
        (30, 89),
        (60, 119),
    ]
    assert all(len(window.frame_ids) == 8 for window in refinement_windows)


def test_build_refinement_windows_skips_clean_transfer_window() -> None:
    base_window = Window(
        window_id=2,
        start_frame=120,
        end_frame=239,
        frame_ids=[120, 137, 154, 171, 188, 205, 222, 239],
    )
    by_wid = {
        2: {
            "window_id": 2,
            "vlm_json": {
                "transitions": [],
                "instructions": ["Pick up the bowl and place it on the tray"],
            },
        }
    }

    refinement_windows = build_refinement_windows(
        [base_window],
        by_wid,
        fps=10.0,
        nframes=240,
        frames_per_window=8,
    )

    assert refinement_windows == []


def test_build_boundary_refinement_windows_centers_short_clip_on_boundary() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 100,
            "instruction": "Add spices to the pot",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 100,
            "end_frame": 220,
            "instruction": "Stir the ingredients in the pot",
            "confidence": 1.0,
        },
    ]

    refinement_windows = build_boundary_refinement_windows(
        segments,
        fps=25.0,
        nframes=400,
        window_sec=4.0,
        frames_per_window=12,
    )

    assert len(refinement_windows) == 1
    assert refinement_windows[0].boundary_id == 0
    assert refinement_windows[0].coarse_boundary_frame == 100
    assert len(refinement_windows[0].frame_ids) == 12
    assert refinement_windows[0].start_frame < 100 < refinement_windows[0].end_frame


def test_apply_boundary_refinement_results_shifts_boundary_to_selected_frame() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 100,
            "instruction": "Add spices to the pot",
            "confidence": 1.0,
        },
        {
            "seg_id": 1,
            "start_frame": 100,
            "end_frame": 220,
            "instruction": "Stir the ingredients in the pot",
            "confidence": 1.0,
        },
    ]
    refinement_results = {
        0: {
            "boundary_id": 0,
            "frame_ids": [60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 140, 150],
            "vlm_json": {"transitions": [4], "instructions": ["Add spices", "Stir the pot"]},
        }
    }

    refined = apply_boundary_refinement_results(segments, refinement_results)

    assert refined[0]["end_frame"] == 95
    assert refined[1]["start_frame"] == 95


def test_apply_boundary_refinement_results_merges_low_support_boundary_when_model_abstains() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 100,
            "instruction": "Prepare ingredients on the counter",
            "confidence": 1.0,
            "boundary_support_after": 0.0,
        },
        {
            "seg_id": 1,
            "start_frame": 100,
            "end_frame": 220,
            "instruction": "Chop the tomatoes on the cutting board",
            "confidence": 1.0,
            "boundary_support_before": 0.0,
        },
    ]
    refinement_results = {
        0: {
            "boundary_id": 0,
            "frame_ids": [60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 140, 150],
            "vlm_json": {"transitions": [], "instructions": ["same task"]},
        }
    }

    refined = apply_boundary_refinement_results(
        segments,
        refinement_results,
        fps=25.0,
        abstain_merge_max_support=0.0,
    )

    assert len(refined) == 1
    assert refined[0]["seg_id"] == 0
    assert refined[0]["start_frame"] == 0
    assert refined[0]["end_frame"] == 220
    assert refined[0]["instruction"] == "Chop the tomatoes on the cutting board"


def test_apply_boundary_refinement_results_keeps_high_support_boundary_when_model_abstains() -> None:
    segments = [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 100,
            "instruction": "Add spices to the pot",
            "confidence": 1.0,
            "boundary_support_after": 1.4,
        },
        {
            "seg_id": 1,
            "start_frame": 100,
            "end_frame": 220,
            "instruction": "Stir the ingredients in the pot",
            "confidence": 1.0,
            "boundary_support_before": 1.4,
        },
    ]
    refinement_results = {
        0: {
            "boundary_id": 0,
            "frame_ids": [60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 140, 150],
            "vlm_json": {"transitions": [], "instructions": ["same task"]},
        }
    }

    refined = apply_boundary_refinement_results(
        segments,
        refinement_results,
        fps=25.0,
        abstain_merge_max_support=0.0,
    )

    assert len(refined) == 2
    assert refined[0]["end_frame"] == 100
    assert refined[1]["start_frame"] == 100
