from video2tasks.server.windowing import merge_task_level_segments


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
