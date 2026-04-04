from video2tasks.vlm.base import normalize_task_window_result


def test_normalize_task_window_result_rejects_transition_outside_allowed_indices() -> None:
    result = normalize_task_window_result(
        {"transitions": [0], "instructions": ["Add potatoes", "Stir the pot"]},
        max_transition_index=7,
        allowed_transition_indices=[2, 3, 4, 5],
    )

    assert result == {}


def test_normalize_task_window_result_rejects_zero_transition_for_regular_window() -> None:
    result = normalize_task_window_result(
        {"transitions": [0], "instructions": ["Before", "After"]},
        max_transition_index=7,
    )

    assert result == {}


def test_normalize_task_window_result_accepts_allowed_candidate_transition() -> None:
    result = normalize_task_window_result(
        {"transitions": [3], "instructions": ["Add potatoes", "Stir the pot"]},
        max_transition_index=7,
        allowed_transition_indices=[2, 3, 4, 5],
    )

    assert result == {
        "thought": "",
        "transitions": [3],
        "instructions": ["Add potatoes", "Stir the pot"],
    }
