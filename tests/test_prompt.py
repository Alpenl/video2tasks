from video2tasks.prompt import prompt_switch_detection


def test_prompt_switch_detection_mentions_repeated_batch_chores() -> None:
    prompt = prompt_switch_detection(24)

    assert "Repeated Batch Chores" in prompt
    assert "folding several clothes" in prompt
    assert "one segment" in prompt


def test_prompt_switch_detection_blocks_narration_style_labels() -> None:
    prompt = prompt_switch_detection(24)

    assert "Do NOT output labels like" in prompt
    assert "Wait" in prompt
    assert "Explain the cooking step" in prompt


def test_prompt_switch_detection_preserves_distinct_ingredient_tasks_in_same_salad() -> None:
    prompt = prompt_switch_detection(24)

    assert "different ingredient" in prompt.lower()
    assert "slice tomatoes" in prompt.lower()
    assert "chop basil" in prompt.lower()


def test_prompt_switch_detection_keeps_brief_sauce_seasoning_steps_merged() -> None:
    prompt = prompt_switch_detection(24)

    assert "quick seasoning" in prompt.lower()
    assert "same sauce" in prompt.lower()
    assert "stay merged" in prompt.lower()
