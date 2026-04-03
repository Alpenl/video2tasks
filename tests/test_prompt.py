from video2tasks.prompt import (
    prompt_boundary_refinement,
    prompt_segment_instruction,
    prompt_switch_detection,
)


def test_prompt_switch_detection_mentions_repeated_batch_chores() -> None:
    prompt = prompt_switch_detection(24)

    assert "Repeated Batch Chores" in prompt
    assert "folding several clothes" in prompt.lower()
    assert "one segment" in prompt


def test_prompt_switch_detection_blocks_narration_style_labels() -> None:
    prompt = prompt_switch_detection(24)

    assert "Do **NOT** output labels like" in prompt
    assert "Stand by" in prompt
    assert "Explain the task" in prompt
    assert "Narrate the action" in prompt


def test_prompt_switch_detection_prioritizes_recall_and_earliest_onset() -> None:
    prompt = prompt_switch_detection(24)

    assert "Missing a true boundary is worse than proposing an extra one" in prompt
    assert "first committed frame" in prompt
    assert "Do **NOT** invent objects that are not yet visible" in prompt
    assert "Over-segmentation is desirable in this pass" in prompt


def test_prompt_switch_detection_uses_objective_labels_when_identity_is_unclear() -> None:
    prompt = prompt_switch_detection(24)

    assert "Dispense granular material" in prompt
    assert "Pour dark liquid" in prompt
    assert "First pour" in prompt
    assert "Second pour" in prompt


def test_prompt_switch_detection_avoids_domain_specific_cooking_bias() -> None:
    prompt = prompt_switch_detection(24).lower()

    assert "salad" not in prompt
    assert "recipe" not in prompt
    assert "dressing" not in prompt
    assert "whisk" not in prompt
    assert "onion" not in prompt
    assert "seasoning" not in prompt
    assert "pepper" not in prompt


def test_prompt_switch_detection_center_scan_restricts_to_one_middle_boundary() -> None:
    prompt = prompt_switch_detection(8, mode="center_scan")

    assert "at most one true boundary" in prompt.lower()
    assert "middle of the clip" in prompt.lower()
    assert "first committed frame" in prompt.lower()
    assert "keep the boundary rather than suppressing it" in prompt.lower()
    assert '"transitions": [3]' in prompt


def test_prompt_switch_detection_multi_probe_scan_limits_boundaries_to_probe_positions() -> None:
    prompt = prompt_switch_detection(8, mode="multi_probe_scan")

    assert "probe positions" in prompt.lower()
    assert "[2, 4, 5]" in prompt
    assert "first committed frame" in prompt.lower()
    assert "only output transitions chosen from those probe positions" in prompt.lower()
    assert '"transitions": [2, 5]' in prompt


def test_prompt_switch_detection_candidate_scan_allows_recall_first_candidates() -> None:
    prompt = prompt_switch_detection(8, mode="candidate_scan")

    assert "candidate-boundary nomination pass" in prompt.lower()
    assert "prioritize recall" in prompt.lower()
    assert "later verification" in prompt.lower()
    assert "up to 3 transitions" in prompt.lower()
    assert "first committed frame" in prompt.lower()
    assert "risk missing a true task switch" in prompt.lower()
    assert "same workspace or container" in prompt.lower()
    assert '"transitions": [2, 5]' in prompt


def test_prompt_switch_detection_contact_sheet_layout_uses_logical_indices() -> None:
    prompt = prompt_switch_detection(48, contact_sheet_rows=4, contact_sheet_cols=4, sheet_count=3)

    assert "Contact Sheet Layout" in prompt
    assert "logical frame index" in prompt.lower()
    assert "left-to-right, top-to-bottom" in prompt
    assert "Use those tile indices in `transitions`" in prompt


def test_prompt_segment_instruction_requests_single_label_without_boundaries() -> None:
    prompt = prompt_segment_instruction(8)

    assert "already a single task segment" in prompt.lower()
    assert "do not split it further" in prompt.lower()
    assert "grounded but coarse names" in prompt.lower()
    assert '"transitions": []' in prompt
    assert '"instructions": ["one concise task instruction"]' in prompt


def test_prompt_segment_instruction_contact_sheet_layout_uses_logical_indices() -> None:
    prompt = prompt_segment_instruction(32, contact_sheet_rows=4, contact_sheet_cols=4, sheet_count=2)

    assert "Contact Sheet Layout" in prompt
    assert "logical frame index" in prompt.lower()
    assert "Use those tile indices in `transitions`" in prompt


def test_prompt_boundary_refinement_prefers_earliest_supported_candidate() -> None:
    prompt = prompt_boundary_refinement(12)

    assert "plausibly true task switch" in prompt.lower()
    assert "earliest supported onset" in prompt.lower()
    assert "closest supported candidate" in prompt.lower()


def test_prompt_boundary_refinement_restricts_output_to_middle_candidates() -> None:
    prompt = prompt_boundary_refinement(12)

    assert "candidate boundary" in prompt.lower()
    assert "refine the boundary location" in prompt.lower()
    assert "[4, 5, 6, 7]" in prompt
    assert "[] or exactly one index" in prompt.lower()


def test_prompt_boundary_refinement_contact_sheet_layout_uses_logical_indices() -> None:
    prompt = prompt_boundary_refinement(32, contact_sheet_rows=4, contact_sheet_cols=4, sheet_count=2)

    assert "Contact Sheet Layout" in prompt
    assert "logical frame index" in prompt.lower()
    assert "Use those tile indices in `transitions`" in prompt
