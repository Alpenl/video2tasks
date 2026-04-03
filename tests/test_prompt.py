from video2tasks.prompt import (
    prompt_boundary_refinement,
    prompt_segment_instruction,
    prompt_switch_detection,
)


def test_prompt_switch_detection_splits_repeated_batch_chores_into_visible_rounds() -> None:
    prompt = prompt_switch_detection(24)

    assert "Repeated Batch Chores" in prompt
    assert "split into separate visible rounds" in prompt
    assert "Arrange the first flexible item" in prompt
    assert "Arrange the second flexible item" in prompt


def test_prompt_switch_detection_blocks_narration_and_passive_bridge_labels() -> None:
    prompt = prompt_switch_detection(24)

    assert "Do **NOT** output labels like" in prompt
    assert "Stand by" in prompt
    assert "Explain the task" in prompt
    assert "Narrate the action" in prompt
    assert "Observe the object" in prompt
    assert "Monitor the target area" in prompt
    assert "View the contents" in prompt
    assert "Show the result" in prompt


def test_prompt_switch_detection_treats_brief_discrete_actions_as_valid_steps() -> None:
    prompt = prompt_switch_detection(24)

    assert "brief but committed discrete manipulation" in prompt.lower()
    assert "Do **NOT** require a true step to stay visually long" in prompt
    assert "placing, removing, covering, uncovering, opening, closing, releasing, or toggling" in prompt


def test_prompt_switch_detection_merges_setup_only_spans_into_following_action() -> None:
    prompt = prompt_switch_detection(24)

    assert "bringing, holding, hovering, aligning, or positioning a source object" in prompt
    assert "merge that setup into the ensuing manipulation" in prompt.lower()
    assert "new segment, not the previous segment" in prompt
    assert "state-only labels" in prompt


def test_prompt_switch_detection_prioritizes_recall_and_earliest_onset() -> None:
    prompt = prompt_switch_detection(24)

    assert "Missing a true boundary is worse than proposing an extra one" in prompt
    assert "first committed frame" in prompt
    assert "Do **NOT** merge such committed setup into the previous task" in prompt
    assert "Do **NOT** invent objects that are not yet visible" in prompt
    assert "Over-segmentation is desirable in this pass" in prompt


def test_prompt_switch_detection_uses_tight_committed_setup_guardrails() -> None:
    prompt = prompt_switch_detection(24)

    assert "within the next few frames with no unrelated action" in prompt
    assert "do **NOT** move the boundary earlier" in prompt
    assert "first visible evidence of the new operation" in prompt


def test_prompt_switch_detection_does_not_merge_distinct_rounds_just_because_workspace_matches() -> None:
    prompt = prompt_switch_detection(24)

    assert "same workspace, same target region" in prompt.lower()
    assert "does **not** by itself justify keeping one segment" in prompt.lower()
    assert "same support surface, target region, receptacle, or workspace is **not** enough" in prompt.lower()


def test_prompt_switch_detection_keeps_early_boundary_with_objective_coarse_labels() -> None:
    prompt = prompt_switch_detection(24)

    assert "Do **NOT** wait for later frames to identify the exact object" in prompt
    assert "First placement" in prompt
    assert "Add the first visible item" in prompt


def test_prompt_switch_detection_discourages_passive_bridge_segments() -> None:
    prompt = prompt_switch_detection(24)

    assert "Do **NOT** create passive bridge segments" in prompt
    assert "intermediate state becoming visible" in prompt
    assert "causally related active manipulation" in prompt
    assert "usually the segment whose action produced that state" in prompt


def test_prompt_switch_detection_separates_onset_from_later_confirmation() -> None:
    prompt = prompt_switch_detection(24)

    assert "### Onset Versus Confirmation" in prompt
    assert "only confirmation, not the onset" in prompt
    assert "mere appearance of a new empty target region" in prompt
    assert "idle tool or source that has not yet committed to the new target" in prompt
    assert "prefer that earlier committed frame over a later clearer-looking release or confirmation frame" in prompt
    assert "prefer the earlier plausible onset" in prompt
    assert "Do **NOT** let one broad label swallow a later visible onset inside it" in prompt
    assert "Empty Target Alone Is Not Enough, But Committed Arrival Can Count" in prompt


def test_prompt_switch_detection_keeps_close_true_boundaries_when_rounds_are_distinct() -> None:
    prompt = prompt_switch_detection(24)

    assert "temporally close" in prompt
    assert "keep both candidate boundaries" in prompt
    assert "one broad bridge segment" in prompt


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


def test_prompt_switch_detection_avoids_household_and_tabletop_scene_bias() -> None:
    prompt = prompt_switch_detection(24).lower()

    assert "household manipulation" not in prompt
    assert "household chore" not in prompt
    assert "cutting board" not in prompt
    assert "drawer" not in prompt
    assert "bowl" not in prompt
    assert "tray" not in prompt
    assert "onto the table" not in prompt
    assert " on the table" not in prompt


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


def test_prompt_switch_detection_center_scan_keeps_same_workspace_and_short_actions_eligible() -> None:
    prompt = prompt_switch_detection(8, mode="center_scan")

    assert "same local work area, target region, support surface, or workspace" in prompt.lower()
    assert "brief but committed discrete manipulation" in prompt.lower()
    assert "clear discrete outcome" in prompt.lower()
    assert "start of the new task rather than the tail of the previous one" in prompt.lower()


def test_prompt_switch_detection_candidate_scan_allows_recall_first_candidates() -> None:
    prompt = prompt_switch_detection(8, mode="candidate_scan")

    assert "candidate-boundary nomination pass" in prompt.lower()
    assert "prioritize recall" in prompt.lower()
    assert "later verification" in prompt.lower()
    assert "up to 3 transitions" in prompt.lower()
    assert "one uninterrupted continuous manipulation trajectory" in prompt.lower()
    assert "first committed frame" in prompt.lower()
    assert "risk missing a true task switch" in prompt.lower()
    assert "same workspace or target region" in prompt.lower()
    assert '"transitions": [2, 5]' in prompt


def test_prompt_switch_detection_multi_probe_scan_splits_same_workspace_rounds() -> None:
    prompt = prompt_switch_detection(8, mode="multi_probe_scan")

    assert "should usually be split" in prompt.lower()
    assert "do **not** merge just because the target region or workspace matches" in prompt.lower()
    assert "placement, removal, release, cover, uncover, open, or close actions can still be valid probe boundaries" in prompt.lower()


def test_prompt_switch_detection_candidate_scan_keeps_close_neighboring_candidates() -> None:
    prompt = prompt_switch_detection(8, mode="candidate_scan")

    assert "nominate both instead of collapsing them into one broader step" in prompt.lower()
    assert "brief but committed discrete manipulation reaches a clear outcome" in prompt.lower()
    assert "part of the new task, not the old one" in prompt.lower()


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


def test_prompt_segment_instruction_prefers_narrow_completed_manipulation() -> None:
    prompt = prompt_segment_instruction(8)

    assert "narrowest completed visible manipulation" in prompt.lower()
    assert "broad scene-level activity summary" in prompt.lower()
    assert "raw json only" in prompt.lower()


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
    assert "earliest plausible candidate already has visible evidence" in prompt.lower()


def test_prompt_scan_modes_share_onset_vs_confirmation_guardrails() -> None:
    center_prompt = prompt_switch_detection(8, mode="center_scan")
    probe_prompt = prompt_switch_detection(8, mode="multi_probe_scan")
    candidate_prompt = prompt_switch_detection(8, mode="candidate_scan")
    refinement_prompt = prompt_boundary_refinement(12)

    for prompt in (center_prompt, probe_prompt, candidate_prompt, refinement_prompt):
        assert "### Onset Versus Confirmation" in prompt
        assert "only confirmation, not the onset" in prompt
        assert "mere appearance of a new empty target region" in prompt
        assert "prefer that earlier committed frame over a later clearer-looking release or confirmation frame" in prompt
        assert "one broad label swallow a later visible onset" in prompt


def test_prompt_boundary_refinement_allows_short_real_switches_in_same_workspace() -> None:
    prompt = prompt_boundary_refinement(12)

    assert "brief but committed discrete manipulation with a clear outcome" in prompt.lower()
    assert "same local work area, target region, support surface, or workspace appears on both sides" in prompt.lower()
    assert "assign it to the new task rather than the previous one" in prompt.lower()


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
