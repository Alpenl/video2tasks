def _boundary_probe_positions(n_images: int) -> list[int]:
    if n_images <= 2:
        return []

    min_idx = 1
    max_idx = max(n_images - 2, min_idx)
    ratios = (0.25, 0.5, 0.75)
    probes = []
    for ratio in ratios:
        idx = int(round((n_images - 1) * ratio))
        idx = max(min_idx, min(max_idx, idx))
        probes.append(idx)
    return sorted(set(probes))


def _boundary_refinement_positions(n_images: int) -> list[int]:
    if n_images <= 2:
        return []

    middle_left = max((n_images // 2) - 1, 0)
    middle_right = min(n_images // 2, max(n_images - 1, 0))
    candidates = {
        max(1, middle_left - 1),
        max(1, middle_left),
        min(n_images - 2, middle_right),
        min(n_images - 2, middle_right + 1),
    }
    return sorted(idx for idx in candidates if 0 < idx < (n_images - 1))


def _shared_earliest_onset_anchoring() -> str:
    return (
        "### Earliest Onset Anchoring\n"
        "Anchor each boundary to the **first committed frame** of the new step.\n"
        "For a pour, cut on the first frame the new liquid starts being poured, not when the pool becomes obvious.\n"
        "For a sprinkle or other granular application, cut on the first committed release, not when the accumulation becomes obvious.\n"
        "For an addition or placement, cut on the first frame the new item starts entering the target workspace or container, not when it is fully inside.\n"
        "For a new tool or source container, cut on the first frame that tool or source begins the new sustained operation.\n"
        "If a new source or tool arrives directly over the target and the addition starts immediately after, anchor at that committed arrival instead of waiting for visible accumulation.\n"
        "If the same source or tool stops, leaves, and later returns to start another visible round, begin a new segment at that return.\n"
        "If the same broad motion happens twice with a different source object, material round, or target phase, split them as separate steps and label them objectively, for example `First pour` and `Second pour` when needed.\n"
        "Do **NOT** invent objects that are not yet visible; anchor the cut to the first visible evidence of the new step.\n\n"
    )


def _shared_labeling_rules() -> str:
    return (
        "### Labeling Rules\n"
        "Do **NOT** output labels like `Wait`, `Stand by`, `Explain the task`, `Describe the scene`, `Narrate the action`, `Prepare the bag`, `Handle the object`, or `Manage the item`.\n"
        "Always anchor the instruction to the visible manipulation goal.\n"
        "Prefer concrete verbs plus visible objects.\n"
        "When identity is ambiguous, prefer grounded but coarse labels like `Dispense granular material`, `Pour dark liquid`, `Place a flat item`, `First pour`, or `Second pour` instead of guessing exact product, material, or brand names.\n"
        "Ignore idle self-adjustment tails such as adjusting clothing, rubbing hands, or resetting posture after a task ends.\n\n"
    )


def _shared_output_format_description() -> str:
    return (
        "Return a valid JSON object with keys `thought`, `transitions`, and `instructions`.\n"
        "Keep the `thought` field extremely short: one sentence, no more than 20 words, and no markdown fences.\n"
        "Instructions should be concise, task-level, and suitable as robot training commands.\n"
        "Prefer concrete verbs plus visible objects. When identity is ambiguous, prefer grounded but coarse names over confident guessing.\n"
    )


def _contact_sheet_layout_guidance(
    contact_sheet_rows: int = 0,
    contact_sheet_cols: int = 0,
    sheet_count: int = 0,
) -> str:
    if contact_sheet_rows <= 0 or contact_sheet_cols <= 0 or sheet_count <= 0:
        return ""

    return (
        "### Contact Sheet Layout\n"
        f"The clip is packed into {sheet_count} uploaded contact sheet image(s).\n"
        f"Each contact sheet is a {contact_sheet_cols}-column by {contact_sheet_rows}-row grid.\n"
        "Each tile already shows its logical frame index in the corner.\n"
        "Read tiles left-to-right, top-to-bottom across the first sheet, then continue to the next sheet.\n"
        "Use those tile indices in `transitions`, not the uploaded image count.\n\n"
    )


def _prompt_switch_detection_freeform(
    n_images: int,
    contact_sheet_rows: int = 0,
    contact_sheet_cols: int = 0,
    sheet_count: int = 0,
) -> str:
    return (
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip of household manipulation tasks.\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n\n"
        f"{_contact_sheet_layout_guidance(contact_sheet_rows, contact_sheet_cols, sheet_count)}"
        "### Goal\n"
        "Detect **task-level boundaries** that are useful for VLA training.\n"
        "Each segment should be one coherent robot instruction from start to finish.\n"
        "Do **not** miss sustained new goals: if the actor clearly finishes or abandons one goal and starts another that continues for several frames, mark a switch.\n"
        "A 'Switch' happens when the robot starts a **new sustained manipulation phase**, not merely when it briefly touches a new object.\n\n"
        "### Boundary Priority\n"
        "Missing a true boundary is worse than proposing an extra one.\n"
        "This is a pre-merge over-segmentation pass: extra adjacent boundaries are acceptable because a later model can merge them.\n"
        "Prefer the smallest objectively visible sustained step, not a broad summary of the whole activity.\n"
        "When uncertain between one broad segment and two narrower adjacent segments, choose the narrower segmentation.\n"
        "Prefer splitting repeated additions, repeated pours, repeated granular applications, repeated placements, and repeated removals into separate visible rounds whenever a committed new round is visible.\n"
        "Place the boundary at the earliest frame where the new sustained operation is already visible.\n"
        "Do **NOT** delay the cut until the new action is fully underway.\n"
        "If the visible operation changes and the new step lasts for several frames, keep the boundary.\n"
        "Do **NOT** collapse separate additions, granular applications, tool changes, container changes, removal steps, or plating steps into one broad instruction.\n"
        "Only keep one segment when the motion is genuinely uninterrupted and there is no visible stop, reset, withdrawal, or re-entry before the next round.\n\n"
        f"{_shared_earliest_onset_anchoring()}"
        "### Core Logic\n"
        "1. **True Switch:** In this pass, treat each new sustained manipulation phase as a switch, even inside one larger household chore.\n"
        "2. **Same Task Across Support Objects:** If the robot is still pursuing the same outcome while touching the carried object, a receptacle, a support object, or the final stack, this is **NOT** a switch.\n"
        "3. **Ignore Micro-Adjustments:** Brief nudges, stabilization, re-grasps, post-placement corrections, and tiny follow-up adjustments around the same task are **NOT** new tasks.\n"
        "4. **Preparation Is Part of the Main Task:** Reaching, hovering, aligning, or partially grasping in preparation for the main action should stay inside that same task.\n"
        "5. **Prefer Separate Visible Rounds:** If the actor changes material, tool, container, workpiece, source vessel, target vessel, or manipulation phase and the new step persists, prefer a new boundary.\n"
        "If the source and target stay the same but there is a visible stop-and-restart, hand reset, source withdrawal and return, or another committed round, prefer a new boundary.\n"
        "6. **Separate Sustained New Work:** If the actor switches to a different tool use, object-focused operation, or work area and that new goal persists, mark a switch even if it happens near the same table, tray, bin, shelf, or stack.\n"
        "7. **Repeated Batch Chores:** Repeating the same high-level chore across several similar items in one work area can stay as **one segment** when the outcome is still the same repeated job.\n"
        "8. **Ignore Narration / Presentation Shots:** Talking to camera, gesturing, pausing, standing by, or briefly showing progress is **NOT** a task boundary if the underlying manipulation task continues.\n"
        "9. **Recall First:** Missing a real sustained step change is worse than over-segmenting; when unsure, keep the boundary if the new operation persists for several frames, and place it at the earliest stable onset of the new task.\n"
        "Over-segmentation is desirable in this pass when it helps expose earlier or narrower candidate boundaries for later merging.\n\n"
        f"{_shared_labeling_rules()}"
        "### Output Format: Strict JSON\n"
        f"{_shared_output_format_description()}\n"
        "### Representative Examples\n"
        "**Example 1: Bowl Stacking (False Switch - Same Goal)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-6: Robot grasps and lifts a blue bowl. Frames 7-12: Robot places the blue bowl onto a bowl stack. Frames 13-15: Robot briefly nudges the stack to stabilize it. This is still one coherent stacking task, so no switch.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Pick up the blue bowl and place it onto the stack\"]\n"
        "}\n\n"
        "**Example 2: Distinct New Goal (True Switch)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-9: Robot finishes placing a bowl onto the stack. Frames 10-11: Gripper leaves the stack. Frames 12-15: Robot reaches to a red cup and begins a new manipulation goal. This is a true task switch at frame 11.\",\n"
        "  \"transitions\": [11],\n"
        "  \"instructions\": [\"Place the bowl onto the stack\", \"Pick up the red cup\"]\n"
        "}\n\n"
        "**Example 3: Preparation + Main Action (Continuous)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-4: Robot hovers and aligns above a cup. Frames 5-15: Robot grasps the cup and moves it to a tray. The early alignment is just preparation for the same task, so no switch.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Pick up the cup and place it on the tray\"]\n"
        "}\n\n"
        "**Example 4: Repeated Batch Chores (Still One Segment)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-15: Person folds several clothes one after another on the same table. Although the hands touch multiple garments, the high-level chore is still folding laundry, so keep it as one segment.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Folding several clothes on the table\"]\n"
        "}\n\n"
        "**Example 5: Same Work Area, Different Sustained Goal (True Switch)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-6: Person wipes the cutting board clean. Frames 7-9: Person puts the cloth down. Frames 10-15: Person opens a drawer and starts placing tools into it. The work area is nearby, but the goal changes from cleaning to organizing, so mark a switch at frame 9.\",\n"
        "  \"transitions\": [9],\n"
        "  \"instructions\": [\"Wipe the cutting board\", \"Place the tools into the drawer\"]\n"
        "}\n\n"
        "**Example 6: Narration Shot Is Not A Task Boundary**\n"
        "{\n"
        "  \"thought\": \"Frames 0-6: Person sorts small parts into a tray. Frames 7-9: Person looks at the camera and talks briefly. Frames 10-15: Person returns to the same sorting task. The talking shot is not a new manipulation goal, so no switch.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Sort the small parts into the tray\"]\n"
        "}\n\n"
        "**Example 7: Earliest Item Onset, Not Late Confirmation**\n"
        "{\n"
        "  \"thought\": \"Frames 0-7: Person keeps arranging objects on a shelf. Frame 8: the first folder starts entering a storage box. Frames 9-15: more of the folder moves inside. Cut at frame 8, not later when the folder is fully inside.\",\n"
        "  \"transitions\": [8],\n"
        "  \"instructions\": [\"Arrange the items on the shelf\", \"Place the folder into the storage box\"]\n"
        "}\n\n"
        "**Example 8: Ambiguous Material, Objective Label**\n"
        "{\n"
        "  \"thought\": \"Frames 0-6: Person holds a container above a tray. Frame 7: a new granular material starts falling into the tray. The exact material is unclear, but the new dispensing step is real.\",\n"
        "  \"transitions\": [7],\n"
        "  \"instructions\": [\"Hold the container above the tray\", \"Dispense granular material into the tray\"]\n"
        "}\n\n"
        "**Example 9: Similar Motion Twice, Separate Objective Steps**\n"
        "{\n"
        "  \"thought\": \"Frames 0-6: Person pours one dark liquid from bottle A into a container. Frame 7: bottle A leaves. Frame 8: bottle B starts a second pour. Even though both look similar, this is a new addition round.\",\n"
        "  \"transitions\": [8],\n"
        "  \"instructions\": [\"First pour dark liquid into the container\", \"Second pour dark liquid into the container\"]\n"
        "}\n\n"
        "**Example 10: Pre-Merge Finer Segmentation Is Preferred**\n"
        "{\n"
        "  \"thought\": \"Frames 0-4: Person places one tool into a storage bin. Frame 5: the hand withdraws. Frame 6: the hand returns with another tool. Because this is a pre-merge pass, split the two visible loading rounds.\",\n"
        "  \"transitions\": [6],\n"
        "  \"instructions\": [\"Place the first tool into the bin\", \"Place the second tool into the bin\"]\n"
        "}\n\n"
        "**Example 11: Same Tool, Stop And Restart Means New Round**\n"
        "{\n"
        "  \"thought\": \"Frames 0-5: A dispenser applies granular material onto a surface. Frame 6: the dispenser leaves. Frame 8: the dispenser returns and starts another application. Split at the restart because it is a new committed round.\",\n"
        "  \"transitions\": [8],\n"
        "  \"instructions\": [\"First application of granular material onto the surface\", \"Second application of granular material onto the surface\"]\n"
        "}"
    )


def _prompt_switch_detection_center_scan(
    n_images: int,
    contact_sheet_rows: int = 0,
    contact_sheet_cols: int = 0,
    sheet_count: int = 0,
) -> str:
    middle_left = max((n_images // 2) - 1, 0)
    middle_right = min(n_images // 2, max(n_images - 1, 0))

    return (
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip of household manipulation tasks.\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n"
        f"Focus on whether there is **at most one true boundary** near the **middle of the clip**, especially around indices {middle_left} and {middle_right}.\n\n"
        f"{_contact_sheet_layout_guidance(contact_sheet_rows, contact_sheet_cols, sheet_count)}"
        "### Goal\n"
        "Judge whether the clip stays as one coherent manipulation task through the middle, or whether it truly switches from one sustained manipulation phase to another near the middle.\n"
        "This is a **local boundary scan**, not a full freeform segmentation of the whole clip.\n"
        "Missing a true middle boundary is worse than proposing one extra plausible middle boundary.\n\n"
        "### Decision Rule\n"
        "1. Return **no boundary** if the action before and after the middle is still one continuous goal.\n"
        "2. Return **one boundary** if a previous task is clearly ending and a different sustained manipulation goal is clearly beginning near the middle of the clip.\n"
        "3. Anchor the chosen boundary to the **first committed frame** of the new step near the middle. Do **NOT** wait until the new action is fully obvious.\n"
        "4. Preparation, alignment, re-grasps, stabilization, brief narration, presentation shots, and post-placement adjustments are part of the same task, not a true switch.\n"
        "5. If the middle shows a plausibly true task switch that persists for several frames, keep the boundary rather than suppressing it.\n"
        "6. Choose the closest supported middle index instead of inventing a far-away boundary.\n\n"
        f"{_shared_labeling_rules()}"
        "### Output Format: Strict JSON\n"
        f"{_shared_output_format_description()}"
        "If there is no true boundary near the middle, output exactly one instruction for the whole clip:\n"
        "{\n"
        "  \"thought\": \"...\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"single coherent task instruction\"]\n"
        "}\n\n"
        "If there is one true boundary near the middle, output exactly one transition and exactly two instructions:\n"
        "{\n"
        "  \"thought\": \"...\",\n"
        f"  \"transitions\": [{middle_left}],\n"
        "  \"instructions\": [\"left-side task instruction\", \"right-side task instruction\"]\n"
        "}\n\n"
        "### Examples\n"
        "**Example 1: No True Boundary Near The Middle**\n"
        "{\n"
        "  \"thought\": \"The person reaches, aligns, grasps a cup, and places it onto a tray. The motion around the middle is still preparation and continuation of the same placement goal, so there is no true boundary near the middle.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Pick up the cup and place it on the tray\"]\n"
        "}\n\n"
        "**Example 2: One True Boundary Near The Middle**\n"
        "{\n"
        "  \"thought\": \"Before the middle, the person finishes placing a bowl onto a stack. Right after the middle, the hands leave the stack and begin reaching for a red cup as a new sustained goal. This is one true boundary near the middle.\",\n"
        f"  \"transitions\": [{middle_left}],\n"
        "  \"instructions\": [\"Place the bowl onto the stack\", \"Pick up the red cup\"]\n"
        "}"
    )


def _prompt_switch_detection_multi_probe_scan(
    n_images: int,
    contact_sheet_rows: int = 0,
    contact_sheet_cols: int = 0,
    sheet_count: int = 0,
) -> str:
    probe_positions = _boundary_probe_positions(n_images)
    probe_positions_str = str(probe_positions)

    return (
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip of household manipulation tasks.\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n"
        f"Use these fixed **probe positions** to judge local task boundaries: {probe_positions_str}.\n\n"
        f"{_contact_sheet_layout_guidance(contact_sheet_rows, contact_sheet_cols, sheet_count)}"
        "### Goal\n"
        "Detect task-level boundaries for VLA training without doing whole-window free segmentation.\n"
        "Judge whether there is a true task boundary near each probe position.\n"
        "Missing a true boundary is worse than keeping one extra probe-supported candidate.\n\n"
        "### Decision Rule\n"
        "1. For each probe position, decide whether the **first committed frame** of a new sustained manipulation phase is visible at or very near that probe.\n"
        "2. Ignore micro-adjustments, alignment, re-grasps, stabilization, narration, presentation shots, and short bridging motion.\n"
        "3. Repeated batch work in the same workspace may stay one task unless a new visible round, tool, source, or work area clearly begins.\n"
        "4. If a probe is near a plausibly true switch that persists for several frames, keep that probe boundary rather than suppressing it.\n"
        "5. Only output transitions chosen from those probe positions.\n"
        "6. Do **NOT** invent objects that are not yet visible, and do **NOT** delay the cut until the new action is fully underway.\n\n"
        f"{_shared_labeling_rules()}"
        "### Output Format: Strict JSON\n"
        f"{_shared_output_format_description()}"
        "The `transitions` list must be a sorted subset of the probe positions.\n"
        "The number of instructions must equal `len(transitions) + 1`.\n\n"
        "### Examples\n"
        "**Example 1: No Probe Is A True Boundary**\n"
        "{\n"
        "  \"thought\": \"All probe positions fall inside one continuous cup-pick-and-place task, so there is no true boundary near any probe.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Pick up the cup and place it on the tray\"]\n"
        "}\n\n"
        "**Example 2: Two Probes Are True Boundaries**\n"
        "{\n"
        "  \"thought\": \"Near probe 2, the person finishes placing one tool into a bin and begins loading a second tool. Near probe 5, the loading task ends and wiping the tray begins. Probe 4 is still part of the second loading round, so only two probe positions are true boundaries.\",\n"
        "  \"transitions\": [2, 5],\n"
        "  \"instructions\": [\"Place the first tool into the bin\", \"Place the second tool into the bin\", \"Wipe the tray\"]\n"
        "}"
    )


def _prompt_switch_detection_candidate_scan(
    n_images: int,
    contact_sheet_rows: int = 0,
    contact_sheet_cols: int = 0,
    sheet_count: int = 0,
) -> str:
    return (
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip of household manipulation tasks.\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n"
        "This is a **candidate-boundary nomination pass**, not the final segmentation decision.\n\n"
        f"{_contact_sheet_layout_guidance(contact_sheet_rows, contact_sheet_cols, sheet_count)}"
        "### Goal\n"
        "Propose candidate boundaries that may separate two sustained manipulation goals useful for VLA training.\n"
        "This first pass is for recall-first nomination, and later verification will confirm or reject the candidates.\n"
        "Prioritize recall over final precision in this first pass, because it is better to keep a plausible candidate than to risk missing a true task switch.\n"
        "Do **NOT** invent tiny micro-boundaries, but if two neighboring time spans plausibly show different sustained goals, include that boundary candidate.\n\n"
        "### Decision Rule\n"
        "1. You may output up to 3 transitions.\n"
        "2. A transition may be any interior image index from 1 to n-2.\n"
        "3. Anchor each candidate to the **first committed frame** of the new step. Do **NOT** wait until the new action is fully obvious.\n"
        "4. Ignore micro-adjustments, alignment, re-grasps, stabilization, narration, presentation shots, and short bridging motion.\n"
        "5. If a boundary looks plausible but not fully certain, you should still nominate it in this first pass rather than risk missing a true task switch.\n"
        "6. If a new source, tool, material, workpiece, or work area begins a sustained new operation, you may nominate a boundary even when everything happens near the same workspace or container.\n"
        "7. Do **NOT** invent objects that are not yet visible.\n\n"
        f"{_shared_labeling_rules()}"
        "### Output Format: Strict JSON\n"
        f"{_shared_output_format_description()}"
        "The `transitions` list must be sorted, contain at most 3 items, and use only interior image indices.\n"
        "The number of instructions must equal `len(transitions) + 1`.\n\n"
        "### Examples\n"
        "{\n"
        "  \"thought\": \"All sampled frames stay within one continuous sorting task, so there is no candidate boundary.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Sort the small parts into the tray\"]\n"
        "}\n\n"
        "{\n"
        "  \"thought\": \"The person finishes placing one tool into a storage bin near frame 2 and later switches from loading tools to wiping the tray near frame 5. Both look like plausible sustained task switches, so I nominate two candidate boundaries for later verification.\",\n"
        "  \"transitions\": [2, 5],\n"
        "  \"instructions\": [\"Place the first tool into the storage bin\", \"Place the second tool into the storage bin\", \"Wipe the tray\"]\n"
        "}"
    )


def prompt_segment_instruction(
    n_images: int,
    contact_sheet_rows: int = 0,
    contact_sheet_cols: int = 0,
    sheet_count: int = 0,
) -> str:
    return (
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip that is already a single task segment.\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n\n"
        f"{_contact_sheet_layout_guidance(contact_sheet_rows, contact_sheet_cols, sheet_count)}"
        "### Goal\n"
        "Describe the one coherent manipulation task shown in this clip.\n"
        "Do not split it further.\n"
        "Ignore narration, presentation gestures, and tiny post-action adjustments when naming the task.\n"
        "Return one concise robot-training instruction for the dominant visible manipulation goal.\n\n"
        f"{_shared_labeling_rules()}"
        "### Output Format: Strict JSON\n"
        f"{_shared_output_format_description()}"
        "This clip is already one segment, so `transitions` must always be empty and `instructions` must contain exactly one string.\n"
        "{\n"
        "  \"thought\": \"...\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"one concise task instruction\"]\n"
        "}"
    )


def prompt_boundary_refinement(
    n_images: int,
    contact_sheet_rows: int = 0,
    contact_sheet_cols: int = 0,
    sheet_count: int = 0,
) -> str:
    candidate_positions = _boundary_refinement_positions(n_images)
    chosen = candidate_positions[1] if len(candidate_positions) > 1 else candidate_positions[0] if candidate_positions else 1
    return (
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip centered around a candidate boundary in a household manipulation task.\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n"
        f"Refine the boundary location using only these middle candidate positions: {candidate_positions}.\n\n"
        f"{_contact_sheet_layout_guidance(contact_sheet_rows, contact_sheet_cols, sheet_count)}"
        "### Goal\n"
        "A coarse task boundary has already been proposed near the middle of this clip.\n"
        "Your job is to refine the boundary location, not to free-segment the whole clip.\n"
        "Choose the single best candidate index if a true or plausibly true task switch happens near the middle.\n"
        "If the middle still looks like one continuous task, return no boundary.\n\n"
        "### Decision Rule\n"
        "1. Return one boundary if the action before and after the candidate plausibly changes from one sustained manipulation goal to another.\n"
        "2. Anchor the refinement to the **earliest supported onset** among the candidate positions. Do **NOT** wait until the new action is fully underway.\n"
        "3. Ignore preparation, alignment, re-grasps, stabilization, narration, presentation shots, and post-action adjustments.\n"
        "4. If the clip shows one continuing task through the middle, return no boundary.\n"
        "5. If there is a plausibly true task switch, prefer selecting the closest supported candidate instead of suppressing the boundary.\n"
        "6. Do **NOT** invent boundaries far away from the middle candidates.\n\n"
        f"{_shared_labeling_rules()}"
        "### Output Format: Strict JSON\n"
        f"{_shared_output_format_description()}"
        "The `transitions` field must be [] or exactly one index from the candidate list.\n"
        "If there is no true boundary, return one instruction.\n"
        "If there is one true boundary, return two short task instructions.\n\n"
        "### Examples\n"
        "{\n"
        "  \"thought\": \"The clip still shows one continuous sorting task through the middle, so there is no true boundary.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Sort the small parts into the tray\"]\n"
        "}\n\n"
        "{\n"
        f"  \"thought\": \"The person finishes placing one tool near the middle and clearly begins wiping the tray right after it, so the best refined boundary is at candidate {chosen}.\",\n"
        f"  \"transitions\": [{chosen}],\n"
        "  \"instructions\": [\"Place the tool into the bin\", \"Wipe the tray\"]\n"
        "}"
    )


def prompt_switch_detection(
    n_images: int,
    mode: str = "freeform",
    contact_sheet_rows: int = 0,
    contact_sheet_cols: int = 0,
    sheet_count: int = 0,
) -> str:
    if mode == "freeform":
        return _prompt_switch_detection_freeform(
            n_images,
            contact_sheet_rows=contact_sheet_rows,
            contact_sheet_cols=contact_sheet_cols,
            sheet_count=sheet_count,
        )
    if mode == "center_scan":
        return _prompt_switch_detection_center_scan(
            n_images,
            contact_sheet_rows=contact_sheet_rows,
            contact_sheet_cols=contact_sheet_cols,
            sheet_count=sheet_count,
        )
    if mode == "multi_probe_scan":
        return _prompt_switch_detection_multi_probe_scan(
            n_images,
            contact_sheet_rows=contact_sheet_rows,
            contact_sheet_cols=contact_sheet_cols,
            sheet_count=sheet_count,
        )
    if mode == "candidate_scan":
        return _prompt_switch_detection_candidate_scan(
            n_images,
            contact_sheet_rows=contact_sheet_rows,
            contact_sheet_cols=contact_sheet_cols,
            sheet_count=sheet_count,
        )
    raise ValueError(f"Unsupported prompt mode: {mode}")
