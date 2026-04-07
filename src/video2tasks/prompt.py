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


def boundary_refinement_candidate_positions(n_images: int) -> list[int]:
    return _boundary_refinement_positions(n_images)


def _shared_earliest_onset_anchoring() -> str:
    return (
        "### Earliest Onset Anchoring\n"
        "Anchor each boundary to the **first committed frame** of the new step.\n"
        "For a pour, cut on the first frame the new liquid starts being poured, not when the pool becomes obvious.\n"
        "For a sprinkle or other granular application, cut on the first committed release, not when the accumulation becomes obvious.\n"
        "For an addition or placement, cut on the first frame the new item starts entering the target area, not when it is fully inside or settled.\n"
        "For a new tool or source object, cut on the first frame that tool or source begins the new distinct operation.\n"
        "If a new source or tool arrives directly over the target and the addition starts immediately after, anchor at that committed arrival instead of waiting for visible accumulation.\n"
        "If the prior goal is clearly abandoned and the hand, tool, or source is already committed near or over the new target such that the new operation starts within the next few frames with no unrelated action, treat that committed setup as the start of the new step.\n"
        "Do **NOT** merge such committed setup into the previous task; merge it into the new task it is serving.\n"
        "If that setup lasts long or the new goal is not yet certain, do **NOT** move the boundary earlier; anchor at the first visible evidence of the new operation, such as entry, release, contact, or motion start.\n"
        "If the same source or tool stops, leaves, and later returns to start another visible round, begin a new segment at that return.\n"
        "If the same broad motion happens twice with a different source object, material round, or target phase, split them as separate steps and label them objectively, for example `First pour` and `Second pour` when needed.\nIf the object, tool, or material becomes easier to recognize a few frames later, still cut at the earlier visible onset instead of the later confirmation frame.\n"
        "Do **NOT** invent objects that are not yet visible; anchor the cut to the first visible evidence of the new step.\n\n"
    )


def _shared_onset_vs_confirmation() -> str:
    return (
        "### Onset Versus Confirmation\n"
        "The onset is the first frame where the new manipulation is visibly underway: first entry, first release, first contact, first committed motion, or the first visible effect caused by that action.\n"
        "A later frame where the object, material, tool, or result becomes easier to recognize is only confirmation, not the onset.\n"
        "Do **NOT** anchor a boundary to the mere appearance of a new empty target region, a camera change, or an idle tool or source that has not yet committed to the new target.\n"
        "If a source or tool is already clearly committed at or over the new target and the new operation starts within the next few frames with no unrelated action, prefer that earlier committed frame over a later clearer-looking release or confirmation frame.\n"
        "In this recall-first pass, when both an earlier plausible onset and a later confirmation describe the same new operation, prefer the earlier plausible onset.\n"
        "Do **NOT** let one broad label swallow a later visible onset inside it. If a later span contains a new release, entry, contact, or restart, split there instead of keeping one umbrella segment.\n"
        "If the new operation has visibly started but the exact identity is still unclear, keep the early boundary and use an objective coarse label.\n\n"
    )


def _shared_fragmentation_bias() -> str:
    return (
        "### Fragmentation Bias\n"
        "This is a recall-first pre-merge pass. Optimize for exposing more real boundaries, not for elegant summarization.\n"
        "Prefer the smallest objectively visible distinct step that a robot could execute, even if a later language model may merge adjacent steps.\n"
        "A brief but committed discrete manipulation can still be its own step if it produces a clear new state, for example a place, remove, cover, uncover, open, close, release, or toggle action.\n"
        "Do **NOT** require a true step to stay visually long once its committed outcome is visible.\n"
        "The same workspace, same target region, same broad activity, or the same long-running workflow does **NOT** by itself justify keeping one segment.\n"
        "If one visible round completes and another round begins with a new object, source, material, target entry, tool use, removal, or restart, prefer a new boundary.\n"
        "If two candidate onsets are temporally close but correspond to two different visible rounds, object entries, or restarts, keep both candidate boundaries in this pass.\nDo **NOT** replace two close true boundaries with one broad bridge segment just because they happen in the same container or workspace.\nIf exact identity is unclear at the onset, keep the early boundary anyway and use an objective coarse label rather than waiting for later confirmation.\n"
        "Good coarse labels include `First placement`, `Second placement`, `First pour`, `Second pour`, `Dispense granular material`, `Add the first visible item`, and `Add the second visible item`.\n"
        "Only keep one segment when the visible manipulation trajectory is truly uninterrupted and the goal has not changed.\n\n"
    )


def _shared_labeling_rules() -> str:
    return (
        "### Labeling Rules\n"
        "Do **NOT** output labels like `Wait`, `Stand by`, `Explain the task`, `Describe the scene`, `Narrate the action`, `Prepare the item`, `Handle the object`, `Manage the item`, `Observe the object`, `Monitor the target area`, `View the contents`, `Show the result`, `Inspect the item`, `Check the state`, `Reveal the inside`, `Continue the task`, `Continue working`, `Process the item`, or `Work on the target region`.\n"
        "Do **NOT** create passive bridge segments whose only content is waiting, showing, viewing, monitoring, inspecting, or revealing a state while no new manipulation has started.\nIf the clip only shows an intermediate state becoming visible, merge that span into the causally related active manipulation, usually the segment whose action produced that state, instead of naming it as its own step.\n"
        "Do **NOT** create a separate segment whose only content is bringing, holding, hovering, aligning, or positioning a source object for an imminent action; merge that setup into the ensuing manipulation unless that setup itself is the completed goal.\nWhen that setup is already clearly committed to a new immediate goal and the new operation starts within the next few frames with no unrelated action, it belongs to the new segment, not the previous segment.\n"
        "Do **NOT** use state-only labels that merely report appearance or location without an active manipulation.\nAlways anchor the instruction to the visible manipulation goal.\n"
        "Prefer concrete verbs plus visible objects.\n"
        "When identity is ambiguous, prefer grounded but coarse labels like `Dispense granular material`, `Pour dark liquid`, `Place a flat item`, `First pour`, or `Second pour` instead of guessing exact product, material, or brand names.\n"
        "Do **NOT** wait for later frames to identify the exact object if the onset is already visible; keep the early cut and use a coarser but objective label.\n"
        "Do **NOT** use an umbrella instruction that spans multiple later visible onsets just because one generic verb could describe all of them.\n"
        "If two adjacent steps look similar, distinguish them by visible order or round, for example `First placement` and `Second placement`.\n"
        "Ignore idle self-adjustment tails such as adjusting clothing, rubbing hands, or resetting posture after a task ends.\n\n"
    )


def _shared_output_format_description() -> str:
    return (
        "Return a valid JSON object with keys `thought`, `transitions`, and `instructions`.\n"
        "The entire reply must be raw JSON only, with no markdown fences and no prose before or after the JSON object.\n"
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
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip of manipulation tasks.\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n\n"
        f"{_contact_sheet_layout_guidance(contact_sheet_rows, contact_sheet_cols, sheet_count)}"
        "### Goal\n"
        "Detect **task-level boundaries** that are useful for VLA training.\n"
        "Each segment should be one coherent robot instruction from start to finish.\n"
        "Do **not** miss new goals: if the actor clearly finishes or abandons one goal and starts another that either persists or completes a discrete manipulation outcome, mark a switch.\n"
        "A 'Switch' happens when the robot starts a **new manipulation phase** or a brief but committed discrete manipulation with a new immediate goal, not merely when it briefly touches a new object without committing to that new goal.\n\n"
        "### Boundary Priority\n"
        "Missing a true boundary is worse than proposing an extra one.\n"
        "This is a pre-merge over-segmentation pass: extra adjacent boundaries are acceptable because a later model can merge them.\n"
        "Prefer the smallest objectively visible distinct step, not a broad summary of the whole activity.\n"
        "When uncertain between one broad segment and two narrower adjacent segments, choose the narrower segmentation.\n"
        "Prefer splitting repeated additions, repeated pours, repeated granular applications, repeated placements, and repeated removals into separate visible rounds whenever a committed new round is visible.\n"
        "Place the boundary at the earliest frame where the new step is already visible, including committed setup that leads into it within the next few frames.\n"
        "Do **NOT** delay the cut until the new action is fully underway.\n"
        "Brief but committed actions like placing, removing, covering, uncovering, opening, closing, releasing, or toggling can still be valid segments even when they are visually short.\n"
        "If the visible operation changes and the new step either persists or completes a discrete manipulation outcome, keep the boundary.\n"
        "Do **NOT** collapse separate additions, granular applications, tool changes, target changes, removal steps, or final transfer-out steps into one broad instruction.\n"
        "Only keep one segment when the motion is genuinely uninterrupted and there is no visible stop, reset, withdrawal, or re-entry before the next round.\n\n"
        f"{_shared_fragmentation_bias()}"
        f"{_shared_earliest_onset_anchoring()}"
        f"{_shared_onset_vs_confirmation()}"
        "### Core Logic\n"
        "1. **True Switch:** In this pass, treat each new distinct manipulation phase as a switch, even inside one larger ongoing workflow.\n"
        "2. **Uninterrupted Transfer Stays Together:** If the robot is moving one carried item through one continuous transfer or placement trajectory, brief contact with a receptacle, support surface, or final stack is **NOT** a switch.\n"
        "The same support surface, target region, receptacle, or workspace is **NOT** enough to merge two visible rounds that are otherwise distinct.\n"
        "3. **Ignore Micro-Adjustments:** Brief nudges, stabilization, re-grasps, post-placement corrections, and tiny follow-up adjustments around the same task are **NOT** new tasks.\n"
        "4. **Preparation Belongs To The New Task It Serves:** Reaching, hovering, aligning, or partially grasping belongs to the new task only when the prior goal is already abandoned and that setup leads directly into the new operation within the next few frames with no unrelated action; otherwise keep the cut at the first visible evidence of the new operation.\n"
        "5. **Prefer Separate Visible Rounds:** If the actor changes material, tool, source object, target region, workpiece, or manipulation phase and the new step persists, prefer a new boundary.\n"
        "If the source and target stay the same but there is a visible stop-and-restart, hand reset, source withdrawal and return, or another committed round, prefer a new boundary.\n"
        "6. **Separate New Work:** If the actor switches to a different tool use, object-focused operation, or work area and that new goal persists or completes a discrete manipulation outcome, mark a switch even if it happens near the same local work area or support region.\n"
        "7. **Repeated Batch Chores:** In this pass, repeated rounds across several similar items should usually be split into separate visible rounds whenever one round completes and the next round begins with a new item or restart.\n"
        "Only keep them as one segment when the motion is genuinely continuous and there is no visible completion, withdrawal, or reset between rounds.\n"
        "8. **Ignore Narration / Presentation Shots:** Talking to camera, gesturing, pausing, standing by, or briefly showing progress is **NOT** a task boundary if the underlying manipulation task continues.\n"
        "9. **Recall First:** Missing a real step change is worse than over-segmenting; when unsure, keep the boundary if the new operation persists or reaches a clear discrete outcome, and place it at the earliest stable onset of the new task.\n"
        "Over-segmentation is desirable in this pass when it helps expose earlier or narrower candidate boundaries for later merging.\n\n"
        f"{_shared_labeling_rules()}"
        "### Output Format: Strict JSON\n"
        f"{_shared_output_format_description()}\n"
        "### Representative Examples\n"
        "**Example 1: Continuous Placement (False Switch - Same Goal)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-6: Robot grasps and lifts a flat item. Frames 7-12: Robot places the item onto a support surface. Frames 13-15: Robot briefly stabilizes it. This is still one coherent placement task, so no switch.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Pick up the item and place it onto the support surface\"]\n"
        "}\n\n"
        "**Example 2: Distinct New Goal (True Switch)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-9: Robot finishes placing one item onto a support surface. Frames 10-11: the hand leaves that target. Frame 12: the hand begins reaching toward a second item as the new task. Frames 13-15: the reach continues. This is a true task switch at frame 12.\",\n"
        "  \"transitions\": [12],\n"
        "  \"instructions\": [\"Place the first item onto the support surface\", \"Pick up the second item\"]\n"
        "}\n\n"
        "**Example 3: Preparation + Main Action (Continuous)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-4: Robot hovers and aligns above an item. Frames 5-15: Robot grasps the item and moves it to a target area. The early alignment is just preparation for the same task, so no switch.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Pick up the item and place it in the target area\"]\n"
        "}\n\n"
        "**Example 4: Repeated Batch Chores (Split Visible Rounds)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-6: Person completes arranging one flexible item. Frame 7: hands withdraw. Frame 8: person starts arranging a second flexible item. In this pre-merge pass, split the visible rounds.\",\n"
        "  \"transitions\": [8],\n"
        "  \"instructions\": [\"Arrange the first flexible item\", \"Arrange the second flexible item\"]\n"
        "}\n\n"
        "**Example 5: Same Work Area, Different Distinct Goal (True Switch)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-6: Person cleans a surface. Frames 7-9: Person puts the cleaning item down. Frame 10: person opens a compartment as the new task begins. Frames 11-15: person keeps placing objects into it. The work area is nearby, but the goal changes from cleaning to organizing, so mark a switch at frame 10.\",\n"
        "  \"transitions\": [10],\n"
        "  \"instructions\": [\"Clean the surface\", \"Place the objects into the compartment\"]\n"
        "}\n\n"
        "**Example 6: Narration Shot Is Not A Task Boundary**\n"
        "{\n"
        "  \"thought\": \"Frames 0-6: Person sorts small items into a target area. Frames 7-9: Person looks at the camera and talks briefly. Frames 10-15: Person returns to the same sorting task. The talking shot is not a new manipulation goal, so no switch.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Sort the small items into the target area\"]\n"
        "}\n\n"
        "**Example 7: Earliest Item Onset, Not Late Confirmation**\n"
        "{\n"
        "  \"thought\": \"Frames 0-7: Person keeps arranging items in one area. Frame 8: the first flat item starts entering a target region. Frames 9-15: more of the item moves inside. Cut at frame 8, not later when the item is fully inside.\",\n"
        "  \"transitions\": [8],\n"
        "  \"instructions\": [\"Arrange the items in the work area\", \"Place the flat item into the target region\"]\n"
        "}\n\n"
        "**Example 8: Ambiguous Material, Objective Label**\n"
        "{\n"
        "  \"thought\": \"Frames 0-6: Person finishes placing a target object in position. Frame 7: a new granular material starts falling into the target region. The exact material is unclear, but the dispensing onset is already visible.\",\n"
        "  \"transitions\": [7],\n"
        "  \"instructions\": [\"Place the target object in position\", \"Dispense granular material into the target region\"]\n"
        "}\n\n"
        "**Example 9: Similar Motion Twice, Separate Objective Steps**\n"
        "{\n"
        "  \"thought\": \"Frames 0-6: Person pours one dark liquid from source A into a target region. Frame 7: source A leaves. Frame 8: source B starts a second pour. Even though both look similar, this is a new addition round.\",\n"
        "  \"transitions\": [8],\n"
        "  \"instructions\": [\"First pour dark liquid into the target region\", \"Second pour dark liquid into the target region\"]\n"
        "}\n\n"
        "**Example 10: Pre-Merge Finer Segmentation Is Preferred**\n"
        "{\n"
        "  \"thought\": \"Frames 0-4: Person places one object into a target region. Frame 5: the hand withdraws. Frame 6: the hand returns with another object. Because this is a pre-merge pass, split the two visible loading rounds.\",\n"
        "  \"transitions\": [6],\n"
        "  \"instructions\": [\"Place the first object into the target region\", \"Place the second object into the target region\"]\n"
        "}\n\n"
        "**Example 11: Same Tool, Stop And Restart Means New Round**\n"
        "{\n"
        "  \"thought\": \"Frames 0-5: A dispenser applies granular material onto a surface. Frame 6: the dispenser leaves. Frame 8: the dispenser returns and starts another application. Split at the restart because it is a new committed round.\",\n"
        "  \"transitions\": [8],\n"
        "  \"instructions\": [\"First application of granular material onto the surface\", \"Second application of granular material onto the surface\"]\n"
        "}\n\n"
        "**Example 12: Empty Target Alone Is Not Enough, But Committed Arrival Can Count**\n"
        "{\n"
        "  \"thought\": \"Frames 0-3: the prior task is ending while an empty target region becomes visible, which is not enough for a new boundary. Frame 4: a source object arrives clearly committed over that target. Frame 5: dispensing begins immediately. Cut at frame 4, not earlier at the empty target and not later at the clearer release.\",\n"
        "  \"transitions\": [4],\n"
        "  \"instructions\": [\"Finish the earlier manipulation\", \"Dispense material into the target region\"]\n"
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
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip of manipulation tasks.\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n"
        f"Focus on whether there is **at most one true boundary** near the **middle of the clip**, especially around indices {middle_left} and {middle_right}.\n\n"
        f"{_contact_sheet_layout_guidance(contact_sheet_rows, contact_sheet_cols, sheet_count)}"
        "### Goal\n"
        "Judge whether the clip stays as one coherent manipulation task through the middle, or whether it truly switches from one distinct manipulation phase to another near the middle.\n"
        "This is a **local boundary scan**, not a full freeform segmentation of the whole clip.\n"
        "Missing a true middle boundary is worse than proposing one extra plausible middle boundary.\n\n"
        "### Decision Rule\n"
        "1. Return **no boundary** if the action before and after the middle is still one continuous goal.\n"
        "2. Return **one boundary** if a previous task is clearly ending and a different manipulation goal, or a brief but committed discrete manipulation with a new immediate goal, is clearly beginning near the middle of the clip.\n"
        "3. Anchor the chosen boundary to the **first committed frame** of the new step near the middle. Do **NOT** wait until the new action is fully obvious.\n"
        "4. Preparation, alignment, re-grasps, stabilization, brief narration, presentation shots, and post-placement adjustments are not separate tasks by themselves.\nIf the middle shows setup that is already clearly committed to a new immediate goal and the new operation starts within the next few frames with no unrelated action, treat it as the start of the new task rather than the tail of the previous one.\n"
        "5. Do **NOT** suppress a middle boundary merely because both sides use the same local work area, target region, support surface, or workspace; if one visible round ends and another begins, keep the boundary.\n"
        "6. If the middle shows a plausibly true task switch that persists or reaches a clear discrete outcome, keep the boundary rather than suppressing it.\n"
        "7. Choose the closest supported middle index instead of inventing a far-away boundary.\n\n"
        f"{_shared_onset_vs_confirmation()}"
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
        "  \"instructions\": [\"Pick up the item and place it in the target area\"]\n"
        "}\n\n"
        "**Example 2: One True Boundary Near The Middle**\n"
        "{\n"
        "  \"thought\": \"Before the middle, the person finishes placing one item onto a support surface. Right after the middle, the hands leave that target and begin reaching for a second item as a new distinct goal. This is one true boundary near the middle.\",\n"
        f"  \"transitions\": [{middle_left}],\n"
        "  \"instructions\": [\"Place the first item onto the support surface\", \"Pick up the second item\"]\n"
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
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip of manipulation tasks.\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n"
        f"Use these fixed **probe positions** to judge local task boundaries: {probe_positions_str}.\n\n"
        f"{_contact_sheet_layout_guidance(contact_sheet_rows, contact_sheet_cols, sheet_count)}"
        "### Goal\n"
        "Detect task-level boundaries for VLA training without doing whole-window free segmentation.\n"
        "Judge whether there is a true task boundary near each probe position.\n"
        "Missing a true boundary is worse than keeping one extra probe-supported candidate.\n\n"
        "### Decision Rule\n"
        "1. For each probe position, decide whether the **first committed frame** of a new distinct manipulation phase or brief but committed discrete manipulation with a new immediate goal is visible at or very near that probe.\n"
        "2. Ignore micro-adjustments, alignment, re-grasps, stabilization, narration, presentation shots, and short bridging motion.\n"
        "If the probe lands on setup that is already clearly committed to a new immediate goal and the new operation starts within the next few frames with no unrelated action, count that probe as part of the new task rather than the old one.\n"
        "3. Repeated batch work in the same workspace should usually be split when one visible round completes and another begins with a new item, restart, source, tool, target entry, or removal. Do **NOT** merge just because the target region or workspace matches.\n"
        "4. Brief but committed placement, removal, release, cover, uncover, open, or close actions can still be valid probe boundaries even when visually short.\n"
        "5. If a probe is near a plausibly true switch that persists or reaches a clear discrete outcome, keep that probe boundary rather than suppressing it.\n"
        "6. Only output transitions chosen from those probe positions.\n"
        "7. Do **NOT** invent objects that are not yet visible, and do **NOT** delay the cut until the new action is fully underway.\n\n"
        f"{_shared_onset_vs_confirmation()}"
        f"{_shared_labeling_rules()}"
        "### Output Format: Strict JSON\n"
        f"{_shared_output_format_description()}"
        "The `transitions` list must be a sorted subset of the probe positions.\n"
        "The number of instructions must equal `len(transitions) + 1`.\n\n"
        "### Examples\n"
        "**Example 1: No Probe Is A True Boundary**\n"
        "{\n"
        "  \"thought\": \"All probe positions fall inside one continuous item pick-and-place task, so there is no true boundary near any probe.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Pick up the item and place it in the target area\"]\n"
        "}\n\n"
        "**Example 2: Two Probes Are True Boundaries**\n"
        "{\n"
        "  \"thought\": \"Near probe 2, the person finishes placing one object into a target region and begins loading a second object. Near probe 5, the loading task ends and surface cleaning begins. Probe 4 is still part of the second loading round, so only two probe positions are true boundaries.\",\n"
        "  \"transitions\": [2, 5],\n"
        "  \"instructions\": [\"Place the first object into the target region\", \"Place the second object into the target region\", \"Clean the surface\"]\n"
        "}"
    )


def _prompt_switch_detection_candidate_scan(
    n_images: int,
    contact_sheet_rows: int = 0,
    contact_sheet_cols: int = 0,
    sheet_count: int = 0,
) -> str:
    return (
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip of manipulation tasks.\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n"
        "This is a **candidate-boundary nomination pass**, not the final segmentation decision.\n\n"
        f"{_contact_sheet_layout_guidance(contact_sheet_rows, contact_sheet_cols, sheet_count)}"
        "### Goal\n"
        "Propose candidate boundaries that may separate two distinct manipulation goals useful for VLA training.\n"
        "This first pass is for recall-first nomination, and later verification will confirm or reject the candidates.\n"
        "Prioritize recall over final precision in this first pass, because it is better to keep a plausible candidate than to risk missing a true task switch.\n"
        "Do **NOT** split inside one uninterrupted continuous manipulation trajectory, such as one pour or one continuous wipe stroke, but if two neighboring time spans plausibly show different manipulation goals, include that boundary candidate.\n\n"
        "### Decision Rule\n"
        "1. You may output up to 3 transitions.\n"
        "2. A transition may be any interior image index from 1 to n-2.\n"
        "3. Anchor each candidate to the **first committed frame** of the new step. Do **NOT** wait until the new action is fully obvious.\n"
        "4. Ignore micro-adjustments, alignment, re-grasps, stabilization, narration, presentation shots, and short bridging motion.\n"
        "If setup is already clearly committed to a new immediate goal and the new operation starts within the next few frames with no unrelated action, nominate the start of that committed setup as part of the new task, not the old one.\n"
        "5. If a boundary looks plausible but not fully certain, you should still nominate it in this first pass rather than risk missing a true task switch.\n"
        "6. If two nearby candidate onsets correspond to different visible rounds, object entries, restarts, or removals, nominate both instead of collapsing them into one broader step.\n"
        "7. If a new source, tool, material, workpiece, or work area begins a distinct new operation, or a brief but committed discrete manipulation reaches a clear outcome, you may nominate a boundary even when everything happens near the same workspace or target region.\n"
        "8. Do **NOT** invent objects that are not yet visible.\n\n"
        f"{_shared_onset_vs_confirmation()}"
        f"{_shared_labeling_rules()}"
        "### Output Format: Strict JSON\n"
        f"{_shared_output_format_description()}"
        "The `transitions` list must be sorted, contain at most 3 items, and use only interior image indices.\n"
        "The number of instructions must equal `len(transitions) + 1`.\n\n"
        "### Examples\n"
        "{\n"
        "  \"thought\": \"All sampled frames stay within one continuous sorting task, so there is no candidate boundary.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Sort the small items into the target area\"]\n"
        "}\n\n"
        "{\n"
        "  \"thought\": \"The person finishes placing one object into a target region near frame 2 and later switches from loading objects to surface cleaning near frame 5. Both look like plausible task switches, so I nominate two candidate boundaries for later verification.\",\n"
        "  \"transitions\": [2, 5],\n"
        "  \"instructions\": [\"Place the first object into the target region\", \"Place the second object into the target region\", \"Clean the surface\"]\n"
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
        "Return one concise robot-training instruction for the dominant visible manipulation goal.\n"
        "Prefer the narrowest completed visible manipulation that fits the clip, not a broad scene-level activity summary.\n\n"
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
    candidate_positions = boundary_refinement_candidate_positions(n_images)
    chosen = candidate_positions[1] if len(candidate_positions) > 1 else candidate_positions[0] if candidate_positions else 1
    return (
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip centered around a candidate boundary in a manipulation task.\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n"
        f"Refine the boundary location using only these middle candidate positions: {candidate_positions}.\n\n"
        f"{_contact_sheet_layout_guidance(contact_sheet_rows, contact_sheet_cols, sheet_count)}"
        "### Goal\n"
        "A coarse task boundary has already been proposed near the middle of this clip.\n"
        "Your job is to refine the boundary location, not to free-segment the whole clip.\n"
        "Choose the single best candidate index if a true or plausibly true task switch happens near the middle.\n"
        "If the middle still looks like one continuous task, return no boundary.\n\n"
        "### Decision Rule\n"
        "1. Return one boundary if the action before and after the candidate plausibly changes from one manipulation goal to another, or to a brief but committed discrete manipulation with a clear outcome.\n"
        "2. Anchor the refinement to the **earliest supported onset** among the candidate positions. Do **NOT** wait until the new action is fully underway.\n"
        "If the earliest plausible candidate already has visible evidence of the new step, choose it instead of a later clearer-looking candidate.\n"
        "3. Ignore preparation, alignment, re-grasps, stabilization, narration, presentation shots, and post-action adjustments as separate tasks by themselves.\nIf the earliest plausible candidate is already committed setup for a new immediate goal and the new operation starts within the next few frames with no unrelated action, assign it to the new task rather than the previous one.\n"
        "4. Do **NOT** reject a middle boundary just because the same local work area, target region, support surface, or workspace appears on both sides; a new visible round can still be real.\n"
        "5. If the clip shows one continuing task through the middle, return no boundary.\n"
        "6. If there is a plausibly true task switch, prefer selecting the closest supported candidate instead of suppressing the boundary.\n"
        "7. Do **NOT** invent boundaries far away from the middle candidates.\n\n"
        f"{_shared_onset_vs_confirmation()}"
        f"{_shared_labeling_rules()}"
        "### Output Format: Strict JSON\n"
        f"{_shared_output_format_description()}"
        "The `transitions` field must be [] or exactly one index from the candidate list.\n"
        "If there is no true boundary, return one instruction.\n"
        "If there is one true boundary, return two short task instructions.\n\n"
        "### Examples\n"
        "{\n"
        "  \"thought\": \"The clip still shows one continuous organizing task through the middle, so there is no true boundary.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Sort the small items into the target area\"]\n"
        "}\n\n"
        "{\n"
        f"  \"thought\": \"The person finishes placing one object near the middle and clearly begins cleaning the surface right after it, so the best refined boundary is at candidate {chosen}.\",\n"
        f"  \"transitions\": [{chosen}],\n"
        "  \"instructions\": [\"Place the object into the target region\", \"Clean the surface\"]\n"
        "}"
    )


def prompt_segment_merge(
    segments: list[dict],
    granularity: str = "guarded",
    boundary_hints: list[dict] | None = None,
) -> str:
    formatted_segments: list[str] = []
    for segment in segments:
        seg_id = int(segment.get("seg_id", len(formatted_segments)))
        start_frame = int(segment.get("start_frame", 0))
        end_frame = int(segment.get("end_frame", start_frame))
        duration_frames = max(0, end_frame - start_frame)
        instruction = str(segment.get("instruction", "")).strip() or "Unknown task step"
        formatted_segments.append(
            f"- seg_id={seg_id}, frames=[{start_frame}, {end_frame}), duration_frames={duration_frames}, instruction={instruction!r}"
        )

    segment_listing = "\n".join(formatted_segments)
    boundary_hint_section = ""

    if granularity == "guarded":
        merge_guidance = (
            "Your job is to merge only the adjacent segments that are **obviously** fragments of the same immediate manipulation step.\n\n"
            "### Goal\n"
            "Reduce obvious over-cutting while preserving true task switches.\n"
            "When uncertain, keep the boundary.\n"
            "Do **NOT** invent new boundaries, delete time, reorder segments, or split a segment further.\n"
            "You may only merge adjacent original segments into larger contiguous ranges.\n\n"
            "### Merge Rules\n"
            "1. Merge only when adjacent segments clearly describe the same immediate manipulation goal and the boundary looks like a fragment caused by setup, short continuation, short tail, or redundant relabeling of the same step.\n"
            "2. Do **NOT** merge across a plausible true task switch such as a new committed action round, a new immediate goal, a clear restart, a distinct transfer outcome, or a tool or target change that indicates a real step change.\n"
            "3. Do **NOT** rely on broad scene similarity. The same workspace, same container, same support surface, or same overall activity does **NOT** by itself justify merging.\n"
            "4. If two adjacent segments could reasonably be separate robot commands, keep them separate.\n"
            "5. Prefer conservative merging. A missed merge is better than deleting a real task boundary.\n"
            "6. Do **NOT** rewrite or relabel the steps. Only output a partition of the original `seg_id` sequence into contiguous merge ranges.\n\n"
        )
        example_thought = "Two short setup fragments clearly belong to the following placement step."
    elif granularity == "coarse":
        formatted_boundary_hints: list[str] = []
        for hint in boundary_hints or []:
            boundary_after_seg_id = int(hint.get("boundary_after_seg_id", -1))
            boundary_frame = int(hint.get("boundary_frame", 0))
            left_instruction = str(hint.get("left_instruction", "")).strip() or "Unknown task step"
            right_instruction = str(hint.get("right_instruction", "")).strip() or "Unknown task step"
            reason_tokens: list[str] = []
            if bool(hint.get("has_boundary_support", False)):
                reason_tokens.append(f"support={float(hint.get('boundary_support', 0.0)):.3f}")
            if bool(hint.get("sequence_markers", False)):
                reason_tokens.append("sequence_change")
            if bool(hint.get("instruction_drift", False)):
                reason_tokens.append("focus_change")
            reason_summary = ", ".join(reason_tokens) if reason_tokens else "candidate_anchor"
            formatted_boundary_hints.append(
                f"- after seg_id={boundary_after_seg_id}, frame={boundary_frame}, signals=[{reason_summary}], "
                f"left={left_instruction!r}, right={right_instruction!r}"
            )

        merge_guidance = (
            "These segments come from a recall-first over-segmentation pass. Your job is to merge them into **coarse task-level steps** that match broad human task annotation. Think in terms of broad robot commands tied to distinct externally visible outcomes, not in terms of domain phases, scene categories, or high-level stage labels.\n\n"
            "### Goal\n"
            "Favor fewer, broader task steps whenever adjacent fine segments still belong to one higher-level immediate goal.\n"
            "The final partition should be much closer to broad human task steps than to fine-grained onset fragments.\n"
            "Aim for one broad robot command per dominant externally visible outcome, not one segment per broad context label or stage name.\n"
            "Do **NOT** invent new boundaries, delete time, reorder segments, or split a segment further.\n"
            "You may only merge adjacent original segments into larger contiguous ranges.\n\n"
            "### Coarse Merge Rules\n"
            "1. Prefer merging setup, approach, alignment, execution, immediate continuation, and short cleanup tails when they all serve one broader task outcome. A coarse step may contain setup plus execution plus immediate cleanup for one dominant outcome.\n"
            "2. Merge repeated micro-fragments, short restarts, and closely spaced sub-actions when a human would still describe them as one broad instruction with one dominant outcome.\n"
            "3. Consecutive fine segments may stay together even if the verb wording changes, as long as they still read as the same broad command with the same main target and dominant visible outcome. Do **NOT** use vague same-stage reasoning to merge across separate completed outcomes.\n"
            "4. Keep a boundary when there is a plausible coarse task switch such as a new main target, a new manipulated object focus, a new transfer destination, a completed outcome followed by a distinct next goal, or any point where a human annotator would still write a separate broad command.\n"
            "5. Do **NOT** keep boundaries that exist only because the earlier pass intentionally split first-versus-second rounds, setup versus execution, or tiny instruction wording drift inside one broader step. But do **NOT** merge a broad stage label if that would absorb multiple separate broad commands with separate visible outcomes.\n"
            "6. When uncertain between two neighboring coarse interpretations, choose the partition that better preserves separate broad commands with distinct visible outcomes. Do **NOT** merge merely to reduce the segment count.\n"
            "7. Place the boundary where the two neighboring broad commands are best separated for a human annotator. Do **NOT** bias toward the earliest possible hint of the next command or the latest residual tail of the previous one. A short lead-in, overlap, or settling tail may stay with the neighboring coarse step when that yields a cleaner broad command boundary.\n"
            "8. Do **NOT** rewrite or relabel the steps. Only output a partition of the original `seg_id` sequence into contiguous merge ranges.\n\n"
        )
        if formatted_boundary_hints:
            boundary_hint_section = (
                "### Objective Boundary Hints\n"
                "The following candidate anchors are inferred from the visual segmentation pass. They are advisory anchors, not hard constraints.\n"
                "Use them to check whether a broad merge would swallow distinct visible outcomes or distinct command switches.\n"
                "Do **NOT** keep every hinted boundary automatically, but do **NOT** ignore a cluster of hinted anchors merely to force a smaller segment count.\n"
                "If a hint introduces a short numbered sub-action that then immediately continues, the cleaner coarse boundary may be after that short completed sub-action rather than at its first introduction.\n"
                f"{chr(10).join(formatted_boundary_hints)}\n\n"
            )
        example_thought = "Several adjacent fine fragments still form one broader task-level step, so they should be merged coarsely."
    else:
        raise ValueError(f"Unsupported merge granularity: {granularity}")

    return (
        "You are reviewing a finalized sequence of adjacent robot task segments that may be over-segmented.\n"
        f"{merge_guidance}"
        f"{boundary_hint_section}"
        "### Output Format\n"
        "Return raw JSON only.\n"
        "Return one object with keys `thought` and `merged_ranges`.\n"
        "`thought` must be one short sentence.\n"
        "`merged_ranges` must cover every original `seg_id` exactly once, in order, with contiguous ranges.\n"
        "Example:\n"
        "{\n"
        f"  \"thought\": \"{example_thought}\",\n"
        "  \"merged_ranges\": [\n"
        "    {\"start_seg_id\": 0, \"end_seg_id\": 1},\n"
        "    {\"start_seg_id\": 2, \"end_seg_id\": 2},\n"
        "    {\"start_seg_id\": 3, \"end_seg_id\": 5}\n"
        "  ]\n"
        "}\n\n"
        "### Segments\n"
        f"{segment_listing}\n"
    )


def prompt_segment_hierarchy(
    segments: list[dict],
    enabled_levels: list[str],
) -> str:
    if not enabled_levels:
        raise ValueError("enabled_levels must not be empty")

    formatted_segments: list[str] = []
    for segment in segments:
        seg_id = int(segment.get("seg_id", len(formatted_segments)))
        start_frame = int(segment.get("start_frame", 0))
        end_frame = int(segment.get("end_frame", start_frame))
        duration_frames = max(0, end_frame - start_frame)
        instruction = str(segment.get("instruction", "")).strip() or "Unknown task step"
        formatted_segments.append(
            f"- seg_id={seg_id}, frames=[{start_frame}, {end_frame}), duration_frames={duration_frames}, instruction={instruction!r}"
        )

    level_rules: list[str] = []
    if "coarse" in enabled_levels:
        level_rules.append(
            "- `coarse`: the broadest meaningful task grouping. Prefer a few outcome-level commands that summarize larger chunks of work."
        )
    if "medium" in enabled_levels:
        level_rules.append(
            "- `medium`: an intermediate abstraction. Keep more structure than `coarse`, but still merge adjacent cleaned segments when they clearly serve one moderate subtask."
        )
    if "fine" in enabled_levels:
        level_rules.append(
            "- `fine`: the most detailed meaningful layer. Stay closest to the cleaned segments and summarize concrete visible actions without scene or stage labels."
        )

    example_lines = [
        "{",
        '  "thought": "The cleaned segments can be summarized at multiple action granularities.",',
    ]
    for level_name in enabled_levels:
        if level_name == "coarse":
            example_lines.extend(
                [
                    '  "coarse": [',
                    '    {"start_seg_id": 0, "end_seg_id": 2, "summary": "Prepare the first work area"},',
                    '    {"start_seg_id": 3, "end_seg_id": 5, "summary": "Complete the transfer and finish the task"}',
                    "  ],",
                ]
            )
        elif level_name == "medium":
            example_lines.extend(
                [
                    '  "medium": [',
                    '    {"start_seg_id": 0, "end_seg_id": 1, "summary": "Position and load the first item"},',
                    '    {"start_seg_id": 2, "end_seg_id": 3, "summary": "Adjust the target and continue loading"},',
                    '    {"start_seg_id": 4, "end_seg_id": 5, "summary": "Transfer out the completed result"}',
                    "  ],",
                ]
            )
        elif level_name == "fine":
            example_lines.extend(
                [
                    '  "fine": [',
                    '    {"start_seg_id": 0, "end_seg_id": 0, "summary": "Pick up the first item"},',
                    '    {"start_seg_id": 1, "end_seg_id": 1, "summary": "Place the first item into the target area"},',
                    '    {"start_seg_id": 2, "end_seg_id": 2, "summary": "Adjust the target area"},',
                    '    {"start_seg_id": 3, "end_seg_id": 5, "summary": "Complete the remaining transfer sequence"}',
                    "  ],",
                ]
            )
    example_lines[-1] = example_lines[-1].rstrip(",")
    example_lines.append("}")
    example_json = "\n".join(example_lines)

    output_keys = ", ".join([f"`{level_name}`" for level_name in enabled_levels])

    return (
        "You are reviewing a cleaned sequence of robot task segments after an over-segmentation cleanup pass.\n"
        "Your job is to keep the cleaned segment order fixed, then summarize the same sequence at one or more abstraction levels.\n\n"
        "### Goal\n"
        "Produce objective hierarchical task summaries that are useful for downstream task aggregation.\n"
        "Do **NOT** imitate scene descriptions, stage labels, recipe phases, narration, or visual presentation shots.\n"
        "Summaries must describe visible manipulation actions or externally visible outcomes.\n"
        "Remove no time, invent no new segments, and reorder nothing.\n"
        "You may only group adjacent original `seg_id` values into larger contiguous ranges for each requested summary level.\n\n"
        "### Level Definitions\n"
        f"{chr(10).join(level_rules)}\n\n"
        "### General Rules\n"
        "1. Every enabled level must partition the full original `seg_id` sequence exactly once, in order, with contiguous inclusive ranges.\n"
        "2. Finer levels must stay nested inside broader levels. Do **NOT** create crossing ranges.\n"
        "3. Use short, objective action summaries with concrete verbs and visible objects or outcomes.\n"
        "4. Do **NOT** use scene labels, stage labels, narration labels, camera labels, or state-only labels such as `Show result`, `Observe`, `Presentation`, or `Preparation stage`.\n"
        "5. If identity is ambiguous, keep the wording grounded but coarse rather than guessing domain-specific terms.\n"
        "6. `coarse` should be the broadest enabled level, `fine` the narrowest enabled level. The three levels should differ in abstraction, not just rephrase the same granularity.\n"
        "7. The cleaned segments already removed obvious failed over-cuts. Focus this pass on meaningful grouping and summarization, not on redoing the first-stage boundary search.\n\n"
        "### Output Format\n"
        "Return raw JSON only.\n"
        f"Return one object with keys `thought` and {output_keys}.\n"
        "`thought` must be one short sentence.\n"
        "Each enabled level must be an array of objects with keys `start_seg_id`, `end_seg_id`, and `summary`.\n"
        "Example:\n"
        f"{example_json}\n\n"
        "### Cleaned Segments\n"
        f"{chr(10).join(formatted_segments)}\n"
    )


def prompt_segment_subtitles(
    segments: list[dict],
    target_language: str,
) -> str:
    normalized_language = str(target_language).strip().lower()
    if normalized_language not in {"zh", "en"}:
        raise ValueError(f"Unsupported subtitle language: {target_language}")

    formatted_segments: list[str] = []
    for index, segment in enumerate(segments):
        seg_id = int(segment.get("seg_id", index))
        start_frame = int(segment.get("start_frame", 0))
        end_frame = int(segment.get("end_frame", start_frame))
        instruction = str(segment.get("instruction", "")).strip() or "Unknown task step"
        formatted_segments.append(
            f"- seg_id={seg_id}, frames=[{start_frame}, {end_frame}), instruction={instruction!r}"
        )

    target_label = "Simplified Chinese" if normalized_language == "zh" else "English"
    target_rule = (
        "Write concise Simplified Chinese robot-task subtitles. Use objective action wording and visible objects. "
        "Preserve ordinal cues such as first/second when they are semantically important."
        if normalized_language == "zh"
        else "Write concise English robot-task subtitles. Preserve the original task meaning and ordering."
    )
    example_subtitles = (
        '[{"seg_id": 0, "subtitle": "拿起第一个物体"}, {"seg_id": 1, "subtitle": "将第一个物体放入目标区域"}]'
        if normalized_language == "zh"
        else '[{"seg_id": 0, "subtitle": "Pick up the first item"}, {"seg_id": 1, "subtitle": "Place the first item into the target area"}]'
    )

    return (
        "You are localizing finalized robot task instructions for exported video subtitles.\n"
        "Do not change the segment count, order, or time ranges.\n"
        "The source `instruction` text must remain semantically unchanged; only generate subtitle text for export.\n\n"
        "### Goal\n"
        f"Translate each segment instruction into {target_label} subtitles.\n"
        f"{target_rule}\n"
        "Do **NOT** merge, split, reorder, summarize across segments, or introduce scene narration.\n"
        "Do **NOT** mention cameras, recipes, stages, or presentation language.\n"
        "Keep one subtitle per input segment.\n\n"
        "### Output Format\n"
        "Return raw JSON only.\n"
        "Return one object with keys `thought` and `subtitles`.\n"
        "`thought` must be one short sentence.\n"
        "`subtitles` must be an array of objects with keys `seg_id` and `subtitle`, covering every original segment exactly once in order.\n"
        "Example:\n"
        "{\n"
        '  "thought": "The finalized instructions can be localized one by one.",\n'
        f'  "subtitles": {example_subtitles}\n'
        "}\n\n"
        "### Segments\n"
        f"{chr(10).join(formatted_segments)}\n"
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
