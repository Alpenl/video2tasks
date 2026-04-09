"""Window planning and segment assembly helpers."""

import re
import sys
from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .segment_semantics import (
    _ACTION_FAMILY_SUPPORT_THRESHOLD,
    _ACTION_FILLERS,
    _COOKING_CONTAINER_TOKENS,
    _DESTINATION_FOCUS_SKIP_TOKENS,
    _DESTINATION_SPLIT_RE,
    _DISH_FOCUS_TOKENS,
    _FILLER_TOKENS,
    _FOCUS_NOISE_TOKENS,
    _GENERIC_ACTION_TOKENS,
    _GENERIC_ADDITIVE_TOKENS,
    _GENERIC_PHASE_NOUN_TOKENS,
    _GENERIC_REFERENCE_TOKENS,
    _MIXING_ACTION_TOKENS,
    _PREP_ACTION_TOKENS,
    _PREP_INGREDIENT_ACTION_TOKENS,
    _RECIPE_ACTION_TOKENS,
    _ROBOT_MOTION_TOKENS,
    _SAUCE_BASE_TOKENS,
    _SEASONING_FOCUS_TOKENS,
    _STEP_ORDER_TOKENS,
    _STOPWORDS,
    _STRONG_ACTION_TOKENS,
    _TOKEN_RE,
    _UTENSIL_TOKENS,
    _action_families,
    _action_tokens,
    _destination_focus_tokens,
    _destination_food_tokens,
    _destination_tokens,
    _generic_reference_subject_tokens,
    _ingredient_tokens,
    _instruction_action_head_tokens,
    _instruction_focus_tokens,
    _instruction_has_generic_phase_noun,
    _instruction_has_mixing_action,
    _instruction_has_prep_ingredient_action,
    _instruction_has_recipe_action,
    _instruction_is_filler_segment,
    _instruction_is_generic,
    _instruction_is_prep_like,
    _instruction_specificity,
    _instruction_tokens,
    _primary_object_tokens,
    _singularize_token,
    _subject_focus_tokens,
    boundary_support_between as _boundary_support_between,
    has_distinct_sequence_markers as _has_distinct_sequence_markers,
    refine_segment_instructions,
    should_split_on_instruction_drift as _should_split_on_instruction_drift,
)
from ..vlm.base import normalize_task_window_result


_WINDOWING_FACADE_MODULE = "video2tasks.server.windowing"


def _resolve_windowing_attr(name: str, fallback: Any) -> Any:
    facade = sys.modules.get(_WINDOWING_FACADE_MODULE)
    if facade is None:
        return fallback
    return getattr(facade, name, fallback)


@dataclass
class Window:
    """Video window definition."""
    window_id: int
    start_frame: int
    end_frame: int
    frame_ids: List[int]


@dataclass
class BoundaryRefinementWindow:
    """Short clip centered on a provisional boundary for local refinement."""
    boundary_id: int
    coarse_boundary_frame: int
    start_frame: int
    end_frame: int
    frame_ids: List[int]


_REFINEMENT_WINDOW_ID_BASE = 1_000_000


def build_windows(
    fps: float,
    nframes: int,
    window_sec: float = 16.0,
    step_sec: float = 8.0,
    frames_per_window: int = 16
) -> List[Window]:
    """Build video windows with frame sampling."""
    if fps < 1e-6:
        fps = 30.0
    
    win_len = max(1, int(round(window_sec * fps)))
    step = max(1, int(round(step_sec * fps)))
    windows: List[Window] = []
    
    def get_frames(s: int, e: int, num: int) -> List[int]:
        idx = np.linspace(s, e, num=num).astype(int)
        return np.clip(idx, 0, nframes - 1).tolist()
    
    s = 0
    wid = 0
    while s < nframes:
        e = min(nframes - 1, s + win_len - 1)
        if (e - s < win_len // 2) and wid > 0:
            break
        windows.append(Window(wid, s, e, get_frames(s, e, frames_per_window)))
        wid += 1
        s += step
    
    return windows


def build_window_prompt_metadata(window: Window, fps: float, nframes: int) -> dict:
    """Build temporal metadata for prompt generation."""
    safe_fps = float(fps) if fps > 1e-6 else 30.0
    return {
        "window_id": int(window.window_id),
        "frame_ids": [int(frame_id) for frame_id in window.frame_ids],
        "fps": safe_fps,
        "window_start_frame": int(window.start_frame),
        "window_end_frame": int(window.end_frame),
        "nframes": max(0, int(nframes)),
    }


def _full_instruction_action_families(text: str) -> set[str]:
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    families = set()
    if tokens & {"add", "pour", "fill", "sprinkle"}:
        families.add("add")
    if tokens & {"season", "stir", "mix", "whisk", "toss"}:
        families.add("mix")
    if tokens & {"cook", "saute", "fry", "boil", "toast", "heat", "microwave", "bake"}:
        families.add("cook")
    if tokens & _PREP_INGREDIENT_ACTION_TOKENS:
        families.add("prep")
    if tokens & {"pick", "lift", "carry", "place", "transfer", "remove", "move", "stack", "insert", "put"}:
        families.add("transfer")
    return families


def _instruction_needs_refinement(text: str) -> bool:
    lower = text.lower()
    families = _full_instruction_action_families(text)
    non_transfer_families = families - {"transfer"}
    has_multi_clause_surface = (" and " in lower) or (" then " in lower) or ("," in lower)
    if len(non_transfer_families) >= 2 and has_multi_clause_surface:
        return True
    if _instruction_has_generic_phase_noun(text) and len(non_transfer_families) >= 1 and has_multi_clause_surface:
        return True
    return False


def _window_should_spawn_refinement(rec: dict) -> bool:
    vlm = rec.get("vlm_json", {})
    transitions = vlm.get("transitions", [])
    if transitions:
        return False

    instructions = [
        str(item).strip()
        for item in vlm.get("instructions", [])
        if str(item).strip()
    ]
    if not instructions or len(instructions) > 2:
        return False
    return any(_instruction_needs_refinement(text) for text in instructions)


def _sample_window_frame_ids(start_frame: int, end_frame: int, frames_per_window: int, nframes: int) -> List[int]:
    if nframes <= 0:
        return []
    idx = np.linspace(start_frame, end_frame, num=frames_per_window).astype(int)
    return np.clip(idx, 0, nframes - 1).tolist()


def _micro_peak_gap_frames(fps: float) -> int:
    safe_fps = float(fps) if fps > 1e-6 else 30.0
    return max(2, min(5, int(round(0.2 * safe_fps))))


def _vote_repeat_window_transitions(
    repeat_transition_sets: List[List[int]],
    frame_ids: List[int],
    fps: float,
) -> Tuple[List[int], Dict[int, float]]:
    """Vote repeated window transitions into weighted local cut indices.

    Each repeat contributes at most one support count to a micro-cluster so that
    a single noisy response cannot inflate support by emitting several nearby cuts.
    """
    if not repeat_transition_sets or not frame_ids:
        return [], {}

    if len(repeat_transition_sets) == 1:
        single = sorted(
            {
                int(item)
                for item in repeat_transition_sets[0]
                if 0 <= int(item) < len(frame_ids)
            }
        )
        return single, {int(local_idx): 1.0 for local_idx in single}

    total_repeats = max(1, len(repeat_transition_sets))
    gap_frames = _micro_peak_gap_frames(fps)
    frame_to_local_indices: Dict[int, List[int]] = {}
    for local_idx, frame in enumerate(frame_ids):
        frame_to_local_indices.setdefault(int(frame), []).append(int(local_idx))

    frame_votes: Dict[int, Dict[str, Any]] = {}
    for repeat_id, transitions in enumerate(repeat_transition_sets):
        for item in transitions:
            try:
                local_idx = int(item)
            except (TypeError, ValueError):
                continue
            if not (0 <= local_idx < len(frame_ids)):
                continue
            frame = int(frame_ids[local_idx])
            bucket = frame_votes.setdefault(
                frame,
                {"repeat_ids": set(), "local_indices": [], "count": 0},
            )
            bucket["repeat_ids"].add(int(repeat_id))
            bucket["local_indices"].append(local_idx)
            bucket["count"] += 1

    if not frame_votes:
        return [], {}

    clustered_frames: List[List[int]] = []
    current_cluster: List[int] = []

    for frame in sorted(frame_votes):
        if not current_cluster:
            current_cluster = [int(frame)]
            continue

        if (int(frame) - current_cluster[-1]) <= gap_frames:
            current_cluster.append(int(frame))
            continue

        clustered_frames.append(current_cluster)
        current_cluster = [int(frame)]

    if current_cluster:
        clustered_frames.append(current_cluster)

    voted_local_indices: List[int] = []
    support_by_local_index: Dict[int, float] = {}

    for cluster in clustered_frames:
        cluster_repeat_ids = set()
        for frame in cluster:
            cluster_repeat_ids.update(frame_votes[frame]["repeat_ids"])

        representative_frame = min(
            cluster,
            key=lambda frame: (
                -len(frame_votes[frame]["repeat_ids"]),
                -int(frame_votes[frame]["count"]),
                frame,
            ),
        )
        representative_locals = frame_to_local_indices.get(int(representative_frame), [])
        if representative_locals:
            representative_local = min(int(item) for item in representative_locals)
        else:
            representative_local = min(
                range(len(frame_ids)),
                key=lambda idx: abs(int(frame_ids[idx]) - int(representative_frame)),
            )

        voted_local_indices.append(int(representative_local))
        support_by_local_index[int(representative_local)] = len(cluster_repeat_ids) / float(total_repeats)

    voted_local_indices = sorted(set(voted_local_indices))
    support_by_local_index = {
        int(local_idx): float(support_by_local_index.get(int(local_idx), 0.0))
        for local_idx in voted_local_indices
    }
    return voted_local_indices, support_by_local_index


def _select_instruction_source_vlm(
    vlms: List[Dict[str, Any]],
    voted_transitions: List[int],
    frame_ids: List[int],
    fps: float,
) -> Dict[str, Any]:
    if not vlms:
        return {}
    if len(vlms) == 1:
        return vlms[0]

    gap_frames = _micro_peak_gap_frames(fps)
    voted_frames = [int(frame_ids[idx]) for idx in voted_transitions if 0 <= int(idx) < len(frame_ids)]

    def record_score(vlm: Dict[str, Any]) -> Tuple[int, float, int, int]:
        record_transitions = [int(item) for item in vlm.get("transitions", []) if 0 <= int(item) < len(frame_ids)]
        record_frames = [int(frame_ids[idx]) for idx in record_transitions]

        if not voted_frames:
            return (-len(record_frames), 0.0, 0, -len(vlm.get("instructions", [])))

        matches = 0
        distance_sum = 0.0
        for voted_frame in voted_frames:
            if not record_frames:
                distance_sum += float(gap_frames + 1)
                continue
            nearest = min(abs(record_frame - voted_frame) for record_frame in record_frames)
            if nearest <= gap_frames:
                matches += 1
            distance_sum += float(nearest)

        return (
            matches,
            -distance_sum,
            -abs(len(record_frames) - len(voted_frames)),
            len(vlm.get("instructions", [])),
        )

    return max(vlms, key=record_score)


def _cluster_cut_votes(raw_cuts: List[Tuple[int, float]], fps: float) -> Tuple[List[int], Dict[int, float]]:
    """Cluster nearby cut votes while preserving distinct local peaks.

    The boundary detector is intentionally over-segmenting. Dense action regions can produce
    multiple candidate cuts inside one broader phase transition. For this pre-merge pass, it is
    safer to keep distinct local peaks than to collapse them into one representative point.

    The clustering rule is therefore intentionally narrow:
    - first aggregate votes landing on the exact same frame
    - then merge only short gaps into a micro-peak
    - emit one representative per micro-peak

    Inside one micro-peak, prefer the strongest frame (highest vote count, then highest support)
    and use the earliest frame only as a tie-breaker. This avoids a weak early singleton dragging
    the cluster representative away from a stronger nearby peak, while still keeping an onset bias
    inside tightly packed votes.
    """
    if not raw_cuts:
        return [], {}

    peak_gap_frames = _micro_peak_gap_frames(fps)

    aggregated_votes: Dict[int, Dict[str, float]] = {}
    for fid, weight in raw_cuts:
        frame = int(fid)
        bucket = aggregated_votes.setdefault(frame, {"count": 0.0, "support": 0.0})
        bucket["count"] += 1.0
        bucket["support"] += float(weight)

    sorted_frames = sorted(aggregated_votes)
    clustered_points: List[int] = []
    cut_support_by_point: Dict[int, float] = {}

    cur_frames: List[int] = []

    def flush_cluster() -> None:
        if not cur_frames:
            return
        point = min(
            cur_frames,
            key=lambda frame: (
                -aggregated_votes[frame]["count"],
                -aggregated_votes[frame]["support"],
                frame,
            ),
        )
        cluster_support = float(sum(aggregated_votes[frame]["support"] for frame in cur_frames))
        clustered_points.append(point)
        cut_support_by_point[point] = max(cut_support_by_point.get(point, 0.0), cluster_support)

    for fid in sorted_frames:
        if not cur_frames:
            cur_frames.append(int(fid))
            continue

        prev_fid = cur_frames[-1]
        if (fid - prev_fid) <= peak_gap_frames:
            cur_frames.append(int(fid))
            continue

        flush_cluster()
        cur_frames = [int(fid)]

    flush_cluster()
    return clustered_points, cut_support_by_point


def sample_segment_frame_ids(
    start_frame: int,
    end_frame: int,
    frames_per_window: int,
    nframes: int,
) -> List[int]:
    """Sample representative frames for a finalized segment.

    `end_frame` follows the segment convention used elsewhere in the pipeline:
    it is exclusive, so the last available frame is `end_frame - 1`.
    """
    if nframes <= 0:
        return []
    safe_start = max(0, int(start_frame))
    safe_end = max(safe_start, int(end_frame) - 1)
    return _sample_window_frame_ids(safe_start, safe_end, frames_per_window, nframes)


def build_refinement_windows(
    base_windows: List[Window],
    by_wid: dict,
    fps: float,
    nframes: int,
    frames_per_window: int,
) -> List[Window]:
    """Build shorter refinement windows only for ambiguous first-pass windows."""
    refinement_windows: List[Window] = []
    safe_fps = float(fps) if fps > 1e-6 else 30.0

    for window in base_windows:
        rec = by_wid.get(window.window_id)
        if not rec or not _window_should_spawn_refinement(rec):
            continue

        base_len = max(1, int(window.end_frame - window.start_frame + 1))
        refine_len = min(
            base_len,
            max(int(round(base_len * 0.5)), int(round(3.0 * safe_fps))),
        )
        refine_step = max(1, refine_len // 2)
        max_start = max(window.start_frame, window.end_frame - refine_len + 1)

        starts = []
        cur_start = int(window.start_frame)
        while cur_start < max_start:
            starts.append(cur_start)
            cur_start += refine_step
        starts.append(int(max_start))

        for offset, sub_start in enumerate(sorted(set(starts))):
            sub_end = min(int(window.end_frame), sub_start + refine_len - 1)
            refinement_windows.append(
                Window(
                    window_id=_REFINEMENT_WINDOW_ID_BASE + int(window.window_id) * 10 + offset,
                    start_frame=sub_start,
                    end_frame=sub_end,
                    frame_ids=_sample_window_frame_ids(
                        sub_start,
                        sub_end,
                        frames_per_window,
                        nframes,
                    ),
                )
            )

    return refinement_windows


def build_boundary_refinement_windows(
    segments: List[dict],
    fps: float,
    nframes: int,
    window_sec: float,
    frames_per_window: int,
) -> List[BoundaryRefinementWindow]:
    """Build short local clips around provisional boundaries for position refinement."""
    if nframes <= 0 or len(segments) <= 1:
        return []

    safe_fps = float(fps) if fps > 1e-6 else 30.0
    window_len = max(1, int(round(window_sec * safe_fps)))
    refinement_windows: List[BoundaryRefinementWindow] = []

    for boundary_id, segment in enumerate(segments[:-1]):
        coarse_boundary = int(segment["end_frame"])
        half = window_len // 2
        start_frame = max(0, coarse_boundary - half)
        end_frame = min(nframes - 1, start_frame + window_len - 1)
        start_frame = max(0, end_frame - window_len + 1)
        frame_ids = _sample_window_frame_ids(
            start_frame,
            end_frame,
            frames_per_window,
            nframes,
        )
        refinement_windows.append(
            BoundaryRefinementWindow(
                boundary_id=boundary_id,
                coarse_boundary_frame=coarse_boundary,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_ids=frame_ids,
            )
        )

    return refinement_windows


def _refinement_abstention_confirms_same_task(record: dict) -> bool:
    vlm_json = record.get("vlm_json", {})
    if not isinstance(vlm_json, dict):
        return False

    instructions = [
        str(item).strip()
        for item in vlm_json.get("instructions", [])
        if str(item).strip()
    ]
    return len(instructions) == 1


def apply_boundary_refinement_results(
    segments: List[dict],
    refinement_results: dict,
    fps: float = 30.0,
    abstain_merge_max_support: float = -1.0,
) -> List[dict]:
    """Shift provisional boundaries using local boundary-refinement predictions.

    When `abstain_merge_max_support` is enabled, a local verifier can also veto a
    weak coarse boundary by returning no transition. This lets the second pass
    filter low-confidence false positives instead of only shifting accepted cuts.
    """
    if not segments:
        return []

    resolved_boundaries: dict[int, int] = {}
    rejected_boundaries: set[int] = set()

    for idx in range(len(segments) - 1):
        record = refinement_results.get(idx) or refinement_results.get(str(idx))
        if not isinstance(record, dict):
            continue

        frame_ids = record.get("frame_ids", [])
        vlm_json = record.get("vlm_json", {})
        transitions = vlm_json.get("transitions", []) if isinstance(vlm_json, dict) else []

        selected_frame: Optional[int] = None
        for transition in transitions:
            try:
                local_idx = int(transition)
            except (TypeError, ValueError):
                continue
            if 0 <= local_idx < len(frame_ids):
                selected_frame = int(frame_ids[local_idx])
                break

        if selected_frame is None:
            if abstain_merge_max_support < 0:
                continue
            if not _refinement_abstention_confirms_same_task(record):
                continue

            support, _ = _boundary_support_between(segments[idx], segments[idx + 1])
            if support <= float(abstain_merge_max_support):
                rejected_boundaries.add(idx)
            continue

        left = segments[idx]
        right = segments[idx + 1]
        clamped_frame = max(int(left["start_frame"]) + 1, min(selected_frame, int(right["end_frame"]) - 1))
        if clamped_frame <= int(left["start_frame"]) or clamped_frame >= int(right["end_frame"]):
            continue
        resolved_boundaries[idx] = clamped_frame

    rebuilt: List[dict] = []
    current = dict(segments[0])

    for idx in range(len(segments) - 1):
        right = dict(segments[idx + 1])

        if idx in rejected_boundaries:
            current["end_frame"] = int(right["end_frame"])
            current["instruction"] = _choose_instruction(current, right, fps)
            if "boundary_support_after" in right:
                current["boundary_support_after"] = right["boundary_support_after"]
            continue

        boundary_frame = int(resolved_boundaries.get(idx, segments[idx]["end_frame"]))
        boundary_frame = max(int(current["start_frame"]) + 1, min(boundary_frame, int(right["end_frame"]) - 1))

        support, has_support = _boundary_support_between(segments[idx], segments[idx + 1])

        finalized = dict(current)
        finalized["end_frame"] = boundary_frame
        if has_support:
            finalized["boundary_support_after"] = support
        rebuilt.append(finalized)

        current = dict(right)
        current["start_frame"] = boundary_frame
        if has_support:
            current["boundary_support_before"] = support

    rebuilt.append(current)

    for idx, segment in enumerate(rebuilt):
        segment["seg_id"] = idx
    return rebuilt


def _instruction_is_bridge_motion(text: str) -> bool:
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    object_tokens = _primary_object_tokens(text)
    if object_tokens:
        return False
    if tokens & _ROBOT_MOTION_TOKENS:
        return True
    return ("reposition" in tokens or "move" in tokens) and "gripper" in tokens


def _instruction_is_preparatory_segment(text: str) -> bool:
    lower = text.lower().strip()
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    completion_tokens = {
        "place", "stack", "insert", "remove", "retrieve", "connect", "plug",
        "fold", "pour", "nest",
    }
    if lower.startswith("reach "):
        return True
    if "align the gripper" in lower or "align gripper" in lower:
        return True
    if "position the gripper" in lower or "position gripper" in lower:
        return True
    if "reposition the gripper" in lower:
        return True
    if "prepare" in tokens and not (tokens & completion_tokens):
        return True
    if (tokens & {"align", "hover", "reach", "approach"}) and not (tokens & completion_tokens):
        return True
    return False


def _token_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(len(left | right))


def _segment_duration_sec(segment: dict, fps: float) -> float:
    if fps < 1e-6:
        fps = 30.0
    return max(0.0, (segment["end_frame"] - segment["start_frame"]) / fps)


def _choose_instruction(left: dict, right: dict, fps: float) -> str:
    shared_ingredient_tokens = _ingredient_tokens(left["instruction"]) & _ingredient_tokens(right["instruction"])
    if (
        shared_ingredient_tokens
        and _instruction_has_prep_ingredient_action(left["instruction"])
        and _instruction_has_prep_ingredient_action(right["instruction"])
        and _action_tokens(left["instruction"]) != _action_tokens(right["instruction"])
    ):
        return right["instruction"]

    left_specificity = _instruction_specificity(left["instruction"])
    right_specificity = _instruction_specificity(right["instruction"])
    if right_specificity > left_specificity:
        return right["instruction"]
    if left_specificity > right_specificity:
        return left["instruction"]

    left_duration = _segment_duration_sec(left, fps)
    right_duration = _segment_duration_sec(right, fps)
    if right_duration >= left_duration * 0.75:
        return right["instruction"]
    return left["instruction"]


def _should_merge_segments(left: dict, right: dict, fps: float, boundary_support_threshold: float = 0.9) -> bool:
    left_tokens = _primary_object_tokens(left["instruction"])
    right_tokens = _primary_object_tokens(right["instruction"])
    similarity = _token_similarity(left_tokens, right_tokens)
    left_dest_tokens = _destination_tokens(left["instruction"])
    right_dest_tokens = _destination_tokens(right["instruction"])
    shared_dest_tokens = left_dest_tokens & right_dest_tokens
    left_ingredient_tokens = _ingredient_tokens(left["instruction"])
    right_ingredient_tokens = _ingredient_tokens(right["instruction"])
    shared_ingredient_tokens = left_ingredient_tokens & right_ingredient_tokens
    left_focus_tokens = _instruction_focus_tokens(left["instruction"])
    right_focus_tokens = _instruction_focus_tokens(right["instruction"])
    shared_focus_tokens = left_focus_tokens & right_focus_tokens
    shared_action_tokens = _action_tokens(left["instruction"]) & _action_tokens(right["instruction"])
    shared_action_families = _action_families(left["instruction"]) & _action_families(right["instruction"])
    boundary_support, has_boundary_support = _boundary_support_between(left, right)
    merge_support_ceiling = (
        float(boundary_support_threshold)
        if boundary_support_threshold > 0.0
        else float(_ACTION_FAMILY_SUPPORT_THRESHOLD)
    )
    strong_boundary = has_boundary_support and boundary_support_threshold > 0.0 and boundary_support >= boundary_support_threshold

    left_recipe = _instruction_has_recipe_action(left["instruction"])
    right_recipe = _instruction_has_recipe_action(right["instruction"])
    left_prep_ingredient = _instruction_has_prep_ingredient_action(left["instruction"])
    right_prep_ingredient = _instruction_has_prep_ingredient_action(right["instruction"])
    left_mixing = _instruction_has_mixing_action(left["instruction"])
    right_mixing = _instruction_has_mixing_action(right["instruction"])
    left_generic = _instruction_is_generic(left["instruction"])
    right_generic = _instruction_is_generic(right["instruction"])
    left_prep = _instruction_is_prep_like(left["instruction"])
    right_prep = _instruction_is_prep_like(right["instruction"])
    left_filler = _instruction_is_filler_segment(left["instruction"])
    right_filler = _instruction_is_filler_segment(right["instruction"])
    left_generic_focus = bool(_generic_reference_subject_tokens(left["instruction"])) and not bool(
        left_focus_tokens - _GENERIC_REFERENCE_TOKENS
    )
    right_generic_focus = bool(_generic_reference_subject_tokens(right["instruction"])) and not bool(
        right_focus_tokens - _GENERIC_REFERENCE_TOKENS
    )
    left_destination_focus = _destination_focus_tokens(left["instruction"])
    right_destination_focus = _destination_focus_tokens(right["instruction"])
    left_additive_only = bool(left_ingredient_tokens) and left_ingredient_tokens <= _SEASONING_FOCUS_TOKENS
    right_additive_only = bool(right_ingredient_tokens) and right_ingredient_tokens <= _SEASONING_FOCUS_TOKENS
    same_ingredient_prep_to_transfer = bool(shared_ingredient_tokens) and (
        (left_prep_ingredient and right_recipe) or (right_prep_ingredient and left_recipe)
    )
    same_ingredient_same_action_prep = (
        bool(shared_ingredient_tokens)
        and bool(shared_action_tokens)
        and left_prep_ingredient
        and right_prep_ingredient
    )
    distinct_same_ingredient_prep_steps = (
        bool(shared_ingredient_tokens)
        and left_prep_ingredient
        and right_prep_ingredient
        and not shared_action_tokens
    )
    distinct_prepped_ingredient_steps = (
        bool(shared_dest_tokens & _COOKING_CONTAINER_TOKENS)
        and left_recipe
        and right_recipe
        and not (left_mixing or right_mixing)
        and bool(left_ingredient_tokens)
        and bool(right_ingredient_tokens)
        and not shared_ingredient_tokens
        and (left_prep_ingredient or right_prep_ingredient)
    )
    same_container_same_family_recipe_steps = (
        bool(shared_dest_tokens & _COOKING_CONTAINER_TOKENS)
        and left_recipe
        and right_recipe
        and bool(shared_action_families & {"add", "mix"})
    )
    generic_phase_to_explicit_ingredient_shift = (
        bool(shared_dest_tokens & {"pot", "pan"})
        and bool(shared_action_families & {"mix", "cook"})
        and has_boundary_support
        and boundary_support >= 1.0
        and not (left_focus_tokens & right_focus_tokens)
        and (
            (_instruction_has_generic_phase_noun(left["instruction"]) and bool(right_focus_tokens))
            or (_instruction_has_generic_phase_noun(right["instruction"]) and bool(left_focus_tokens))
        )
    )
    heated_distinct_add_sequence = (
        bool(shared_dest_tokens & {"pot", "pan"})
        and shared_action_families == {"add"}
        and bool(left_ingredient_tokens)
        and bool(right_ingredient_tokens)
        and not shared_ingredient_tokens
    )
    same_container_shared_focus_recipe_steps = (
        bool(shared_dest_tokens & _COOKING_CONTAINER_TOKENS)
        and bool(shared_focus_tokens)
        and not heated_distinct_add_sequence
        and (
            bool(shared_action_families)
            or bool(shared_action_tokens)
            or (left_recipe and right_recipe)
            or (left_mixing and right_recipe)
            or (right_mixing and left_recipe)
            or (left_prep and right_recipe)
            or (right_prep and left_recipe)
        )
    )
    same_container_generic_focus_bridge = (
        bool(shared_dest_tokens & _COOKING_CONTAINER_TOKENS)
        and (
            (left_generic_focus and bool(right_focus_tokens - _GENERIC_REFERENCE_TOKENS))
            or (right_generic_focus and bool(left_focus_tokens - _GENERIC_REFERENCE_TOKENS))
        )
        and (
            bool(shared_action_families)
            or (left_prep_ingredient and right_prep_ingredient)
            or (left_prep_ingredient and right_recipe)
            or (right_prep_ingredient and left_recipe)
            or (left_prep_ingredient and right_mixing)
            or (right_prep_ingredient and left_mixing)
        )
    )
    same_container_additive_mix_bridge = (
        bool(shared_dest_tokens & _COOKING_CONTAINER_TOKENS)
        and (
            (left_additive_only and bool(left_destination_focus) and right_mixing and bool(right_focus_tokens))
            or (right_additive_only and bool(right_destination_focus) and left_mixing and bool(left_focus_tokens))
        )
    )
    repeated_sequence_markers = _has_distinct_sequence_markers(
        left["instruction"],
        right["instruction"],
    )

    if similarity < 0.34:
        # Adjacent sub-steps can stay mergeable when there is shared structure,
        # but recall-first means we should be conservative once the link is weak.
        if not (
            left_prep
            or right_prep
            or left_filler
            or right_filler
            or (
                same_container_shared_focus_recipe_steps
                and (not has_boundary_support or boundary_support < merge_support_ceiling)
            )
            or (
                same_container_generic_focus_bridge
                and (not has_boundary_support or boundary_support < merge_support_ceiling)
            )
            or (
                same_container_additive_mix_bridge
                and (not has_boundary_support or boundary_support < merge_support_ceiling)
            )
            or (
                shared_dest_tokens & _COOKING_CONTAINER_TOKENS
                and (left_recipe or right_recipe)
                and (shared_ingredient_tokens or shared_action_tokens)
                and not repeated_sequence_markers
            )
            or same_ingredient_prep_to_transfer
            or same_ingredient_same_action_prep
            or (distinct_same_ingredient_prep_steps and has_boundary_support and not strong_boundary)
            or (
                same_container_same_family_recipe_steps
                and not heated_distinct_add_sequence
                and has_boundary_support
                and boundary_support < merge_support_ceiling
                and not repeated_sequence_markers
            )
        ):
            return False

    left_duration = _segment_duration_sec(left, fps)
    right_duration = _segment_duration_sec(right, fps)

    if strong_boundary:
        return False
    if distinct_prepped_ingredient_steps:
        return False
    if generic_phase_to_explicit_ingredient_shift:
        return False
    if (
        same_container_shared_focus_recipe_steps
        and (not has_boundary_support or boundary_support < merge_support_ceiling)
        and max(left_duration, right_duration) <= 18.0
    ):
        return True
    if (
        same_container_generic_focus_bridge
        and (not has_boundary_support or boundary_support < merge_support_ceiling)
        and min(left_duration, right_duration) <= 4.5
        and max(left_duration, right_duration) <= 14.0
    ):
        return True
    if (
        same_container_additive_mix_bridge
        and (not has_boundary_support or boundary_support < merge_support_ceiling)
        and min(left_duration, right_duration) <= 12.0
        and max(left_duration, right_duration) <= 18.0
    ):
        return True
    if distinct_same_ingredient_prep_steps:
        if not has_boundary_support:
            return False
        if min(left_duration, right_duration) <= 8.0:
            return True
        return False

    if left_generic and right_generic and not (left_prep or right_prep or left_filler or right_filler):
        return True
    if left_generic and left_duration <= 4.5 and not (left_prep or left_filler):
        return True
    if right_generic and right_duration <= 4.5 and not (right_prep or right_filler):
        return True
    if shared_dest_tokens & _COOKING_CONTAINER_TOKENS and (left_recipe or right_recipe):
        if shared_ingredient_tokens and min(left_duration, right_duration) <= 8.0 and not repeated_sequence_markers:
            return True
        if shared_action_tokens and max(left_duration, right_duration) <= 8.0 and not repeated_sequence_markers:
            return True
        if (
            (left_mixing or right_mixing)
            and shared_ingredient_tokens
            and max(left_duration, right_duration) <= 20.0
            and not repeated_sequence_markers
        ):
            return True
    if same_ingredient_prep_to_transfer:
        if min(left_duration, right_duration) <= 8.0:
            return True
    if same_ingredient_same_action_prep:
        if min(left_duration, right_duration) <= 8.0:
            return True
    if (
        same_container_same_family_recipe_steps
        and not heated_distinct_add_sequence
        and has_boundary_support
        and boundary_support < merge_support_ceiling
        and not repeated_sequence_markers
    ):
        if min(left_duration, right_duration) <= 16.0:
            return True
    if similarity >= 0.6 and min(left_duration, right_duration) <= 6.5 and not repeated_sequence_markers:
        return True
    return False


def merge_task_level_segments(
    segments: List[dict],
    fps: float,
    boundary_support_threshold: float = 0.9,
) -> List[dict]:
    """Merge over-segmented adjacent spans into task-level segments."""
    if not segments:
        return []

    merged: List[dict] = []

    for segment in segments:
        current = dict(segment)
        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        if _should_merge_segments(previous, current, fps, boundary_support_threshold=boundary_support_threshold):
            chosen_instruction = _choose_instruction(previous, current, fps)
            previous["end_frame"] = current["end_frame"]
            previous["instruction"] = chosen_instruction
            previous["confidence"] = max(
                float(previous.get("confidence", 0.0)),
                float(current.get("confidence", 0.0)),
            )
            previous["boundary_support_after"] = current.get(
                "boundary_support_after",
                previous.get("boundary_support_after", 0.0),
            )
        else:
            merged.append(current)

    return cleanup_auxiliary_segments(merged, fps)


def cleanup_auxiliary_segments(segments: List[dict], fps: float) -> List[dict]:
    """Absorb bridge/prep filler spans without semantic task merging."""
    if not segments:
        return []

    bridge_cleaned: List[dict] = []
    pending_bridge: Optional[dict] = None
    for segment in segments:
        current = dict(segment)
        if _instruction_is_bridge_motion(current["instruction"]):
            if pending_bridge is None:
                pending_bridge = current
            else:
                pending_bridge["end_frame"] = current["end_frame"]
                pending_bridge["confidence"] = max(
                    float(pending_bridge.get("confidence", 0.0)),
                    float(current.get("confidence", 0.0)),
                )
            continue
        if (
            _instruction_is_filler_segment(current["instruction"])
            and _segment_duration_sec(current, fps) <= 5.0
        ):
            if pending_bridge is None:
                pending_bridge = current
            else:
                pending_bridge["end_frame"] = current["end_frame"]
                pending_bridge["confidence"] = max(
                    float(pending_bridge.get("confidence", 0.0)),
                    float(current.get("confidence", 0.0)),
                )
            continue

        if pending_bridge is not None:
            current["start_frame"] = pending_bridge["start_frame"]
            current["confidence"] = max(
                float(current.get("confidence", 0.0)),
                float(pending_bridge.get("confidence", 0.0)),
            )
            pending_bridge = None

        bridge_cleaned.append(current)

    if pending_bridge is not None and bridge_cleaned:
        bridge_cleaned[-1]["end_frame"] = pending_bridge["end_frame"]
        bridge_cleaned[-1]["confidence"] = max(
            float(bridge_cleaned[-1].get("confidence", 0.0)),
            float(pending_bridge.get("confidence", 0.0)),
        )

    filler_adjusted: List[dict] = []
    for idx, segment in enumerate(bridge_cleaned):
        current = dict(segment)
        current_duration = _segment_duration_sec(current, fps)
        if (
            _instruction_is_filler_segment(current["instruction"])
            and 5.0 < current_duration <= 12.0
            and filler_adjusted
            and idx + 1 < len(bridge_cleaned)
        ):
            previous = filler_adjusted[-1]
            following = bridge_cleaned[idx + 1]
            previous_specific = _is_specific_instruction(previous["instruction"])
            following_specific = _is_specific_instruction(following["instruction"])
            previous_dest = _destination_tokens(previous["instruction"]) & _COOKING_CONTAINER_TOKENS
            following_dest = _destination_tokens(following["instruction"]) & _COOKING_CONTAINER_TOKENS
            workspace_shift = (
                previous_dest != following_dest
                or _token_similarity(
                    _primary_object_tokens(previous["instruction"]),
                    _primary_object_tokens(following["instruction"]),
                ) < 0.25
            )
            if previous_specific and following_specific and workspace_shift:
                previous["end_frame"] = current["end_frame"]
                previous["confidence"] = max(
                    float(previous.get("confidence", 0.0)),
                    float(current.get("confidence", 0.0)),
                )
                continue

        filler_adjusted.append(current)

    prep_cleaned: List[dict] = []
    pending_prep: Optional[dict] = None
    for segment in (filler_adjusted or bridge_cleaned or segments):
        current = dict(segment)
        if _instruction_is_preparatory_segment(current["instruction"]) and _segment_duration_sec(current, fps) <= 5.0:
            if pending_prep is None:
                pending_prep = current
            else:
                pending_prep["end_frame"] = current["end_frame"]
                pending_prep["confidence"] = max(
                    float(pending_prep.get("confidence", 0.0)),
                    float(current.get("confidence", 0.0)),
                )
            continue

        if pending_prep is not None:
            current["start_frame"] = pending_prep["start_frame"]
            current["confidence"] = max(
                float(current.get("confidence", 0.0)),
                float(pending_prep.get("confidence", 0.0)),
            )
            pending_prep = None

        prep_cleaned.append(current)

    if pending_prep is not None and prep_cleaned:
        if _segment_duration_sec(pending_prep, fps) > 3.0:
            prep_cleaned[-1]["end_frame"] = pending_prep["end_frame"]
            prep_cleaned[-1]["confidence"] = max(
                float(prep_cleaned[-1].get("confidence", 0.0)),
                float(pending_prep.get("confidence", 0.0)),
            )

    final_segments = prep_cleaned or filler_adjusted or bridge_cleaned or segments

    for idx, segment in enumerate(final_segments):
        segment["seg_id"] = idx

    return final_segments


def _segment_overlap_frames(left: dict, right: dict) -> int:
    return max(
        0,
        min(int(left["end_frame"]), int(right["end_frame"])) - max(int(left["start_frame"]), int(right["start_frame"]))
    )


def _contributors_for_segment(segment: dict, source_segments: List[dict]) -> List[dict]:
    contributors = []
    for candidate in source_segments:
        overlap = _segment_overlap_frames(segment, candidate)
        if overlap > 0:
            enriched = dict(candidate)
            enriched["_overlap_frames"] = overlap
            contributors.append(enriched)
    return contributors


def _is_specific_instruction(text: str) -> bool:
    return not (
        _instruction_is_generic(text)
        or _instruction_is_prep_like(text)
        or _instruction_is_filler_segment(text)
    )


def _segment_boundary_points(segments: List[dict]) -> List[int]:
    return [int(segment["end_frame"]) for segment in segments[:-1]]


def _strong_boundary_points(segments: List[dict], boundary_support_threshold: float) -> List[int]:
    if boundary_support_threshold <= 0.0:
        return []

    points: List[int] = []
    for segment in segments[:-1]:
        try:
            support = float(segment.get("boundary_support_after", 0.0))
        except (TypeError, ValueError):
            support = 0.0
        if support >= boundary_support_threshold:
            points.append(int(segment["end_frame"]))
    return points


def _missing_boundary_points(
    protected_points: List[int],
    merged_segments: List[dict],
    tolerance_frames: int,
) -> List[int]:
    if not protected_points:
        return []

    candidate_points = _segment_boundary_points(merged_segments)
    missing: List[int] = []
    for point in protected_points:
        if any(abs(int(candidate) - int(point)) <= tolerance_frames for candidate in candidate_points):
            continue
        missing.append(int(point))
    return missing


def _missing_strong_boundary_points(
    light_segments: List[dict],
    merged_segments: List[dict],
    boundary_support_threshold: float,
    tolerance_frames: int,
) -> List[int]:
    strong_points = _strong_boundary_points(light_segments, boundary_support_threshold)
    return _missing_boundary_points(strong_points, merged_segments, tolerance_frames)


def _merged_segment_looks_overcollapsed(
    segment: dict,
    source_segments: List[dict],
    fps: float,
    median_source_duration_sec: float,
) -> bool:
    contributors = _contributors_for_segment(segment, source_segments)
    if len(contributors) < 3:
        return False

    distinct_specific = {
        item["instruction"].strip().lower()
        for item in contributors
        if _is_specific_instruction(item["instruction"])
    }
    if len(distinct_specific) < 2:
        return False

    duration_sec = _segment_duration_sec(segment, fps)
    return duration_sec >= max(10.0, median_source_duration_sec * 1.8)


def _should_fallback_to_light_cleanup(
    light_segments: List[dict],
    merged_segments: List[dict],
    fps: float,
    min_segments: int,
    collapse_ratio: float,
) -> bool:
    if len(light_segments) < min_segments:
        return False
    if not merged_segments:
        return False
    if len(merged_segments) >= max(1, int(np.ceil(len(light_segments) * collapse_ratio))):
        return False

    source_durations = [_segment_duration_sec(segment, fps) for segment in light_segments]
    median_source_duration_sec = float(np.median(source_durations)) if source_durations else 0.0
    return any(
        _merged_segment_looks_overcollapsed(segment, light_segments, fps, median_source_duration_sec)
        for segment in merged_segments
    )


def _score_instruction_candidate(segment: dict, candidate: dict, current_instruction: str) -> float:
    overlap = float(candidate.get("_overlap_frames", 0))
    text = candidate["instruction"]
    score = overlap
    score += float(_instruction_specificity(text)) * 24.0
    if text == current_instruction:
        score += 18.0
    if _instruction_is_generic(text):
        score -= 12.0
    if _instruction_is_prep_like(text):
        score -= 16.0
    if _instruction_is_filler_segment(text):
        score -= 24.0
    return score


def apply_deferred_segment_labels(final_segments: List[dict], label_results: dict) -> List[dict]:
    """Override final segment instructions with a second-pass segment labeling result."""
    labeled: List[dict] = []
    for segment in final_segments:
        current = dict(segment)
        seg_id = int(current.get("seg_id", len(labeled)))
        label_payload = label_results.get(seg_id) or label_results.get(str(seg_id))
        if isinstance(label_payload, dict):
            instructions = label_payload.get("instructions", [])
            if isinstance(instructions, list) and instructions:
                text = str(instructions[0]).strip()
                if text and text.lower() != "unknown":
                    current["instruction"] = text
        labeled.append(current)

    for idx, segment in enumerate(labeled):
        segment["seg_id"] = idx
    return labeled


def _dominant_instruction_from_candidates(candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None

    counts = Counter(candidates)
    return max(
        counts.items(),
        key=lambda item: (
            item[1],
            _instruction_specificity(item[0]),
            len(_ingredient_tokens(item[0])),
            len(_action_families(item[0])),
        ),
    )[0]


def _phase_container_tokens(text: str, fallback_containers: Optional[set[str]] = None) -> set[str]:
    explicit_containers = _destination_tokens(text) & _COOKING_CONTAINER_TOKENS
    if explicit_containers:
        return explicit_containers

    fallback = set(fallback_containers or ())
    if not fallback:
        return set()

    tokens = set(_instruction_tokens(text.replace("/", " ")))
    if tokens & (_SAUCE_BASE_TOKENS | {"sausage"}):
        return fallback
    return set()


def _is_sauce_base_phase(text: str, fallback_containers: Optional[set[str]] = None) -> bool:
    containers = _phase_container_tokens(text, fallback_containers) & {"pot", "pan"}
    if not containers:
        return False

    action_families = _action_families(text)
    if not action_families or not action_families <= {"add", "mix", "cook"}:
        return False

    ingredients = _ingredient_tokens(text)
    if not ingredients:
        return False

    return bool(ingredients & _SAUCE_BASE_TOKENS) and ingredients <= (_SAUCE_BASE_TOKENS | {"sauteed"})


def _instruction_phase_signature(text: str, fallback_containers: Optional[set[str]] = None) -> tuple:
    action_families = tuple(sorted(_action_families(text)))
    if not action_families:
        action_families = tuple(sorted(_action_tokens(text)))

    dest_tokens = tuple(sorted(_phase_container_tokens(text, fallback_containers)))
    if _is_sauce_base_phase(text, fallback_containers):
        return (("sauce_base",), dest_tokens[:1], ("sauce_base",))

    focus_tokens = list(sorted(_ingredient_tokens(text)))
    if not focus_tokens:
        focus_tokens = list(sorted(_primary_object_tokens(text) - _COOKING_CONTAINER_TOKENS))

    return (
        action_families,
        dest_tokens[:1],
        tuple(focus_tokens[:2]),
    )


def _instruction_runs_for_segment(
    segment: dict,
    instruction_timeline: List[List[str]],
    fps: float,
    bin_sec: float = 2.0,
) -> List[dict]:
    bin_frames = max(1, int(round(max(0.5, bin_sec) * max(fps, 1.0))))
    runs: List[dict] = []
    context_containers = _destination_tokens(segment["instruction"]) & _COOKING_CONTAINER_TOKENS

    for start in range(int(segment["start_frame"]), int(segment["end_frame"]), bin_frames):
        end = min(int(segment["end_frame"]), start + bin_frames)
        candidates: List[str] = []
        for fid in range(start, end):
            if 0 <= fid < len(instruction_timeline):
                candidates.extend(instruction_timeline[fid])

        instruction = _dominant_instruction_from_candidates(candidates)
        if not instruction:
            continue

        signature = _instruction_phase_signature(instruction, fallback_containers=context_containers)
        if runs and runs[-1]["signature"] == signature:
            runs[-1]["end_frame"] = end
            runs[-1]["candidates"].append(instruction)
            runs[-1]["instruction"] = _dominant_instruction_from_candidates(runs[-1]["candidates"]) or runs[-1]["instruction"]
            continue

        runs.append({
            "start_frame": start,
            "end_frame": end,
            "instruction": instruction,
            "signature": signature,
            "candidates": [instruction],
        })

    return runs


def _split_segment_on_instruction_drift(
    segment: dict,
    instruction_timeline: List[List[str]],
    fps: float,
    min_segment_frames: int,
    min_phase_frames: int,
    depth: int,
) -> List[dict]:
    current = dict(segment)
    if depth <= 0 or (int(current["end_frame"]) - int(current["start_frame"])) < min_segment_frames:
        return [current]

    context_containers = _destination_tokens(current["instruction"]) & _COOKING_CONTAINER_TOKENS
    runs = _instruction_runs_for_segment(current, instruction_timeline, fps)
    if len(runs) < 2:
        return [current]

    candidates = []
    for idx in range(len(runs) - 1):
        left_run = runs[idx]
        right_run = runs[idx + 1]
        left_duration = int(left_run["end_frame"]) - int(left_run["start_frame"])
        right_duration = int(right_run["end_frame"]) - int(right_run["start_frame"])
        if left_duration < min_phase_frames or right_duration < min_phase_frames:
            continue
        if not _should_split_on_instruction_drift(
            left_run["instruction"],
            right_run["instruction"],
            fallback_containers=context_containers,
        ):
            continue

        boundary = int((int(left_run["end_frame"]) + int(right_run["start_frame"])) / 2)
        if boundary - int(current["start_frame"]) < min_phase_frames:
            continue
        if int(current["end_frame"]) - boundary < min_phase_frames:
            continue

        score = float(min(left_duration, right_duration))
        if _action_families(left_run["instruction"]) != _action_families(right_run["instruction"]):
            score += float(min_phase_frames)
        if _is_specific_instruction(left_run["instruction"]):
            score += 0.25 * min_phase_frames
        if _is_specific_instruction(right_run["instruction"]):
            score += 0.25 * min_phase_frames

        candidates.append((score, boundary, left_run, right_run))

    if not candidates:
        return [current]

    _, boundary, left_run, right_run = max(candidates, key=lambda item: item[0])
    left_segment = dict(current)
    left_segment["end_frame"] = boundary
    left_segment["instruction"] = left_run["instruction"]
    left_segment["boundary_support_after"] = 0.0

    right_segment = dict(current)
    right_segment["start_frame"] = boundary
    right_segment["instruction"] = right_run["instruction"]
    right_segment["boundary_support_before"] = 0.0

    return (
        _split_segment_on_instruction_drift(
            left_segment,
            instruction_timeline,
            fps,
            min_segment_frames,
            min_phase_frames,
            depth - 1,
        )
        + _split_segment_on_instruction_drift(
            right_segment,
            instruction_timeline,
            fps,
            min_segment_frames,
            min_phase_frames,
            depth - 1,
        )
    )


def split_long_raw_segments_on_instruction_drift(
    raw_segments: List[dict],
    instruction_timeline: List[List[str]],
    fps: float,
    min_segment_sec: float = 30.0,
    min_phase_sec: float = 5.0,
) -> List[dict]:
    if not raw_segments:
        return []

    if fps < 1e-6:
        fps = 30.0

    min_segment_frames = max(1, int(round(min_segment_sec * fps)))
    min_phase_frames = max(1, int(round(min_phase_sec * fps)))
    split_segments: List[dict] = []

    for segment in raw_segments:
        split_segments.extend(
            _split_segment_on_instruction_drift(
                segment,
                instruction_timeline,
                fps,
                min_segment_frames,
                min_phase_frames,
                depth=3,
            )
        )

    for idx, segment in enumerate(split_segments):
        segment["seg_id"] = idx

    return split_segments


def _recover_dense_transition_micro_boundaries(
    windows: List[Window],
    by_wid: dict,
    fps: float,
    nframes: int,
    existing_cut_points: List[int],
) -> List[int]:
    if nframes <= 0:
        return []

    safe_fps = float(fps) if fps > 1e-6 else 30.0
    micro_gap_frames = max(1, min(5, int(round(0.15 * safe_fps))))
    existing_points = {int(point) for point in existing_cut_points}
    recovered: List[int] = []

    for window in windows:
        rec = by_wid.get(window.window_id)
        if not rec:
            continue

        cur_len = len(window.frame_ids)
        if cur_len == 0:
            continue

        repeat_records = rec.get("repeat_records") if isinstance(rec, dict) else None
        normalized_repeat_vlms: List[Dict[str, Any]] = []
        if isinstance(repeat_records, list) and repeat_records:
            for repeat_record in repeat_records:
                vlm = normalize_task_window_result(
                    repeat_record.get("vlm_json", {}),
                    max_transition_index=cur_len - 1,
                )
                if vlm:
                    normalized_repeat_vlms.append(vlm)
        else:
            vlm = normalize_task_window_result(
                rec.get("vlm_json", {}),
                max_transition_index=cur_len - 1,
            )
            if vlm:
                normalized_repeat_vlms.append(vlm)

        if not normalized_repeat_vlms:
            continue

        transitions, _ = _vote_repeat_window_transitions(
            [vlm.get("transitions", []) for vlm in normalized_repeat_vlms],
            window.frame_ids,
            fps,
        )
        if len(transitions) < 2:
            continue

        global_points = [int(window.frame_ids[idx]) for idx in transitions]
        for idx, point in enumerate(global_points):
            prev_point = global_points[idx - 1] if idx > 0 else None
            next_point = global_points[idx + 1] if idx + 1 < len(global_points) else None
            close_prev = prev_point is not None and 0 < (point - prev_point) <= micro_gap_frames
            close_next = next_point is not None and 0 < (next_point - point) <= micro_gap_frames
            if not (close_prev or close_next):
                continue
            if point <= 0 or point >= nframes or point in existing_points:
                continue
            recovered.append(point)

    return sorted(set(recovered))


def build_segments_via_cuts(
    sample_id: str,
    windows: List[Window],
    by_wid: dict,
    fps: float,
    nframes: int,
    frames_per_window: int = 16,
    boundary_prompt_mode: str = "freeform",
    adaptive_merge_guard: bool = True,
    adaptive_merge_min_segments: int = 8,
    adaptive_merge_collapse_ratio: float = 0.6,
    boundary_support_threshold: float = 0.9,
    refine_final_instructions: bool = True,
) -> dict:
    """Build final segments from window results."""
    if nframes == 0:
        return {}

    split_long_segments = _resolve_windowing_attr(
        "split_long_raw_segments_on_instruction_drift",
        split_long_raw_segments_on_instruction_drift,
    )
    cleanup_segments = _resolve_windowing_attr(
        "cleanup_auxiliary_segments",
        cleanup_auxiliary_segments,
    )
    merge_segments = _resolve_windowing_attr(
        "merge_task_level_segments",
        merge_task_level_segments,
    )
    missing_strong_boundary_points = _resolve_windowing_attr(
        "_missing_strong_boundary_points",
        _missing_strong_boundary_points,
    )
    missing_boundary_points = _resolve_windowing_attr(
        "_missing_boundary_points",
        _missing_boundary_points,
    )
    should_fallback_to_light_cleanup = _resolve_windowing_attr(
        "_should_fallback_to_light_cleanup",
        _should_fallback_to_light_cleanup,
    )
    refine_final_segment_instructions = _resolve_windowing_attr(
        "refine_segment_instructions",
        refine_segment_instructions,
    )

    if fps < 1e-6:
        fps = 30.0
    
    raw_cuts = []
    raw_cut_support_by_frame: Dict[int, float] = {}
    raw_cut_window_ids_by_frame: Dict[int, set[int]] = {}
    instruction_timeline = [[] for _ in range(nframes)]
    center_weights = np.hanning(frames_per_window + 2)[1:-1]
    
    for w in windows:
        rec = by_wid.get(w.window_id)
        if not rec:
            continue

        f_ids = w.frame_ids
        cur_len = len(f_ids)

        if cur_len == 0:
            continue

        repeat_records = rec.get("repeat_records") if isinstance(rec, dict) else None
        normalized_repeat_vlms: List[Dict[str, Any]] = []
        if isinstance(repeat_records, list) and repeat_records:
            for repeat_record in repeat_records:
                vlm = normalize_task_window_result(
                    repeat_record.get("vlm_json", {}),
                    max_transition_index=cur_len - 1,
                )
                if vlm:
                    normalized_repeat_vlms.append(vlm)
        else:
            vlm = normalize_task_window_result(
                rec.get("vlm_json", {}),
                max_transition_index=cur_len - 1,
            )
            if vlm:
                normalized_repeat_vlms.append(vlm)

        if not normalized_repeat_vlms:
            continue

        voted_transitions, voted_support = _vote_repeat_window_transitions(
            [vlm.get("transitions", []) for vlm in normalized_repeat_vlms],
            f_ids,
            fps,
        )
        instruction_vlm = _select_instruction_source_vlm(
            normalized_repeat_vlms,
            voted_transitions,
            f_ids,
            fps,
        )

        transitions = voted_transitions
        instructions = instruction_vlm.get("instructions", [])
        instruction_boundaries = instruction_vlm.get("transitions", [])

        # Collect cut points
        for t_idx in transitions:
            try:
                idx = int(t_idx)
                if 0 <= idx < cur_len:
                    global_fid = f_ids[idx]
                    if cur_len == frames_per_window:
                        w_val = center_weights[idx]
                    else:
                        w_val = 1.0 if min(idx, cur_len - 1 - idx) > 2 else 0.5
                    weighted_support = float(w_val) * float(voted_support.get(idx, 1.0))
                    raw_cuts.append((global_fid, weighted_support))
                    raw_cut_support_by_frame[global_fid] = raw_cut_support_by_frame.get(global_fid, 0.0) + weighted_support
                    raw_cut_window_ids_by_frame.setdefault(int(global_fid), set()).add(int(w.window_id))
            except (ValueError, IndexError):
                pass

        # Collect instructions
        try:
            boundaries = [0] + [int(t) for t in instruction_boundaries if 0 <= int(t) < cur_len] + [cur_len]
            boundaries = sorted(list(set(boundaries)))
            
            for i in range(len(boundaries) - 1):
                if i < len(instructions):
                    inst = str(instructions[i]).strip()
                    if inst and inst.lower() != "unknown":
                        s_local, e_local = boundaries[i], boundaries[i + 1]
                        for k in range(s_local, e_local):
                            if k < cur_len:
                                global_fid = f_ids[k]
                                if global_fid < nframes:
                                    instruction_timeline[global_fid].append(inst)
        except (ValueError, IndexError):
            pass
    
    # Cluster cuts
    final_cut_points = [0]
    cut_support_by_point: dict[int, float] = {}
    
    if raw_cuts:
        clustered_points, clustered_support = _cluster_cut_votes(raw_cuts, fps)
        final_cut_points.extend(clustered_points)
        cut_support_by_point.update(clustered_support)
    
    final_cut_points.append(nframes)
    final_cut_points = sorted(list(set(final_cut_points)))

    if boundary_prompt_mode in {"center_scan", "multi_probe_scan"}:
        # Local probe-style prompting is much less reliable when only one window votes
        # for a boundary. Require corroboration from overlapping windows before
        # promoting an interior cut into the final segmentation.
        min_cluster_support = max(1.0, float(np.max(center_weights))) + 1e-6
        corroboration_gap_frames = max(1, int(round(fps)))

        def supporting_window_count(point: int) -> int:
            supporting_windows: set[int] = set()
            for raw_point, window_ids in raw_cut_window_ids_by_frame.items():
                if abs(int(raw_point) - int(point)) <= corroboration_gap_frames:
                    supporting_windows.update(int(window_id) for window_id in window_ids)
            return len(supporting_windows)

        interior_points = sorted(int(point) for point in final_cut_points[1:-1])
        corroborated_support_by_point: dict[int, float] = {}
        corroborated_points: List[int] = []

        for point in interior_points:
            if supporting_window_count(point) < 2:
                continue
            neighborhood_support = float(
                sum(
                    float(cut_support_by_point.get(other, 0.0))
                    for other in interior_points
                    if abs(other - point) <= corroboration_gap_frames
                )
            )
            if neighborhood_support >= min_cluster_support:
                corroborated_points.append(point)
                corroborated_support_by_point[point] = neighborhood_support

        merged_probe_points: List[int] = []
        if corroborated_points:
            cur_probe_cluster: List[int] = []

            def flush_probe_cluster() -> None:
                if not cur_probe_cluster:
                    return
                representative = min(
                    cur_probe_cluster,
                    key=lambda point: (
                        -corroborated_support_by_point.get(point, 0.0),
                        point,
                    ),
                )
                merged_probe_points.append(representative)

            for point in sorted(set(corroborated_points)):
                if not cur_probe_cluster or (point - cur_probe_cluster[-1]) <= corroboration_gap_frames:
                    cur_probe_cluster.append(point)
                    continue
                flush_probe_cluster()
                cur_probe_cluster = [point]

            flush_probe_cluster()

        final_cut_points = [0] + merged_probe_points + [nframes]
        cut_support_by_point = {
            point: float(corroborated_support_by_point.get(point, cut_support_by_point.get(point, 0.0)))
            for point in merged_probe_points
        }

    recovered_micro_cut_points = _recover_dense_transition_micro_boundaries(
        windows,
        by_wid,
        fps,
        nframes,
        final_cut_points,
    )
    if boundary_prompt_mode in {"center_scan", "multi_probe_scan"}:
        recovered_micro_cut_points = [
            int(point)
            for point in recovered_micro_cut_points
            if len(
                {
                    int(window_id)
                    for raw_point, window_ids in raw_cut_window_ids_by_frame.items()
                    if abs(int(raw_point) - int(point)) <= max(1, int(round(fps)))
                    for window_id in window_ids
                }
            ) >= 2
        ]
    if recovered_micro_cut_points:
        final_cut_points = sorted(set(final_cut_points) | set(recovered_micro_cut_points))
        for point in recovered_micro_cut_points:
            cut_support_by_point.setdefault(int(point), float(raw_cut_support_by_frame.get(int(point), 0.0)))

    # Build segments
    raw_segments = []
    seg_id = 0
    
    for i in range(len(final_cut_points) - 1):
        s, e = int(final_cut_points[i]), int(final_cut_points[i + 1])
        if e <= s:
            continue

        margin = int((e - s) * 0.2) if e > s else 0
        mid_s, mid_e = s + margin, e - margin
        mid_start = min(max(s, mid_s), max(s, e - 1))
        mid_end = min(e, max(mid_start + 1, mid_e))

        candidates = []
        for f in range(mid_start, mid_end):
            if f < nframes:
                candidates.extend(instruction_timeline[f])

        if not candidates:
            for f in range(s, e):
                if f < nframes:
                    candidates.extend(instruction_timeline[f])

        best_inst = Counter(candidates).most_common(1)[0][0] if candidates else "Unknown task step"
        raw_segments.append({
            "seg_id": seg_id,
            "start_frame": s,
            "end_frame": e,
            "instruction": best_inst,
            "confidence": 1.0,
            "boundary_support_before": float(cut_support_by_point.get(s, 0.0)),
            "boundary_support_after": float(cut_support_by_point.get(e, 0.0)),
        })
        seg_id += 1

    raw_segments = split_long_segments(
        raw_segments,
        instruction_timeline,
        fps,
    )
    light_segments = cleanup_segments(raw_segments, fps)
    merged_segments = merge_segments(
        light_segments,
        fps,
        boundary_support_threshold=boundary_support_threshold,
    )
    strong_boundary_tolerance_frames = max(1, min(5, int(round(0.2 * max(fps, 1.0)))))
    strong_boundary_points = _strong_boundary_points(light_segments, boundary_support_threshold)
    missing_strong_boundaries_if_merged = missing_strong_boundary_points(
        light_segments,
        merged_segments,
        boundary_support_threshold,
        strong_boundary_tolerance_frames,
    )
    recovered_micro_boundary_tolerance_frames = 0
    missing_recovered_micro_boundaries_if_merged = missing_boundary_points(
        recovered_micro_cut_points,
        merged_segments,
        recovered_micro_boundary_tolerance_frames,
    )
    adaptive_light_fallback = adaptive_merge_guard and should_fallback_to_light_cleanup(
        light_segments,
        merged_segments,
        fps,
        min_segments=adaptive_merge_min_segments,
        collapse_ratio=adaptive_merge_collapse_ratio,
    )

    if not merged_segments:
        final_output = light_segments
        selected_segment_source = "light_cleanup"
        selection_policy = "light_cleanup_empty_merge"
    elif missing_strong_boundaries_if_merged:
        final_output = light_segments
        selected_segment_source = "light_cleanup"
        selection_policy = "light_cleanup_strong_boundary_guard"
    elif missing_recovered_micro_boundaries_if_merged:
        final_output = light_segments
        selected_segment_source = "light_cleanup"
        selection_policy = "light_cleanup_micro_boundary_guard"
    elif adaptive_light_fallback:
        final_output = light_segments
        selected_segment_source = "light_cleanup"
        selection_policy = "light_cleanup_fallback"
    else:
        final_output = merged_segments
        selected_segment_source = "merged"
        selection_policy = "merged_default"

    if refine_final_instructions:
        final_output = refine_final_segment_instructions(final_output, light_segments)

    return {
        "sample_id": sample_id,
        "nframes": nframes,
        "segments": final_output,
        "diagnostics": {
            "light_segment_count": len(light_segments),
            "merged_segment_count": len(merged_segments),
            "selected_segment_count": len(final_output),
            "selected_segment_source": selected_segment_source,
            "selection_policy": selection_policy,
            "strong_boundary_count": len(strong_boundary_points),
            "strong_boundary_points": strong_boundary_points,
            "missing_strong_boundaries_if_merged": missing_strong_boundaries_if_merged,
            "missing_recovered_micro_boundaries_if_merged": missing_recovered_micro_boundaries_if_merged,
            "strong_boundary_tolerance_frames": strong_boundary_tolerance_frames,
            "recovered_micro_boundary_count": len(recovered_micro_cut_points),
            "recovered_micro_boundary_points": recovered_micro_cut_points,
        },
    }
