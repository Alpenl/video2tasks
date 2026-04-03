from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Optional


@dataclass(frozen=True)
class BoundaryMatch:
    gt_index: int
    gt_frame: int
    matched_pred_frame: Optional[int]
    delta_frames: Optional[int]
    hit: bool


@dataclass(frozen=True)
class BoundaryRecallSummary:
    tolerance_frames: int
    gt_boundary_count: int
    pred_boundary_count: int
    hit_count: int
    miss_count: int
    recall: float
    mean_abs_delta_on_hits: Optional[float]
    median_abs_delta_on_hits: Optional[float]
    max_abs_delta_on_hits: Optional[int]
    matches: list[BoundaryMatch]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["matches"] = [asdict(match) for match in self.matches]
        return data


def predicted_boundary_frames_from_segments_file(path: str | Path) -> list[int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    return [int(segment["end_frame"]) for segment in segments[:-1]]


def official_boundary_frames_from_file(path: str | Path) -> list[int]:
    segments = json.loads(Path(path).read_text(encoding="utf-8"))
    return [int(segment["end_frame"]) for segment in segments[:-1]]


def _nearest_predicted_boundary(gt_frame: int, pred_boundaries: list[int]) -> tuple[Optional[int], Optional[int]]:
    if not pred_boundaries:
        return None, None
    nearest = min(pred_boundaries, key=lambda pred_frame: abs(pred_frame - gt_frame))
    return nearest, nearest - gt_frame


def score_boundary_recall(
    gt_boundaries: list[int],
    pred_boundaries: list[int],
    tolerance_frames: int = 5,
) -> BoundaryRecallSummary:
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames must be non-negative")

    normalized_pred = sorted(int(frame) for frame in pred_boundaries)
    matches: list[BoundaryMatch] = []
    hit_deltas: list[int] = []

    for gt_index, gt_frame in enumerate(int(frame) for frame in gt_boundaries):
        matched_pred_frame, delta_frames = _nearest_predicted_boundary(gt_frame, normalized_pred)
        hit = delta_frames is not None and abs(delta_frames) <= tolerance_frames
        if hit and delta_frames is not None:
            hit_deltas.append(abs(delta_frames))
        matches.append(
            BoundaryMatch(
                gt_index=gt_index,
                gt_frame=gt_frame,
                matched_pred_frame=matched_pred_frame,
                delta_frames=delta_frames,
                hit=bool(hit),
            )
        )

    gt_boundary_count = len(gt_boundaries)
    hit_count = sum(1 for match in matches if match.hit)
    miss_count = gt_boundary_count - hit_count
    recall = hit_count / gt_boundary_count if gt_boundary_count else 0.0

    return BoundaryRecallSummary(
        tolerance_frames=tolerance_frames,
        gt_boundary_count=gt_boundary_count,
        pred_boundary_count=len(normalized_pred),
        hit_count=hit_count,
        miss_count=miss_count,
        recall=recall,
        mean_abs_delta_on_hits=(mean(hit_deltas) if hit_deltas else None),
        median_abs_delta_on_hits=(median(hit_deltas) if hit_deltas else None),
        max_abs_delta_on_hits=(max(hit_deltas) if hit_deltas else None),
        matches=matches,
    )
