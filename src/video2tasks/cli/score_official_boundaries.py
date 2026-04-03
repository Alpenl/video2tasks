"""CLI for official boundary recall scoring."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ..eval.official_boundaries import (
    official_boundary_frames_from_file,
    predicted_boundary_frames_from_segments_file,
    score_boundary_recall,
)


@click.command()
@click.option(
    "--pred-segments",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to predicted segments.json",
)
@click.option(
    "--official-segments",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to official_segments.json",
)
@click.option(
    "--tolerance-frames",
    type=int,
    default=5,
    show_default=True,
    help="Frame tolerance for counting an official boundary as hit",
)
@click.option(
    "--json-output/--text-output",
    default=False,
    show_default=True,
    help="Emit machine-readable JSON instead of text summary",
)
def main(
    pred_segments: Path,
    official_segments: Path,
    tolerance_frames: int,
    json_output: bool,
) -> None:
    """Score official boundary recall without penalizing over-segmentation."""
    pred_boundaries = predicted_boundary_frames_from_segments_file(pred_segments)
    gt_boundaries = official_boundary_frames_from_file(official_segments)
    summary = score_boundary_recall(
        gt_boundaries=gt_boundaries,
        pred_boundaries=pred_boundaries,
        tolerance_frames=tolerance_frames,
    )

    if json_output:
        click.echo(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return

    click.echo(f"Tolerance: +/-{summary.tolerance_frames} frames")
    click.echo(f"Official boundaries: {summary.gt_boundary_count}")
    click.echo(f"Predicted boundaries: {summary.pred_boundary_count} (reported only, not scored)")
    click.echo(f"Hits: {summary.hit_count}")
    click.echo(f"Misses: {summary.miss_count}")
    click.echo(f"Recall: {summary.recall:.3f}")
    if summary.mean_abs_delta_on_hits is not None:
        click.echo(
            "Hit abs delta stats: "
            f"mean={summary.mean_abs_delta_on_hits:.2f}, "
            f"median={summary.median_abs_delta_on_hits:.2f}, "
            f"max={summary.max_abs_delta_on_hits}"
        )

    missed = [match for match in summary.matches if not match.hit]
    if missed:
        click.echo("Missed official boundaries:")
        for match in missed:
            pred = "none" if match.matched_pred_frame is None else str(match.matched_pred_frame)
            delta = "none" if match.delta_frames is None else f"{match.delta_frames:+d}"
            click.echo(
                f"- gt[{match.gt_index}] frame={match.gt_frame}, nearest_pred={pred}, delta={delta}"
            )


if __name__ == "__main__":
    main()
