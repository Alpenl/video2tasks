"""Stable facade for Stage 2 text post-processing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..config import LLMMergeConfig
from .stage2_merge import run_llm_merge_pass, validate_merged_partition, validate_merged_ranges
from .stage2_subtitles import (
    attach_stage2_subtitles_to_segments,
    run_export_subtitle_localization_pass,
    run_llm_stage2_pass,
    run_llm_subtitle_localization_pass,
)
from .stage2_summary import run_llm_summary_pass, validate_summary_partitions


def run_llm_postprocess_pass(
    sample_id: str,
    segments: List[dict],
    merge_config: LLMMergeConfig,
    backend: Any = None,
) -> Tuple[List[dict], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Legacy Stage 2 API kept as a compatibility facade.

    App-side orchestration should use `run_llm_stage2_pass(...)` as the canonical
    Stage 2 contract. This wrapper remains for module consumers that still expect
    the older merge+summary tuple shape.

    Returns:
      - cleaned_segments: merge output (or original segments on merge failure)
      - task_hierarchy: optional summary hierarchy
      - diagnostics: merged diagnostics from merge + summary

    Notes:
      - Summary is independent from merge: merge failures do not suppress summary.
      - Subtitle localization is intentionally not part of this legacy return shape.
    """

    cleaned_segments, merge_diagnostics = run_llm_merge_pass(
        sample_id,
        segments,
        merge_config,
        backend=backend,
    )

    task_hierarchy, summary_diagnostics = run_llm_summary_pass(
        sample_id,
        cleaned_segments,
        merge_config,
        backend=backend,
    )

    diagnostics = dict(merge_diagnostics)
    diagnostics.update(summary_diagnostics)
    return cleaned_segments, task_hierarchy, diagnostics


__all__ = [
    'attach_stage2_subtitles_to_segments',
    'run_export_subtitle_localization_pass',
    'run_llm_merge_pass',
    'run_llm_postprocess_pass',
    'run_llm_stage2_pass',
    'run_llm_subtitle_localization_pass',
    'run_llm_summary_pass',
    'validate_merged_partition',
    'validate_merged_ranges',
    'validate_summary_partitions',
]
