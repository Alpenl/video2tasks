from video2tasks.server import stage2_merge, stage2_subtitles, stage2_summary
from video2tasks.server.llm_merge import (
    attach_stage2_subtitles_to_segments,
    run_export_subtitle_localization_pass,
    run_llm_merge_pass,
    run_llm_stage2_pass,
    run_llm_subtitle_localization_pass,
    run_llm_summary_pass,
    validate_merged_partition,
    validate_merged_ranges,
    validate_summary_partitions,
)


def test_llm_merge_facade_reexports_stage2_module_entrypoints() -> None:
    assert run_llm_merge_pass is stage2_merge.run_llm_merge_pass
    assert validate_merged_ranges is stage2_merge.validate_merged_ranges
    assert validate_merged_partition is stage2_merge.validate_merged_partition

    assert run_llm_summary_pass is stage2_summary.run_llm_summary_pass
    assert validate_summary_partitions is stage2_summary.validate_summary_partitions

    assert run_llm_subtitle_localization_pass is stage2_subtitles.run_llm_subtitle_localization_pass
    assert run_export_subtitle_localization_pass is stage2_subtitles.run_export_subtitle_localization_pass
    assert attach_stage2_subtitles_to_segments is stage2_subtitles.attach_stage2_subtitles_to_segments
    assert run_llm_stage2_pass is stage2_subtitles.run_llm_stage2_pass
