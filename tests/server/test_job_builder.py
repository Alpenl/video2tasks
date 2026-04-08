import video2tasks.server.app as app_module

from video2tasks.server.job_builder import JobBuilder
from video2tasks.server.windowing import BoundaryRefinementWindow, Window


class _Batch:
    def __init__(self, manifest_path: str, image_path: str) -> None:
        record = type("ArtifactRecord", (), {"path": image_path})()
        self.records = [record]
        self.manifest_path = manifest_path


class _ArtifactExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def get_many_b64_with_artifacts(self, *_args, **_kwargs):
        self.calls += 1
        return [], _Batch(
            manifest_path=f"/tmp/manifest_{self.calls}.json",
            image_path=f"/tmp/image_{self.calls}.png",
        )


def _make_builder() -> JobBuilder:
    return JobBuilder(
        target_width=720,
        target_height=480,
        png_compression=0,
        use_contact_sheets=True,
        contact_sheet_rows=4,
        contact_sheet_cols=4,
    )


def test_build_window_job_reuses_cached_artifact_with_explicit_producer_consumer_fields() -> None:
    extractor = _ArtifactExtractor()
    builder = _make_builder()
    window = Window(window_id=0, start_frame=0, end_frame=15, frame_ids=[0, 5, 10, 15])
    reuse_cache = {}

    first_job = builder.build_window_boundary_job(
        extractor,
        task_id="demo::sample_w0_r0",
        subset="demo",
        sample_id="sample",
        window=window,
        fps=30.0,
        nframes=16,
        repeat_index=0,
        repeat_count=2,
        reuse_cache=reuse_cache,
    )
    second_job = builder.build_window_boundary_job(
        extractor,
        task_id="demo::sample_w0_r1",
        subset="demo",
        sample_id="sample",
        window=window,
        fps=30.0,
        nframes=16,
        repeat_index=1,
        repeat_count=2,
        reuse_cache=reuse_cache,
    )

    assert extractor.calls == 1
    assert first_job.meta["artifact_reuse"] is False
    assert second_job.meta["artifact_reuse"] is True
    assert second_job.meta["artifact_producer_task_id"] == "demo::sample_w0_r0"
    assert second_job.meta["artifact_consumer_task_id"] == "demo::sample_w0_r1"
    assert second_job.meta["artifact_reuse_group"] == first_job.meta["artifact_reuse_group"]
    assert (
        second_job.image_transport.artifact_manifest_path
        == first_job.image_transport.artifact_manifest_path
    )


def test_build_window_job_reuse_group_excludes_refinement_pass() -> None:
    extractor = _ArtifactExtractor()
    builder = _make_builder()
    coarse_window = Window(window_id=0, start_frame=0, end_frame=15, frame_ids=[0, 5, 10, 15])
    refinement_window = Window(window_id=0, start_frame=0, end_frame=15, frame_ids=[0, 5, 10, 15])
    reuse_cache = {}

    builder.build_window_boundary_job(
        extractor,
        task_id="demo::sample_w0_r0",
        subset="demo",
        sample_id="sample",
        window=coarse_window,
        fps=30.0,
        nframes=16,
        repeat_index=0,
        repeat_count=1,
        reuse_cache=reuse_cache,
    )
    refinement_job = builder.build_window_boundary_job(
        extractor,
        task_id="demo::sample_rw0_r0",
        subset="demo",
        sample_id="sample",
        window=refinement_window,
        fps=30.0,
        nframes=16,
        repeat_index=0,
        repeat_count=1,
        window_pass="refinement",
        reuse_cache=reuse_cache,
    )

    assert extractor.calls == 2
    assert refinement_job.meta["artifact_reuse"] is False
    assert refinement_job.meta["window_pass"] == "refinement"


def test_build_window_job_reuse_group_changes_with_reuse_boundary_inputs() -> None:
    extractor = _ArtifactExtractor()
    builder = _make_builder()

    first_job = builder.build_window_boundary_job(
        extractor,
        task_id="demo::sample_w0_r0",
        subset="demo",
        sample_id="sample",
        window=Window(window_id=0, start_frame=0, end_frame=15, frame_ids=[0, 5, 10, 15]),
        fps=30.0,
        nframes=16,
        repeat_index=0,
        repeat_count=1,
    )
    second_job = builder.build_window_boundary_job(
        extractor,
        task_id="demo::sample_w0_shifted_r0",
        subset="demo",
        sample_id="sample",
        window=Window(window_id=0, start_frame=0, end_frame=15, frame_ids=[1, 6, 11, 15]),
        fps=30.0,
        nframes=16,
        repeat_index=0,
        repeat_count=1,
    )

    assert first_job.meta["artifact_reuse_group"] != second_job.meta["artifact_reuse_group"]


def test_build_boundary_refinement_and_segment_label_jobs_include_transport_metadata() -> None:
    extractor = _ArtifactExtractor()
    builder = _make_builder()

    boundary_job = builder.build_boundary_refinement_job(
        extractor,
        task_id="demo::sample_b3",
        subset="demo",
        sample_id="sample",
        boundary_window=BoundaryRefinementWindow(
            boundary_id=3,
            coarse_boundary_frame=12,
            start_frame=10,
            end_frame=14,
            frame_ids=[10, 11, 12, 13],
        ),
    )
    segment_job = builder.build_segment_label_job(
        extractor,
        task_id="demo::sample_seg7",
        subset="demo",
        sample_id="sample",
        segment={"seg_id": 7, "start_frame": 20, "end_frame": 35},
        frame_ids=[20, 25, 30, 35],
    )

    assert boundary_job.meta["job_type"] == "boundary_refinement"
    assert boundary_job.meta["artifact_reuse"] is False
    assert boundary_job.meta["artifact_producer_task_id"] == "demo::sample_b3"
    assert segment_job.meta["job_type"] == "segment_label"
    assert segment_job.meta["artifact_consumer_task_id"] == "demo::sample_seg7"


def test_app_no_longer_exposes_legacy_job_builder_helpers() -> None:
    assert not hasattr(app_module, "_clone_shared_fs_transport")
    assert not hasattr(app_module, "_repeat_artifact_reuse_key")
    assert not hasattr(app_module, "_build_job_payload")
