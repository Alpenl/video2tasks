"""FastAPI server for job queue management."""

from fastapi import FastAPI
import uvicorn

from ..config import Config
from ..logging_utils import configure_logging, get_logger
from .exporter import export_sample_outputs
from .llm_merge import attach_stage2_subtitles_to_segments, run_llm_stage2_pass
from .producer import create_producer_loop
from .routes import register_routes
from .runtime import ThreadRuntime
from .runtime_state import RuntimeDependencies, build_runtime_state
from .windowing import (
    FrameExtractor,
    apply_boundary_refinement_results,
    apply_deferred_segment_labels,
    build_boundary_refinement_windows,
    build_refinement_windows,
    build_segments_via_cuts,
    build_windows,
    read_video_info,
    sample_segment_frame_ids,
)


logger = get_logger(__name__)


def create_app(config: Config) -> FastAPI:
    """Create and configure FastAPI application."""

    configure_logging(config.logging.level)
    app = FastAPI(title="Video2Tasks Server")

    runtime_state = build_runtime_state(
        config=config,
        logger=logger,
        dependencies=RuntimeDependencies(
            read_video_info_resolver=lambda: read_video_info,
            build_windows_resolver=lambda: build_windows,
            frame_extractor_cls_resolver=lambda: FrameExtractor,
            build_refinement_windows_resolver=lambda: build_refinement_windows,
            build_segments_via_cuts_resolver=lambda: build_segments_via_cuts,
            build_boundary_refinement_windows_resolver=lambda: build_boundary_refinement_windows,
            apply_boundary_refinement_results_resolver=lambda: apply_boundary_refinement_results,
            sample_segment_frame_ids_resolver=lambda: sample_segment_frame_ids,
            apply_deferred_segment_labels_resolver=lambda: apply_deferred_segment_labels,
            run_llm_stage2_pass_resolver=lambda: run_llm_stage2_pass,
            attach_stage2_subtitles_to_segments_resolver=lambda: attach_stage2_subtitles_to_segments,
            export_sample_outputs_resolver=lambda: export_sample_outputs,
        ),
    )
    runtime_state.attach_app_state(app)
    runtime_state.initialize_runtime_artifacts()
    register_routes(app, runtime_state)
    app.state.runtime = ThreadRuntime(
        name="video2tasks-producer",
        target=create_producer_loop(runtime_state),
        daemon=True,
    )
    return app


def run_server(config: Config) -> None:
    """Run the server with given configuration."""

    app = create_app(config)
    runtime = app.state.runtime
    runtime.start()
    try:
        uvicorn.run(
            app,
            host=config.server.host,
            port=config.server.port,
            log_level=config.logging.level.lower(),
        )
    finally:
        runtime.stop()
        runtime.join()
