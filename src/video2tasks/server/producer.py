"""Producer runtime orchestration for the server app."""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logging_utils import log_event
from .runtime_state import (
    ServerRuntimeState,
    count_failed_samples,
    final_exit_code,
    job_payload_meta,
    job_queue_contains_task_id,
)
from .task_artifacts import (
    ArtifactPayloadValidationError,
    artifact_validation_error_details,
)


@dataclass
class _SampleArtifactDispatchFailure(Exception):
    reason: str
    phase: str
    error: ArtifactPayloadValidationError
    details: Dict[str, Any]


def _artifact_failure_details(
    error: ArtifactPayloadValidationError,
    *,
    phase: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {"phase": phase, **artifact_validation_error_details(error)}
    payload["error"] = str(error).strip() or type(error).__name__
    payload["exception_type"] = type(error).__name__
    if details:
        payload.update(details)
    return payload


def _fail_sample_for_invalid_artifacts(
    runtime_state: ServerRuntimeState,
    state: Dict[str, Any],
    subset: str,
    sample_id: str,
    *,
    reason: str,
    phase: str,
    error: ArtifactPayloadValidationError,
    log_message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    runtime_state.fail_sample(
        state,
        subset,
        sample_id,
        reason,
        _artifact_failure_details(error, phase=phase, details=details),
        log_message=log_message,
    )


def create_producer_loop(runtime_state: ServerRuntimeState):
    """Build the producer loop bound to a shared runtime facade."""

    def producer_loop(stop_event: threading.Event) -> None:
        config = runtime_state.config
        deps = runtime_state.dependencies
        dataset_ctxs = runtime_state.dataset_ctxs

        total = sum(len(ctx.sample_ids) for ctx in dataset_ctxs)
        progress_total = config.progress.total_override if config.progress.total_override > 0 else total

        done = 0
        for ctx in dataset_ctxs:
            for sample_id in ctx.sample_ids:
                if Path(runtime_state.done_marker_path(ctx.subset, sample_id)).exists():
                    done += 1

        runtime_state.logger.info(
            f"[Server] Started. IMG=PNG, "
            f"FIXED={config.windowing.target_width}x{config.windowing.target_height}, "
            f"FRAMES_PER_WINDOW={config.windowing.frames_per_window}\n"
            f"[Plan] DATASETS={[(ctx.data_dir, ctx.subset) for ctx in dataset_ctxs]}\n"
            f"[Resume] Already done: {done}/{progress_total} (computed_total={total})"
        )

        states = {
            ctx.subset: {
                "cur_idx": 0,
                "sample_status": {sample_id: 0 for sample_id in ctx.sample_ids},
            }
            for ctx in dataset_ctxs
        }

        dataset_idx = 0
        global_done = done

        while not stop_event.is_set():
            now = time.time()
            exhausted_timeouts: List[tuple[str, str, Dict[str, Any], int, str]] = []
            timeout_retries_to_log: List[tuple[str, str, Dict[str, Any], int, str]] = []
            with runtime_state.queue_lock:
                expired = [
                    task_id
                    for task_id, info in runtime_state.inflight.items()
                    if now - info["ts"] > config.server.inflight_timeout_sec
                ]
                for task_id in expired:
                    inflight_info = runtime_state.inflight.pop(task_id)
                    job = inflight_info["job"]
                    dispatch_id = str(inflight_info.get("dispatch_id", ""))
                    meta = job_payload_meta(job)
                    runtime_state.timeout_retry_counts[task_id] = runtime_state.timeout_retry_counts.get(task_id, 0) + 1
                    attempt = runtime_state.timeout_retry_counts[task_id]
                    limit = config.server.max_retries_per_job
                    limit_label = "inf" if limit <= 0 else str(limit)
                    if limit <= 0 or attempt <= limit:
                        runtime_state.job_queue.append(job)
                        runtime_state.record_sample_retry(
                            str(meta.get("subset", "")),
                            str(meta.get("sample_id", "")),
                            "timeout_retries",
                        )
                        timeout_retries_to_log.append((task_id, dispatch_id, meta, attempt, limit_label))
                    else:
                        exhausted_timeouts.append((task_id, dispatch_id, meta, attempt, limit_label))

            for task_id, dispatch_id, meta, attempt, limit_label in timeout_retries_to_log:
                runtime_state.persist_retry_evidence(
                    str(meta.get("subset", "")),
                    str(meta.get("sample_id", "")),
                )
                log_event(
                    runtime_state.logger,
                    "result_timeout_retry",
                    task_id=task_id,
                    dispatch_id=dispatch_id,
                    subset=str(meta.get("subset", "")),
                    sample_id=str(meta.get("sample_id", "")),
                    job_type=str(meta.get("job_type", "")),
                    attempt=attempt,
                    retry_limit=limit_label,
                )
                runtime_state.logger.warning(
                    f"[Warn] Task {task_id} timed out, re-queueing to tail "
                    f"(timeout attempt {attempt}/{limit_label})"
                )

            for task_id, dispatch_id, meta, attempt, limit_label in exhausted_timeouts:
                runtime_state.logger.error(
                    f"[Err] Task {task_id} timed out and exhausted retry budget "
                    f"(timeout attempt {attempt}/{limit_label}); recording terminal timeout result"
                )
                runtime_state.mark_task_terminal_failure(
                    task_id,
                    dispatch_id,
                    {**meta, "dispatch_id": dispatch_id},
                    "timeout_retry_exhausted",
                )

            if dataset_idx >= len(dataset_ctxs):
                if config.server.auto_exit_after_all_done:
                    failed_samples = count_failed_samples(states)
                    exit_code = final_exit_code(states)
                    if failed_samples:
                        runtime_state.logger.warning(
                            f"[All Done] {global_done}/{progress_total} succeeded, "
                            f"{failed_samples} failed. Exiting with code {exit_code}."
                        )
                    else:
                        runtime_state.logger.info(f"[All Done] {global_done}/{progress_total}. Exiting.")
                    os._exit(exit_code)
                if stop_event.wait(1.0):
                    break
                continue

            ctx = dataset_ctxs[dataset_idx]
            state = states[ctx.subset]
            cur_idx = state["cur_idx"]
            sample_status = state["sample_status"]
            sample_ids = ctx.sample_ids

            if cur_idx >= len(sample_ids):
                with runtime_state.queue_lock:
                    if not runtime_state.job_queue and not runtime_state.inflight:
                        runtime_state.logger.info(f"[Dataset] Completed {ctx.subset}. Switching to next...")
                        dataset_idx += 1
                if stop_event.wait(0.2):
                    break
                continue

            with runtime_state.queue_lock:
                queue_length = len(runtime_state.job_queue)

            if queue_length < config.server.max_queue:
                sample_id = sample_ids[cur_idx]
                sample_dir = Path(ctx.data_dir) / sample_id

                if Path(runtime_state.done_marker_path(ctx.subset, sample_id)).exists():
                    runtime_state.ensure_sample_runtime_for_terminal_artifact(ctx.subset, sample_id)
                    runtime_state.clear_step_a_repeat_artifact_reuse_cache(ctx.subset, sample_id)
                    sample_status[sample_id] = 3
                    state["cur_idx"] += 1
                    time.sleep(0.01)
                    continue

                if Path(runtime_state.failed_marker_path(ctx.subset, sample_id)).exists():
                    runtime_state.ensure_sample_runtime_for_terminal_artifact(ctx.subset, sample_id)
                    runtime_state.clear_step_a_repeat_artifact_reuse_cache(ctx.subset, sample_id)
                    sample_status[sample_id] = 4
                    state["cur_idx"] += 1
                    time.sleep(0.01)
                    continue

                mp4s = list(sample_dir.glob("Frame_*.mp4"))
                if not mp4s:
                    runtime_state.fail_sample(
                        state,
                        ctx.subset,
                        sample_id,
                        "missing_input_video",
                        {
                            "sample_dir": str(sample_dir),
                            "expected_glob": "Frame_*.mp4",
                        },
                        log_message=f"[Fail] {ctx.subset}/{sample_id}: missing Frame_*.mp4 input",
                    )
                    time.sleep(0.01)
                    continue
                mp4 = str(mp4s[0])

                if sample_status[sample_id] == 0:
                    try:
                        fps, nframes = deps.read_video_info(mp4)
                        windows = deps.build_windows(
                            fps,
                            nframes,
                            config.windowing.window_sec,
                            config.windowing.step_sec,
                            config.windowing.frames_per_window,
                        )

                        window_results, failed_window_results = runtime_state.load_window_results(ctx.subset, sample_id)
                        if failed_window_results:
                            runtime_state.fail_sample(
                                state,
                                ctx.subset,
                                sample_id,
                                "window_boundary_failed",
                                {
                                    "failed_window_ids": sorted(int(window_id) for window_id in failed_window_results),
                                    "errors": {
                                        str(window_id): str(record.get("terminal_error", "unknown"))
                                        for window_id, record in sorted(failed_window_results.items())
                                    },
                                },
                                log_message=f"[Fail] {ctx.subset}/{sample_id}: terminal window failure before finalize",
                            )
                            time.sleep(0.01)
                            continue

                        done_window_ids = runtime_state.completed_window_ids(window_results)
                        repeat_target = max(1, int(config.windowing.window_repeat_count))
                        repeat_artifact_reuse_cache = runtime_state.step_a_repeat_artifact_reuse_caches.setdefault(
                            (ctx.subset, sample_id),
                            {},
                        )

                        step_a_batch_limit = runtime_state.current_step_a_producer_batch_limit()
                        with deps.frame_extractor_cls(mp4, artifact_writer=runtime_state.task_artifact_writer) as extractor:
                            produced_count = 0

                            for window in windows:
                                if window.window_id in done_window_ids:
                                    continue
                                success_indices = set(
                                    int(item)
                                    for item in window_results.get(window.window_id, {}).get("repeat_indices", [])
                                )

                                for repeat_index in range(repeat_target):
                                    if repeat_index in success_indices:
                                        continue

                                    task_id = f"{ctx.subset}::{sample_id}_w{window.window_id}_r{repeat_index}"

                                    active = False
                                    with runtime_state.queue_lock:
                                        if job_queue_contains_task_id(runtime_state.job_queue, task_id) or task_id in runtime_state.inflight:
                                            active = True

                                    if active:
                                        continue

                                    job = runtime_state.job_builder.build_window_boundary_job(
                                        extractor,
                                        task_id=task_id,
                                        subset=ctx.subset,
                                        sample_id=sample_id,
                                        window=window,
                                        fps=fps,
                                        nframes=nframes,
                                        repeat_index=repeat_index,
                                        repeat_count=repeat_target,
                                        reuse_cache=repeat_artifact_reuse_cache,
                                    )

                                    with runtime_state.queue_lock:
                                        runtime_state.job_queue.append(job)

                                    produced_count += 1
                                    if produced_count > step_a_batch_limit:
                                        break

                                if produced_count > step_a_batch_limit:
                                    break

                        if produced_count == 0 and runtime_state.all_windows_completed(windows, window_results):
                            runtime_state.clear_step_a_repeat_artifact_reuse_cache(ctx.subset, sample_id)
                            sample_status[sample_id] = 2

                    except ArtifactPayloadValidationError as exc:
                        _fail_sample_for_invalid_artifacts(
                            runtime_state,
                            state,
                            ctx.subset,
                            sample_id,
                            reason="artifact_extraction_failed",
                            phase="step_a",
                            error=exc,
                            log_message=f"[Fail] {ctx.subset}/{sample_id}: invalid extraction artifacts blocked dispatch",
                        )
                    except Exception as exc:
                        runtime_state.logger.exception(f"[Err] {ctx.subset}/{sample_id}: {exc}")
                        runtime_state.fail_sample(
                            state,
                            ctx.subset,
                            sample_id,
                            "step_a_exception",
                            {
                                "phase": "step_a",
                                "error": str(exc).strip() or type(exc).__name__,
                                "exception_type": type(exc).__name__,
                            },
                            log_message=f"[Fail] {ctx.subset}/{sample_id}: Step A crashed",
                        )

                if sample_status[sample_id] == 2:
                    try:
                        finalize_start = time.perf_counter()
                        fps, nframes = deps.read_video_info(mp4)
                        windows = deps.build_windows(
                            fps,
                            nframes,
                            config.windowing.window_sec,
                            config.windowing.step_sec,
                            config.windowing.frames_per_window,
                        )

                        by_window_id, failed_window_results = runtime_state.load_window_results(ctx.subset, sample_id)
                        if failed_window_results:
                            runtime_state.fail_sample(
                                state,
                                ctx.subset,
                                sample_id,
                                "window_boundary_failed",
                                {
                                    "failed_window_ids": sorted(int(window_id) for window_id in failed_window_results),
                                    "errors": {
                                        str(window_id): str(record.get("terminal_error", "unknown"))
                                        for window_id, record in sorted(failed_window_results.items())
                                    },
                                },
                                log_message=f"[Fail] {ctx.subset}/{sample_id}: terminal window failure blocks finalize",
                            )
                            time.sleep(0.01)
                            continue

                        if runtime_state.all_windows_completed(windows, by_window_id):
                            refinement_windows: List[Any] = []
                            if config.windowing.enable_refinement_pass:
                                refinement_frames = (
                                    config.windowing.refinement_frames_per_window
                                    or config.windowing.frames_per_window
                                )
                                refinement_windows = deps.build_refinement_windows(
                                    windows,
                                    by_window_id,
                                    fps,
                                    nframes,
                                    refinement_frames,
                                )
                                if refinement_windows:
                                    completed_refinement = runtime_state.completed_window_ids(by_window_id)
                                    missing_refinement = [
                                        refinement_window
                                        for refinement_window in refinement_windows
                                        if refinement_window.window_id not in completed_refinement
                                    ]
                                    if missing_refinement:
                                        with deps.frame_extractor_cls(mp4, artifact_writer=runtime_state.task_artifact_writer) as extractor:
                                            produced_count = 0

                                            for refinement_window in missing_refinement:
                                                existing_success_indices = set(
                                                    int(item)
                                                    for item in by_window_id.get(refinement_window.window_id, {}).get("repeat_indices", [])
                                                )

                                                for repeat_index in range(max(1, int(config.windowing.window_repeat_count))):
                                                    if repeat_index in existing_success_indices:
                                                        continue

                                                    task_id = f"{ctx.subset}::{sample_id}_rw{refinement_window.window_id}_r{repeat_index}"
                                                    active = False
                                                    with runtime_state.queue_lock:
                                                        if job_queue_contains_task_id(runtime_state.job_queue, task_id) or task_id in runtime_state.inflight:
                                                            active = True

                                                    if active:
                                                        continue

                                                    try:
                                                        job = runtime_state.job_builder.build_window_boundary_job(
                                                            extractor,
                                                            task_id=task_id,
                                                            subset=ctx.subset,
                                                            sample_id=sample_id,
                                                            window=refinement_window,
                                                            fps=fps,
                                                            nframes=nframes,
                                                            repeat_index=repeat_index,
                                                            repeat_count=max(1, int(config.windowing.window_repeat_count)),
                                                            window_pass="refinement",
                                                        )
                                                    except ArtifactPayloadValidationError as exc:
                                                        raise _SampleArtifactDispatchFailure(
                                                            reason="artifact_preparation_failed",
                                                            phase="refinement_dispatch",
                                                            error=exc,
                                                            details={"task_id": task_id, "job_type": "window_boundary"},
                                                        ) from exc

                                                    with runtime_state.queue_lock:
                                                        runtime_state.job_queue.append(job)

                                                    produced_count += 1
                                                    if produced_count > 20:
                                                        break

                                                if produced_count > 20:
                                                    break

                                        if produced_count > 0:
                                            time.sleep(0.05)
                                            continue

                                    if not runtime_state.all_windows_completed(windows + refinement_windows, by_window_id):
                                        time.sleep(0.05)
                                        continue

                            required_stages = runtime_state.required_stages_for_subset(ctx.subset)
                            completed_stages: List[str] = []

                            runtime_state.logger.info(f"[Finalize] {ctx.subset}/{sample_id}...")

                            provisional_res = deps.build_segments_via_cuts(
                                sample_id,
                                windows + refinement_windows,
                                by_window_id,
                                fps,
                                nframes,
                                config.windowing.frames_per_window,
                                boundary_prompt_mode=config.windowing.boundary_prompt_mode,
                                adaptive_merge_guard=config.windowing.adaptive_merge_guard,
                                adaptive_merge_min_segments=config.windowing.adaptive_merge_min_segments,
                                adaptive_merge_collapse_ratio=config.windowing.adaptive_merge_collapse_ratio,
                                boundary_support_threshold=config.windowing.boundary_support_threshold,
                                refine_final_instructions=config.windowing.refine_final_instructions,
                            )
                            if not provisional_res.get("segments"):
                                runtime_state.fail_sample(
                                    state,
                                    ctx.subset,
                                    sample_id,
                                    "finalize_empty_segments",
                                    {
                                        "phase": "cuts",
                                        "window_count": len(windows),
                                        "completed_window_count": len(by_window_id),
                                    },
                                    log_message=f"[Fail] {ctx.subset}/{sample_id}: finalize produced no segments",
                                )
                                time.sleep(0.01)
                                continue

                            final_res = provisional_res
                            if config.windowing.enable_boundary_refinement:
                                boundary_results, boundary_failures = runtime_state.load_boundary_refinement_results(ctx.subset, sample_id)
                                if boundary_failures:
                                    runtime_state.fail_sample(
                                        state,
                                        ctx.subset,
                                        sample_id,
                                        "boundary_refinement_failed",
                                        {
                                            "failed_boundary_ids": [int(boundary_id) for boundary_id in sorted(boundary_failures)],
                                            "errors": {
                                                str(boundary_id): str(record.get("terminal_error", "unknown"))
                                                for boundary_id, record in sorted(boundary_failures.items())
                                            },
                                        },
                                        log_message=f"[Fail] {ctx.subset}/{sample_id}: terminal boundary refinement failure blocks finalize",
                                    )
                                    time.sleep(0.01)
                                    continue
                                boundary_refinement_frames = (
                                    config.windowing.boundary_refinement_frames_per_window
                                    or config.windowing.frames_per_window
                                )
                                boundary_refinement_windows = deps.build_boundary_refinement_windows(
                                    provisional_res.get("segments", []),
                                    fps,
                                    nframes,
                                    config.windowing.boundary_refinement_window_sec,
                                    boundary_refinement_frames,
                                )
                                missing_boundaries = [
                                    boundary_window
                                    for boundary_window in boundary_refinement_windows
                                    if boundary_window.boundary_id not in boundary_results
                                    and boundary_window.boundary_id not in boundary_failures
                                ]

                                if missing_boundaries:
                                    with deps.frame_extractor_cls(mp4, artifact_writer=runtime_state.task_artifact_writer) as extractor:
                                        produced_count = 0
                                        for boundary_window in missing_boundaries:
                                            boundary_id = int(boundary_window.boundary_id)
                                            task_id = f"{ctx.subset}::{sample_id}_b{boundary_id}"
                                            active = False
                                            with runtime_state.queue_lock:
                                                if job_queue_contains_task_id(runtime_state.job_queue, task_id) or task_id in runtime_state.inflight:
                                                    active = True

                                            if active:
                                                continue

                                            try:
                                                job = runtime_state.job_builder.build_boundary_refinement_job(
                                                    extractor,
                                                    task_id=task_id,
                                                    subset=ctx.subset,
                                                    sample_id=sample_id,
                                                    boundary_window=boundary_window,
                                                )
                                            except ArtifactPayloadValidationError as exc:
                                                raise _SampleArtifactDispatchFailure(
                                                    reason="artifact_preparation_failed",
                                                    phase="boundary_refinement_dispatch",
                                                    error=exc,
                                                    details={"task_id": task_id, "job_type": "boundary_refinement"},
                                                ) from exc

                                            with runtime_state.queue_lock:
                                                runtime_state.job_queue.append(job)

                                            produced_count += 1
                                            if produced_count > 20:
                                                break

                                    if produced_count > 0:
                                        time.sleep(0.05)
                                        continue

                                    time.sleep(0.05)
                                    continue

                                final_res = dict(provisional_res)
                                final_res["segments"] = deps.apply_boundary_refinement_results(
                                    provisional_res.get("segments", []),
                                    boundary_results,
                                    fps=fps,
                                    abstain_merge_max_support=config.windowing.boundary_refinement_abstain_merge_max_support,
                                )
                                diagnostics = dict(final_res.get("diagnostics", {}))
                                diagnostics["boundary_refinement_enabled"] = True
                                diagnostics["boundary_refinement_count"] = len(boundary_results)
                                diagnostics["boundary_refinement_failed_count"] = len(boundary_failures)
                                diagnostics["boundary_refinement_failed_ids"] = [
                                    int(boundary_id) for boundary_id in sorted(boundary_failures)
                                ]
                                final_res["diagnostics"] = diagnostics

                            if config.windowing.segment_labeling_mode == "deferred":
                                label_results, label_failures = runtime_state.load_segment_label_results(ctx.subset, sample_id)
                                if label_failures:
                                    runtime_state.fail_sample(
                                        state,
                                        ctx.subset,
                                        sample_id,
                                        "segment_label_failed",
                                        {
                                            "failed_segment_ids": [int(segment_id) for segment_id in sorted(label_failures)],
                                            "errors": {
                                                str(segment_id): str(record.get("terminal_error", "unknown"))
                                                for segment_id, record in sorted(label_failures.items())
                                            },
                                        },
                                        log_message=f"[Fail] {ctx.subset}/{sample_id}: terminal deferred labeling failure blocks finalize",
                                    )
                                    time.sleep(0.01)
                                    continue
                                missing_segments = [
                                    segment
                                    for segment in final_res.get("segments", [])
                                    if int(segment.get("seg_id", -1)) not in label_results
                                    and int(segment.get("seg_id", -1)) not in label_failures
                                ]

                                if missing_segments:
                                    with deps.frame_extractor_cls(mp4, artifact_writer=runtime_state.task_artifact_writer) as extractor:
                                        produced_count = 0
                                        for segment in missing_segments:
                                            seg_id = int(segment.get("seg_id", -1))
                                            task_id = f"{ctx.subset}::{sample_id}_seg{seg_id}"
                                            active = False
                                            with runtime_state.queue_lock:
                                                if job_queue_contains_task_id(runtime_state.job_queue, task_id) or task_id in runtime_state.inflight:
                                                    active = True

                                            if active:
                                                continue

                                            frame_ids = deps.sample_segment_frame_ids(
                                                int(segment["start_frame"]),
                                                int(segment["end_frame"]),
                                                config.windowing.frames_per_window,
                                                nframes,
                                            )
                                            try:
                                                job = runtime_state.job_builder.build_segment_label_job(
                                                    extractor,
                                                    task_id=task_id,
                                                    subset=ctx.subset,
                                                    sample_id=sample_id,
                                                    segment=segment,
                                                    frame_ids=frame_ids,
                                                )
                                            except ArtifactPayloadValidationError as exc:
                                                raise _SampleArtifactDispatchFailure(
                                                    reason="artifact_preparation_failed",
                                                    phase="segment_label_dispatch",
                                                    error=exc,
                                                    details={"task_id": task_id, "job_type": "segment_label"},
                                                ) from exc

                                            with runtime_state.queue_lock:
                                                runtime_state.job_queue.append(job)

                                            produced_count += 1
                                            if produced_count > 20:
                                                break

                                    if produced_count > 0:
                                        time.sleep(0.05)
                                        continue

                                    time.sleep(0.05)
                                    continue

                                final_res = dict(final_res)
                                final_res["segments"] = deps.apply_deferred_segment_labels(
                                    final_res.get("segments", []),
                                    label_results,
                                )
                                diagnostics = dict(final_res.get("diagnostics", {}))
                                diagnostics["segment_labeling_mode"] = "deferred"
                                diagnostics["segment_label_count"] = len(label_results)
                                diagnostics["segment_label_failed_count"] = len(label_failures)
                                diagnostics["segment_label_failed_ids"] = [
                                    int(segment_id) for segment_id in sorted(label_failures)
                                ]
                                final_res["diagnostics"] = diagnostics

                            completed_stages.append("stage1_segments")
                            early_done = runtime_state.required_stages_satisfied(required_stages, completed_stages)
                            if early_done:
                                global_done = runtime_state.mark_sample_done(
                                    state,
                                    ctx.subset,
                                    sample_id,
                                    final_res,
                                    required_stages=required_stages,
                                    completed_stages=completed_stages,
                                    finalize_start=finalize_start,
                                    global_done=global_done,
                                    progress_total=progress_total,
                                )

                            stage2_writeback_required = bool(config.llm_merge.enabled) and (
                                early_done or "stage2_text" in required_stages
                            )
                            if stage2_writeback_required:
                                try:
                                    stage2_res = runtime_state.apply_stage2_text_writeback(sample_id, final_res)
                                except Exception as exc:
                                    if early_done:
                                        runtime_state.logger.warning(
                                            f"[Warn] {ctx.subset}/{sample_id}: optional Stage 2 writeback skipped after terminal success: {exc}"
                                        )
                                        continue
                                    raise

                                if not stage2_res.get("segments"):
                                    if early_done:
                                        runtime_state.logger.warning(
                                            f"[Warn] {ctx.subset}/{sample_id}: optional Stage 2 writeback produced no segments after terminal success"
                                        )
                                        continue
                                    runtime_state.fail_sample(
                                        state,
                                        ctx.subset,
                                        sample_id,
                                        "finalize_empty_segments",
                                        {
                                            "phase": "postprocess",
                                            "window_count": len(windows),
                                            "completed_window_count": len(by_window_id),
                                        },
                                        log_message=f"[Fail] {ctx.subset}/{sample_id}: finalize produced no segments after postprocess",
                                    )
                                    time.sleep(0.01)
                                    continue

                                final_res = stage2_res
                                if early_done:
                                    runtime_state.persist_sample_writeback(
                                        ctx.subset,
                                        sample_id,
                                        final_res,
                                        required_stages=required_stages,
                                        completed_stages=completed_stages,
                                    )
                                    continue

                                if "stage2_text" in required_stages:
                                    completed_stages.append("stage2_text")
                                    if runtime_state.required_stages_satisfied(required_stages, completed_stages):
                                        global_done = runtime_state.mark_sample_done(
                                            state,
                                            ctx.subset,
                                            sample_id,
                                            final_res,
                                            required_stages=required_stages,
                                            completed_stages=completed_stages,
                                            finalize_start=finalize_start,
                                            global_done=global_done,
                                            progress_total=progress_total,
                                        )
                                        continue

                            if "export" in required_stages:
                                export_segments = [
                                    dict(segment)
                                    for segment in final_res.get("segments", [])
                                    if isinstance(segment, dict)
                                ]

                                try:
                                    export_diagnostics = deps.export_sample_outputs(
                                        run_dir=ctx.run_dir,
                                        sample_id=sample_id,
                                        video_path=mp4,
                                        fps=fps,
                                        segments=export_segments,
                                        export_config=config.export,
                                    )
                                except Exception as exc:
                                    export_diagnostics = {
                                        "export_enabled": bool(config.export.enabled),
                                        "export_attempted": bool(config.export.enabled),
                                        "export_mode": str(config.export.mode),
                                        "export_reason": "failed_before_export_completion",
                                        "export_error": str(exc).strip() or type(exc).__name__,
                                    }

                                diagnostics = dict(final_res.get("diagnostics", {}))
                                diagnostics.update(export_diagnostics)
                                final_res["diagnostics"] = diagnostics
                                if not runtime_state.export_stage_succeeded(export_diagnostics):
                                    runtime_state.fail_sample(
                                        state,
                                        ctx.subset,
                                        sample_id,
                                        "export_failed",
                                        {
                                            "stage": "export",
                                            "required_stages": list(required_stages),
                                            "completed_stages": list(completed_stages),
                                            "diagnostics": dict(diagnostics),
                                            "export_enabled": bool(export_diagnostics.get("export_enabled", bool(config.export.enabled))),
                                            "export_attempted": bool(export_diagnostics.get("export_attempted", bool(config.export.enabled))),
                                            "export_mode": str(export_diagnostics.get("export_mode", str(config.export.mode))).strip(),
                                            "export_reason": str(export_diagnostics.get("export_reason", "")).strip(),
                                            "export_errors": list(export_diagnostics.get("export_errors", [])),
                                            "export_error": str(export_diagnostics.get("export_error", "")).strip(),
                                        },
                                        log_message=f"[Fail] {ctx.subset}/{sample_id}: required export stage did not complete",
                                    )
                                    time.sleep(0.01)
                                    continue
                                completed_stages.append("export")
                                global_done = runtime_state.mark_sample_done(
                                    state,
                                    ctx.subset,
                                    sample_id,
                                    final_res,
                                    required_stages=required_stages,
                                    completed_stages=completed_stages,
                                    finalize_start=finalize_start,
                                    global_done=global_done,
                                    progress_total=progress_total,
                                )
                                continue

                            if early_done:
                                continue

                            global_done = runtime_state.mark_sample_done(
                                state,
                                ctx.subset,
                                sample_id,
                                final_res,
                                required_stages=required_stages,
                                completed_stages=completed_stages,
                                finalize_start=finalize_start,
                                global_done=global_done,
                                progress_total=progress_total,
                            )

                    except _SampleArtifactDispatchFailure as exc:
                        _fail_sample_for_invalid_artifacts(
                            runtime_state,
                            state,
                            ctx.subset,
                            sample_id,
                            reason=exc.reason,
                            phase=exc.phase,
                            error=exc.error,
                            log_message=f"[Fail] {ctx.subset}/{sample_id}: invalid preparation artifacts blocked dispatch",
                            details=exc.details,
                        )
                    except Exception as exc:
                        runtime_state.logger.error(f"[Err-Finalize] {ctx.subset}/{sample_id}: {exc}")
                        runtime_state.fail_sample(
                            state,
                            ctx.subset,
                            sample_id,
                            "finalize_exception",
                            {
                                "phase": "finalize",
                                "error": str(exc).strip() or type(exc).__name__,
                                "exception_type": type(exc).__name__,
                            },
                            log_message=f"[Fail] {ctx.subset}/{sample_id}: finalize crashed",
                        )

            if stop_event.wait(0.1):
                break

    return producer_loop
