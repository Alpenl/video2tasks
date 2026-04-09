"""FastAPI server for job queue management."""

import os
import json
import time
import glob
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
import uvicorn

from ..config import Config, DatasetConfig
from ..logging_utils import configure_logging, get_logger, log_event
from ..prompt import boundary_refinement_candidate_positions
from .windowing import (
    read_video_info, build_windows, FrameExtractor,
    apply_boundary_refinement_results, apply_deferred_segment_labels,
    build_boundary_refinement_windows, build_segments_via_cuts,
    build_refinement_windows, Window,
    sample_segment_frame_ids,
)
from .exporter import export_sample_outputs
from .llm_merge import attach_stage2_subtitles_to_segments, run_llm_stage2_pass
from .job_builder import ArtifactReuseEntry, JobBuilder
from .run_manifest import ensure_run_manifest, load_run_manifest, run_manifest_path
from .run_summary import build_run_summary, build_sample_runtime_record, write_run_summary
from .protocol import (
    JobEnvelope,
    ProtocolValidationError,
    ResultEnvelope,
)
from .runtime import ThreadRuntime
from .sample_store import SampleStore
from .task_artifacts import (
    ArtifactPayloadValidationError,
    TaskArtifactWriter,
    artifact_validation_error_details,
)
from ..vlm.base import normalize_task_window_result


logger = get_logger(__name__)

_STEP_A_PRODUCER_BATCH_LIMIT = 20


@dataclass
class DatasetCtx:
    """Dataset context for processing."""
    data_root: str
    subset: str
    data_dir: str
    run_dir: str
    run_dir_nonempty_before_prepare: bool
    sample_ids: List[str]


@dataclass
class _SampleArtifactDispatchFailure(Exception):
    reason: str
    phase: str
    error: ArtifactPayloadValidationError
    details: Dict[str, Any]


def parse_datasets(config: Config) -> List[DatasetCtx]:
    """Parse dataset configurations into contexts."""
    ctxs = []
    for ds in config.datasets:
        data_dir = Path(ds.root) / ds.subset
        run_dir = Path(config.run.base_dir) / ds.subset / config.run.run_id
        run_dir_nonempty_before_prepare = run_dir.exists() and any(run_dir.iterdir())
        samples_dir = run_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # List sample IDs
        if data_dir.exists():
            sample_ids = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
        else:
            sample_ids = []
        
        ctxs.append(DatasetCtx(
            data_root=ds.root,
            subset=ds.subset,
            data_dir=str(data_dir),
            run_dir=str(run_dir),
            run_dir_nonempty_before_prepare=run_dir_nonempty_before_prepare,
            sample_ids=sample_ids
        ))
    return ctxs


def _requeue_empty_result(
    job_queue: List[Any],
    retry_counts: Dict[str, int],
    task_id: str,
    job: Optional[Any],
    max_retries_per_job: int = 0,
) -> tuple[int, bool]:
    if not job:
        return 0, False

    retry_counts[task_id] = retry_counts.get(task_id, 0) + 1
    attempt = retry_counts[task_id]
    if max_retries_per_job > 0 and attempt > max_retries_per_job:
        return attempt, False

    job_queue.append(job)
    return attempt, True


def _job_payload_task_id(job: Any) -> str:
    if isinstance(job, JobEnvelope):
        return job.task_id
    if isinstance(job, dict):
        return str(job.get("task_id", "")).strip()
    return ""



def _job_payload_meta(job: Any) -> Dict[str, Any]:
    if isinstance(job, JobEnvelope):
        return dict(job.meta)
    if isinstance(job, dict):
        raw_meta = job.get("meta", {})
        if isinstance(raw_meta, dict):
            return dict(raw_meta)
    return {}



def _coerce_job_envelope(job: Any) -> JobEnvelope:
    if isinstance(job, JobEnvelope):
        return job
    return JobEnvelope.parse_payload(job)



def _job_queue_contains_task_id(job_queue: List[Any], task_id: str) -> bool:
    return any(_job_payload_task_id(job) == task_id for job in job_queue)



def _logical_frame_count_from_meta(meta: Dict[str, Any]) -> Optional[int]:
    if not isinstance(meta, dict):
        return None

    try:
        logical_frame_count = int(meta.get("logical_frame_count") or 0)
    except (TypeError, ValueError):
        logical_frame_count = 0
    if logical_frame_count > 0:
        return logical_frame_count

    frame_ids = meta.get("frame_ids", [])
    if isinstance(frame_ids, list) and frame_ids:
        return len(frame_ids)
    return None


def _merge_result_meta(authoritative_meta: Dict[str, Any], worker_meta: Dict[str, Any], dispatch_id: str) -> Dict[str, Any]:
    merged = dict(authoritative_meta)
    for key, value in worker_meta.items():
        if key not in merged:
            merged[key] = value
    merged["dispatch_id"] = dispatch_id
    return merged


def _normalize_submitted_vlm_json(vlm_json: Dict[str, Any], authoritative_meta: Dict[str, Any]) -> Dict[str, Any]:
    logical_frame_count = _logical_frame_count_from_meta(authoritative_meta)
    max_transition_index = (logical_frame_count - 1) if logical_frame_count and logical_frame_count > 0 else None

    allowed_transition_indices = None
    if str(authoritative_meta.get("job_type", "window_boundary")) == "boundary_refinement":
        if logical_frame_count and logical_frame_count > 0:
            allowed_transition_indices = boundary_refinement_candidate_positions(logical_frame_count)

    return normalize_task_window_result(
        vlm_json,
        max_transition_index=max_transition_index,
        allowed_transition_indices=allowed_transition_indices,
    )



def _logical_frame_count_from_record(record: Dict[str, Any]) -> Optional[int]:
    if not isinstance(record, dict):
        return None

    logical_frame_count = _logical_frame_count_from_meta(record)
    if logical_frame_count and logical_frame_count > 0:
        return logical_frame_count

    return _logical_frame_count_from_meta(record.get("meta", {}))



def _normalize_loaded_window_vlm_json(record: Dict[str, Any]) -> Dict[str, Any]:
    logical_frame_count = _logical_frame_count_from_record(record)
    max_transition_index = (logical_frame_count - 1) if logical_frame_count and logical_frame_count > 0 else None
    return normalize_task_window_result(
        record.get("vlm_json", {}),
        max_transition_index=max_transition_index,
    )


def _normalize_loaded_boundary_refinement_vlm_json(record: Dict[str, Any]) -> Dict[str, Any]:
    frame_ids = record.get("frame_ids", [])
    if isinstance(frame_ids, list) and frame_ids:
        logical_frame_count = len(frame_ids)
    else:
        logical_frame_count = _logical_frame_count_from_record(record)

    max_transition_index = (logical_frame_count - 1) if logical_frame_count and logical_frame_count > 0 else None
    allowed_transition_indices = (
        boundary_refinement_candidate_positions(logical_frame_count)
        if logical_frame_count and logical_frame_count > 0
        else None
    )
    return normalize_task_window_result(
        record.get("vlm_json", {}),
        max_transition_index=max_transition_index,
        allowed_transition_indices=allowed_transition_indices,
    )


def _count_failed_samples(states: Dict[str, Dict[str, Any]]) -> int:
    return sum(
        1
        for state in states.values()
        for status in dict(state.get("sample_status", {})).values()
        if int(status) == 4
    )


def _final_exit_code(states: Dict[str, Dict[str, Any]]) -> int:
    return 1 if _count_failed_samples(states) > 0 else 0


def create_app(config: Config) -> FastAPI:
    """Create and configure FastAPI application."""
    configure_logging(config.logging.level)
    app = FastAPI(title="Video2Tasks Server")
    
    # Initialize dataset contexts
    dataset_ctxs = parse_datasets(config)
    data_dir_by_subset = {ctx.subset: ctx.data_dir for ctx in dataset_ctxs}
    run_dir_by_subset = {ctx.subset: ctx.run_dir for ctx in dataset_ctxs}
    sample_ids_by_subset = {ctx.subset: list(ctx.sample_ids) for ctx in dataset_ctxs}
    
    # Thread-safe job management
    queue_lock = threading.Lock()
    job_queue: List[Any] = []
    inflight: Dict[str, Dict[str, Any]] = {}
    timeout_retry_counts: Dict[str, int] = {}
    empty_retry_counts: Dict[str, int] = {}
    dispatch_counts: Dict[str, int] = {}
    sample_retry_stats: Dict[Tuple[str, str], Dict[str, int]] = {}
    completed_dispatch_ids: Dict[str, str] = {}
    step_a_repeat_artifact_reuse_caches: Dict[
        Tuple[str, str],
        Dict[Tuple[Any, ...], ArtifactReuseEntry],
    ] = {}
    artifact_root_dir = os.getenv("VIDEO2TASKS_TMP_DIR", "tmp").strip() or "tmp"
    task_artifact_writer = TaskArtifactWriter(root_dir=artifact_root_dir)
    sample_store = SampleStore(
        base_dir=config.run.base_dir,
        run_id=config.run.run_id,
        initial_samples_dir_by_subset={
            ctx.subset: str(Path(ctx.run_dir) / "samples")
            for ctx in dataset_ctxs
        },
        default_subset=dataset_ctxs[0].subset if dataset_ctxs else "default",
        window_repeat_count=config.windowing.window_repeat_count,
        normalize_window_record=_normalize_loaded_window_vlm_json,
        normalize_boundary_refinement_record=_normalize_loaded_boundary_refinement_vlm_json,
        normalize_segment_label_payload=normalize_task_window_result,
    )
    job_builder = JobBuilder(
        target_width=config.windowing.target_width,
        target_height=config.windowing.target_height,
        png_compression=config.windowing.png_compression,
        use_contact_sheets=config.windowing.use_contact_sheets,
        contact_sheet_rows=config.windowing.contact_sheet_rows,
        contact_sheet_cols=config.windowing.contact_sheet_cols,
    )
    run_manifest_paths: Dict[str, str] = {}
    run_manifest_status_by_subset: Dict[str, Dict[str, Any]] = {}

    for ctx in dataset_ctxs:
        manifest_status = ensure_run_manifest(
            run_dir=ctx.run_dir,
            subset=ctx.subset,
            data_root=ctx.data_root,
            config=config,
            force_resume=bool(config.run.force_resume),
            run_dir_nonempty_before_start=bool(ctx.run_dir_nonempty_before_prepare),
        )
        run_manifest_paths[ctx.subset] = str(run_manifest_path(ctx.run_dir))
        run_manifest_status_by_subset[ctx.subset] = manifest_status.model_dump(mode="json")
        if manifest_status.resume.force_resume and manifest_status.resume.mismatch_fields:
            logger.warning(
                f"[Resume-Override] subset={ctx.subset} action={manifest_status.action} "
                f"mismatches={manifest_status.resume.mismatch_fields}"
            )
    
    def sample_out_dir(subset: str, sample_id: str) -> str:
        return sample_store.sample_out_dir(subset, sample_id)

    def windows_jsonl_path(subset: str, sample_id: str) -> str:
        return sample_store.windows_jsonl_path(subset, sample_id)

    def segments_path(subset: str, sample_id: str) -> str:
        return sample_store.segments_path(subset, sample_id)

    def segment_labels_jsonl_path(subset: str, sample_id: str) -> str:
        return sample_store.segment_labels_jsonl_path(subset, sample_id)

    def boundary_refinements_jsonl_path(subset: str, sample_id: str) -> str:
        return sample_store.boundary_refinements_jsonl_path(subset, sample_id)

    def done_marker_path(subset: str, sample_id: str) -> str:
        return sample_store.done_marker_path(subset, sample_id)

    def failed_marker_path(subset: str, sample_id: str) -> str:
        return sample_store.failed_marker_path(subset, sample_id)

    def failure_report_path(subset: str, sample_id: str) -> str:
        return sample_store.failure_report_path(subset, sample_id)

    def load_window_results(subset: str, sample_id: str) -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        return sample_store.load_window_results(subset, sample_id)

    def _completed_window_ids(results: Dict[int, Dict[str, Any]]) -> set[int]:
        completed: set[int] = set()
        repeat_target = max(1, int(config.windowing.window_repeat_count))
        for window_id, record in results.items():
            if int(record.get("repeat_success_count", 1)) >= repeat_target:
                completed.add(int(window_id))
        return completed

    def _all_windows_completed(windows: List[Window], results: Dict[int, Dict[str, Any]]) -> bool:
        completed = _completed_window_ids(results)
        return all(int(window.window_id) in completed for window in windows)

    def load_segment_label_results(subset: str, sample_id: str) -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        return sample_store.load_segment_label_results(subset, sample_id)

    def load_boundary_refinement_results(subset: str, sample_id: str) -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        return sample_store.load_boundary_refinement_results(subset, sample_id)

    def _resolve_samples_dir(subset: str) -> str:
        return sample_store.resolve_samples_dir(subset)

    def _persist_result_record(
        task_id: str,
        dispatch_id: str,
        vlm_json: Dict[str, Any],
        meta: Dict[str, Any],
        terminal_error: str = "",
    ) -> None:
        sample_store.persist_result_record(
            task_id,
            dispatch_id,
            vlm_json,
            meta,
            terminal_error=terminal_error,
        )

    def _sample_retry_key(subset: str, sample_id: str) -> Tuple[str, str]:
        return (str(subset), str(sample_id))

    def _persisted_retry_summary(subset: str, sample_id: str) -> Dict[str, int]:
        payload = sample_store.load_sample_runtime(subset, sample_id) or {}
        retry = payload.get("retry", {}) if isinstance(payload, dict) else {}
        if not isinstance(retry, dict):
            retry = {}
        return {
            "total_retries": int(retry.get("total_retries", 0) or 0),
            "empty_result_retries": int(retry.get("empty_result_retries", 0) or 0),
            "timeout_retries": int(retry.get("timeout_retries", 0) or 0),
            "dispatch_count": int(retry.get("dispatch_count", 0) or 0),
        }

    def _record_sample_retry(subset: str, sample_id: str, retry_field: str) -> None:
        key = _sample_retry_key(subset, sample_id)
        stats = sample_retry_stats.setdefault(key, _persisted_retry_summary(subset, sample_id))
        stats[retry_field] = int(stats.get(retry_field, 0)) + 1
        stats["total_retries"] = int(stats.get("empty_result_retries", 0)) + int(stats.get("timeout_retries", 0))

    def _sample_dispatch_count(subset: str, sample_id: str) -> int:
        prefix = f"{subset}::{sample_id}_"
        with queue_lock:
            return sum(
                int(count)
                for task_id, count in dispatch_counts.items()
                if str(task_id).startswith(prefix)
            )

    def _sample_retry_summary(subset: str, sample_id: str) -> Dict[str, int]:
        persisted = _persisted_retry_summary(subset, sample_id)
        stats = dict(persisted)
        stats.update(sample_retry_stats.get(_sample_retry_key(subset, sample_id), {}))
        total_retries = max(int(stats.get("total_retries", 0) or 0), int(persisted.get("total_retries", 0) or 0))
        empty_result_retries = max(int(stats.get("empty_result_retries", 0) or 0), int(persisted.get("empty_result_retries", 0) or 0))
        timeout_retries = max(int(stats.get("timeout_retries", 0) or 0), int(persisted.get("timeout_retries", 0) or 0))
        dispatch_count = max(
            _sample_dispatch_count(subset, sample_id),
            int(stats.get("dispatch_count", 0) or 0),
            int(persisted.get("dispatch_count", 0) or 0),
        )
        if dispatch_count <= 0 and total_retries > 0:
            dispatch_count = total_retries + 1
        return {
            "total_retries": total_retries,
            "empty_result_retries": empty_result_retries,
            "timeout_retries": timeout_retries,
            "dispatch_count": dispatch_count,
        }

    def _persist_retry_evidence(subset: str, sample_id: str) -> None:
        existing = sample_store.load_sample_runtime(subset, sample_id) or {}
        stages = existing.get("stages", {}) if isinstance(existing, dict) else {}
        failure = existing.get("failure", {}) if isinstance(existing, dict) else {}
        payload = build_sample_runtime_record(
            subset=subset,
            sample_id=sample_id,
            terminal_state=str(existing.get("terminal_state", "running") or "running"),
            required_stages=(stages.get("required", []) if isinstance(stages, dict) else []),
            completed_stages=(stages.get("completed", []) if isinstance(stages, dict) else []),
            diagnostics={},
            retry_summary=_sample_retry_summary(subset, sample_id),
            failure_reason=(str(failure.get("reason", "")).strip() if isinstance(failure, dict) else ""),
            failure_details=(dict(failure.get("details", {})) if isinstance(failure, dict) else {}),
            failure_report_path=(str(failure.get("report_path", "")).strip() if isinstance(failure, dict) else ""),
        )
        if isinstance(existing.get("fallback"), dict):
            payload["fallback"] = dict(existing["fallback"])
        if isinstance(existing.get("export"), dict):
            payload["export"] = dict(existing["export"])
        sample_store.persist_sample_runtime(subset, sample_id, payload)

    def _read_json_object(path_str: str) -> Dict[str, Any]:
        path = Path(path_str)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _persist_run_summary(subset: str) -> None:
        manifest_path = run_manifest_paths.get(subset, "")
        run_dir = run_dir_by_subset.get(subset, "")
        if not manifest_path or not run_dir:
            return
        try:
            manifest = load_run_manifest(manifest_path)
        except Exception as exc:
            logger.warning(f"[Warn] subset={subset}: failed to load run manifest for summary: {exc}")
            return

        sample_runtime_records = []
        for sample_id in sample_ids_by_subset.get(subset, []):
            sample_runtime = sample_store.load_sample_runtime(subset, sample_id)
            if sample_runtime is not None:
                sample_runtime_records.append(sample_runtime)

        summary = build_run_summary(
            run_manifest=manifest,
            sample_runtime_records=sample_runtime_records,
            total_samples=len(sample_ids_by_subset.get(subset, [])),
        )
        write_run_summary(run_dir, summary)

    def _build_sample_runtime(
        subset: str,
        sample_id: str,
        *,
        terminal_state: str,
        required_stages: List[str],
        completed_stages: List[str],
        diagnostics: Optional[Dict[str, Any]] = None,
        failure_reason: str = "",
        failure_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return build_sample_runtime_record(
            subset=subset,
            sample_id=sample_id,
            terminal_state=terminal_state,
            required_stages=required_stages,
            completed_stages=completed_stages,
            diagnostics=dict(diagnostics or {}),
            retry_summary=_sample_retry_summary(subset, sample_id),
            failure_reason=failure_reason,
            failure_details=dict(failure_details or {}),
            failure_report_path=(Path(sample_store.failure_report_path(subset, sample_id)).name if failure_reason else ""),
        )

    def _ensure_sample_runtime_for_terminal_artifact(subset: str, sample_id: str) -> None:
        existing_runtime = sample_store.load_sample_runtime(subset, sample_id)
        done_exists = Path(done_marker_path(subset, sample_id)).exists()
        failed_exists = Path(failed_marker_path(subset, sample_id)).exists()
        if existing_runtime is not None:
            existing_state = str(existing_runtime.get("terminal_state", "")).strip()
            if (done_exists and existing_state == "done") or (failed_exists and existing_state == "failed"):
                return

        required_stages = _required_stages_for_subset(subset)
        if done_exists:
            payload = _read_json_object(segments_path(subset, sample_id))
            diagnostics = dict(payload.get("diagnostics", {}))
            completed_stages = diagnostics.get("completed_stages", required_stages)
            runtime_payload = _build_sample_runtime(
                subset,
                sample_id,
                terminal_state="done",
                required_stages=list(diagnostics.get("required_stages", required_stages)),
                completed_stages=list(completed_stages),
                diagnostics=diagnostics,
            )
            sample_store.persist_sample_runtime(subset, sample_id, runtime_payload)
            return

        if Path(failed_marker_path(subset, sample_id)).exists():
            report = _read_json_object(failure_report_path(subset, sample_id))
            details = dict(report.get("details", {}))
            runtime_payload = _build_sample_runtime(
                subset,
                sample_id,
                terminal_state="failed",
                required_stages=list(details.get("required_stages", required_stages)),
                completed_stages=list(details.get("completed_stages", [])),
                failure_reason=str(report.get("reason", "")).strip(),
                failure_details=details,
            )
            sample_store.persist_sample_runtime(subset, sample_id, runtime_payload)

    def _persist_sample_failure(
        subset: str,
        sample_id: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details_payload = dict(details or {})
        runtime_payload = _build_sample_runtime(
            subset,
            sample_id,
            terminal_state="failed",
            required_stages=list(details_payload.get("required_stages", _required_stages_for_subset(subset))),
            completed_stages=list(details_payload.get("completed_stages", [])),
            diagnostics=dict(details_payload.get("diagnostics", {})) if isinstance(details_payload.get("diagnostics", {}), dict) else {},
            failure_reason=reason,
            failure_details=details_payload,
        )
        sample_store.persist_sample_failure(
            subset,
            sample_id,
            reason,
            details_payload,
            sample_runtime=runtime_payload,
        )
        _persist_run_summary(subset)

    def _clear_sample_jobs(subset: str, sample_id: str) -> None:
        removed_task_ids: set[str] = set()
        with queue_lock:
            retained_jobs: List[Any] = []
            for job in job_queue:
                meta = _job_payload_meta(job)
                if str(meta.get("subset", "")) == subset and str(meta.get("sample_id", "")) == sample_id:
                    task_id = _job_payload_task_id(job)
                    if task_id:
                        removed_task_ids.add(task_id)
                    continue
                retained_jobs.append(job)
            if len(retained_jobs) != len(job_queue):
                job_queue[:] = retained_jobs

            inflight_task_ids = [
                str(task_id)
                for task_id, info in inflight.items()
                if str(_job_payload_meta(info.get("job")).get("subset", "")) == subset
                and str(_job_payload_meta(info.get("job")).get("sample_id", "")) == sample_id
            ]
            for task_id in inflight_task_ids:
                inflight.pop(task_id, None)
                removed_task_ids.add(task_id)

            for task_id in removed_task_ids:
                timeout_retry_counts.pop(task_id, None)
                empty_retry_counts.pop(task_id, None)
                dispatch_counts.pop(task_id, None)
                completed_dispatch_ids.pop(task_id, None)

    def _clear_step_a_repeat_artifact_reuse_cache(subset: str, sample_id: str) -> None:
        step_a_repeat_artifact_reuse_caches.pop((subset, sample_id), None)

    def _log_fallback_applied(subset: str, sample_id: str, diagnostics: Dict[str, Any]) -> None:
        fallback_fields: Dict[str, Any] = {}
        fallback_reasons: List[str] = []
        if str(diagnostics.get("selection_policy", "")) == "light_cleanup_fallback":
            fallback_fields["selection_policy"] = "light_cleanup_fallback"

        for key, value in diagnostics.items():
            if key.endswith("_fallback_used") and bool(value):
                fallback_fields[key] = True
                reason_key = key.replace("_fallback_used", "_fallback_reason")
                if reason_key in diagnostics:
                    reason_value = str(diagnostics[reason_key]).strip()
                    if reason_value:
                        fallback_fields[reason_key] = reason_value
                        fallback_reasons.append(reason_value)
            if key.endswith("used_subtitle_fallback") and bool(value):
                fallback_fields[key] = True

        if fallback_reasons:
            fallback_fields["fallback_reason"] = "; ".join(sorted(set(fallback_reasons)))

        if fallback_fields:
            log_event(
                logger,
                "fallback_applied",
                subset=subset,
                sample_id=sample_id,
                **fallback_fields,
            )

    def _fail_sample(
        state: Dict[str, Any],
        subset: str,
        sample_id: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
        *,
        log_message: str = "",
    ) -> None:
        _clear_sample_jobs(subset, sample_id)
        _clear_step_a_repeat_artifact_reuse_cache(subset, sample_id)
        _persist_sample_failure(subset, sample_id, reason, details)
        log_event(
            logger,
            "sample_failed",
            subset=subset,
            sample_id=sample_id,
            reason=reason,
            details=details or {},
        )
        state["sample_status"][sample_id] = 4
        state["cur_idx"] += 1
        if log_message:
            logger.error(log_message)

    def _mark_task_terminal_failure(
        task_id: str,
        dispatch_id: str,
        meta: Dict[str, Any],
        terminal_error: str,
    ) -> None:
        with queue_lock:
            if dispatch_id:
                completed_dispatch_ids[task_id] = dispatch_id
            timeout_retry_counts.pop(task_id, None)
            empty_retry_counts.pop(task_id, None)
        _persist_result_record(
            task_id,
            dispatch_id,
            {},
            meta,
            terminal_error=terminal_error,
        )

    def _required_stages_for_subset(subset: str) -> list[str]:
        manifest_path = run_manifest_paths.get(subset, "")
        if manifest_path:
            try:
                manifest = load_run_manifest(manifest_path)
                return [
                    str(stage).strip()
                    for stage in manifest.required_stages
                    if str(stage).strip()
                ]
            except Exception as exc:
                logger.warning(
                    f"[Warn] subset={subset}: failed to read required stages from manifest {manifest_path}: {exc}"
                )

        required_stages = ["stage1_segments"]
        if bool(config.llm_merge.enabled):
            required_stages.append("stage2_text")
        if bool(config.export.enabled):
            required_stages.append("export")
        return required_stages

    def _required_stages_satisfied(required_stages: List[str], completed_stages: List[str]) -> bool:
        completed_stage_set = {str(stage).strip() for stage in completed_stages if str(stage).strip()}
        return all(str(stage).strip() in completed_stage_set for stage in required_stages)

    def _payload_with_terminal_diagnostics(
        final_res: Dict[str, Any],
        *,
        required_stages: List[str],
        completed_stages: List[str],
    ) -> Dict[str, Any]:
        payload_to_persist = dict(final_res)
        diagnostics = dict(payload_to_persist.get("diagnostics", {}))
        diagnostics["required_stages"] = list(required_stages)
        diagnostics["completed_stages"] = list(completed_stages)
        payload_to_persist["diagnostics"] = diagnostics
        return payload_to_persist

    def _persist_sample_writeback(
        subset: str,
        sample_id: str,
        final_res: Dict[str, Any],
        *,
        required_stages: List[str],
        completed_stages: List[str],
    ) -> Dict[str, Any]:
        payload_to_persist = _payload_with_terminal_diagnostics(
            final_res,
            required_stages=required_stages,
            completed_stages=completed_stages,
        )
        diagnostics = dict(payload_to_persist.get("diagnostics", {}))
        _log_fallback_applied(subset, sample_id, diagnostics)
        sample_store.persist_sample_payload(subset, sample_id, payload_to_persist)
        sample_store.persist_sample_runtime(
            subset,
            sample_id,
            _build_sample_runtime(
                subset,
                sample_id,
                terminal_state="done",
                required_stages=required_stages,
                completed_stages=completed_stages,
                diagnostics=diagnostics,
            ),
        )
        _persist_run_summary(subset)
        return payload_to_persist

    def _mark_sample_done(
        state: Dict[str, Any],
        subset: str,
        sample_id: str,
        final_res: Dict[str, Any],
        *,
        required_stages: List[str],
        completed_stages: List[str],
        finalize_start: float,
        global_done: int,
        progress_total: int,
    ) -> int:
        payload_to_persist = _payload_with_terminal_diagnostics(
            final_res,
            required_stages=required_stages,
            completed_stages=completed_stages,
        )
        diagnostics = dict(payload_to_persist.get("diagnostics", {}))
        _log_fallback_applied(subset, sample_id, diagnostics)
        already_done = sample_store.finalize_sample_success(
            subset,
            sample_id,
            payload_to_persist,
            required_stages=required_stages,
            completed_stages=completed_stages,
            sample_runtime=_build_sample_runtime(
                subset,
                sample_id,
                terminal_state="done",
                required_stages=required_stages,
                completed_stages=completed_stages,
                diagnostics=diagnostics,
            ),
        )
        _persist_run_summary(subset)

        state["sample_status"][sample_id] = 3
        state["cur_idx"] += 1

        if not already_done:
            global_done += 1
        log_event(
            logger,
            "finalize_done",
            subset=subset,
            sample_id=sample_id,
            finalize_ms=int(round((time.perf_counter() - finalize_start) * 1000.0)),
            segment_count=len(payload_to_persist.get("segments", [])),
        )
        logger.info(f"[Progress] {global_done}/{progress_total} (finished: {subset}/{sample_id})")
        return global_done

    def _apply_stage2_text_writeback(sample_id: str, final_res: Dict[str, Any]) -> Dict[str, Any]:
        stage2_res = run_llm_stage2_pass(
            sample_id,
            final_res.get("segments", []),
            config.llm_merge,
            target_language=str(config.export.subtitles.language),
        )

        merge_payload = dict(stage2_res.get("merge", {}))
        summary_payload = dict(stage2_res.get("summary", {}))
        subtitles_payload = dict(stage2_res.get("subtitles", {}))

        cleaned_segments = [
            dict(segment)
            for segment in merge_payload.get("segments", [])
            if isinstance(segment, dict)
        ]
        next_res = dict(final_res)
        next_res["segments"] = attach_stage2_subtitles_to_segments(
            cleaned_segments,
            subtitles_payload.get("items", []),
        )

        task_hierarchy = summary_payload.get("hierarchy")
        if isinstance(task_hierarchy, dict):
            next_res["task_hierarchy"] = dict(task_hierarchy)

        diagnostics = dict(next_res.get("diagnostics", {}))
        diagnostics.update(dict(merge_payload.get("diagnostics", {})))
        diagnostics.update(dict(summary_payload.get("diagnostics", {})))
        diagnostics.update(dict(subtitles_payload.get("diagnostics", {})))
        next_res["diagnostics"] = diagnostics
        return next_res

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
        _fail_sample(
            state,
            subset,
            sample_id,
            reason,
            _artifact_failure_details(error, phase=phase, details=details),
            log_message=log_message,
        )

    def _export_stage_succeeded(export_diagnostics: Dict[str, Any]) -> bool:
        return bool(export_diagnostics.get("export_attempted")) and str(export_diagnostics.get("export_reason", "")).strip() == "applied"

    app.state.job_queue = job_queue
    app.state.inflight = inflight
    app.state.timeout_retry_counts = timeout_retry_counts
    app.state.empty_retry_counts = empty_retry_counts
    app.state.dispatch_counts = dispatch_counts
    app.state.sample_retry_stats = sample_retry_stats
    app.state.completed_dispatch_ids = completed_dispatch_ids
    app.state.task_artifact_writer = task_artifact_writer
    app.state.sample_store = sample_store
    app.state.step_a_producer_batch_limit = _STEP_A_PRODUCER_BATCH_LIMIT
    app.state.run_manifest_paths = run_manifest_paths
    app.state.run_manifest_status_by_subset = run_manifest_status_by_subset

    for ctx in dataset_ctxs:
        for sample_id in ctx.sample_ids:
            _ensure_sample_runtime_for_terminal_artifact(ctx.subset, sample_id)
        _persist_run_summary(ctx.subset)

    @app.get("/get_job")
    def get_job() -> Dict[str, Any]:
        with queue_lock:
            if not job_queue:
                return {"status": "empty"}
            base_job_payload = job_queue.pop(0)
            try:
                base_job = _coerce_job_envelope(base_job_payload)
            except ProtocolValidationError as exc:
                raise HTTPException(status_code=500, detail=f"invalid queued job payload: {exc}") from exc
            task_id = base_job.task_id
            dispatch_counts[task_id] = dispatch_counts.get(task_id, 0) + 1
            dispatch_id = f"d{dispatch_counts[task_id]}"
            dispatched_job = base_job.with_dispatch(dispatch_id)
            inflight[task_id] = {"ts": time.time(), "job": base_job, "dispatch_id": dispatch_id}
            log_event(
                logger,
                "job_dispatched",
                task_id=task_id,
                dispatch_id=dispatch_id,
                subset=str(base_job.meta.get("subset", "")),
                sample_id=str(base_job.meta.get("sample_id", "")),
                job_type=str(base_job.meta.get("job_type", "unknown") or "unknown"),
                source_count=base_job.source_count,
                transport_mode=str(base_job.image_transport.mode),
                artifact_reuse=bool(base_job.meta.get("artifact_reuse", False)),
            )
            return {"status": "ok", "data": dispatched_job.model_dump_payload()}
    
    @app.post("/submit_result")
    def submit_result(payload: Dict[str, Any] = Body(...)) -> Dict[str, str]:
        submit_start = time.perf_counter()
        try:
            res = ResultEnvelope.parse_payload(payload)
        except ProtocolValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        tid = res.task_id
        dispatch_id = res.resolved_dispatch_id

        with queue_lock:
            accepted_dispatch_id = completed_dispatch_ids.get(tid)
            inflight_info = inflight.get(tid)

            if dispatch_id and accepted_dispatch_id == dispatch_id:
                return {"status": "already_received"}

            if not inflight_info:
                return {"status": "stale_ignored"}

            expected_dispatch_id = str(inflight_info.get("dispatch_id", ""))
            if not dispatch_id or dispatch_id != expected_dispatch_id:
                return {"status": "stale_ignored"}

            job_info = inflight.pop(tid)

        authoritative_meta = _job_payload_meta(job_info["job"])
        result_meta = _merge_result_meta(authoritative_meta, dict(res.meta), dispatch_id)
        infer_ms = int(round(max(0.0, float(res.latency_s)) * 1000.0))
        log_event(
            logger,
            "infer_attempt",
            task_id=tid,
            dispatch_id=dispatch_id,
            subset=str(result_meta.get("subset", "")),
            sample_id=str(result_meta.get("sample_id", "")),
            job_type=str(result_meta.get("job_type", "")),
            infer_ms=infer_ms,
        )
        normalized_vlm_json = _normalize_submitted_vlm_json(
            res.vlm_json,
            authoritative_meta,
        )

        if not normalized_vlm_json:
            retry_subset = str(result_meta.get("subset", ""))
            retry_sample_id = str(result_meta.get("sample_id", ""))
            with queue_lock:
                attempt, requeued = _requeue_empty_result(
                    job_queue,
                    empty_retry_counts,
                    tid,
                    job_info["job"],
                    config.server.max_empty_retries_per_job,
                )
                limit = config.server.max_empty_retries_per_job
                limit_label = "inf" if limit <= 0 else str(limit)
                if requeued:
                    _record_sample_retry(
                        retry_subset,
                        retry_sample_id,
                        "empty_result_retries",
                    )

            if requeued:
                _persist_retry_evidence(retry_subset, retry_sample_id)
                log_event(
                    logger,
                    "result_empty_retry",
                    task_id=tid,
                    dispatch_id=dispatch_id,
                    subset=retry_subset,
                    sample_id=retry_sample_id,
                    job_type=str(result_meta.get("job_type", "")),
                    attempt=attempt,
                    retry_limit=limit_label,
                    infer_ms=infer_ms,
                    submit_ms=int(round((time.perf_counter() - submit_start) * 1000.0)),
                )
                logger.warning(
                    f"[Warn] Task {tid} empty or invalid, re-queueing to tail "
                    f"(empty attempt {attempt}/{limit_label})"
                )
                return {"status": "retry_triggered"}

            logger.error(
                f"[Err] Task {tid} empty or invalid retry budget exhausted "
                f"(empty attempt {attempt}/{limit_label}); recording terminal empty result"
            )
            _mark_task_terminal_failure(
                tid,
                dispatch_id,
                result_meta,
                "empty_retry_exhausted",
            )
            return {"status": "empty_retry_exhausted"}

        with queue_lock:
            completed_dispatch_ids[tid] = dispatch_id
            timeout_retry_counts.pop(tid, None)
            empty_retry_counts.pop(tid, None)

        _persist_result_record(tid, dispatch_id, normalized_vlm_json, result_meta)
        log_event(
            logger,
            "job_done",
            task_id=tid,
            dispatch_id=dispatch_id,
            subset=str(result_meta.get("subset", "")),
            sample_id=str(result_meta.get("sample_id", "")),
            job_type=str(result_meta.get("job_type", "")),
            infer_ms=infer_ms,
            submit_ms=int(round((time.perf_counter() - submit_start) * 1000.0)),
        )
        return {"status": "received"}
    
    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}
    
    # Producer loop
    def producer_loop(stop_event: threading.Event):
        # Compute progress totals
        total = sum(len(ctx.sample_ids) for ctx in dataset_ctxs)
        progress_total = config.progress.total_override if config.progress.total_override > 0 else total
        
        done = 0
        for ctx in dataset_ctxs:
            for sid in ctx.sample_ids:
                if Path(done_marker_path(ctx.subset, sid)).exists():
                    done += 1
        
        logger.info(
            f"[Server] Started. IMG=PNG, "
            f"FIXED={config.windowing.target_width}x{config.windowing.target_height}, "
            f"FRAMES_PER_WINDOW={config.windowing.frames_per_window}\n"
            f"[Plan] DATASETS={[(c.data_dir, c.subset) for c in dataset_ctxs]}\n"
            f"[Resume] Already done: {done}/{progress_total} (computed_total={total})"
        )
        
        # Initialize states
        states = {}
        for ctx in dataset_ctxs:
            states[ctx.subset] = {
                "cur_idx": 0,
                "sample_status": {sid: 0 for sid in ctx.sample_ids},
            }
        
        dataset_idx = 0
        global_done = done
        
        while not stop_event.is_set():
            # Check inflight timeouts
            now = time.time()
            exhausted_timeouts: List[tuple[str, str, Dict[str, Any], int, str]] = []
            timeout_retries_to_log: List[tuple[str, str, Dict[str, Any], int, str]] = []
            with queue_lock:
                expired = [
                    tid for tid, info in inflight.items()
                    if now - info["ts"] > config.server.inflight_timeout_sec
                ]
                for tid in expired:
                    inflight_info = inflight.pop(tid)
                    job = inflight_info["job"]
                    dispatch_id = str(inflight_info.get("dispatch_id", ""))
                    meta = _job_payload_meta(job)
                    timeout_retry_counts[tid] = timeout_retry_counts.get(tid, 0) + 1
                    attempt = timeout_retry_counts[tid]
                    limit = config.server.max_retries_per_job
                    limit_label = "inf" if limit <= 0 else str(limit)
                    if limit <= 0 or attempt <= limit:
                        job_queue.append(job)
                        _record_sample_retry(
                            str(meta.get("subset", "")),
                            str(meta.get("sample_id", "")),
                            "timeout_retries",
                        )
                        timeout_retries_to_log.append((tid, dispatch_id, meta, attempt, limit_label))
                    else:
                        exhausted_timeouts.append((tid, dispatch_id, meta, attempt, limit_label))

            for tid, dispatch_id, meta, attempt, limit_label in timeout_retries_to_log:
                _persist_retry_evidence(
                    str(meta.get("subset", "")),
                    str(meta.get("sample_id", "")),
                )
                log_event(
                    logger,
                    "result_timeout_retry",
                    task_id=tid,
                    dispatch_id=dispatch_id,
                    subset=str(meta.get("subset", "")),
                    sample_id=str(meta.get("sample_id", "")),
                    job_type=str(meta.get("job_type", "")),
                    attempt=attempt,
                    retry_limit=limit_label,
                )
                logger.warning(
                    f"[Warn] Task {tid} timed out, re-queueing to tail "
                    f"(timeout attempt {attempt}/{limit_label})"
                )

            for tid, dispatch_id, meta, attempt, limit_label in exhausted_timeouts:
                logger.error(
                    f"[Err] Task {tid} timed out and exhausted retry budget "
                    f"(timeout attempt {attempt}/{limit_label}); recording terminal timeout result"
                )
                _mark_task_terminal_failure(
                    tid,
                    dispatch_id,
                    {**meta, "dispatch_id": dispatch_id},
                    "timeout_retry_exhausted",
                )
            
            # All datasets done
            if dataset_idx >= len(dataset_ctxs):
                if config.server.auto_exit_after_all_done:
                    failed_samples = _count_failed_samples(states)
                    exit_code = _final_exit_code(states)
                    if failed_samples:
                        logger.warning(
                            f"[All Done] {global_done}/{progress_total} succeeded, "
                            f"{failed_samples} failed. Exiting with code {exit_code}."
                        )
                    else:
                        logger.info(f"[All Done] {global_done}/{progress_total}. Exiting.")
                    os._exit(exit_code)
                if stop_event.wait(1.0):
                    break
                continue
            
            ctx = dataset_ctxs[dataset_idx]
            st = states[ctx.subset]
            cur_idx = st["cur_idx"]
            sample_status = st["sample_status"]
            sample_ids = ctx.sample_ids
            
            # Current dataset done, wait for queue to clear
            if cur_idx >= len(sample_ids):
                with queue_lock:
                    if not job_queue and not inflight:
                        logger.info(f"[Dataset] Completed {ctx.subset}. Switching to next...")
                        dataset_idx += 1
                if stop_event.wait(0.2):
                    break
                continue
            
            # Produce jobs if queue not full
            with queue_lock:
                q_len = len(job_queue)
            
            if q_len < config.server.max_queue:
                sid = sample_ids[cur_idx]
                s_dir = Path(ctx.data_dir) / sid
                
                # Skip if already done
                if Path(done_marker_path(ctx.subset, sid)).exists():
                    _ensure_sample_runtime_for_terminal_artifact(ctx.subset, sid)
                    _clear_step_a_repeat_artifact_reuse_cache(ctx.subset, sid)
                    sample_status[sid] = 3
                    st["cur_idx"] += 1
                    time.sleep(0.01)
                    continue

                if Path(failed_marker_path(ctx.subset, sid)).exists():
                    _ensure_sample_runtime_for_terminal_artifact(ctx.subset, sid)
                    _clear_step_a_repeat_artifact_reuse_cache(ctx.subset, sid)
                    sample_status[sid] = 4
                    st["cur_idx"] += 1
                    time.sleep(0.01)
                    continue
                
                # Find video
                mp4s = list(s_dir.glob("Frame_*.mp4"))
                if not mp4s:
                    _fail_sample(
                        st,
                        ctx.subset,
                        sid,
                        "missing_input_video",
                        {
                            "sample_dir": str(s_dir),
                            "expected_glob": "Frame_*.mp4",
                        },
                        log_message=f"[Fail] {ctx.subset}/{sid}: missing Frame_*.mp4 input",
                    )
                    time.sleep(0.01)
                    continue
                mp4 = str(mp4s[0])
                
                w_path = windows_jsonl_path(ctx.subset, sid)
                
                # Step A: Generate window tasks
                if sample_status[sid] == 0:
                    try:
                        fps, nframes = read_video_info(mp4)
                        windows = build_windows(
                            fps, nframes,
                            config.windowing.window_sec,
                            config.windowing.step_sec,
                            config.windowing.frames_per_window
                        )
                        
                        # Load completed windows
                        window_results, failed_window_results = load_window_results(ctx.subset, sid)
                        if failed_window_results:
                            _fail_sample(
                                st,
                                ctx.subset,
                                sid,
                                "window_boundary_failed",
                                {
                                    "failed_window_ids": sorted(int(wid) for wid in failed_window_results),
                                    "errors": {
                                        str(wid): str(record.get("terminal_error", "unknown"))
                                        for wid, record in sorted(failed_window_results.items())
                                    },
                                },
                                log_message=f"[Fail] {ctx.subset}/{sid}: terminal window failure before finalize",
                            )
                            time.sleep(0.01)
                            continue

                        done_wids = _completed_window_ids(window_results)
                        repeat_target = max(1, int(config.windowing.window_repeat_count))
                        repeat_artifact_reuse_cache = step_a_repeat_artifact_reuse_caches.setdefault(
                            (ctx.subset, sid),
                            {},
                        )

                        step_a_producer_batch_limit = int(
                            getattr(app.state, "step_a_producer_batch_limit", _STEP_A_PRODUCER_BATCH_LIMIT)
                        )
                        with FrameExtractor(mp4, artifact_writer=task_artifact_writer) as extractor:
                            cnt = 0
                            
                            for w in windows:
                                if w.window_id in done_wids:
                                    continue
                                success_indices = set(
                                    int(item)
                                    for item in window_results.get(w.window_id, {}).get("repeat_indices", [])
                                )

                                for repeat_index in range(repeat_target):
                                    if repeat_index in success_indices:
                                        continue

                                    tid = f"{ctx.subset}::{sid}_w{w.window_id}_r{repeat_index}"

                                    active = False
                                    with queue_lock:
                                        if _job_queue_contains_task_id(job_queue, tid) or tid in inflight:
                                            active = True

                                    if active:
                                        continue

                                    job = job_builder.build_window_boundary_job(
                                        extractor,
                                        task_id=tid,
                                        subset=ctx.subset,
                                        sample_id=sid,
                                        window=w,
                                        fps=fps,
                                        nframes=nframes,
                                        repeat_index=repeat_index,
                                        repeat_count=repeat_target,
                                        reuse_cache=repeat_artifact_reuse_cache,
                                    )

                                    with queue_lock:
                                        job_queue.append(job)

                                    cnt += 1
                                    if cnt > step_a_producer_batch_limit:
                                        break

                                if cnt > step_a_producer_batch_limit:
                                    break
                        
                        if cnt == 0 and _all_windows_completed(windows, window_results):
                            _clear_step_a_repeat_artifact_reuse_cache(ctx.subset, sid)
                            sample_status[sid] = 2
                    
                    except ArtifactPayloadValidationError as exc:
                        _fail_sample_for_invalid_artifacts(
                            st,
                            ctx.subset,
                            sid,
                            reason="artifact_extraction_failed",
                            phase="step_a",
                            error=exc,
                            log_message=f"[Fail] {ctx.subset}/{sid}: invalid extraction artifacts blocked dispatch",
                        )
                    except Exception as e:
                        logger.exception(f"[Err] {ctx.subset}/{sid}: {e}")
                        _fail_sample(
                            st,
                            ctx.subset,
                            sid,
                            "step_a_exception",
                            {
                                "phase": "step_a",
                                "error": str(e).strip() or type(e).__name__,
                                "exception_type": type(e).__name__,
                            },
                            log_message=f"[Fail] {ctx.subset}/{sid}: Step A crashed",
                        )
                
                # Step B: Finalize
                if sample_status[sid] == 2:
                    try:
                        finalize_start = time.perf_counter()
                        fps, nframes = read_video_info(mp4)
                        windows = build_windows(
                            fps, nframes,
                            config.windowing.window_sec,
                            config.windowing.step_sec,
                            config.windowing.frames_per_window
                        )
                        
                        by_wid, failed_window_results = load_window_results(ctx.subset, sid)
                        if failed_window_results:
                            _fail_sample(
                                st,
                                ctx.subset,
                                sid,
                                "window_boundary_failed",
                                {
                                    "failed_window_ids": sorted(int(wid) for wid in failed_window_results),
                                    "errors": {
                                        str(wid): str(record.get("terminal_error", "unknown"))
                                        for wid, record in sorted(failed_window_results.items())
                                    },
                                },
                                log_message=f"[Fail] {ctx.subset}/{sid}: terminal window failure blocks finalize",
                            )
                            time.sleep(0.01)
                            continue

                        if _all_windows_completed(windows, by_wid):
                            refinement_windows: List[Window] = []
                            if config.windowing.enable_refinement_pass:
                                refinement_frames = (
                                    config.windowing.refinement_frames_per_window
                                    or config.windowing.frames_per_window
                                )
                                refinement_windows = build_refinement_windows(
                                    windows,
                                    by_wid,
                                    fps,
                                    nframes,
                                    refinement_frames,
                                )
                                if refinement_windows:
                                    completed_refinement = _completed_window_ids(by_wid)
                                    missing_refinement = [
                                        refinement_window
                                        for refinement_window in refinement_windows
                                        if refinement_window.window_id not in completed_refinement
                                    ]
                                    if missing_refinement:
                                        with FrameExtractor(mp4, artifact_writer=task_artifact_writer) as extractor:
                                            cnt = 0

                                            for refinement_window in missing_refinement:
                                                existing_success_indices = set(
                                                    int(item)
                                                    for item in by_wid.get(refinement_window.window_id, {}).get("repeat_indices", [])
                                                )

                                                for repeat_index in range(max(1, int(config.windowing.window_repeat_count))):
                                                    if repeat_index in existing_success_indices:
                                                        continue

                                                    tid = f"{ctx.subset}::{sid}_rw{refinement_window.window_id}_r{repeat_index}"
                                                    active = False
                                                    with queue_lock:
                                                        if _job_queue_contains_task_id(job_queue, tid) or tid in inflight:
                                                            active = True

                                                    if active:
                                                        continue

                                                    try:
                                                        job = job_builder.build_window_boundary_job(
                                                            extractor,
                                                            task_id=tid,
                                                            subset=ctx.subset,
                                                            sample_id=sid,
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
                                                            details={"task_id": tid, "job_type": "window_boundary"},
                                                        ) from exc

                                                    with queue_lock:
                                                        job_queue.append(job)

                                                    cnt += 1
                                                    if cnt > 20:
                                                        break

                                                if cnt > 20:
                                                    break

                                        if cnt > 0:
                                            time.sleep(0.05)
                                            continue

                                    if not _all_windows_completed(windows + refinement_windows, by_wid):
                                        time.sleep(0.05)
                                        continue

                            required_stages = _required_stages_for_subset(ctx.subset)
                            completed_stages: List[str] = []

                            logger.info(f"[Finalize] {ctx.subset}/{sid}...")

                            provisional_res = build_segments_via_cuts(
                                sid, windows + refinement_windows, by_wid, fps, nframes,
                                config.windowing.frames_per_window,
                                boundary_prompt_mode=config.windowing.boundary_prompt_mode,
                                adaptive_merge_guard=config.windowing.adaptive_merge_guard,
                                adaptive_merge_min_segments=config.windowing.adaptive_merge_min_segments,
                                adaptive_merge_collapse_ratio=config.windowing.adaptive_merge_collapse_ratio,
                                boundary_support_threshold=config.windowing.boundary_support_threshold,
                                refine_final_instructions=config.windowing.refine_final_instructions,
                            )
                            if not provisional_res.get("segments"):
                                _fail_sample(
                                    st,
                                    ctx.subset,
                                    sid,
                                    "finalize_empty_segments",
                                    {
                                        "phase": "cuts",
                                        "window_count": len(windows),
                                        "completed_window_count": len(by_wid),
                                    },
                                    log_message=f"[Fail] {ctx.subset}/{sid}: finalize produced no segments",
                                )
                                time.sleep(0.01)
                                continue

                            final_res = provisional_res
                            if config.windowing.enable_boundary_refinement:
                                boundary_results, boundary_failures = load_boundary_refinement_results(ctx.subset, sid)
                                if boundary_failures:
                                    _fail_sample(
                                        st,
                                        ctx.subset,
                                        sid,
                                        "boundary_refinement_failed",
                                        {
                                            "failed_boundary_ids": [int(boundary_id) for boundary_id in sorted(boundary_failures)],
                                            "errors": {
                                                str(boundary_id): str(record.get("terminal_error", "unknown"))
                                                for boundary_id, record in sorted(boundary_failures.items())
                                            },
                                        },
                                        log_message=f"[Fail] {ctx.subset}/{sid}: terminal boundary refinement failure blocks finalize",
                                    )
                                    time.sleep(0.01)
                                    continue
                                boundary_refinement_frames = (
                                    config.windowing.boundary_refinement_frames_per_window
                                    or config.windowing.frames_per_window
                                )
                                boundary_refinement_windows = build_boundary_refinement_windows(
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
                                    with FrameExtractor(mp4, artifact_writer=task_artifact_writer) as extractor:
                                        cnt = 0
                                        for boundary_window in missing_boundaries:
                                            boundary_id = int(boundary_window.boundary_id)
                                            tid = f"{ctx.subset}::{sid}_b{boundary_id}"
                                            active = False
                                            with queue_lock:
                                                if _job_queue_contains_task_id(job_queue, tid) or tid in inflight:
                                                    active = True

                                            if active:
                                                continue

                                            try:
                                                job = job_builder.build_boundary_refinement_job(
                                                    extractor,
                                                    task_id=tid,
                                                    subset=ctx.subset,
                                                    sample_id=sid,
                                                    boundary_window=boundary_window,
                                                )
                                            except ArtifactPayloadValidationError as exc:
                                                raise _SampleArtifactDispatchFailure(
                                                    reason="artifact_preparation_failed",
                                                    phase="boundary_refinement_dispatch",
                                                    error=exc,
                                                    details={"task_id": tid, "job_type": "boundary_refinement"},
                                                ) from exc

                                            with queue_lock:
                                                job_queue.append(job)

                                            cnt += 1
                                            if cnt > 20:
                                                break

                                    if cnt > 0:
                                        time.sleep(0.05)
                                        continue

                                    time.sleep(0.05)
                                    continue

                                final_res = dict(provisional_res)
                                final_res["segments"] = apply_boundary_refinement_results(
                                    provisional_res.get("segments", []),
                                    boundary_results,
                                    fps=fps,
                                    abstain_merge_max_support=(
                                        config.windowing.boundary_refinement_abstain_merge_max_support
                                    ),
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
                                label_results, label_failures = load_segment_label_results(ctx.subset, sid)
                                if label_failures:
                                    _fail_sample(
                                        st,
                                        ctx.subset,
                                        sid,
                                        "segment_label_failed",
                                        {
                                            "failed_segment_ids": [int(segment_id) for segment_id in sorted(label_failures)],
                                            "errors": {
                                                str(segment_id): str(record.get("terminal_error", "unknown"))
                                                for segment_id, record in sorted(label_failures.items())
                                            },
                                        },
                                        log_message=f"[Fail] {ctx.subset}/{sid}: terminal deferred labeling failure blocks finalize",
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
                                    with FrameExtractor(mp4, artifact_writer=task_artifact_writer) as extractor:
                                        cnt = 0
                                        for segment in missing_segments:
                                            seg_id = int(segment.get("seg_id", -1))
                                            tid = f"{ctx.subset}::{sid}_seg{seg_id}"
                                            active = False
                                            with queue_lock:
                                                if _job_queue_contains_task_id(job_queue, tid) or tid in inflight:
                                                    active = True

                                            if active:
                                                continue

                                            frame_ids = sample_segment_frame_ids(
                                                int(segment["start_frame"]),
                                                int(segment["end_frame"]),
                                                config.windowing.frames_per_window,
                                                nframes,
                                            )
                                            try:
                                                job = job_builder.build_segment_label_job(
                                                    extractor,
                                                    task_id=tid,
                                                    subset=ctx.subset,
                                                    sample_id=sid,
                                                    segment=segment,
                                                    frame_ids=frame_ids,
                                                )
                                            except ArtifactPayloadValidationError as exc:
                                                raise _SampleArtifactDispatchFailure(
                                                    reason="artifact_preparation_failed",
                                                    phase="segment_label_dispatch",
                                                    error=exc,
                                                    details={"task_id": tid, "job_type": "segment_label"},
                                                ) from exc

                                            with queue_lock:
                                                job_queue.append(job)

                                            cnt += 1
                                            if cnt > 20:
                                                break

                                    if cnt > 0:
                                        time.sleep(0.05)
                                        continue

                                    time.sleep(0.05)
                                    continue

                                final_res = dict(final_res)
                                final_res["segments"] = apply_deferred_segment_labels(
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
                            early_done = _required_stages_satisfied(required_stages, completed_stages)
                            if early_done:
                                global_done = _mark_sample_done(
                                    st,
                                    ctx.subset,
                                    sid,
                                    final_res,
                                    required_stages=required_stages,
                                    completed_stages=completed_stages,
                                    finalize_start=finalize_start,
                                    global_done=global_done,
                                    progress_total=progress_total,
                                )

                            stage2_writeback_required = bool(config.llm_merge.enabled) and (early_done or "stage2_text" in required_stages)
                            if stage2_writeback_required:
                                try:
                                    stage2_res = _apply_stage2_text_writeback(sid, final_res)
                                except Exception as exc:
                                    if early_done:
                                        logger.warning(
                                            f"[Warn] {ctx.subset}/{sid}: optional Stage 2 writeback skipped after terminal success: {exc}"
                                        )
                                        continue
                                    raise

                                if not stage2_res.get("segments"):
                                    if early_done:
                                        logger.warning(
                                            f"[Warn] {ctx.subset}/{sid}: optional Stage 2 writeback produced no segments after terminal success"
                                        )
                                        continue
                                    _fail_sample(
                                        st,
                                        ctx.subset,
                                        sid,
                                        "finalize_empty_segments",
                                        {
                                            "phase": "postprocess",
                                            "window_count": len(windows),
                                            "completed_window_count": len(by_wid),
                                        },
                                        log_message=f"[Fail] {ctx.subset}/{sid}: finalize produced no segments after postprocess",
                                    )
                                    time.sleep(0.01)
                                    continue

                                final_res = stage2_res
                                if early_done:
                                    _persist_sample_writeback(
                                        ctx.subset,
                                        sid,
                                        final_res,
                                        required_stages=required_stages,
                                        completed_stages=completed_stages,
                                    )
                                    continue

                                if "stage2_text" in required_stages:
                                    completed_stages.append("stage2_text")
                                    if _required_stages_satisfied(required_stages, completed_stages):
                                        global_done = _mark_sample_done(
                                            st,
                                            ctx.subset,
                                            sid,
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
                                    export_diagnostics = export_sample_outputs(
                                        run_dir=ctx.run_dir,
                                        sample_id=sid,
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
                                if not _export_stage_succeeded(export_diagnostics):
                                    _fail_sample(
                                        st,
                                        ctx.subset,
                                        sid,
                                        "export_failed",
                                        {
                                            "stage": "export",
                                            "required_stages": list(required_stages),
                                            "completed_stages": list(completed_stages),
                                            "export_enabled": bool(export_diagnostics.get("export_enabled", bool(config.export.enabled))),
                                            "export_attempted": bool(export_diagnostics.get("export_attempted", bool(config.export.enabled))),
                                            "export_mode": str(export_diagnostics.get("export_mode", str(config.export.mode))).strip(),
                                            "export_reason": str(export_diagnostics.get("export_reason", "")).strip(),
                                            "export_errors": list(export_diagnostics.get("export_errors", [])),
                                            "export_error": str(export_diagnostics.get("export_error", "")).strip(),
                                        },
                                        log_message=f"[Fail] {ctx.subset}/{sid}: required export stage did not complete",
                                    )
                                    time.sleep(0.01)
                                    continue
                                completed_stages.append("export")
                                global_done = _mark_sample_done(
                                    st,
                                    ctx.subset,
                                    sid,
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

                            global_done = _mark_sample_done(
                                st,
                                ctx.subset,
                                sid,
                                final_res,
                                required_stages=required_stages,
                                completed_stages=completed_stages,
                                finalize_start=finalize_start,
                                global_done=global_done,
                                progress_total=progress_total,
                            )

                    except _SampleArtifactDispatchFailure as exc:
                        _fail_sample_for_invalid_artifacts(
                            st,
                            ctx.subset,
                            sid,
                            reason=exc.reason,
                            phase=exc.phase,
                            error=exc.error,
                            log_message=f"[Fail] {ctx.subset}/{sid}: invalid preparation artifacts blocked dispatch",
                            details=exc.details,
                        )
                    except Exception as e:
                        logger.error(f"[Err-Finalize] {ctx.subset}/{sid}: {e}")
                        _fail_sample(
                            st,
                            ctx.subset,
                            sid,
                            "finalize_exception",
                            {
                                "phase": "finalize",
                                "error": str(e).strip() or type(e).__name__,
                                "exception_type": type(e).__name__,
                            },
                            log_message=f"[Fail] {ctx.subset}/{sid}: finalize crashed",
                        )
            
            if stop_event.wait(0.1):
                break

    app.state.runtime = ThreadRuntime(
        name="video2tasks-producer",
        target=producer_loop,
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
            log_level=config.logging.level.lower()
        )
    finally:
        runtime.stop()
        runtime.join()
