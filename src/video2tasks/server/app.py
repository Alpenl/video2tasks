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
from ..logging_utils import configure_logging, get_logger
from ..prompt import boundary_refinement_candidate_positions
from .windowing import (
    read_video_info, build_windows, FrameExtractor,
    apply_boundary_refinement_results, apply_deferred_segment_labels,
    build_boundary_refinement_windows, build_segments_via_cuts,
    build_refinement_windows, build_window_prompt_metadata, Window,
    sample_segment_frame_ids,
)
from .exporter import export_sample_outputs
from .llm_merge import run_export_subtitle_localization_pass, run_llm_postprocess_pass
from .run_manifest import ensure_run_manifest, run_manifest_path
from .protocol import (
    InlineImageTransport,
    JobEnvelope,
    ProtocolValidationError,
    ResultEnvelope,
    SharedFSImageTransport,
)
from .runtime import ThreadRuntime
from .task_artifacts import TaskArtifactWriter
from ..vlm.base import normalize_task_window_result


logger = get_logger(__name__)


@dataclass
class DatasetCtx:
    """Dataset context for processing."""
    data_root: str
    subset: str
    data_dir: str
    run_dir: str
    samples_dir: str
    run_dir_nonempty_before_prepare: bool
    sample_ids: List[str]


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
            samples_dir=str(samples_dir),
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



def _clone_shared_fs_transport(transport: SharedFSImageTransport) -> SharedFSImageTransport:
    return SharedFSImageTransport(
        image_paths=list(transport.image_paths),
        artifact_manifest_path=transport.artifact_manifest_path,
    )



def _repeat_artifact_reuse_key(
    *,
    meta: Dict[str, Any],
    frame_ids: List[int],
    artifact_image_kind: str,
    target_width: int,
    target_height: int,
    png_compression: int,
    use_contact_sheets: bool,
    contact_sheet_rows: int,
    contact_sheet_cols: int,
) -> Optional[Tuple[Any, ...]]:
    if not use_contact_sheets:
        return None

    window_id = meta.get("window_id")
    if window_id is None:
        return None

    try:
        normalized_window_id = int(window_id)
    except (TypeError, ValueError):
        return None

    return (
        str(meta.get("subset", "")),
        str(meta.get("sample_id", "")),
        str(meta.get("job_type", "")),
        str(meta.get("window_pass", "coarse") or "coarse"),
        normalized_window_id,
        tuple(int(frame_id) for frame_id in frame_ids),
        str(artifact_image_kind),
        int(target_width),
        int(target_height),
        int(png_compression),
        int(contact_sheet_rows),
        int(contact_sheet_cols),
    )



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
    samples_dir_by_subset = {ctx.subset: ctx.samples_dir for ctx in dataset_ctxs}
    data_dir_by_subset = {ctx.subset: ctx.data_dir for ctx in dataset_ctxs}
    
    # Thread-safe job management
    queue_lock = threading.Lock()
    job_queue: List[Any] = []
    inflight: Dict[str, Dict[str, Any]] = {}
    timeout_retry_counts: Dict[str, int] = {}
    empty_retry_counts: Dict[str, int] = {}
    dispatch_counts: Dict[str, int] = {}
    completed_dispatch_ids: Dict[str, str] = {}
    step_a_repeat_artifact_reuse_caches: Dict[
        Tuple[str, str],
        Dict[Tuple[Any, ...], SharedFSImageTransport],
    ] = {}
    artifact_root_dir = os.getenv("VIDEO2TASKS_TMP_DIR", "tmp").strip() or "tmp"
    task_artifact_writer = TaskArtifactWriter(root_dir=artifact_root_dir)
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
    
    # Per-sample locks
    _sample_locks: Dict[str, threading.Lock] = {}
    _sample_locks_lock = threading.Lock()
    
    def get_sample_lock(sample_key: str) -> threading.Lock:
        with _sample_locks_lock:
            if sample_key not in _sample_locks:
                _sample_locks[sample_key] = threading.Lock()
            return _sample_locks[sample_key]
    
    def sample_out_dir(samples_dir: str, sample_id: str) -> str:
        p = Path(samples_dir) / sample_id
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    
    def windows_jsonl_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "windows.jsonl")
    
    def segments_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "segments.json")

    def segment_labels_jsonl_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "segment_labels.jsonl")

    def boundary_refinements_jsonl_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "boundary_refinements.jsonl")
    
    def done_marker_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / ".DONE")

    def failed_marker_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / ".FAILED")

    def failure_report_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "failure.json")

    def _load_indexed_result_records(path: Path, index_key: str) -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        results: Dict[int, Dict[str, Any]] = {}
        failures: Dict[int, Dict[str, Any]] = {}
        if not path.exists():
            return results, failures

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    index = int(record[index_key])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue

                terminal_error = str(record.get("terminal_error", "")).strip()
                if terminal_error:
                    failures[index] = record
                    results.pop(index, None)
                    continue

                results[index] = record
                failures.pop(index, None)

        return results, failures

    def load_window_results(samples_dir: str, sample_id: str) -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        repeat_target = max(1, int(config.windowing.window_repeat_count))
        path = Path(windows_jsonl_path(samples_dir, sample_id))
        results: Dict[int, Dict[str, Any]] = {}
        failures: Dict[int, Dict[str, Any]] = {}
        if not path.exists():
            return results, failures

        grouped_records: Dict[int, Dict[int, Dict[str, Any]]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    window_id = int(record["window_id"])
                    repeat_index = int(record.get("repeat_index", 0) or 0)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                grouped_records.setdefault(window_id, {})[repeat_index] = record

        for window_id, repeat_records in grouped_records.items():
            normalized_repeats: List[Dict[str, Any]] = []
            failed_repeats: Dict[int, Dict[str, Any]] = {}

            for repeat_index, record in sorted(repeat_records.items()):
                terminal_error = str(record.get("terminal_error", "")).strip()
                if terminal_error:
                    failed_repeats[repeat_index] = record
                    continue

                vlm_json = _normalize_loaded_window_vlm_json(record)
                if not vlm_json:
                    failed_repeats[repeat_index] = {**record, "terminal_error": "invalid_vlm_json"}
                    continue

                normalized_record = dict(record)
                normalized_record["repeat_index"] = repeat_index
                normalized_record["vlm_json"] = vlm_json
                normalized_repeats.append(normalized_record)

            if normalized_repeats:
                representative = max(
                    normalized_repeats,
                    key=lambda record: (
                        len(record.get("vlm_json", {}).get("transitions", [])),
                        -int(record.get("repeat_index", 0)),
                    ),
                )
                aggregated_record = dict(representative)
                aggregated_record["repeat_records"] = normalized_repeats
                aggregated_record["repeat_success_count"] = len(normalized_repeats)
                aggregated_record["repeat_target_count"] = repeat_target
                aggregated_record["repeat_indices"] = [
                    int(record.get("repeat_index", 0))
                    for record in normalized_repeats
                ]
                results[window_id] = aggregated_record

            if failed_repeats:
                first_failed_index = min(failed_repeats)
                failure_record = dict(failed_repeats[first_failed_index])
                failure_record["repeat_indices_failed"] = sorted(int(idx) for idx in failed_repeats)
                failures[window_id] = failure_record

        return results, failures

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

    def load_segment_label_results(samples_dir: str, sample_id: str) -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        records, failures = _load_indexed_result_records(
            Path(segment_labels_jsonl_path(samples_dir, sample_id)),
            "segment_id",
        )
        results: Dict[int, Dict[str, Any]] = {}
        for segment_id, record in records.items():
            vlm_json = normalize_task_window_result(record.get("vlm_json", {}))
            if not vlm_json:
                failures[segment_id] = {**record, "terminal_error": "invalid_vlm_json"}
                continue
            results[segment_id] = vlm_json
        return results, failures

    def load_boundary_refinement_results(samples_dir: str, sample_id: str) -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        records, failures = _load_indexed_result_records(
            Path(boundary_refinements_jsonl_path(samples_dir, sample_id)),
            "boundary_id",
        )
        results: Dict[int, Dict[str, Any]] = {}
        for boundary_id, record in records.items():
            vlm_json = _normalize_loaded_boundary_refinement_vlm_json(record)
            if not vlm_json:
                failures[boundary_id] = {**record, "terminal_error": "invalid_vlm_json"}
                continue
            normalized_record = dict(record)
            normalized_record["vlm_json"] = vlm_json
            results[boundary_id] = normalized_record
        return results, failures

    def _resolve_samples_dir(subset: str) -> str:
        samples_dir = samples_dir_by_subset.get(subset)
        if not samples_dir:
            samples_dir = str(Path(config.run.base_dir) / subset / config.run.run_id / "samples")
            Path(samples_dir).mkdir(parents=True, exist_ok=True)
            samples_dir_by_subset[subset] = samples_dir
        return samples_dir

    def _append_jsonl_record(path: str, record: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _persist_result_record(
        task_id: str,
        dispatch_id: str,
        vlm_json: Dict[str, Any],
        meta: Dict[str, Any],
        terminal_error: str = "",
    ) -> None:
        subset = str(meta.get("subset", dataset_ctxs[0].subset if dataset_ctxs else "default"))
        sid = str(meta.get("sample_id", "unknown"))
        samples_dir = _resolve_samples_dir(subset)
        job_type = str(meta.get("job_type", "window_boundary"))
        common_fields: Dict[str, Any] = {
            "task_id": task_id,
            "dispatch_id": dispatch_id,
            "vlm_json": vlm_json,
        }
        logical_frame_count = int(meta.get("logical_frame_count", 0) or 0)
        if logical_frame_count > 0:
            common_fields["logical_frame_count"] = logical_frame_count
        if terminal_error:
            common_fields["terminal_error"] = terminal_error

        sample_key = f"{subset}::{sid}"
        with get_sample_lock(sample_key):
            if job_type == "segment_label":
                rec = {
                    **common_fields,
                    "segment_id": int(meta.get("segment_id", -1)),
                }
                _append_jsonl_record(segment_labels_jsonl_path(samples_dir, sid), rec)
                return

            if job_type == "boundary_refinement":
                rec = {
                    **common_fields,
                    "boundary_id": int(meta.get("boundary_id", -1)),
                    "coarse_boundary_frame": int(meta.get("coarse_boundary_frame", -1)),
                    "frame_ids": [int(frame_id) for frame_id in meta.get("frame_ids", [])],
                }
                _append_jsonl_record(boundary_refinements_jsonl_path(samples_dir, sid), rec)
                return

            rec = {
                **common_fields,
                "window_id": meta.get("window_id"),
                "repeat_index": int(meta.get("repeat_index", 0) or 0),
            }
            _append_jsonl_record(windows_jsonl_path(samples_dir, sid), rec)

    def _persist_sample_failure(
        subset: str,
        sample_id: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        samples_dir = _resolve_samples_dir(subset)
        sample_key = f"{subset}::{sample_id}"
        payload = {
            "subset": subset,
            "sample_id": sample_id,
            "reason": reason,
            "details": details or {},
        }

        with get_sample_lock(sample_key):
            for stale_path in (segments_path(samples_dir, sample_id), done_marker_path(samples_dir, sample_id)):
                try:
                    Path(stale_path).unlink()
                except FileNotFoundError:
                    pass
            Path(failed_marker_path(samples_dir, sample_id)).touch()
            with open(failure_report_path(samples_dir, sample_id), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

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

    def _build_job_payload(
        extractor: FrameExtractor,
        *,
        task_id: str,
        frame_ids: List[int],
        meta: Dict[str, Any],
        artifact_image_kind: str,
        repeat_artifact_reuse_cache: Optional[Dict[Tuple[Any, ...], SharedFSImageTransport]] = None,
    ) -> JobEnvelope:
        cache_key = _repeat_artifact_reuse_key(
            meta=meta,
            frame_ids=frame_ids,
            artifact_image_kind=artifact_image_kind,
            target_width=config.windowing.target_width,
            target_height=config.windowing.target_height,
            png_compression=config.windowing.png_compression,
            use_contact_sheets=config.windowing.use_contact_sheets,
            contact_sheet_rows=config.windowing.contact_sheet_rows,
            contact_sheet_cols=config.windowing.contact_sheet_cols,
        )
        if repeat_artifact_reuse_cache is not None and cache_key is not None:
            cached_transport = repeat_artifact_reuse_cache.get(cache_key)
            if cached_transport is not None:
                return JobEnvelope(
                    task_id=task_id,
                    meta=dict(meta),
                    image_transport=_clone_shared_fs_transport(cached_transport),
                )

        images, artifact_batch = extractor.get_many_b64_with_artifacts(
            frame_ids,
            config.windowing.target_width,
            config.windowing.target_height,
            config.windowing.png_compression,
            use_contact_sheets=config.windowing.use_contact_sheets,
            contact_sheet_rows=config.windowing.contact_sheet_rows,
            contact_sheet_cols=config.windowing.contact_sheet_cols,
            artifact_metadata={**meta, "task_id": task_id},
            artifact_image_kind=artifact_image_kind,
            return_images=getattr(extractor, "artifact_writer", None) is None,
        )
        if artifact_batch is not None and artifact_batch.records:
            image_transport = SharedFSImageTransport(
                image_paths=[record.path for record in artifact_batch.records],
                artifact_manifest_path=artifact_batch.manifest_path,
            )
            if repeat_artifact_reuse_cache is not None and cache_key is not None:
                repeat_artifact_reuse_cache.setdefault(cache_key, _clone_shared_fs_transport(image_transport))
        else:
            image_transport = InlineImageTransport(images=images)
        return JobEnvelope(
            task_id=task_id,
            meta=dict(meta),
            image_transport=image_transport,
        )

    app.state.job_queue = job_queue
    app.state.inflight = inflight
    app.state.timeout_retry_counts = timeout_retry_counts
    app.state.empty_retry_counts = empty_retry_counts
    app.state.dispatch_counts = dispatch_counts
    app.state.completed_dispatch_ids = completed_dispatch_ids
    app.state.task_artifact_writer = task_artifact_writer
    app.state.samples_dir_by_subset = samples_dir_by_subset
    app.state.run_manifest_paths = run_manifest_paths
    app.state.run_manifest_status_by_subset = run_manifest_status_by_subset

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
            return {"status": "ok", "data": dispatched_job.model_dump_payload()}
    
    @app.post("/submit_result")
    def submit_result(payload: Dict[str, Any] = Body(...)) -> Dict[str, str]:
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
        normalized_vlm_json = _normalize_submitted_vlm_json(
            res.vlm_json,
            authoritative_meta,
        )

        if not normalized_vlm_json:
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
                if Path(done_marker_path(ctx.samples_dir, sid)).exists():
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
            with queue_lock:
                expired = [
                    tid for tid, info in inflight.items()
                    if now - info["ts"] > config.server.inflight_timeout_sec
                ]
                for tid in expired:
                    inflight_info = inflight.pop(tid)
                    job = inflight_info["job"]
                    dispatch_id = str(inflight_info.get("dispatch_id", ""))
                    timeout_retry_counts[tid] = timeout_retry_counts.get(tid, 0) + 1
                    attempt = timeout_retry_counts[tid]
                    limit = config.server.max_retries_per_job
                    limit_label = "inf" if limit <= 0 else str(limit)
                    if limit <= 0 or attempt <= limit:
                        job_queue.append(job)
                        logger.warning(
                            f"[Warn] Task {tid} timed out, re-queueing to tail "
                            f"(timeout attempt {attempt}/{limit_label})"
                        )
                    else:
                        exhausted_timeouts.append((tid, dispatch_id, _job_payload_meta(job), attempt, limit_label))

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
                if Path(done_marker_path(ctx.samples_dir, sid)).exists():
                    _clear_step_a_repeat_artifact_reuse_cache(ctx.subset, sid)
                    sample_status[sid] = 3
                    st["cur_idx"] += 1
                    time.sleep(0.01)
                    continue

                if Path(failed_marker_path(ctx.samples_dir, sid)).exists():
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
                
                w_path = windows_jsonl_path(ctx.samples_dir, sid)
                
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
                        window_results, failed_window_results = load_window_results(ctx.samples_dir, sid)
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

                                    meta = {
                                        "subset": ctx.subset,
                                        "sample_id": sid,
                                        "job_type": "window_boundary",
                                        "repeat_index": repeat_index,
                                        "window_repeat_count": repeat_target,
                                        "logical_frame_count": len(w.frame_ids),
                                        **build_window_prompt_metadata(w, fps, nframes),
                                        "use_contact_sheets": config.windowing.use_contact_sheets,
                                        "contact_sheet_rows": (
                                            config.windowing.contact_sheet_rows
                                            if config.windowing.use_contact_sheets else 0
                                        ),
                                        "contact_sheet_cols": (
                                            config.windowing.contact_sheet_cols
                                            if config.windowing.use_contact_sheets else 0
                                        ),
                                    }
                                    job = _build_job_payload(
                                        extractor,
                                        task_id=tid,
                                        frame_ids=w.frame_ids,
                                        meta=meta,
                                        artifact_image_kind=(
                                            "window_contact_sheet"
                                            if config.windowing.use_contact_sheets else "window_frame"
                                        ),
                                        repeat_artifact_reuse_cache=repeat_artifact_reuse_cache,
                                    )

                                    with queue_lock:
                                        job_queue.append(job)

                                    cnt += 1
                                    if cnt > 20:
                                        break

                                if cnt > 20:
                                    break
                        
                        if cnt == 0 and _all_windows_completed(windows, window_results):
                            _clear_step_a_repeat_artifact_reuse_cache(ctx.subset, sid)
                            sample_status[sid] = 2
                    
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
                        fps, nframes = read_video_info(mp4)
                        windows = build_windows(
                            fps, nframes,
                            config.windowing.window_sec,
                            config.windowing.step_sec,
                            config.windowing.frames_per_window
                        )
                        
                        by_wid, failed_window_results = load_window_results(ctx.samples_dir, sid)
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

                                                    meta = {
                                                        "subset": ctx.subset,
                                                        "sample_id": sid,
                                                        "job_type": "window_boundary",
                                                        "window_pass": "refinement",
                                                        "repeat_index": repeat_index,
                                                        "window_repeat_count": max(1, int(config.windowing.window_repeat_count)),
                                                        "logical_frame_count": len(refinement_window.frame_ids),
                                                        **build_window_prompt_metadata(refinement_window, fps, nframes),
                                                        "use_contact_sheets": config.windowing.use_contact_sheets,
                                                        "contact_sheet_rows": (
                                                            config.windowing.contact_sheet_rows
                                                            if config.windowing.use_contact_sheets else 0
                                                        ),
                                                        "contact_sheet_cols": (
                                                            config.windowing.contact_sheet_cols
                                                            if config.windowing.use_contact_sheets else 0
                                                        ),
                                                    }
                                                    job = _build_job_payload(
                                                        extractor,
                                                        task_id=tid,
                                                        frame_ids=refinement_window.frame_ids,
                                                        meta=meta,
                                                        artifact_image_kind=(
                                                            "refinement_contact_sheet"
                                                            if config.windowing.use_contact_sheets else "refinement_frame"
                                                        ),
                                                    )

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
                                boundary_results, boundary_failures = load_boundary_refinement_results(ctx.samples_dir, sid)
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

                                            meta = {
                                                "subset": ctx.subset,
                                                "sample_id": sid,
                                                "job_type": "boundary_refinement",
                                                "boundary_id": boundary_id,
                                                "coarse_boundary_frame": int(boundary_window.coarse_boundary_frame),
                                                "frame_ids": [int(frame_id) for frame_id in boundary_window.frame_ids],
                                                "logical_frame_count": len(boundary_window.frame_ids),
                                                "window_start_frame": int(boundary_window.start_frame),
                                                "window_end_frame": int(boundary_window.end_frame),
                                                "use_contact_sheets": config.windowing.use_contact_sheets,
                                                "contact_sheet_rows": (
                                                    config.windowing.contact_sheet_rows
                                                    if config.windowing.use_contact_sheets else 0
                                                ),
                                                "contact_sheet_cols": (
                                                    config.windowing.contact_sheet_cols
                                                    if config.windowing.use_contact_sheets else 0
                                                ),
                                            }
                                            job = _build_job_payload(
                                                extractor,
                                                task_id=tid,
                                                frame_ids=boundary_window.frame_ids,
                                                meta=meta,
                                                artifact_image_kind=(
                                                    "boundary_refinement_contact_sheet"
                                                    if config.windowing.use_contact_sheets else "boundary_refinement_frame"
                                                ),
                                            )

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
                                label_results, label_failures = load_segment_label_results(ctx.samples_dir, sid)
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
                                            meta = {
                                                "subset": ctx.subset,
                                                "sample_id": sid,
                                                "job_type": "segment_label",
                                                "segment_id": seg_id,
                                                "segment_start_frame": int(segment["start_frame"]),
                                                "segment_end_frame": int(segment["end_frame"]),
                                                "frame_ids": [int(frame_id) for frame_id in frame_ids],
                                                "logical_frame_count": len(frame_ids),
                                                "use_contact_sheets": config.windowing.use_contact_sheets,
                                                "contact_sheet_rows": (
                                                    config.windowing.contact_sheet_rows
                                                    if config.windowing.use_contact_sheets else 0
                                                ),
                                                "contact_sheet_cols": (
                                                    config.windowing.contact_sheet_cols
                                                    if config.windowing.use_contact_sheets else 0
                                                ),
                                            }
                                            job = _build_job_payload(
                                                extractor,
                                                task_id=tid,
                                                frame_ids=frame_ids,
                                                meta=meta,
                                                artifact_image_kind=(
                                                    "segment_label_contact_sheet"
                                                    if config.windowing.use_contact_sheets else "segment_label_frame"
                                                ),
                                            )

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

                            cleaned_segments, task_hierarchy, llm_postprocess_diagnostics = run_llm_postprocess_pass(
                                sid,
                                final_res.get("segments", []),
                                config.llm_merge,
                            )
                            final_res = dict(final_res)
                            final_res["segments"] = cleaned_segments
                            if task_hierarchy is not None:
                                final_res["task_hierarchy"] = task_hierarchy
                            diagnostics = dict(final_res.get("diagnostics", {}))
                            diagnostics.update(llm_postprocess_diagnostics)
                            final_res["diagnostics"] = diagnostics
                            if not final_res.get("segments"):
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

                            export_segments = [
                                dict(segment)
                                for segment in final_res.get("segments", [])
                                if isinstance(segment, dict)
                            ]
                            subtitle_segments, subtitle_localization_diagnostics = run_export_subtitle_localization_pass(
                                sid,
                                export_segments,
                                config.llm_merge,
                                str(config.export.subtitles.language),
                            )
                            export_segments = subtitle_segments
                            final_res["segments"] = [
                                dict(segment)
                                for segment in export_segments
                                if isinstance(segment, dict)
                            ]
                            diagnostics.update(subtitle_localization_diagnostics)

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

                            diagnostics.update(export_diagnostics)
                            final_res["diagnostics"] = diagnostics
                            
                            with open(segments_path(ctx.samples_dir, sid), "w", encoding="utf-8") as f:
                                json.dump(final_res, f, indent=2, ensure_ascii=False)
                            
                            done_path = done_marker_path(ctx.samples_dir, sid)
                            already_done = Path(done_path).exists()
                            for stale_failure_path in (failed_marker_path(ctx.samples_dir, sid), failure_report_path(ctx.samples_dir, sid)):
                                try:
                                    Path(stale_failure_path).unlink()
                                except FileNotFoundError:
                                    pass
                            Path(done_path).touch()
                            
                            sample_status[sid] = 3
                            st["cur_idx"] += 1
                            
                            if not already_done:
                                global_done += 1
                            logger.info(f"[Progress] {global_done}/{progress_total} (finished: {ctx.subset}/{sid})")

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
