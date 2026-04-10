"""Shared runtime facade for the server app."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import FastAPI

from ..config import Config
from ..logging_utils import log_event
from ..prompt import boundary_refinement_candidate_positions
from ..vlm.base import normalize_task_window_result
from .job_builder import ArtifactReuseEntry, JobBuilder
from .protocol import JobEnvelope
from .run_manifest import ensure_run_manifest, load_run_manifest, run_manifest_path
from .run_summary import build_run_summary, build_sample_runtime_record, build_sample_timing_record, write_run_summary
from .sample_store import SampleStore
from .task_artifacts import TaskArtifactWriter


STEP_A_PRODUCER_BATCH_LIMIT = 20


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
class RuntimeDependencies:
    """Dependency bundle captured from app.py to preserve monkeypatch entrypoints."""

    read_video_info_resolver: Callable[[], Callable[..., Any]]
    build_windows_resolver: Callable[[], Callable[..., Any]]
    frame_extractor_cls_resolver: Callable[[], Any]
    build_refinement_windows_resolver: Callable[[], Callable[..., Any]]
    build_segments_via_cuts_resolver: Callable[[], Callable[..., Any]]
    build_boundary_refinement_windows_resolver: Callable[[], Callable[..., Any]]
    apply_boundary_refinement_results_resolver: Callable[[], Callable[..., Any]]
    sample_segment_frame_ids_resolver: Callable[[], Callable[..., Any]]
    apply_deferred_segment_labels_resolver: Callable[[], Callable[..., Any]]
    run_llm_stage2_pass_resolver: Callable[[], Callable[..., Any]]
    attach_stage2_subtitles_to_segments_resolver: Callable[[], Callable[..., Any]]
    export_sample_outputs_resolver: Callable[[], Callable[..., Any]]

    @property
    def read_video_info(self) -> Callable[..., Any]:
        return self.read_video_info_resolver()

    @property
    def build_windows(self) -> Callable[..., Any]:
        return self.build_windows_resolver()

    @property
    def frame_extractor_cls(self) -> Any:
        return self.frame_extractor_cls_resolver()

    @property
    def build_refinement_windows(self) -> Callable[..., Any]:
        return self.build_refinement_windows_resolver()

    @property
    def build_segments_via_cuts(self) -> Callable[..., Any]:
        return self.build_segments_via_cuts_resolver()

    @property
    def build_boundary_refinement_windows(self) -> Callable[..., Any]:
        return self.build_boundary_refinement_windows_resolver()

    @property
    def apply_boundary_refinement_results(self) -> Callable[..., Any]:
        return self.apply_boundary_refinement_results_resolver()

    @property
    def sample_segment_frame_ids(self) -> Callable[..., Any]:
        return self.sample_segment_frame_ids_resolver()

    @property
    def apply_deferred_segment_labels(self) -> Callable[..., Any]:
        return self.apply_deferred_segment_labels_resolver()

    @property
    def run_llm_stage2_pass(self) -> Callable[..., Any]:
        return self.run_llm_stage2_pass_resolver()

    @property
    def attach_stage2_subtitles_to_segments(self) -> Callable[..., Any]:
        return self.attach_stage2_subtitles_to_segments_resolver()

    @property
    def export_sample_outputs(self) -> Callable[..., Any]:
        return self.export_sample_outputs_resolver()


def parse_datasets(config: Config) -> List[DatasetCtx]:
    """Parse dataset configurations into contexts."""

    ctxs: List[DatasetCtx] = []
    for ds in config.datasets:
        data_dir = Path(ds.root) / ds.subset
        run_dir = Path(config.run.base_dir) / ds.subset / config.run.run_id
        run_dir_nonempty_before_prepare = run_dir.exists() and any(run_dir.iterdir())
        samples_dir = run_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        if data_dir.exists():
            sample_ids = sorted([path.name for path in data_dir.iterdir() if path.is_dir()])
        else:
            sample_ids = []

        ctxs.append(
            DatasetCtx(
                data_root=ds.root,
                subset=ds.subset,
                data_dir=str(data_dir),
                run_dir=str(run_dir),
                run_dir_nonempty_before_prepare=run_dir_nonempty_before_prepare,
                sample_ids=sample_ids,
            )
        )
    return ctxs


def requeue_empty_result(
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


def job_payload_task_id(job: Any) -> str:
    if isinstance(job, JobEnvelope):
        return job.task_id
    if isinstance(job, dict):
        return str(job.get("task_id", "")).strip()
    return ""


def job_payload_meta(job: Any) -> Dict[str, Any]:
    if isinstance(job, JobEnvelope):
        return dict(job.meta)
    if isinstance(job, dict):
        raw_meta = job.get("meta", {})
        if isinstance(raw_meta, dict):
            return dict(raw_meta)
    return {}


def coerce_job_envelope(job: Any) -> JobEnvelope:
    if isinstance(job, JobEnvelope):
        return job
    return JobEnvelope.parse_payload(job)


def job_queue_contains_task_id(job_queue: List[Any], task_id: str) -> bool:
    return any(job_payload_task_id(job) == task_id for job in job_queue)


def logical_frame_count_from_meta(meta: Dict[str, Any]) -> Optional[int]:
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


def merge_result_meta(authoritative_meta: Dict[str, Any], worker_meta: Dict[str, Any], dispatch_id: str) -> Dict[str, Any]:
    merged = dict(authoritative_meta)
    for key, value in worker_meta.items():
        if key not in merged:
            merged[key] = value
    merged["dispatch_id"] = dispatch_id
    return merged


def normalize_submitted_vlm_json(vlm_json: Dict[str, Any], authoritative_meta: Dict[str, Any]) -> Dict[str, Any]:
    logical_frame_count = logical_frame_count_from_meta(authoritative_meta)
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


def logical_frame_count_from_record(record: Dict[str, Any]) -> Optional[int]:
    if not isinstance(record, dict):
        return None

    logical_frame_count = logical_frame_count_from_meta(record)
    if logical_frame_count and logical_frame_count > 0:
        return logical_frame_count
    return logical_frame_count_from_meta(record.get("meta", {}))


def normalize_loaded_window_vlm_json(record: Dict[str, Any]) -> Dict[str, Any]:
    logical_frame_count = logical_frame_count_from_record(record)
    max_transition_index = (logical_frame_count - 1) if logical_frame_count and logical_frame_count > 0 else None
    return normalize_task_window_result(
        record.get("vlm_json", {}),
        max_transition_index=max_transition_index,
    )


def normalize_loaded_boundary_refinement_vlm_json(record: Dict[str, Any]) -> Dict[str, Any]:
    frame_ids = record.get("frame_ids", [])
    if isinstance(frame_ids, list) and frame_ids:
        logical_frame_count = len(frame_ids)
    else:
        logical_frame_count = logical_frame_count_from_record(record)

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


def count_failed_samples(states: Dict[str, Dict[str, Any]]) -> int:
    return sum(
        1
        for state in states.values()
        for status in dict(state.get("sample_status", {})).values()
        if int(status) == 4
    )


def final_exit_code(states: Dict[str, Dict[str, Any]]) -> int:
    return 1 if count_failed_samples(states) > 0 else 0


class ServerRuntimeState:
    """State and helper facade shared by app assembly, routes, and producer."""

    def __init__(
        self,
        *,
        config: Config,
        logger: Any,
        dependencies: RuntimeDependencies,
        dataset_ctxs: List[DatasetCtx],
        queue_lock: threading.Lock,
        job_queue: List[Any],
        inflight: Dict[str, Dict[str, Any]],
        timeout_retry_counts: Dict[str, int],
        empty_retry_counts: Dict[str, int],
        dispatch_counts: Dict[str, int],
        sample_retry_stats: Dict[Tuple[str, str], Dict[str, int]],
        completed_dispatch_ids: Dict[str, str],
        step_a_repeat_artifact_reuse_caches: Dict[Tuple[str, str], Dict[Tuple[Any, ...], ArtifactReuseEntry]],
        task_artifact_writer: TaskArtifactWriter,
        sample_store: SampleStore,
        job_builder: JobBuilder,
        run_dir_by_subset: Dict[str, str],
        sample_ids_by_subset: Dict[str, List[str]],
        run_manifest_paths: Dict[str, str],
        run_manifest_status_by_subset: Dict[str, Dict[str, Any]],
        step_a_producer_batch_limit: int,
    ) -> None:
        self.config = config
        self.logger = logger
        self.dependencies = dependencies
        self.dataset_ctxs = dataset_ctxs
        self.queue_lock = queue_lock
        self.job_queue = job_queue
        self.inflight = inflight
        self.timeout_retry_counts = timeout_retry_counts
        self.empty_retry_counts = empty_retry_counts
        self.dispatch_counts = dispatch_counts
        self.sample_retry_stats = sample_retry_stats
        self.completed_dispatch_ids = completed_dispatch_ids
        self.step_a_repeat_artifact_reuse_caches = step_a_repeat_artifact_reuse_caches
        self.task_artifact_writer = task_artifact_writer
        self.sample_store = sample_store
        self.job_builder = job_builder
        self.run_dir_by_subset = run_dir_by_subset
        self.sample_ids_by_subset = sample_ids_by_subset
        self.run_manifest_paths = run_manifest_paths
        self.run_manifest_status_by_subset = run_manifest_status_by_subset
        self.step_a_producer_batch_limit = int(step_a_producer_batch_limit)
        self._app_state: Any = None
        self._stage_timing_lock = threading.Lock()
        self._stage_start_ts: Dict[Tuple[str, str, str], float] = {}
        self._stage_done_keys: set[Tuple[str, str, str]] = set()

    def attach_app_state(self, app: FastAPI) -> None:
        self._app_state = app.state
        app.state.runtime_state = self
        app.state.job_queue = self.job_queue
        app.state.inflight = self.inflight
        app.state.timeout_retry_counts = self.timeout_retry_counts
        app.state.empty_retry_counts = self.empty_retry_counts
        app.state.dispatch_counts = self.dispatch_counts
        app.state.sample_retry_stats = self.sample_retry_stats
        app.state.completed_dispatch_ids = self.completed_dispatch_ids
        app.state.task_artifact_writer = self.task_artifact_writer
        app.state.sample_store = self.sample_store
        app.state.step_a_producer_batch_limit = self.step_a_producer_batch_limit
        app.state.run_manifest_paths = self.run_manifest_paths
        app.state.run_manifest_status_by_subset = self.run_manifest_status_by_subset

    def initialize_runtime_artifacts(self) -> None:
        for ctx in self.dataset_ctxs:
            for sample_id in ctx.sample_ids:
                self.ensure_sample_runtime_for_terminal_artifact(ctx.subset, sample_id)
            self.persist_run_summary(ctx.subset)

    def current_step_a_producer_batch_limit(self) -> int:
        if self._app_state is not None:
            try:
                return int(getattr(self._app_state, "step_a_producer_batch_limit", self.step_a_producer_batch_limit))
            except (TypeError, ValueError):
                return self.step_a_producer_batch_limit
        return self.step_a_producer_batch_limit

    def sample_out_dir(self, subset: str, sample_id: str) -> str:
        return self.sample_store.sample_out_dir(subset, sample_id)

    def windows_jsonl_path(self, subset: str, sample_id: str) -> str:
        return self.sample_store.windows_jsonl_path(subset, sample_id)

    def segments_path(self, subset: str, sample_id: str) -> str:
        return self.sample_store.segments_path(subset, sample_id)

    def segment_labels_jsonl_path(self, subset: str, sample_id: str) -> str:
        return self.sample_store.segment_labels_jsonl_path(subset, sample_id)

    def boundary_refinements_jsonl_path(self, subset: str, sample_id: str) -> str:
        return self.sample_store.boundary_refinements_jsonl_path(subset, sample_id)

    def done_marker_path(self, subset: str, sample_id: str) -> str:
        return self.sample_store.done_marker_path(subset, sample_id)

    def failed_marker_path(self, subset: str, sample_id: str) -> str:
        return self.sample_store.failed_marker_path(subset, sample_id)

    def failure_report_path(self, subset: str, sample_id: str) -> str:
        return self.sample_store.failure_report_path(subset, sample_id)

    def load_window_results(self, subset: str, sample_id: str) -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        return self.sample_store.load_window_results(subset, sample_id)

    def completed_window_ids(self, results: Dict[int, Dict[str, Any]]) -> set[int]:
        completed: set[int] = set()
        repeat_target = max(1, int(self.config.windowing.window_repeat_count))
        for window_id, record in results.items():
            if int(record.get("repeat_success_count", 1)) >= repeat_target:
                completed.add(int(window_id))
        return completed

    def all_windows_completed(self, windows: List[Any], results: Dict[int, Dict[str, Any]]) -> bool:
        completed = self.completed_window_ids(results)
        return all(int(window.window_id) in completed for window in windows)

    def load_segment_label_results(self, subset: str, sample_id: str) -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        return self.sample_store.load_segment_label_results(subset, sample_id)

    def load_boundary_refinement_results(self, subset: str, sample_id: str) -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        return self.sample_store.load_boundary_refinement_results(subset, sample_id)

    def resolve_samples_dir(self, subset: str) -> str:
        return self.sample_store.resolve_samples_dir(subset)

    def persist_result_record(
        self,
        task_id: str,
        dispatch_id: str,
        vlm_json: Dict[str, Any],
        meta: Dict[str, Any],
        terminal_error: str = "",
    ) -> None:
        self.sample_store.persist_result_record(
            task_id,
            dispatch_id,
            vlm_json,
            meta,
            terminal_error=terminal_error,
        )

    def sample_retry_key(self, subset: str, sample_id: str) -> Tuple[str, str]:
        return (str(subset), str(sample_id))

    def persisted_retry_summary(self, subset: str, sample_id: str) -> Dict[str, int]:
        payload = self.sample_store.load_sample_runtime(subset, sample_id) or {}
        retry = payload.get("retry", {}) if isinstance(payload, dict) else {}
        if not isinstance(retry, dict):
            retry = {}
        return {
            "total_retries": int(retry.get("total_retries", 0) or 0),
            "empty_result_retries": int(retry.get("empty_result_retries", 0) or 0),
            "timeout_retries": int(retry.get("timeout_retries", 0) or 0),
            "dispatch_count": int(retry.get("dispatch_count", 0) or 0),
        }

    def record_sample_retry(self, subset: str, sample_id: str, retry_field: str) -> None:
        key = self.sample_retry_key(subset, sample_id)
        stats = self.sample_retry_stats.setdefault(key, self.persisted_retry_summary(subset, sample_id))
        stats[retry_field] = int(stats.get(retry_field, 0)) + 1
        stats["total_retries"] = int(stats.get("empty_result_retries", 0)) + int(stats.get("timeout_retries", 0))

    def sample_dispatch_count(self, subset: str, sample_id: str) -> int:
        prefix = f"{subset}::{sample_id}_"
        with self.queue_lock:
            return sum(
                int(count)
                for task_id, count in self.dispatch_counts.items()
                if str(task_id).startswith(prefix)
            )

    def sample_retry_summary(self, subset: str, sample_id: str) -> Dict[str, int]:
        persisted = self.persisted_retry_summary(subset, sample_id)
        stats = dict(persisted)
        stats.update(self.sample_retry_stats.get(self.sample_retry_key(subset, sample_id), {}))
        total_retries = max(int(stats.get("total_retries", 0) or 0), int(persisted.get("total_retries", 0) or 0))
        empty_result_retries = max(
            int(stats.get("empty_result_retries", 0) or 0),
            int(persisted.get("empty_result_retries", 0) or 0),
        )
        timeout_retries = max(
            int(stats.get("timeout_retries", 0) or 0),
            int(persisted.get("timeout_retries", 0) or 0),
        )
        dispatch_count = max(
            self.sample_dispatch_count(subset, sample_id),
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

    def persist_retry_evidence(self, subset: str, sample_id: str) -> None:
        existing = self.sample_store.load_sample_runtime(subset, sample_id) or {}
        stages = existing.get("stages", {}) if isinstance(existing, dict) else {}
        failure = existing.get("failure", {}) if isinstance(existing, dict) else {}
        payload = build_sample_runtime_record(
            subset=subset,
            sample_id=sample_id,
            terminal_state=str(existing.get("terminal_state", "running") or "running"),
            required_stages=(stages.get("required", []) if isinstance(stages, dict) else []),
            completed_stages=(stages.get("completed", []) if isinstance(stages, dict) else []),
            diagnostics={},
            retry_summary=self.sample_retry_summary(subset, sample_id),
            timing=self.sample_timing_summary(subset, sample_id),
            failure_reason=(str(failure.get("reason", "")).strip() if isinstance(failure, dict) else ""),
            failure_details=(dict(failure.get("details", {})) if isinstance(failure, dict) else {}),
            failure_report_path=(str(failure.get("report_path", "")).strip() if isinstance(failure, dict) else ""),
        )
        if isinstance(existing.get("fallback"), dict):
            payload["fallback"] = dict(existing["fallback"])
        if isinstance(existing.get("export"), dict):
            payload["export"] = dict(existing["export"])
        self.sample_store.persist_sample_runtime(subset, sample_id, payload)

    def read_json_object(self, path_str: str) -> Dict[str, Any]:
        path = Path(path_str)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def persist_run_summary(self, subset: str) -> None:
        manifest_path = self.run_manifest_paths.get(subset, "")
        run_dir = self.run_dir_by_subset.get(subset, "")
        if not manifest_path or not run_dir:
            return
        try:
            manifest = load_run_manifest(manifest_path)
        except Exception as exc:
            self.logger.warning(f"[Warn] subset={subset}: failed to load run manifest for summary: {exc}")
            return

        sample_runtime_records = []
        for sample_id in self.sample_ids_by_subset.get(subset, []):
            sample_runtime = self.sample_store.load_sample_runtime(subset, sample_id)
            if sample_runtime is not None:
                sample_runtime_records.append(sample_runtime)

        summary = build_run_summary(
            run_manifest=manifest,
            sample_runtime_records=sample_runtime_records,
            total_samples=len(self.sample_ids_by_subset.get(subset, [])),
        )
        write_run_summary(run_dir, summary)

    def runtime_diagnostics_payload(self, diagnostics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = dict(diagnostics or {})
        for key, value in list(payload.items()):
            if not key.endswith("_fallback_used") or not bool(value):
                continue
            prefix = key[: -len("_fallback_used")]
            reason_key = f"{prefix}_fallback_reason"
            if str(payload.get(reason_key, "")).strip():
                continue
            legacy_reason_key = f"{prefix}_reason"
            legacy_reason = str(payload.get(legacy_reason_key, "")).strip()
            if legacy_reason:
                payload[reason_key] = legacy_reason
        return payload

    def build_sample_runtime(
        self,
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
            diagnostics=self.runtime_diagnostics_payload(diagnostics),
            retry_summary=self.sample_retry_summary(subset, sample_id),
            timing=self.sample_timing_summary(subset, sample_id),
            failure_reason=failure_reason,
            failure_details=dict(failure_details or {}),
            failure_report_path=(Path(self.sample_store.failure_report_path(subset, sample_id)).name if failure_reason else ""),
        )

    def _event_timestamp_fields(self) -> Dict[str, Any]:
        now = datetime.now().astimezone()
        return {
            "ts": now.isoformat(timespec="milliseconds"),
            "ts_unix_ms": int(now.timestamp() * 1000),
        }

    def persist_structured_event_record(self, event: str, **fields: Any) -> None:
        subset = str(fields.get("subset", "")).strip()
        sample_id = str(fields.get("sample_id", "")).strip()
        if not subset or not sample_id:
            return

        payload: Dict[str, Any] = {
            "event": str(event),
            **self._event_timestamp_fields(),
        }
        for key, value in fields.items():
            if value is None:
                continue
            payload[str(key)] = value
        self.sample_store.persist_event_record(subset, sample_id, payload)

    def sample_timing_summary(self, subset: str, sample_id: str) -> Dict[str, Any]:
        return dict(build_sample_timing_record(self.sample_store.load_sample_event_records(subset, sample_id)) or {})

    def ensure_stage_started(self, subset: str, sample_id: str, stage: str) -> None:
        stage_name = str(stage).strip()
        if not stage_name:
            return

        key = (str(subset), str(sample_id), stage_name)
        with self._stage_timing_lock:
            if key in self._stage_start_ts or key in self._stage_done_keys:
                return
            self._stage_start_ts[key] = time.perf_counter()

        log_event(
            self.logger,
            "sample_stage_start",
            subset=str(subset),
            sample_id=str(sample_id),
            stage=stage_name,
        )
        self.persist_structured_event_record(
            "sample_stage_start",
            subset=str(subset),
            sample_id=str(sample_id),
            stage=stage_name,
        )

    def mark_stage_done(self, subset: str, sample_id: str, stage: str) -> None:
        stage_name = str(stage).strip()
        if not stage_name:
            return

        key = (str(subset), str(sample_id), stage_name)
        with self._stage_timing_lock:
            if key in self._stage_done_keys:
                return
            start_ts = self._stage_start_ts.pop(key, None)
            self._stage_done_keys.add(key)

        elapsed_ms = 0
        if start_ts is not None:
            elapsed_ms = int(round((time.perf_counter() - start_ts) * 1000.0))

        log_event(
            self.logger,
            "sample_stage_done",
            subset=str(subset),
            sample_id=str(sample_id),
            stage=stage_name,
            elapsed_ms=elapsed_ms,
        )
        self.persist_structured_event_record(
            "sample_stage_done",
            subset=str(subset),
            sample_id=str(sample_id),
            stage=stage_name,
            elapsed_ms=elapsed_ms,
        )

    def with_minimal_done_export_diagnostics(
        self,
        *,
        required_stages: List[str],
        completed_stages: List[str],
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(diagnostics or {})
        required_set = {str(stage).strip() for stage in required_stages if str(stage).strip()}
        completed_set = {str(stage).strip() for stage in completed_stages if str(stage).strip()}
        export_required = "export" in required_set
        export_completed = "export" in completed_set
        has_export_evidence = any(str(key).startswith("export_") for key in payload)
        if export_required and export_completed and not has_export_evidence:
            payload["export_enabled"] = True
            payload["export_attempted"] = True
            payload["export_reason"] = "applied"
        return payload

    def ensure_sample_runtime_for_terminal_artifact(self, subset: str, sample_id: str) -> None:
        existing_runtime = self.sample_store.load_sample_runtime(subset, sample_id)
        done_exists = Path(self.done_marker_path(subset, sample_id)).exists()
        failed_exists = Path(self.failed_marker_path(subset, sample_id)).exists()
        if existing_runtime is not None:
            existing_state = str(existing_runtime.get("terminal_state", "")).strip()
            if (done_exists and existing_state == "done") or (failed_exists and existing_state == "failed"):
                return

        required_stages = self.required_stages_for_subset(subset)
        if done_exists:
            payload = self.read_json_object(self.segments_path(subset, sample_id))
            raw_diagnostics = dict(payload.get("diagnostics", {}))
            normalized_required_stages = list(raw_diagnostics.get("required_stages", required_stages))
            normalized_completed_stages = list(raw_diagnostics.get("completed_stages", normalized_required_stages))
            diagnostics = self.with_minimal_done_export_diagnostics(
                required_stages=normalized_required_stages,
                completed_stages=normalized_completed_stages,
                diagnostics=raw_diagnostics,
            )
            runtime_payload = self.build_sample_runtime(
                subset,
                sample_id,
                terminal_state="done",
                required_stages=normalized_required_stages,
                completed_stages=normalized_completed_stages,
                diagnostics=diagnostics,
            )
            self.sample_store.persist_sample_runtime(subset, sample_id, runtime_payload)
            return

        if Path(self.failed_marker_path(subset, sample_id)).exists():
            report = self.read_json_object(self.failure_report_path(subset, sample_id))
            details = dict(report.get("details", {}))
            diagnostics = dict(details.get("diagnostics", {})) if isinstance(details.get("diagnostics", {}), dict) else {}
            runtime_payload = self.build_sample_runtime(
                subset,
                sample_id,
                terminal_state="failed",
                required_stages=list(details.get("required_stages", required_stages)),
                completed_stages=list(details.get("completed_stages", [])),
                diagnostics=diagnostics,
                failure_reason=str(report.get("reason", "")).strip(),
                failure_details=details,
            )
            self.sample_store.persist_sample_runtime(subset, sample_id, runtime_payload)

    def persist_sample_failure(
        self,
        subset: str,
        sample_id: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details_payload = dict(details or {})
        runtime_payload = self.build_sample_runtime(
            subset,
            sample_id,
            terminal_state="failed",
            required_stages=list(details_payload.get("required_stages", self.required_stages_for_subset(subset))),
            completed_stages=list(details_payload.get("completed_stages", [])),
            diagnostics=dict(details_payload.get("diagnostics", {})) if isinstance(details_payload.get("diagnostics", {}), dict) else {},
            failure_reason=reason,
            failure_details=details_payload,
        )
        self.sample_store.persist_sample_failure(
            subset,
            sample_id,
            reason,
            details_payload,
            sample_runtime=runtime_payload,
            publish_failed_marker=False,
        )
        self.persist_run_summary(subset)
        self.sample_store.publish_failed_marker(subset, sample_id)

    def clear_sample_jobs(self, subset: str, sample_id: str) -> None:
        removed_task_ids: set[str] = set()
        with self.queue_lock:
            retained_jobs: List[Any] = []
            for job in self.job_queue:
                meta = job_payload_meta(job)
                if str(meta.get("subset", "")) == subset and str(meta.get("sample_id", "")) == sample_id:
                    task_id = job_payload_task_id(job)
                    if task_id:
                        removed_task_ids.add(task_id)
                    continue
                retained_jobs.append(job)
            if len(retained_jobs) != len(self.job_queue):
                self.job_queue[:] = retained_jobs

            inflight_task_ids = [
                str(task_id)
                for task_id, info in self.inflight.items()
                if str(job_payload_meta(info.get("job")).get("subset", "")) == subset
                and str(job_payload_meta(info.get("job")).get("sample_id", "")) == sample_id
            ]
            for task_id in inflight_task_ids:
                self.inflight.pop(task_id, None)
                removed_task_ids.add(task_id)

            for task_id in removed_task_ids:
                self.timeout_retry_counts.pop(task_id, None)
                self.empty_retry_counts.pop(task_id, None)
                self.dispatch_counts.pop(task_id, None)
                self.completed_dispatch_ids.pop(task_id, None)

    def clear_step_a_repeat_artifact_reuse_cache(self, subset: str, sample_id: str) -> None:
        self.step_a_repeat_artifact_reuse_caches.pop((subset, sample_id), None)

    def log_fallback_applied(self, subset: str, sample_id: str, diagnostics: Dict[str, Any]) -> None:
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
                self.logger,
                "fallback_applied",
                subset=subset,
                sample_id=sample_id,
                **fallback_fields,
            )
            self.persist_structured_event_record(
                "fallback_applied",
                subset=subset,
                sample_id=sample_id,
                **fallback_fields,
            )

    def fail_sample(
        self,
        state: Dict[str, Any],
        subset: str,
        sample_id: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
        *,
        log_message: str = "",
    ) -> None:
        self.clear_sample_jobs(subset, sample_id)
        self.clear_step_a_repeat_artifact_reuse_cache(subset, sample_id)
        self.persist_sample_failure(subset, sample_id, reason, details)
        log_event(
            self.logger,
            "sample_failed",
            subset=subset,
            sample_id=sample_id,
            reason=reason,
            details=details or {},
        )
        self.persist_structured_event_record(
            "sample_failed",
            subset=subset,
            sample_id=sample_id,
            reason=reason,
            details=details or {},
        )
        state["sample_status"][sample_id] = 4
        state["cur_idx"] += 1
        if log_message:
            self.logger.error(log_message)

    def mark_task_terminal_failure(
        self,
        task_id: str,
        dispatch_id: str,
        meta: Dict[str, Any],
        terminal_error: str,
    ) -> None:
        with self.queue_lock:
            if dispatch_id:
                self.completed_dispatch_ids[task_id] = dispatch_id
            self.timeout_retry_counts.pop(task_id, None)
            self.empty_retry_counts.pop(task_id, None)
        self.persist_result_record(
            task_id,
            dispatch_id,
            {},
            meta,
            terminal_error=terminal_error,
        )

    def required_stages_for_subset(self, subset: str) -> list[str]:
        manifest_path = self.run_manifest_paths.get(subset, "")
        if manifest_path:
            try:
                manifest = load_run_manifest(manifest_path)
                return [
                    str(stage).strip()
                    for stage in manifest.required_stages
                    if str(stage).strip()
                ]
            except Exception as exc:
                self.logger.warning(
                    f"[Warn] subset={subset}: failed to read required stages from manifest {manifest_path}: {exc}"
                )

        required_stages = ["stage1_segments"]
        if bool(self.config.llm_merge.enabled):
            required_stages.append("stage2_text")
        if bool(self.config.export.enabled):
            required_stages.append("export")
        return required_stages

    def required_stages_satisfied(self, required_stages: List[str], completed_stages: List[str]) -> bool:
        completed_stage_set = {str(stage).strip() for stage in completed_stages if str(stage).strip()}
        return all(str(stage).strip() in completed_stage_set for stage in required_stages)

    def result_payload_for_segments(self, final_res: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(final_res)
        payload.pop("diagnostics", None)
        return payload

    def persist_sample_writeback(
        self,
        subset: str,
        sample_id: str,
        final_res: Dict[str, Any],
        *,
        required_stages: List[str],
        completed_stages: List[str],
    ) -> Dict[str, Any]:
        diagnostics = dict(final_res.get("diagnostics", {}))
        payload_to_persist = self.result_payload_for_segments(final_res)
        self.log_fallback_applied(subset, sample_id, diagnostics)
        self.sample_store.persist_sample_payload(subset, sample_id, payload_to_persist)
        self.sample_store.persist_sample_runtime(
            subset,
            sample_id,
            self.build_sample_runtime(
                subset,
                sample_id,
                terminal_state="done",
                required_stages=required_stages,
                completed_stages=completed_stages,
                diagnostics=diagnostics,
            ),
        )
        self.persist_run_summary(subset)
        return payload_to_persist

    def mark_sample_done(
        self,
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
        diagnostics = dict(final_res.get("diagnostics", {}))
        payload_to_persist = self.result_payload_for_segments(final_res)
        self.log_fallback_applied(subset, sample_id, diagnostics)
        already_done = self.sample_store.finalize_sample_success(
            subset,
            sample_id,
            payload_to_persist,
            required_stages=required_stages,
            completed_stages=completed_stages,
            sample_runtime=self.build_sample_runtime(
                subset,
                sample_id,
                terminal_state="done",
                required_stages=required_stages,
                completed_stages=completed_stages,
                diagnostics=diagnostics,
            ),
            publish_done_marker=False,
        )
        self.persist_run_summary(subset)
        self.sample_store.publish_done_marker(subset, sample_id)

        state["sample_status"][sample_id] = 3
        state["cur_idx"] += 1

        if not already_done:
            global_done += 1
        finalize_ms = int(round((time.perf_counter() - finalize_start) * 1000.0))
        log_event(
            self.logger,
            "finalize_done",
            subset=subset,
            sample_id=sample_id,
            finalize_ms=finalize_ms,
            segment_count=len(payload_to_persist.get("segments", [])),
        )
        self.persist_structured_event_record(
            "finalize_done",
            subset=subset,
            sample_id=sample_id,
            finalize_ms=finalize_ms,
            segment_count=len(payload_to_persist.get("segments", [])),
        )
        self.logger.info(f"[Progress] {global_done}/{progress_total} (finished: {subset}/{sample_id})")
        return global_done

    def apply_stage2_text_writeback(self, sample_id: str, final_res: Dict[str, Any]) -> Dict[str, Any]:
        stage2_res = self.dependencies.run_llm_stage2_pass(
            sample_id,
            final_res.get("segments", []),
            self.config.llm_merge,
            target_language=str(self.config.export.subtitles.language),
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
        next_res["segments"] = self.dependencies.attach_stage2_subtitles_to_segments(
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

    def export_stage_succeeded(self, export_diagnostics: Dict[str, Any]) -> bool:
        return bool(export_diagnostics.get("export_attempted")) and str(export_diagnostics.get("export_reason", "")).strip() == "applied"


def build_runtime_state(config: Config, logger: Any, dependencies: RuntimeDependencies) -> ServerRuntimeState:
    dataset_ctxs = parse_datasets(config)
    run_dir_by_subset = {ctx.subset: ctx.run_dir for ctx in dataset_ctxs}
    sample_ids_by_subset = {ctx.subset: list(ctx.sample_ids) for ctx in dataset_ctxs}

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
        normalize_window_record=normalize_loaded_window_vlm_json,
        normalize_boundary_refinement_record=normalize_loaded_boundary_refinement_vlm_json,
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
        manifest_file = run_manifest_path(ctx.run_dir)
        previous_required_stages: list[str] = []
        if manifest_file.exists():
            try:
                previous_manifest = load_run_manifest(manifest_file)
                previous_required_stages = [
                    str(stage).strip()
                    for stage in previous_manifest.required_stages
                    if str(stage).strip()
                ]
            except Exception:
                previous_required_stages = []

        if bool(config.run.force_resume) and ("stage2_text" in previous_required_stages) and not bool(config.llm_merge.enabled):
            raise ValueError(
                f"Cannot force-resume subset={ctx.subset}: manifest requires stage2_text but current llm_merge.enabled=false"
            )

        manifest_status = ensure_run_manifest(
            run_dir=ctx.run_dir,
            subset=ctx.subset,
            data_root=ctx.data_root,
            config=config,
            force_resume=bool(config.run.force_resume),
            run_dir_nonempty_before_start=bool(ctx.run_dir_nonempty_before_prepare),
        )
        run_manifest_paths[ctx.subset] = str(manifest_file)
        run_manifest_status_by_subset[ctx.subset] = manifest_status.model_dump(mode="json")
        if manifest_status.resume.force_resume and manifest_status.resume.mismatch_fields:
            logger.warning(
                f"[Resume-Override] subset={ctx.subset} action={manifest_status.action} "
                f"mismatches={manifest_status.resume.mismatch_fields}"
            )

    runtime_state = ServerRuntimeState(
        config=config,
        logger=logger,
        dependencies=dependencies,
        dataset_ctxs=dataset_ctxs,
        queue_lock=queue_lock,
        job_queue=job_queue,
        inflight=inflight,
        timeout_retry_counts=timeout_retry_counts,
        empty_retry_counts=empty_retry_counts,
        dispatch_counts=dispatch_counts,
        sample_retry_stats=sample_retry_stats,
        completed_dispatch_ids=completed_dispatch_ids,
        step_a_repeat_artifact_reuse_caches=step_a_repeat_artifact_reuse_caches,
        task_artifact_writer=task_artifact_writer,
        sample_store=sample_store,
        job_builder=job_builder,
        run_dir_by_subset=run_dir_by_subset,
        sample_ids_by_subset=sample_ids_by_subset,
        run_manifest_paths=run_manifest_paths,
        run_manifest_status_by_subset=run_manifest_status_by_subset,
        step_a_producer_batch_limit=STEP_A_PRODUCER_BATCH_LIMIT,
    )
    job_builder.bind_event_recorder(lambda event, fields: runtime_state.persist_structured_event_record(event, **fields))
    return runtime_state
