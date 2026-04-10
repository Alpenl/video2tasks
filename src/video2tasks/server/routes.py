"""Route registration for the server app."""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException

from ..logging_utils import log_event
from .protocol import ProtocolValidationError, ResultEnvelope
from .runtime_state import (
    ServerRuntimeState,
    coerce_job_envelope,
    job_payload_meta,
    merge_result_meta,
    normalize_submitted_vlm_json,
    requeue_empty_result,
)


def register_routes(app: FastAPI, runtime_state: ServerRuntimeState) -> None:
    """Register API routes against a shared runtime facade."""

    @app.get("/get_job")
    def get_job() -> Dict[str, Any]:
        with runtime_state.queue_lock:
            if not runtime_state.job_queue:
                return {"status": "empty"}
            base_job_payload = runtime_state.job_queue.pop(0)
            try:
                base_job = coerce_job_envelope(base_job_payload)
            except ProtocolValidationError as exc:
                raise HTTPException(status_code=500, detail=f"invalid queued job payload: {exc}") from exc
            task_id = base_job.task_id
            runtime_state.dispatch_counts[task_id] = runtime_state.dispatch_counts.get(task_id, 0) + 1
            dispatch_id = f"d{runtime_state.dispatch_counts[task_id]}"
            dispatched_job = base_job.with_dispatch(dispatch_id)
            runtime_state.inflight[task_id] = {
                "ts": time.time(),
                "job": base_job,
                "dispatch_id": dispatch_id,
            }
            log_event(
                runtime_state.logger,
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
            runtime_state.persist_structured_event_record(
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
            result = ResultEnvelope.parse_payload(payload)
        except ProtocolValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        task_id = result.task_id
        dispatch_id = result.resolved_dispatch_id

        with runtime_state.queue_lock:
            accepted_dispatch_id = runtime_state.completed_dispatch_ids.get(task_id)
            inflight_info = runtime_state.inflight.get(task_id)

            if dispatch_id and accepted_dispatch_id == dispatch_id:
                return {"status": "already_received"}

            if not inflight_info:
                return {"status": "stale_ignored"}

            expected_dispatch_id = str(inflight_info.get("dispatch_id", ""))
            if not dispatch_id or dispatch_id != expected_dispatch_id:
                return {"status": "stale_ignored"}

            job_info = runtime_state.inflight.pop(task_id)

        authoritative_meta = job_payload_meta(job_info["job"])
        result_meta = merge_result_meta(authoritative_meta, dict(result.meta), dispatch_id)
        infer_ms = int(round(max(0.0, float(result.latency_s)) * 1000.0))
        log_event(
            runtime_state.logger,
            "infer_attempt",
            task_id=task_id,
            dispatch_id=dispatch_id,
            subset=str(result_meta.get("subset", "")),
            sample_id=str(result_meta.get("sample_id", "")),
            job_type=str(result_meta.get("job_type", "")),
            infer_ms=infer_ms,
        )
        runtime_state.persist_structured_event_record(
            "infer_attempt",
            task_id=task_id,
            dispatch_id=dispatch_id,
            subset=str(result_meta.get("subset", "")),
            sample_id=str(result_meta.get("sample_id", "")),
            job_type=str(result_meta.get("job_type", "")),
            infer_ms=infer_ms,
        )
        normalized_vlm_json = normalize_submitted_vlm_json(result.vlm_json, authoritative_meta)

        if not normalized_vlm_json:
            retry_subset = str(result_meta.get("subset", ""))
            retry_sample_id = str(result_meta.get("sample_id", ""))
            with runtime_state.queue_lock:
                attempt, requeued = requeue_empty_result(
                    runtime_state.job_queue,
                    runtime_state.empty_retry_counts,
                    task_id,
                    job_info["job"],
                    runtime_state.config.server.max_empty_retries_per_job,
                )
                limit = runtime_state.config.server.max_empty_retries_per_job
                limit_label = "inf" if limit <= 0 else str(limit)
                if requeued:
                    runtime_state.record_sample_retry(
                        retry_subset,
                        retry_sample_id,
                        "empty_result_retries",
                    )

            if requeued:
                submit_ms = int(round((time.perf_counter() - submit_start) * 1000.0))
                log_event(
                    runtime_state.logger,
                    "result_empty_retry",
                    task_id=task_id,
                    dispatch_id=dispatch_id,
                    subset=retry_subset,
                    sample_id=retry_sample_id,
                    job_type=str(result_meta.get("job_type", "")),
                    attempt=attempt,
                    retry_limit=limit_label,
                    infer_ms=infer_ms,
                    submit_ms=submit_ms,
                )
                runtime_state.persist_structured_event_record(
                    "result_empty_retry",
                    task_id=task_id,
                    dispatch_id=dispatch_id,
                    subset=retry_subset,
                    sample_id=retry_sample_id,
                    job_type=str(result_meta.get("job_type", "")),
                    attempt=attempt,
                    retry_limit=limit_label,
                    infer_ms=infer_ms,
                    submit_ms=submit_ms,
                )
                runtime_state.persist_retry_evidence(retry_subset, retry_sample_id)
                runtime_state.logger.warning(
                    f"[Warn] Task {task_id} empty or invalid, re-queueing to tail "
                    f"(empty attempt {attempt}/{limit_label})"
                )
                return {"status": "retry_triggered"}

            runtime_state.logger.error(
                f"[Err] Task {task_id} empty or invalid retry budget exhausted "
                f"(empty attempt {attempt}/{limit_label}); recording terminal empty result"
            )
            runtime_state.mark_task_terminal_failure(
                task_id,
                dispatch_id,
                result_meta,
                "empty_retry_exhausted",
            )
            return {"status": "empty_retry_exhausted"}

        with runtime_state.queue_lock:
            runtime_state.completed_dispatch_ids[task_id] = dispatch_id
            runtime_state.timeout_retry_counts.pop(task_id, None)
            runtime_state.empty_retry_counts.pop(task_id, None)

        runtime_state.persist_result_record(task_id, dispatch_id, normalized_vlm_json, result_meta)
        submit_ms = int(round((time.perf_counter() - submit_start) * 1000.0))
        log_event(
            runtime_state.logger,
            "job_done",
            task_id=task_id,
            dispatch_id=dispatch_id,
            subset=str(result_meta.get("subset", "")),
            sample_id=str(result_meta.get("sample_id", "")),
            job_type=str(result_meta.get("job_type", "")),
            infer_ms=infer_ms,
            submit_ms=submit_ms,
        )
        runtime_state.persist_structured_event_record(
            "job_done",
            task_id=task_id,
            dispatch_id=dispatch_id,
            subset=str(result_meta.get("subset", "")),
            sample_id=str(result_meta.get("sample_id", "")),
            job_type=str(result_meta.get("job_type", "")),
            infer_ms=infer_ms,
            submit_ms=submit_ms,
        )
        return {"status": "received"}

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}
