"""Runtime evidence helpers for sample-level and run-level summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .run_manifest import RunManifest


RUN_SUMMARY_FILENAME = "run_summary.json"
SAMPLE_RUNTIME_SCHEMA_VERSION = 1
RUN_SUMMARY_SCHEMA_VERSION = 1


def _normalize_stage_names(stages: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for stage in stages:
        name = str(stage).strip()
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return normalized


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return max(default, int(value))
    except (TypeError, ValueError):
        return default


def _fallback_overview(diagnostics: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    reasons: list[str] = []

    if str(diagnostics.get("selection_policy", "")).strip() == "light_cleanup_fallback":
        fields["selection_policy"] = "light_cleanup_fallback"

    for key, value in diagnostics.items():
        if key.endswith("_fallback_used") and bool(value):
            fields[key] = True
            reason_key = key.replace("_fallback_used", "_fallback_reason")
            reason = str(diagnostics.get(reason_key, "")).strip()
            if reason:
                fields[reason_key] = reason
                reasons.append(reason)
        elif key.endswith("used_subtitle_fallback") and bool(value):
            fields[key] = True

    unique_reasons = sorted(set(reasons))
    return {
        "applied": bool(fields),
        "reasons": unique_reasons,
        "fields": fields,
    }


def _export_overview(
    required_stages: list[str],
    diagnostics: dict[str, Any],
    *,
    failure_reason: str = "",
) -> dict[str, Any]:
    export_required = "export" in required_stages
    export_enabled = bool(diagnostics.get("export_enabled", export_required))
    export_attempted = bool(diagnostics.get("export_attempted", False))
    export_reason = str(diagnostics.get("export_reason", "")).strip()

    if export_reason == "applied":
        status = "applied"
    elif failure_reason == "export_failed" or export_reason in {"failed", "partial_failure", "failed_before_export_completion"}:
        status = "failed"
    elif not export_enabled:
        status = "disabled"
    elif export_required and not export_attempted:
        status = "not_attempted"
    elif not export_required:
        status = "not_required"
    else:
        status = "unknown"

    payload = {
        "required": export_required,
        "enabled": export_enabled,
        "attempted": export_attempted,
        "status": status,
        "reason": export_reason or status,
    }

    errors = diagnostics.get("export_errors", [])
    if isinstance(errors, list) and errors:
        payload["errors"] = [str(item).strip() for item in errors if str(item).strip()]
    error = str(diagnostics.get("export_error", "")).strip()
    if error:
        payload["error"] = error
    return payload


def build_sample_runtime_record(
    *,
    subset: str,
    sample_id: str,
    terminal_state: str,
    required_stages: Iterable[str],
    completed_stages: Iterable[str],
    diagnostics: dict[str, Any] | None = None,
    retry_summary: dict[str, Any] | None = None,
    failure_reason: str = "",
    failure_details: dict[str, Any] | None = None,
    failure_report_path: str = "",
) -> dict[str, Any]:
    normalized_required = _normalize_stage_names(required_stages)
    normalized_completed = _normalize_stage_names(completed_stages)
    pending_stages = [
        stage
        for stage in normalized_required
        if stage not in normalized_completed
    ]
    diagnostics_payload = dict(diagnostics or {})
    retry_payload = dict(retry_summary or {})
    normalized_failure_reason = str(failure_reason).strip()
    normalized_failure_path = str(failure_report_path).strip()
    details_payload = dict(failure_details or {})

    failure_payload: dict[str, Any] | None = None
    if normalized_failure_reason or normalized_failure_path or details_payload:
        failure_payload = {}
        if normalized_failure_reason:
            failure_payload["reason"] = normalized_failure_reason
        if details_payload:
            failure_payload["details"] = details_payload
        if normalized_failure_path:
            failure_payload["report_path"] = normalized_failure_path

    total_retries = _int_value(retry_payload.get("total_retries"))
    empty_retries = _int_value(retry_payload.get("empty_result_retries"))
    timeout_retries = _int_value(retry_payload.get("timeout_retries"))
    dispatch_count = _int_value(retry_payload.get("dispatch_count"))
    if dispatch_count <= 0:
        dispatch_count = total_retries + 1 if terminal_state else 0

    return {
        "schema_version": SAMPLE_RUNTIME_SCHEMA_VERSION,
        "subset": str(subset),
        "sample_id": str(sample_id),
        "terminal_state": str(terminal_state).strip() or "unknown",
        "stages": {
            "required": normalized_required,
            "completed": normalized_completed,
            "pending": pending_stages,
        },
        "fallback": _fallback_overview(diagnostics_payload),
        "retry": {
            "total_retries": total_retries,
            "empty_result_retries": empty_retries,
            "timeout_retries": timeout_retries,
            "dispatch_count": dispatch_count,
        },
        "export": _export_overview(
            normalized_required,
            diagnostics_payload,
            failure_reason=normalized_failure_reason,
        ),
        "failure": failure_payload,
    }


def build_run_summary(
    *,
    run_manifest: RunManifest,
    sample_runtime_records: Iterable[dict[str, Any]],
    total_samples: int | None = None,
) -> dict[str, Any]:
    records = [dict(record) for record in sample_runtime_records if isinstance(record, dict)]
    normalized_total = _int_value(total_samples, 0)
    total = max(normalized_total, len(records))

    done_count = sum(1 for record in records if str(record.get("terminal_state", "")).strip() == "done")
    failed_count = sum(1 for record in records if str(record.get("terminal_state", "")).strip() == "failed")
    pending_count = max(0, total - done_count - failed_count)

    required_stages = _normalize_stage_names(run_manifest.required_stages)
    stage_completion: dict[str, dict[str, int]] = {}
    for stage in required_stages:
        completed = 0
        for record in records:
            stages = record.get("stages", {})
            completed_stages = stages.get("completed", []) if isinstance(stages, dict) else []
            if stage in {str(item).strip() for item in completed_stages if str(item).strip()}:
                completed += 1
        stage_completion[stage] = {
            "completed": completed,
            "missing": max(0, total - completed),
        }

    fallback_reason_counts: dict[str, int] = {}
    fallback_applied_sample_count = 0
    retry_totals = {
        "samples_with_retries": 0,
        "total_retries": 0,
        "empty_result_retries": 0,
        "timeout_retries": 0,
    }
    export_status_counts: dict[str, int] = {}
    failure_reasons: dict[str, int] = {}

    for record in records:
        fallback = record.get("fallback", {}) if isinstance(record.get("fallback"), dict) else {}
        if bool(fallback.get("applied")):
            fallback_applied_sample_count += 1
        for reason in fallback.get("reasons", []) if isinstance(fallback.get("reasons"), list) else []:
            normalized_reason = str(reason).strip()
            if normalized_reason:
                fallback_reason_counts[normalized_reason] = fallback_reason_counts.get(normalized_reason, 0) + 1

        retry = record.get("retry", {}) if isinstance(record.get("retry"), dict) else {}
        total_retries_value = _int_value(retry.get("total_retries"))
        if total_retries_value > 0:
            retry_totals["samples_with_retries"] += 1
        retry_totals["total_retries"] += total_retries_value
        retry_totals["empty_result_retries"] += _int_value(retry.get("empty_result_retries"))
        retry_totals["timeout_retries"] += _int_value(retry.get("timeout_retries"))

        export = record.get("export", {}) if isinstance(record.get("export"), dict) else {}
        export_status = str(export.get("status", "")).strip()
        if export_status:
            export_status_counts[export_status] = export_status_counts.get(export_status, 0) + 1

        failure = record.get("failure", {}) if isinstance(record.get("failure"), dict) else {}
        failure_reason = str(failure.get("reason", "")).strip()
        if failure_reason:
            failure_reasons[failure_reason] = failure_reasons.get(failure_reason, 0) + 1

    return {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "run_id": run_manifest.run_id,
        "subset": run_manifest.subset,
        "run_dir": run_manifest.run_dir,
        "required_stages": required_stages,
        "sample_counts": {
            "total": total,
            "done": done_count,
            "failed": failed_count,
            "pending": pending_count,
        },
        "stage_completion": stage_completion,
        "fallback": {
            "applied_sample_count": fallback_applied_sample_count,
            "reason_counts": fallback_reason_counts,
        },
        "retry": retry_totals,
        "export": {
            "required": "export" in required_stages,
            "status_counts": export_status_counts,
        },
        "failure_reasons": failure_reasons,
    }


def run_summary_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / RUN_SUMMARY_FILENAME


def write_run_summary(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    path = run_summary_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


__all__ = [
    "RUN_SUMMARY_FILENAME",
    "build_run_summary",
    "build_sample_runtime_record",
    "run_summary_path",
    "write_run_summary",
]
