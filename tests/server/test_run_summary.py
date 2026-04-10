import json
from pathlib import Path

from video2tasks.config import Config
from video2tasks.server.run_manifest import build_run_manifest
from video2tasks.server.run_summary import (
    build_run_summary,
    build_sample_runtime_record,
    build_sample_timing_record,
    run_summary_path,
    write_run_summary,
)


def _build_manifest(tmp_path: Path, *, export_enabled: bool = False):
    data_root = tmp_path / "data"
    run_dir = tmp_path / "demo" / "testrun"
    config = Config(
        datasets=[{"root": str(data_root), "subset": "demo"}],
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        export={"enabled": export_enabled},
    )
    return build_run_manifest(
        run_dir=run_dir,
        subset="demo",
        data_root=str(data_root),
        config=config,
    )


def test_build_sample_runtime_record_captures_terminal_evidence() -> None:
    payload = build_sample_runtime_record(
        subset="demo",
        sample_id="sample",
        terminal_state="done",
        required_stages=["stage1_segments"],
        completed_stages=["stage1_segments"],
        diagnostics={
            "selection_policy": "light_cleanup_fallback",
            "llm_summary_fallback_used": True,
            "llm_summary_fallback_reason": "empty_response",
            "export_enabled": False,
            "export_attempted": False,
            "export_reason": "disabled",
        },
        retry_summary={
            "total_retries": 2,
            "empty_result_retries": 1,
            "timeout_retries": 1,
            "dispatch_count": 3,
        },
    )

    assert payload["terminal_state"] == "done"
    assert payload["stages"]["pending"] == []
    assert payload["fallback"]["applied"] is True
    assert payload["fallback"]["reasons"] == ["empty_response"]
    assert payload["retry"]["total_retries"] == 2
    assert payload["export"]["status"] == "disabled"
    assert payload["failure"] is None


def test_build_sample_runtime_record_references_failure_details() -> None:
    payload = build_sample_runtime_record(
        subset="demo",
        sample_id="sample",
        terminal_state="failed",
        required_stages=["stage1_segments", "export"],
        completed_stages=["stage1_segments"],
        diagnostics={
            "export_enabled": True,
            "export_attempted": True,
            "export_reason": "failed",
        },
        retry_summary={"dispatch_count": 1},
        failure_reason="export_failed",
        failure_details={"stage": "export"},
        failure_report_path="failure.json",
    )

    assert payload["terminal_state"] == "failed"
    assert payload["stages"]["pending"] == ["export"]
    assert payload["export"]["status"] == "failed"
    assert payload["failure"] == {
        "reason": "export_failed",
        "details": {"stage": "export"},
        "report_path": "failure.json",
    }


def test_build_sample_timing_record_aggregates_stage_and_request_elapsed() -> None:
    payload = build_sample_timing_record(
        [
            {
                "event": "sample_stage_start",
                "subset": "demo",
                "sample_id": "sample",
                "stage": "stage1_segments",
                "ts": "2026-04-10T12:00:00.000+08:00",
                "ts_unix_ms": 1000,
            },
            {
                "event": "artifact_extract_done",
                "subset": "demo",
                "sample_id": "sample",
                "task_id": "demo::sample_w0_r0",
                "job_type": "window_boundary",
                "artifact_extract_ms": 120,
                "image_count": 8,
                "transport_mode": "shared_fs",
                "artifact_reuse": False,
                "ts": "2026-04-10T12:00:00.200+08:00",
                "ts_unix_ms": 1200,
            },
            {
                "event": "infer_attempt",
                "subset": "demo",
                "sample_id": "sample",
                "task_id": "demo::sample_w0_r0",
                "dispatch_id": "d1",
                "job_type": "window_boundary",
                "infer_ms": 2300,
                "ts": "2026-04-10T12:00:02.700+08:00",
                "ts_unix_ms": 3700,
            },
            {
                "event": "job_done",
                "subset": "demo",
                "sample_id": "sample",
                "task_id": "demo::sample_w0_r0",
                "dispatch_id": "d1",
                "job_type": "window_boundary",
                "infer_ms": 2300,
                "submit_ms": 40,
                "ts": "2026-04-10T12:00:02.740+08:00",
                "ts_unix_ms": 3740,
            },
            {
                "event": "sample_stage_done",
                "subset": "demo",
                "sample_id": "sample",
                "stage": "stage1_segments",
                "elapsed_ms": 4200,
                "ts": "2026-04-10T12:00:04.200+08:00",
                "ts_unix_ms": 5200,
            },
            {
                "event": "finalize_done",
                "subset": "demo",
                "sample_id": "sample",
                "finalize_ms": 180,
                "segment_count": 7,
                "ts": "2026-04-10T12:00:04.400+08:00",
                "ts_unix_ms": 5400,
            },
        ]
    )

    assert payload == {
        "first_event_at": "2026-04-10T12:00:00.000+08:00",
        "last_event_at": "2026-04-10T12:00:04.400+08:00",
        "total_elapsed_ms": 4400,
        "stage_elapsed_ms": {"stage1_segments": 4200},
        "job_elapsed_ms": {
            "artifact_extract_total": 120,
            "infer_total": 2300,
            "submit_total": 40,
            "finalize_total": 180,
        },
        "event_counts": {
            "sample_stage_start": 1,
            "artifact_extract_done": 1,
            "infer_attempt": 1,
            "job_done": 1,
            "sample_stage_done": 1,
            "finalize_done": 1,
        },
    }


def test_build_sample_runtime_record_includes_timing_payload() -> None:
    payload = build_sample_runtime_record(
        subset="demo",
        sample_id="sample",
        terminal_state="done",
        required_stages=["stage1_segments"],
        completed_stages=["stage1_segments"],
        diagnostics={},
        retry_summary={"dispatch_count": 1},
        timing={
            "first_event_at": "2026-04-10T12:00:00.000+08:00",
            "last_event_at": "2026-04-10T12:00:04.400+08:00",
            "total_elapsed_ms": 4400,
            "stage_elapsed_ms": {"stage1_segments": 4200},
            "job_elapsed_ms": {
                "artifact_extract_total": 120,
                "infer_total": 2300,
                "submit_total": 40,
                "finalize_total": 180,
            },
            "event_counts": {"finalize_done": 1},
        },
    )

    assert payload["timing"]["total_elapsed_ms"] == 4400
    assert payload["timing"]["stage_elapsed_ms"] == {"stage1_segments": 4200}
    assert payload["timing"]["job_elapsed_ms"]["infer_total"] == 2300
    assert payload["timing"]["event_counts"] == {"finalize_done": 1}


def test_build_sample_runtime_record_preserves_export_failure_diagnostics_from_failure_details() -> None:
    payload = build_sample_runtime_record(
        subset="demo",
        sample_id="sample",
        terminal_state="failed",
        required_stages=["stage1_segments", "export"],
        completed_stages=["stage1_segments"],
        diagnostics={},
        retry_summary={"dispatch_count": 2, "total_retries": 1},
        failure_reason="export_failed",
        failure_details={
            "stage": "export",
            "export_enabled": True,
            "export_attempted": True,
            "export_mode": "clips",
            "export_reason": "failed_before_export_completion",
            "export_error": "ffmpeg exploded",
            "export_errors": ["clips:degraded"],
        },
        failure_report_path="failure.json",
    )

    assert payload["export"] == {
        "required": True,
        "enabled": True,
        "attempted": True,
        "status": "failed",
        "reason": "failed_before_export_completion",
        "mode": "clips",
        "errors": ["clips:degraded"],
        "error": "ffmpeg exploded",
    }


def test_build_run_summary_aggregates_terminal_runtime_records(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, export_enabled=False)
    summary = build_run_summary(
        run_manifest=manifest,
        sample_runtime_records=[
            build_sample_runtime_record(
                subset="demo",
                sample_id="sample_success",
                terminal_state="done",
                required_stages=["stage1_segments"],
                completed_stages=["stage1_segments"],
                diagnostics={
                    "selection_policy": "light_cleanup_fallback",
                    "llm_summary_fallback_used": True,
                    "llm_summary_fallback_reason": "empty_response",
                    "export_enabled": False,
                    "export_attempted": False,
                    "export_reason": "disabled",
                },
                retry_summary={
                    "total_retries": 2,
                    "empty_result_retries": 1,
                    "timeout_retries": 1,
                    "dispatch_count": 3,
                },
            ),
            build_sample_runtime_record(
                subset="demo",
                sample_id="sample_failed",
                terminal_state="failed",
                required_stages=["stage1_segments"],
                completed_stages=[],
                diagnostics={
                    "export_enabled": False,
                    "export_attempted": False,
                    "export_reason": "disabled",
                },
                retry_summary={"dispatch_count": 1},
                failure_reason="window_boundary_failed",
                failure_report_path="failure.json",
            ),
        ],
        total_samples=3,
    )

    assert summary["sample_counts"] == {"total": 3, "done": 1, "failed": 1, "pending": 1}
    assert summary["fallback"]["applied_sample_count"] == 1
    assert summary["fallback"]["reason_counts"] == {"empty_response": 1}
    assert summary["retry"] == {
        "samples_with_retries": 1,
        "total_retries": 2,
        "empty_result_retries": 1,
        "timeout_retries": 1,
    }
    assert summary["export"]["status_counts"] == {"disabled": 2}
    assert summary["failure_reasons"] == {"window_boundary_failed": 1}
    assert summary["stage_completion"]["stage1_segments"] == {"completed": 1, "missing": 2}


def test_write_run_summary_persists_json_artifact(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, export_enabled=True)
    summary = build_run_summary(
        run_manifest=manifest,
        sample_runtime_records=[
            build_sample_runtime_record(
                subset="demo",
                sample_id="sample_success",
                terminal_state="done",
                required_stages=["stage1_segments", "export"],
                completed_stages=["stage1_segments", "export"],
                diagnostics={
                    "export_enabled": True,
                    "export_attempted": True,
                    "export_reason": "applied",
                },
                retry_summary={"dispatch_count": 1},
            ),
            build_sample_runtime_record(
                subset="demo",
                sample_id="sample_failed",
                terminal_state="failed",
                required_stages=["stage1_segments", "export"],
                completed_stages=["stage1_segments"],
                diagnostics={
                    "export_enabled": True,
                    "export_attempted": True,
                    "export_reason": "failed",
                },
                retry_summary={"dispatch_count": 2, "total_retries": 1},
                failure_reason="export_failed",
                failure_report_path="failure.json",
            ),
        ],
        total_samples=2,
    )

    write_run_summary(manifest.run_dir, summary)

    payload = json.loads(run_summary_path(manifest.run_dir).read_text(encoding="utf-8"))
    assert payload["export"]["status_counts"] == {"applied": 1, "failed": 1}
    assert payload["failure_reasons"] == {"export_failed": 1}
