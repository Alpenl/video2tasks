import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def _resolve_repo_relative_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def test_official_smoke_demo_contract() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config.smoke.yaml"

    assert config_path.exists(), "Missing official smoke config: config.smoke.yaml"

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    datasets = config_payload.get("datasets") or []
    assert len(datasets) == 1, "Smoke demo must use exactly one dataset entry"
    dataset = datasets[0]
    subset = str(dataset["subset"])
    dataset_root = _resolve_repo_relative_path(repo_root, str(dataset["root"]))

    sample_id = "sample_001"
    sample_video = dataset_root / subset / sample_id / "Frame_demo.mp4"
    assert sample_video.exists(), f"Missing smoke fixture video: {sample_video}"

    run_cfg = config_payload.get("run") or {}
    run_base_dir = _resolve_repo_relative_path(repo_root, str(run_cfg["base_dir"]))
    run_id = str(run_cfg["run_id"])
    run_dir = run_base_dir / subset / run_id
    sample_out_dir = run_dir / "samples" / sample_id

    worker_cfg = config_payload.get("worker") or {}
    assert str(worker_cfg.get("backend")) == "dummy", "Smoke demo must force dummy backend"

    server_cfg = config_payload.get("server") or {}
    assert bool(server_cfg.get("auto_exit_after_all_done")) is True, (
        "Smoke demo must enable auto_exit_after_all_done for one-command completion"
    )

    shutil.rmtree(run_dir, ignore_errors=True)

    env = os.environ.copy()
    src_dir = repo_root / "src"
    env["PYTHONPATH"] = (
        f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(src_dir)
    )

    cmd = [
        sys.executable,
        "-m",
        "video2tasks.cli.cluster",
        "--config",
        str(config_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        "Official smoke command failed.\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )

    done_marker = sample_out_dir / ".DONE"
    failed_marker = sample_out_dir / ".FAILED"
    segments_path = sample_out_dir / "segments.json"
    run_manifest_path = run_dir / "run_manifest.json"

    assert done_marker.exists(), f"Missing done marker: {done_marker}"
    assert not failed_marker.exists(), f"Unexpected failed marker: {failed_marker}"
    assert segments_path.exists(), f"Missing sample segments output: {segments_path}"
    assert run_manifest_path.exists(), f"Missing run manifest: {run_manifest_path}"

    segments_payload = json.loads(segments_path.read_text(encoding="utf-8"))
    assert segments_payload.get("sample_id") == sample_id
    assert isinstance(segments_payload.get("nframes"), int)
    assert segments_payload["nframes"] > 0
    assert isinstance(segments_payload.get("segments"), list)
    assert len(segments_payload["segments"]) >= 1
    first_segment = segments_payload["segments"][0]
    for required_key in ("seg_id", "start_frame", "end_frame", "instruction"):
        assert required_key in first_segment

    run_manifest_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert run_manifest_payload.get("schema_version") == 1
    assert run_manifest_payload.get("subset") == subset
    assert run_manifest_payload.get("run_id") == run_id
    assert isinstance(run_manifest_payload.get("required_stages"), list)
    assert len(run_manifest_payload["required_stages"]) >= 1
    assert run_manifest_payload["backend_summary"]["stage1"]["backend"] == "dummy"
