import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import yaml


def _resolve_repo_relative_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_official_smoke_demo_contract(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config.smoke.yaml"

    assert config_path.exists(), "Missing official smoke config: config.smoke.yaml"
    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'v2t-cluster = "video2tasks.cli.cluster:main"' in pyproject_text, (
        "Missing console-script contract for v2t-cluster"
    )

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
    run_id = str(run_cfg["run_id"])

    worker_cfg = config_payload.get("worker") or {}
    assert str(worker_cfg.get("backend")) == "dummy", "Smoke demo must force dummy backend"

    server_cfg = config_payload.get("server") or {}
    assert bool(server_cfg.get("auto_exit_after_all_done")) is True, (
        "Smoke demo must enable auto_exit_after_all_done for one-command completion"
    )

    # Keep config.smoke.yaml as the public contract, but execute tests in an isolated
    # temporary run directory to avoid wiping/reusing the repo-local documented path.
    test_port = _pick_free_tcp_port()
    test_base_dir = tmp_path / "smoke_runs"
    test_run_id = f"{run_id}_it"
    test_cfg_payload = dict(config_payload)
    test_cfg_payload["datasets"] = [dict(dataset)]
    test_cfg_payload["datasets"][0]["root"] = str(dataset_root)
    test_cfg_payload["run"] = dict(run_cfg)
    test_cfg_payload["run"]["base_dir"] = str(test_base_dir)
    test_cfg_payload["run"]["run_id"] = test_run_id
    test_cfg_payload["server"] = dict(server_cfg)
    test_cfg_payload["server"]["host"] = "127.0.0.1"
    test_cfg_payload["server"]["port"] = int(test_port)
    test_cfg_payload["worker"] = dict(worker_cfg)
    test_cfg_payload["worker"]["server_url"] = f"http://127.0.0.1:{test_port}"

    test_config_path = tmp_path / "config.smoke.test.yaml"
    test_config_path.write_text(
        yaml.safe_dump(test_cfg_payload, sort_keys=False),
        encoding="utf-8",
    )

    run_dir = test_base_dir / subset / test_run_id
    sample_out_dir = run_dir / "samples" / sample_id

    env = os.environ.copy()
    src_dir = repo_root / "src"
    env["PYTHONPATH"] = (
        f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(src_dir)
    )

    cluster_bin = shutil.which("v2t-cluster")
    if cluster_bin:
        cmd = [cluster_bin, "--config", str(test_config_path)]
    else:
        cmd = [
            sys.executable,
            "-m",
            "video2tasks.cli.cluster",
            "--config",
            str(test_config_path),
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
        f"COMMAND: {' '.join(cmd)}\n"
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
    assert run_manifest_payload.get("run_id") == test_run_id
    assert isinstance(run_manifest_payload.get("required_stages"), list)
    assert len(run_manifest_payload["required_stages"]) >= 1
    assert run_manifest_payload["backend_summary"]["stage1"]["backend"] == "dummy"
