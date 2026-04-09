import json
from pathlib import Path

import pytest

from video2tasks.config import Config
from video2tasks.server.app import create_app


def _make_config(
    tmp_path: Path,
    *,
    backend: str = "dummy",
    boundary_prompt_mode: str = "freeform",
    force_resume: bool = False,
) -> Config:
    data_root = tmp_path / "data"
    (data_root / "demo").mkdir(parents=True, exist_ok=True)
    return Config(
        datasets=[{"root": str(data_root), "subset": "demo"}],
        run={
            "base_dir": str(tmp_path),
            "run_id": "testrun",
            "force_resume": force_resume,
        },
        server={"auto_exit_after_all_done": False},
        worker={"backend": backend},
        windowing={"boundary_prompt_mode": boundary_prompt_mode},
    )


def _manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "demo" / "testrun" / "run_manifest.json"


def _load_manifest(tmp_path: Path) -> dict:
    return json.loads(_manifest_path(tmp_path).read_text(encoding="utf-8"))


def _write_manifest(tmp_path: Path, payload: dict) -> None:
    _manifest_path(tmp_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _seed_existing_run_artifact(tmp_path: Path) -> None:
    sample_dir = tmp_path / "demo" / "testrun" / "samples" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / ".DONE").write_text("", encoding="utf-8")


def _seed_existing_run_dirs_only(tmp_path: Path) -> None:
    (tmp_path / "demo" / "testrun" / "samples" / "sample").mkdir(parents=True, exist_ok=True)


def test_create_app_writes_run_manifest_at_start(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path))

    manifest = _load_manifest(tmp_path)

    assert manifest["schema_version"] == 1
    assert manifest["deployment_mode"] == "single_machine_shared_fs"
    assert manifest["config_hash"]
    assert manifest["prompt_hash"]
    assert manifest["git_version"]
    assert manifest["backend_summary"]["stage1"]["backend"] == "dummy"
    assert manifest["resume"]["force_resume"] is False
    assert manifest["resume"]["mismatch_fields"] == []


def test_create_app_exposes_fresh_manifest_status_contract(tmp_path: Path) -> None:
    app = create_app(_make_config(tmp_path))

    status = app.state.run_manifest_status_by_subset["demo"]

    assert status["action"] == "created_fresh_manifest"
    assert status["manifest_present_before_start"] is False
    assert status["run_dir_nonempty_before_start"] is False
    assert status["path"] == str(_manifest_path(tmp_path))
    assert status["resume"]["validated_against_existing_manifest"] is False
    assert status["resume"]["mismatch_fields"] == []


def test_create_app_rejects_resume_when_run_manifest_identity_mismatches(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path, backend="dummy"))

    with pytest.raises(ValueError, match="backend_summary"):
        create_app(_make_config(tmp_path, backend="openai"))


def test_create_app_exposes_validated_resume_status_for_identity_match(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path))

    app = create_app(_make_config(tmp_path))

    status = app.state.run_manifest_status_by_subset["demo"]

    assert status["action"] == "validated_existing_manifest"
    assert status["manifest_present_before_start"] is True
    assert status["resume"]["validated_against_existing_manifest"] is True
    assert status["resume"]["force_resume"] is False
    assert status["resume"]["mismatch_fields"] == []


def test_create_app_allows_forced_resume_and_records_override(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path, backend="dummy"))

    create_app(_make_config(tmp_path, backend="openai", force_resume=True))

    manifest = _load_manifest(tmp_path)

    assert manifest["backend_summary"]["stage1"]["backend"] == "openai"
    assert manifest["resume"]["force_resume"] is True
    assert "backend_summary" in manifest["resume"]["mismatch_fields"]
    assert manifest["resume"]["previous_config_hash"]


def test_create_app_force_resume_records_exact_identity_mismatch_surface(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path))
    manifest = _load_manifest(tmp_path)
    manifest["deployment_mode"] = "legacy_mode"
    manifest["run_id"] = "legacy_run"
    manifest["required_stages"] = ["stage1_segments"]
    _write_manifest(tmp_path, manifest)

    app = create_app(_make_config(tmp_path, force_resume=True))

    status = app.state.run_manifest_status_by_subset["demo"]
    resume_status = status["resume"]

    assert status["action"] == "force_resume_overrode_identity_mismatch"
    assert resume_status["validated_against_existing_manifest"] is True
    assert resume_status["force_resume"] is True
    assert resume_status["mismatch_fields"] == [
        "deployment_mode",
        "run_id",
    ]


def test_create_app_rejects_resume_when_config_hash_mismatches(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path, boundary_prompt_mode="freeform"))

    with pytest.raises(ValueError, match="config_hash"):
        create_app(_make_config(tmp_path, boundary_prompt_mode="center_scan"))


def test_create_app_rejects_resume_when_required_stages_mismatch(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path))

    cfg = _make_config(tmp_path)
    cfg.export.enabled = True

    with pytest.raises(ValueError, match="required_stages"):
        create_app(cfg)


def test_create_app_rejects_resume_when_prompt_hash_mismatches(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path))
    manifest = _load_manifest(tmp_path)
    manifest["prompt_hash"] = "tampered-prompt-hash"
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="prompt_hash"):
        create_app(_make_config(tmp_path))


def test_create_app_rejects_resume_when_git_version_mismatches(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path))
    manifest = _load_manifest(tmp_path)
    manifest["git_version"]["git_commit"] = "deadbeef"
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="git_version"):
        create_app(_make_config(tmp_path))


def test_create_app_rejects_resume_when_manifest_is_malformed(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path))
    _manifest_path(tmp_path).write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_schema"):
        create_app(_make_config(tmp_path))


def test_create_app_force_resume_overrides_malformed_manifest(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path))
    _manifest_path(tmp_path).write_text("{not valid json", encoding="utf-8")

    create_app(_make_config(tmp_path, force_resume=True))

    manifest = _load_manifest(tmp_path)

    assert manifest["resume"]["force_resume"] is True
    assert manifest["resume"]["mismatch_fields"] == ["manifest_schema"]


def test_create_app_rejects_resume_when_manifest_schema_is_invalid(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path))
    manifest = _load_manifest(tmp_path)
    manifest.pop("config_hash")
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="manifest_schema"):
        create_app(_make_config(tmp_path))


def test_create_app_force_resume_overrides_schema_invalid_manifest(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path))
    manifest = _load_manifest(tmp_path)
    manifest["required_stages"] = "stage1_segments"
    _write_manifest(tmp_path, manifest)

    create_app(_make_config(tmp_path, force_resume=True))

    manifest = _load_manifest(tmp_path)

    assert manifest["resume"]["force_resume"] is True
    assert manifest["resume"]["mismatch_fields"] == ["manifest_schema"]


def test_create_app_rejects_nonempty_run_dir_without_manifest(tmp_path: Path) -> None:
    _seed_existing_run_artifact(tmp_path)

    with pytest.raises(ValueError, match="missing_manifest"):
        create_app(_make_config(tmp_path))


def test_create_app_force_resume_allows_nonempty_run_dir_without_manifest_and_records_reason(tmp_path: Path) -> None:
    _seed_existing_run_artifact(tmp_path)

    app = create_app(_make_config(tmp_path, force_resume=True))

    manifest = _load_manifest(tmp_path)

    assert manifest["resume"]["force_resume"] is True
    assert manifest["resume"]["mismatch_fields"] == ["missing_manifest"]
    assert app.state.run_manifest_status_by_subset["demo"]["resume"]["mismatch_fields"] == ["missing_manifest"]


def test_create_app_rejects_resume_when_remote_api_headers_change(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.worker.backend = "remote_api"
    config.worker.remote_api.headers = {"X-Tenant": "alpha"}
    create_app(config)

    changed = _make_config(tmp_path)
    changed.worker.backend = "remote_api"
    changed.worker.remote_api.headers = {"X-Tenant": "beta"}

    with pytest.raises(ValueError, match="config_hash"):
        create_app(changed)


def test_create_app_exposes_force_resume_override_summary_in_app_state(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path, backend="dummy"))

    app = create_app(_make_config(tmp_path, backend="openai", force_resume=True))

    resume_status = app.state.run_manifest_status_by_subset["demo"]["resume"]

    assert resume_status["force_resume"] is True
    assert "backend_summary" in resume_status["mismatch_fields"]


def test_create_app_rejects_directory_only_nonempty_run_dir_without_manifest(tmp_path: Path) -> None:
    _seed_existing_run_dirs_only(tmp_path)

    with pytest.raises(ValueError, match="missing_manifest"):
        create_app(_make_config(tmp_path))


def test_create_app_force_resume_allows_directory_only_nonempty_run_dir_without_manifest(tmp_path: Path) -> None:
    _seed_existing_run_dirs_only(tmp_path)

    app = create_app(_make_config(tmp_path, force_resume=True))

    assert app.state.run_manifest_status_by_subset["demo"]["resume"]["mismatch_fields"] == ["missing_manifest"]
