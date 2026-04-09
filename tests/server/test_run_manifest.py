import json
from pathlib import Path
from typing import Callable

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


def _status_for_demo_subset(app: object) -> dict:
    return app.state.run_manifest_status_by_subset["demo"]


def test_create_app_writes_operator_facing_manifest_and_status_for_fresh_run(tmp_path: Path) -> None:
    app = create_app(_make_config(tmp_path))

    manifest = _load_manifest(tmp_path)
    status = _status_for_demo_subset(app)

    assert manifest["schema_version"] == 1
    assert manifest["deployment_mode"] == "single_machine_shared_fs"
    assert manifest["run_id"] == "testrun"
    assert manifest["subset"] == "demo"
    assert manifest["backend_summary"]["stage1"]["backend"] == "dummy"
    assert manifest["required_stages"] == ["stage1_segments"]
    assert manifest["resume"]["force_resume"] is False
    assert manifest["resume"]["mismatch_fields"] == []

    assert status["action"] == "created_fresh_manifest"
    assert status["manifest_present_before_start"] is False
    assert status["run_dir_nonempty_before_start"] is False
    assert status["path"] == str(_manifest_path(tmp_path))
    assert status["resume"]["validated_against_existing_manifest"] is False
    assert status["resume"]["mismatch_fields"] == []


def test_create_app_validates_existing_manifest_for_same_resume_identity(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path))

    app = create_app(_make_config(tmp_path))
    status = _status_for_demo_subset(app)

    assert status["action"] == "validated_existing_manifest"
    assert status["manifest_present_before_start"] is True
    assert status["resume"]["validated_against_existing_manifest"] is True
    assert status["resume"]["force_resume"] is False
    assert status["resume"]["mismatch_fields"] == []


def test_create_app_rejects_resume_when_backend_identity_changes(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path, backend="dummy"))

    with pytest.raises(ValueError, match="backend_summary"):
        create_app(_make_config(tmp_path, backend="openai"))


@pytest.mark.parametrize(
    ("mutate_manifest", "expected_mismatch"),
    [
        (
            lambda manifest: {**manifest, "prompt_hash": "tampered-prompt-hash"},
            "prompt_hash",
        ),
        (
            lambda manifest: {
                **manifest,
                "git_version": {**manifest["git_version"], "git_commit": "deadbeef"},
            },
            "git_version",
        ),
    ],
)
def test_create_app_rejects_resume_when_manifest_identity_markers_are_tampered(
    tmp_path: Path,
    mutate_manifest: Callable[[dict], dict],
    expected_mismatch: str,
) -> None:
    create_app(_make_config(tmp_path))
    _write_manifest(tmp_path, mutate_manifest(_load_manifest(tmp_path)))

    with pytest.raises(ValueError, match=expected_mismatch):
        create_app(_make_config(tmp_path))


def test_create_app_rejects_resume_when_config_hash_changes(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path, boundary_prompt_mode="freeform"))

    with pytest.raises(ValueError, match="config_hash"):
        create_app(_make_config(tmp_path, boundary_prompt_mode="center_scan"))


def test_create_app_rejects_resume_when_required_stages_change(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path))

    changed = _make_config(tmp_path)
    changed.export.enabled = True

    with pytest.raises(ValueError, match="required_stages"):
        create_app(changed)


def test_create_app_rejects_resume_when_remote_api_identity_headers_change(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.worker.backend = "remote_api"
    config.worker.remote_api.headers = {"X-Tenant": "alpha"}
    create_app(config)

    changed = _make_config(tmp_path)
    changed.worker.backend = "remote_api"
    changed.worker.remote_api.headers = {"X-Tenant": "beta"}

    with pytest.raises(ValueError, match="config_hash"):
        create_app(changed)


def test_create_app_force_resume_overrides_backend_mismatch_and_records_resume_surface(
    tmp_path: Path,
) -> None:
    create_app(_make_config(tmp_path, backend="dummy"))

    app = create_app(_make_config(tmp_path, backend="openai", force_resume=True))
    manifest = _load_manifest(tmp_path)
    status = _status_for_demo_subset(app)

    assert manifest["backend_summary"]["stage1"]["backend"] == "openai"
    assert manifest["resume"]["force_resume"] is True
    assert "backend_summary" in manifest["resume"]["mismatch_fields"]
    assert manifest["resume"]["previous_config_hash"]

    assert status["action"] == "force_resume_overrode_identity_mismatch"
    assert status["resume"]["validated_against_existing_manifest"] is True
    assert status["resume"]["force_resume"] is True
    assert "backend_summary" in status["resume"]["mismatch_fields"]


def test_create_app_force_resume_records_exact_identity_mismatch_fields(tmp_path: Path) -> None:
    create_app(_make_config(tmp_path))
    manifest = _load_manifest(tmp_path)
    manifest["deployment_mode"] = "legacy_mode"
    manifest["run_id"] = "legacy_run"
    manifest["required_stages"] = ["stage1_segments"]
    _write_manifest(tmp_path, manifest)

    app = create_app(_make_config(tmp_path, force_resume=True))
    status = _status_for_demo_subset(app)

    assert status["action"] == "force_resume_overrode_identity_mismatch"
    assert set(status["resume"]["mismatch_fields"]) == {"deployment_mode", "run_id"}


@pytest.mark.parametrize(
    "prepare_bad_manifest",
    [
        lambda tmp_path: _manifest_path(tmp_path).write_text("{not valid json", encoding="utf-8"),
        lambda tmp_path: _write_manifest(
            tmp_path,
            {
                **_load_manifest(tmp_path),
                "required_stages": "stage1_segments",
            },
        ),
    ],
)
def test_create_app_rejects_resume_when_existing_manifest_is_unreadable_or_schema_invalid(
    tmp_path: Path,
    prepare_bad_manifest: Callable[[Path], None],
) -> None:
    create_app(_make_config(tmp_path))
    prepare_bad_manifest(tmp_path)

    with pytest.raises(ValueError, match="manifest_schema"):
        create_app(_make_config(tmp_path))


@pytest.mark.parametrize(
    "prepare_bad_manifest",
    [
        lambda tmp_path: _manifest_path(tmp_path).write_text("{not valid json", encoding="utf-8"),
        lambda tmp_path: _write_manifest(
            tmp_path,
            {
                **_load_manifest(tmp_path),
                "required_stages": "stage1_segments",
            },
        ),
    ],
)
def test_create_app_force_resume_rebuilds_manifest_when_existing_manifest_is_invalid(
    tmp_path: Path,
    prepare_bad_manifest: Callable[[Path], None],
) -> None:
    create_app(_make_config(tmp_path))
    prepare_bad_manifest(tmp_path)

    app = create_app(_make_config(tmp_path, force_resume=True))
    status = _status_for_demo_subset(app)
    manifest = _load_manifest(tmp_path)

    assert status["action"] == "force_resume_overrode_identity_mismatch"
    assert status["resume"]["mismatch_fields"] == ["manifest_schema"]
    assert manifest["resume"]["force_resume"] is True
    assert manifest["resume"]["mismatch_fields"] == ["manifest_schema"]


@pytest.mark.parametrize(
    "seed_nonempty_run_dir",
    [_seed_existing_run_artifact, _seed_existing_run_dirs_only],
)
def test_create_app_rejects_nonempty_run_dir_without_manifest(
    tmp_path: Path,
    seed_nonempty_run_dir: Callable[[Path], None],
) -> None:
    seed_nonempty_run_dir(tmp_path)

    with pytest.raises(ValueError, match="missing_manifest"):
        create_app(_make_config(tmp_path))


@pytest.mark.parametrize(
    "seed_nonempty_run_dir",
    [_seed_existing_run_artifact, _seed_existing_run_dirs_only],
)
def test_create_app_force_resume_rebuilds_missing_manifest_for_nonempty_run_dir(
    tmp_path: Path,
    seed_nonempty_run_dir: Callable[[Path], None],
) -> None:
    seed_nonempty_run_dir(tmp_path)

    app = create_app(_make_config(tmp_path, force_resume=True))
    status = _status_for_demo_subset(app)
    manifest = _load_manifest(tmp_path)

    assert status["action"] == "force_resume_rebuilt_missing_manifest"
    assert status["run_dir_nonempty_before_start"] is True
    assert status["resume"]["mismatch_fields"] == ["missing_manifest"]
    assert manifest["resume"]["force_resume"] is True
    assert manifest["resume"]["mismatch_fields"] == ["missing_manifest"]
