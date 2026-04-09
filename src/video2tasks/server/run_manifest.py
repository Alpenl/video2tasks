"""Run manifest schema and resume identity helpers."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import Config


SCHEMA_VERSION = 1
DEPLOYMENT_MODE = "single_machine_shared_fs"
RUN_MANIFEST_FILENAME = "run_manifest.json"


class VersionMarker(BaseModel):
    package_version: str
    git_commit: str = ""
    git_dirty: bool = False


class ResumeMetadata(BaseModel):
    force_resume: bool = False
    validated_against_existing_manifest: bool = False
    mismatch_fields: list[str] = Field(default_factory=list)
    previous_config_hash: str = ""


class RunManifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    deployment_mode: str = DEPLOYMENT_MODE
    run_id: str
    subset: str
    data_root: str
    run_dir: str
    config_hash: str
    prompt_hash: str
    git_version: VersionMarker
    backend_summary: dict[str, Any]
    required_stages: list[str]
    resume: ResumeMetadata = Field(default_factory=ResumeMetadata)


class RunManifestStatus(BaseModel):
    path: str
    action: str
    manifest_present_before_start: bool
    run_dir_nonempty_before_start: bool
    resume: ResumeMetadata


def run_manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / RUN_MANIFEST_FILENAME


def ensure_run_manifest(
    *,
    run_dir: str | Path,
    subset: str,
    data_root: str,
    config: Config,
    force_resume: bool = False,
    run_dir_nonempty_before_start: bool | None = None,
) -> RunManifestStatus:
    path = run_manifest_path(run_dir)
    manifest_present_before_start = path.exists()
    if run_dir_nonempty_before_start is None:
        run_dir_nonempty_before_start = _run_dir_is_nonempty(run_dir)
    current = build_run_manifest(
        run_dir=run_dir,
        subset=subset,
        data_root=data_root,
        config=config,
        force_resume=force_resume,
    )

    if not manifest_present_before_start:
        if run_dir_nonempty_before_start:
            current.resume.validated_against_existing_manifest = True
            current.resume.mismatch_fields = ["missing_manifest"]
            if not force_resume:
                raise ValueError(_format_resume_mismatch_error(path, current.resume.mismatch_fields))
            _write_run_manifest(path, current)
            return RunManifestStatus(
                path=str(path),
                action="force_resume_rebuilt_missing_manifest",
                manifest_present_before_start=False,
                run_dir_nonempty_before_start=True,
                resume=current.resume.model_copy(deep=True),
            )

        _write_run_manifest(path, current)
        return RunManifestStatus(
            path=str(path),
            action="created_fresh_manifest",
            manifest_present_before_start=False,
            run_dir_nonempty_before_start=False,
            resume=current.resume.model_copy(deep=True),
        )

    existing: RunManifest | None = None
    mismatches: list[str] = []
    try:
        existing = load_run_manifest(path)
        mismatches = compare_manifest_identity(existing, current)
    except (OSError, json.JSONDecodeError, ValueError):
        mismatches = ["manifest_schema"]

    current.resume.validated_against_existing_manifest = True
    current.resume.mismatch_fields = mismatches
    if existing is not None:
        current.resume.previous_config_hash = existing.config_hash

    if mismatches and not force_resume:
        raise ValueError(_format_resume_mismatch_error(path, mismatches))

    _write_run_manifest(path, current)
    return RunManifestStatus(
        path=str(path),
        action=(
            "force_resume_overrode_identity_mismatch"
            if mismatches and force_resume
            else "validated_existing_manifest"
        ),
        manifest_present_before_start=True,
        run_dir_nonempty_before_start=bool(run_dir_nonempty_before_start),
        resume=current.resume.model_copy(deep=True),
    )


def build_run_manifest(
    *,
    run_dir: str | Path,
    subset: str,
    data_root: str,
    config: Config,
    force_resume: bool = False,
) -> RunManifest:
    return RunManifest(
        run_id=str(config.run.run_id),
        subset=str(subset),
        data_root=str(data_root),
        run_dir=str(Path(run_dir)),
        config_hash=_hash_payload(_identity_config_payload(config)),
        prompt_hash=_prompt_hash(),
        git_version=_version_marker(),
        backend_summary=_backend_summary(config),
        required_stages=_required_stages(config),
        resume=ResumeMetadata(force_resume=force_resume),
    )


def load_run_manifest(path: str | Path) -> RunManifest:
    return RunManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def compare_manifest_identity(existing: RunManifest, current: RunManifest) -> list[str]:
    mismatches: list[str] = []
    for field_name in (
        "schema_version",
        "deployment_mode",
        "run_id",
        "subset",
        "data_root",
        "config_hash",
        "prompt_hash",
        "git_version",
        "backend_summary",
        "required_stages",
    ):
        if getattr(existing, field_name) != getattr(current, field_name):
            mismatches.append(field_name)
    return mismatches


def _write_run_manifest(path: Path, manifest: RunManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _format_resume_mismatch_error(path: Path, mismatches: list[str]) -> str:
    mismatch_text = ", ".join(mismatches)
    return (
        f"Resume refused for {path}: manifest identity mismatch in [{mismatch_text}]. "
        "Set run.force_resume=true or env RUN_FORCE_RESUME=true to override explicitly."
    )


def _identity_config_payload(config: Config) -> dict[str, Any]:
    return {
        "deployment_mode": DEPLOYMENT_MODE,
        "datasets": [
            {"root": str(dataset.root), "subset": str(dataset.subset)}
            for dataset in config.datasets
        ],
        "windowing": config.windowing.model_dump(),
        "export": config.export.model_dump(),
        "worker": {
            "backend": str(config.worker.backend),
            "selected_backend_config": _selected_worker_backend_config(config),
        },
        "llm_merge": _llm_merge_identity_config(config),
        "required_stages": _required_stages(config),
    }


def _selected_worker_backend_config(config: Config) -> dict[str, Any]:
    backend = str(config.worker.backend)
    if backend == "dummy":
        return {}
    if backend == "qwen3vl":
        return config.worker.qwen3vl.model_dump()
    if backend == "remote_api":
        return config.worker.remote_api.model_dump(exclude={"api_key"})
    if backend == "openai":
        return config.worker.openai.model_dump(exclude={"api_key", "organization", "project"})
    if backend == "gemini":
        return config.worker.gemini.model_dump(exclude={"api_key"})
    return {}


def _llm_merge_identity_config(config: Config) -> dict[str, Any]:
    return config.llm_merge.model_dump(exclude={"api_key", "organization", "project"})


def _backend_summary(config: Config) -> dict[str, Any]:
    worker_backend = str(config.worker.backend)
    worker_summary: dict[str, Any] = {"backend": worker_backend}
    if worker_backend == "qwen3vl":
        worker_summary.update(
            {
                "model_path": str(config.worker.qwen3vl.model_path),
                "device_map": str(config.worker.qwen3vl.device_map),
            }
        )
    elif worker_backend == "remote_api":
        worker_summary.update(
            {
                "api_url": str(config.worker.remote_api.api_url),
                "timeout_sec": float(config.worker.remote_api.timeout_sec),
            }
        )
    elif worker_backend == "openai":
        worker_summary.update(
            {
                "model": str(config.worker.openai.model),
                "base_url": str(config.worker.openai.base_url),
                "reasoning_effort": str(config.worker.openai.reasoning_effort),
            }
        )
    elif worker_backend == "gemini":
        worker_summary.update(
            {
                "model": str(config.worker.gemini.model),
                "base_url": str(config.worker.gemini.base_url),
                "api_mode": str(config.worker.gemini.api_mode),
            }
        )

    llm_merge_summary: dict[str, Any] = {"enabled": bool(config.llm_merge.enabled)}
    if bool(config.llm_merge.enabled):
        llm_merge_summary.update(
            {
                "backend": str(config.llm_merge.backend),
                "model": str(config.llm_merge.model),
                "base_url": str(config.llm_merge.base_url),
                "reasoning_effort": str(config.llm_merge.reasoning_effort),
            }
        )

    return {
        "stage1": worker_summary,
        "stage2": llm_merge_summary,
    }


def _required_stages(config: Config) -> list[str]:
    stages = ["stage1_segments"]
    if bool(config.llm_merge.enabled):
        stages.append("stage2_text")
    if bool(config.export.enabled):
        stages.append("export")
    return stages


def _prompt_hash() -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "prompt.py"
    return hashlib.sha256(prompt_path.read_bytes()).hexdigest()


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _version_marker() -> VersionMarker:
    return VersionMarker(
        package_version=_package_version(),
        git_commit=_git_output("rev-parse", "HEAD"),
        git_dirty=bool(_git_output("status", "--short")),
    )


def _package_version() -> str:
    try:
        return metadata.version("video2tasks")
    except metadata.PackageNotFoundError:
        pyproject_path = _repo_root() / "pyproject.toml"
        try:
            content = pyproject_path.read_text(encoding="utf-8")
        except OSError:
            return "unknown"
        match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
        return match.group(1) if match else "unknown"


def _git_output(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=_repo_root(),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _run_dir_is_nonempty(run_dir: str | Path) -> bool:
    base = Path(run_dir)
    if not base.exists():
        return False

    return any(base.iterdir())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
