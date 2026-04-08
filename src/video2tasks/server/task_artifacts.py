"""Task-scoped intermediate artifact persistence helpers."""

from __future__ import annotations

import base64
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import cv2
import numpy as np


_SAFE_PATH_TOKEN_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize_path_token(value: Any, fallback: str) -> str:
    raw = str(value).strip()
    if not raw:
        return fallback
    sanitized = _SAFE_PATH_TOKEN_RE.sub("_", raw).strip("._-")
    return sanitized or fallback


def _decode_b64_payload(image_b64: Any) -> tuple[bytes, Optional[str]]:
    if image_b64 is None:
        return b"", "empty_payload"
    if isinstance(image_b64, bytes):
        try:
            image_b64 = image_b64.decode("utf-8")
        except UnicodeDecodeError:
            return b"", "invalid_payload_type"
    elif not isinstance(image_b64, str):
        return b"", "invalid_payload_type"

    if not image_b64:
        return b"", "empty_payload"

    payload = image_b64
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    payload = "".join(payload.split())
    if not payload:
        return b"", "empty_payload"
    try:
        decoded = base64.b64decode(payload, validate=True)
    except Exception:
        return b"", "base64_decode_failed"
    if not decoded:
        return b"", "empty_payload"
    return decoded, None


def _validate_image_payload(image_bytes: bytes) -> Optional[str]:
    if not image_bytes:
        return "empty_payload"
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    if encoded.size == 0:
        return "empty_payload"
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.size == 0:
        return "image_decode_failed"
    return None


def _remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


@dataclass
class ArtifactPayloadIssue:
    index: int
    reason: str
    byte_size: int
    source: str


class ArtifactPayloadValidationError(ValueError):
    """Raised when one or more artifact payloads are empty or undecodable."""

    def __init__(self, issues: List[ArtifactPayloadIssue]) -> None:
        self.issues = list(issues)
        preview = ", ".join(
            f"index={issue.index}:reason={issue.reason}:source={issue.source or 'unknown'}"
            for issue in self.issues[:3]
        )
        super().__init__(f"invalid image payloads: {preview}")


def validate_image_payloads_or_raise(
    image_payloads: List[bytes],
    *,
    source_tags: Optional[List[str]] = None,
    validation_errors: Optional[List[Optional[str]]] = None,
) -> None:
    issues: List[ArtifactPayloadIssue] = []
    for index, payload in enumerate(image_payloads):
        image_bytes = bytes(payload or b"")
        reason = None
        if validation_errors and index < len(validation_errors):
            reason = validation_errors[index]
        if reason is None:
            reason = _validate_image_payload(image_bytes)
        if reason is None:
            continue

        source = ""
        if source_tags and index < len(source_tags):
            source = str(source_tags[index])
        issues.append(
            ArtifactPayloadIssue(
                index=index,
                reason=reason,
                byte_size=len(image_bytes),
                source=source,
            )
        )

    if issues:
        raise ArtifactPayloadValidationError(issues)


@dataclass
class ArtifactImageRecord:
    index: int
    path: str
    byte_size: int
    frame_ids: List[int]
    source: str
    decode_ok: bool


@dataclass
class ArtifactBatch:
    root_dir: str
    task_dir: str
    images_dir: str
    manifest_path: str
    image_kind: str
    image_count: int
    records: List[ArtifactImageRecord]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_dir": self.root_dir,
            "task_dir": self.task_dir,
            "images_dir": self.images_dir,
            "manifest_path": self.manifest_path,
            "image_kind": self.image_kind,
            "image_count": self.image_count,
            "records": [asdict(record) for record in self.records],
            "metadata": dict(self.metadata),
        }


class TaskArtifactWriter:
    """Persist per-task intermediate images to disk under a shared root."""

    def __init__(self, root_dir: str = "tmp") -> None:
        self.root_dir = Path(root_dir)

    def _task_dir(self, metadata: Mapping[str, Any]) -> Path:
        subset = _sanitize_path_token(metadata.get("subset", ""), "subset")
        sample_id = _sanitize_path_token(metadata.get("sample_id", ""), "sample")
        task_id = _sanitize_path_token(metadata.get("task_id", ""), "task")
        return self.root_dir / subset / sample_id / task_id

    def _staging_dir(self, task_dir: Path) -> Path:
        return task_dir.parent / f".{task_dir.name}.staging-{time.time_ns()}"

    def _backup_dir(self, task_dir: Path) -> Path:
        return task_dir.parent / f".{task_dir.name}.backup-{time.time_ns()}"

    def _manifest_json(
        self,
        *,
        metadata: Mapping[str, Any],
        image_kind: str,
        records: List[ArtifactImageRecord],
    ) -> str:
        payload = {
            "created_at_ms": int(time.time() * 1000),
            "metadata": dict(metadata),
            "image_kind": image_kind,
            "image_count": len(records),
            "records": [asdict(record) for record in records],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _write_images_payloads(
        self,
        *,
        metadata: Mapping[str, Any],
        image_payloads: List[bytes],
        image_kind: str,
        frame_groups: Optional[List[List[int]]] = None,
        source_tags: Optional[List[str]] = None,
        extension: str = "png",
        validation_errors: Optional[List[Optional[str]]] = None,
    ) -> ArtifactBatch:
        validate_image_payloads_or_raise(
            image_payloads,
            source_tags=source_tags,
            validation_errors=validation_errors,
        )

        task_dir = self._task_dir(metadata)
        final_images_dir = task_dir / "images"
        manifest_path = task_dir / "manifest.json"
        safe_kind = _sanitize_path_token(image_kind, "artifact")
        safe_ext = _sanitize_path_token(extension, "png")

        records: List[ArtifactImageRecord] = []
        for index, payload in enumerate(image_payloads):
            image_bytes = bytes(payload)
            frame_ids: List[int] = []
            if frame_groups and index < len(frame_groups):
                frame_ids = [int(frame_id) for frame_id in frame_groups[index]]

            source = ""
            if source_tags and index < len(source_tags):
                source = str(source_tags[index])

            image_name = f"{safe_kind}_{index:04d}.{safe_ext}"
            records.append(
                ArtifactImageRecord(
                    index=index,
                    path=str(final_images_dir / image_name),
                    byte_size=len(image_bytes),
                    frame_ids=frame_ids,
                    source=source,
                    decode_ok=True,
                )
            )

        manifest_text = self._manifest_json(
            metadata=metadata,
            image_kind=image_kind,
            records=records,
        )

        stage_dir = self._staging_dir(task_dir)
        stage_images_dir = stage_dir / "images"
        stage_manifest_path = stage_dir / "manifest.json"
        stage_images_dir.mkdir(parents=True, exist_ok=True)

        try:
            for record, payload in zip(records, image_payloads):
                stage_image_path = stage_images_dir / Path(record.path).name
                stage_image_path.write_bytes(bytes(payload))
            stage_manifest_path.write_text(manifest_text, encoding="utf-8")
        except Exception:
            _remove_tree(stage_dir)
            raise

        backup_dir: Optional[Path] = None
        try:
            if task_dir.exists():
                backup_dir = self._backup_dir(task_dir)
                task_dir.replace(backup_dir)
            stage_dir.replace(task_dir)
        except Exception:
            _remove_tree(stage_dir)
            if backup_dir is not None and backup_dir.exists() and not task_dir.exists():
                try:
                    backup_dir.replace(task_dir)
                except OSError:
                    pass
            raise

        if backup_dir is not None:
            _remove_tree(backup_dir)

        return ArtifactBatch(
            root_dir=str(self.root_dir),
            task_dir=str(task_dir),
            images_dir=str(final_images_dir),
            manifest_path=str(manifest_path),
            image_kind=image_kind,
            image_count=len(records),
            records=records,
            metadata=dict(metadata),
        )

    def write_images_bytes(
        self,
        *,
        metadata: Mapping[str, Any],
        images_bytes: List[bytes],
        image_kind: str,
        frame_groups: Optional[List[List[int]]] = None,
        source_tags: Optional[List[str]] = None,
        extension: str = "png",
    ) -> ArtifactBatch:
        return self._write_images_payloads(
            metadata=metadata,
            image_payloads=images_bytes,
            image_kind=image_kind,
            frame_groups=frame_groups,
            source_tags=source_tags,
            extension=extension,
        )

    def write_images_b64(
        self,
        *,
        metadata: Mapping[str, Any],
        images_b64: List[str],
        image_kind: str,
        frame_groups: Optional[List[List[int]]] = None,
        source_tags: Optional[List[str]] = None,
        extension: str = "png",
    ) -> ArtifactBatch:
        decoded_payloads: List[bytes] = []
        validation_errors: List[Optional[str]] = []
        for image_b64 in images_b64:
            payload, reason = _decode_b64_payload(image_b64)
            decoded_payloads.append(payload)
            validation_errors.append(reason)

        return self._write_images_payloads(
            metadata=metadata,
            image_payloads=decoded_payloads,
            image_kind=image_kind,
            frame_groups=frame_groups,
            source_tags=source_tags,
            extension=extension,
            validation_errors=validation_errors,
        )
