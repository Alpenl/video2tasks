"""Task-scoped intermediate artifact persistence helpers."""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


_SAFE_PATH_TOKEN_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize_path_token(value: Any, fallback: str) -> str:
    raw = str(value).strip()
    if not raw:
        return fallback
    sanitized = _SAFE_PATH_TOKEN_RE.sub("_", raw).strip("._-")
    return sanitized or fallback


def _decode_b64_payload(image_b64: str) -> bytes:
    if not image_b64:
        return b""
    payload = image_b64
    if image_b64.startswith("data:") and "," in image_b64:
        payload = image_b64.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=False)
    except Exception:
        return b""


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

    def _write_manifest(
        self,
        *,
        task_dir: Path,
        metadata: Mapping[str, Any],
        image_kind: str,
        records: List[ArtifactImageRecord],
    ) -> str:
        manifest_path = task_dir / "manifest.json"
        payload = {
            "created_at_ms": int(time.time() * 1000),
            "metadata": dict(metadata),
            "image_kind": image_kind,
            "image_count": len(records),
            "records": [asdict(record) for record in records],
        }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(manifest_path)

    def _write_images_payloads(
        self,
        *,
        metadata: Mapping[str, Any],
        image_payloads: List[bytes],
        image_kind: str,
        frame_groups: Optional[List[List[int]]] = None,
        source_tags: Optional[List[str]] = None,
        extension: str = "png",
    ) -> ArtifactBatch:
        task_dir = self._task_dir(metadata)
        images_dir = task_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        records: List[ArtifactImageRecord] = []
        safe_kind = _sanitize_path_token(image_kind, "artifact")
        safe_ext = _sanitize_path_token(extension, "png")

        for index, payload in enumerate(image_payloads):
            image_bytes = bytes(payload or b"")
            image_name = f"{safe_kind}_{index:04d}.{safe_ext}"
            image_path = images_dir / image_name
            image_path.write_bytes(image_bytes)

            frame_ids: List[int] = []
            if frame_groups and index < len(frame_groups):
                frame_ids = [int(frame_id) for frame_id in frame_groups[index]]

            source = ""
            if source_tags and index < len(source_tags):
                source = str(source_tags[index])

            records.append(
                ArtifactImageRecord(
                    index=index,
                    path=str(image_path),
                    byte_size=len(image_bytes),
                    frame_ids=frame_ids,
                    source=source,
                    decode_ok=bool(image_bytes),
                )
            )

        manifest_path = self._write_manifest(
            task_dir=task_dir,
            metadata=metadata,
            image_kind=image_kind,
            records=records,
        )
        return ArtifactBatch(
            root_dir=str(self.root_dir),
            task_dir=str(task_dir),
            images_dir=str(images_dir),
            manifest_path=manifest_path,
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
        return self._write_images_payloads(
            metadata=metadata,
            image_payloads=[_decode_b64_payload(image_b64) for image_b64 in images_b64],
            image_kind=image_kind,
            frame_groups=frame_groups,
            source_tags=source_tags,
            extension=extension,
        )
