"""Job payload and metadata assembly helpers extracted from the server app."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

from ..logging_utils import get_logger, log_event
from .protocol import InlineImageTransport, JobEnvelope, SharedFSImageTransport
from .windowing import BoundaryRefinementWindow, Window, build_window_prompt_metadata


logger = get_logger(__name__)


@dataclass
class ArtifactReuseEntry:
    transport: SharedFSImageTransport
    producer_task_id: str
    producer_meta: Dict[str, Any]
    reuse_group: str


def _clone_shared_fs_transport(transport: SharedFSImageTransport) -> SharedFSImageTransport:
    return SharedFSImageTransport(
        image_paths=list(transport.image_paths),
        artifact_manifest_path=transport.artifact_manifest_path,
    )


class JobBuilder:
    def __init__(
        self,
        *,
        target_width: int,
        target_height: int,
        png_compression: int,
        use_contact_sheets: bool,
        contact_sheet_rows: int,
        contact_sheet_cols: int,
    ) -> None:
        self.target_width = int(target_width)
        self.target_height = int(target_height)
        self.png_compression = int(png_compression)
        self.use_contact_sheets = bool(use_contact_sheets)
        self.contact_sheet_rows = int(contact_sheet_rows)
        self.contact_sheet_cols = int(contact_sheet_cols)

    def _contact_sheet_meta(self) -> Dict[str, Any]:
        return {
            "use_contact_sheets": self.use_contact_sheets,
            "contact_sheet_rows": self.contact_sheet_rows if self.use_contact_sheets else 0,
            "contact_sheet_cols": self.contact_sheet_cols if self.use_contact_sheets else 0,
        }

    def _repeat_artifact_reuse_key(
        self,
        *,
        meta: Mapping[str, Any],
        frame_ids: List[int],
        artifact_image_kind: str,
    ) -> Optional[Tuple[Any, ...]]:
        if not self.use_contact_sheets:
            return None

        window_id = meta.get("window_id")
        if window_id is None:
            return None

        try:
            normalized_window_id = int(window_id)
        except (TypeError, ValueError):
            return None

        return (
            str(meta.get("subset", "")),
            str(meta.get("sample_id", "")),
            str(meta.get("job_type", "")),
            str(meta.get("window_pass", "coarse") or "coarse"),
            normalized_window_id,
            tuple(int(frame_id) for frame_id in frame_ids),
            str(artifact_image_kind),
            self.target_width,
            self.target_height,
            self.png_compression,
            self.contact_sheet_rows,
            self.contact_sheet_cols,
        )

    def _artifact_reuse_group(self, cache_key: Optional[Tuple[Any, ...]]) -> str:
        if cache_key is None:
            return ""
        encoded_key = json.dumps(
            list(cache_key),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(encoded_key).hexdigest()[:16]
        return f"reuse::{digest}"

    def _with_artifact_provenance(
        self,
        *,
        meta: Mapping[str, Any],
        task_id: str,
        artifact_reuse: bool,
        producer_task_id: str,
        producer_meta: Mapping[str, Any],
        reuse_group: str,
    ) -> Dict[str, Any]:
        normalized = dict(meta)
        normalized["artifact_reuse"] = bool(artifact_reuse)
        normalized["artifact_reuse_group"] = reuse_group
        normalized["artifact_producer_task_id"] = producer_task_id
        normalized["artifact_producer_job_type"] = str(producer_meta.get("job_type", normalized.get("job_type", "")))
        normalized["artifact_producer_repeat_index"] = int(producer_meta.get("repeat_index", 0) or 0)
        normalized["artifact_producer_window_pass"] = str(producer_meta.get("window_pass", "coarse") or "coarse")
        normalized["artifact_consumer_task_id"] = task_id
        normalized["artifact_consumer_job_type"] = str(normalized.get("job_type", ""))
        normalized["artifact_consumer_repeat_index"] = int(normalized.get("repeat_index", 0) or 0)
        normalized["artifact_consumer_window_pass"] = str(normalized.get("window_pass", "coarse") or "coarse")
        return normalized

    def _artifact_metadata(self, meta: Mapping[str, Any], task_id: str, reuse_group: str) -> Dict[str, Any]:
        return {
            **dict(meta),
            "task_id": task_id,
            "artifact_producer_task_id": task_id,
            "artifact_producer_job_type": str(meta.get("job_type", "")),
            "artifact_producer_repeat_index": int(meta.get("repeat_index", 0) or 0),
            "artifact_producer_window_pass": str(meta.get("window_pass", "coarse") or "coarse"),
            "artifact_reuse_group": reuse_group,
        }

    def _build_job_payload(
        self,
        extractor: Any,
        *,
        task_id: str,
        frame_ids: List[int],
        meta: Dict[str, Any],
        artifact_image_kind: str,
        reuse_cache: Optional[MutableMapping[Tuple[Any, ...], ArtifactReuseEntry]] = None,
    ) -> JobEnvelope:
        cache_key = self._repeat_artifact_reuse_key(
            meta=meta,
            frame_ids=frame_ids,
            artifact_image_kind=artifact_image_kind,
        )
        reuse_group = self._artifact_reuse_group(cache_key)
        if reuse_cache is not None and cache_key is not None:
            cached_entry = reuse_cache.get(cache_key)
            if cached_entry is not None:
                job_meta = self._with_artifact_provenance(
                    meta=meta,
                    task_id=task_id,
                    artifact_reuse=True,
                    producer_task_id=cached_entry.producer_task_id,
                    producer_meta=cached_entry.producer_meta,
                    reuse_group=cached_entry.reuse_group,
                )
                log_event(
                    logger,
                    "artifact_reuse_hit",
                    task_id=task_id,
                    subset=str(job_meta.get("subset", "")),
                    sample_id=str(job_meta.get("sample_id", "")),
                    job_type=str(job_meta.get("job_type", "")),
                    artifact_reuse_group=str(job_meta.get("artifact_reuse_group", "")),
                    artifact_producer_task_id=str(job_meta.get("artifact_producer_task_id", "")),
                    artifact_consumer_task_id=str(job_meta.get("artifact_consumer_task_id", "")),
                )
                return JobEnvelope(
                    task_id=task_id,
                    meta=job_meta,
                    image_transport=_clone_shared_fs_transport(cached_entry.transport),
                )

        extract_start = time.perf_counter()
        images, artifact_batch = extractor.get_many_b64_with_artifacts(
            frame_ids,
            self.target_width,
            self.target_height,
            self.png_compression,
            use_contact_sheets=self.use_contact_sheets,
            contact_sheet_rows=self.contact_sheet_rows,
            contact_sheet_cols=self.contact_sheet_cols,
            artifact_metadata=self._artifact_metadata(meta, task_id, reuse_group),
            artifact_image_kind=artifact_image_kind,
            return_images=getattr(extractor, "artifact_writer", None) is None,
        )
        artifact_extract_ms = int(round((time.perf_counter() - extract_start) * 1000.0))
        job_meta = self._with_artifact_provenance(
            meta=meta,
            task_id=task_id,
            artifact_reuse=False,
            producer_task_id=task_id,
            producer_meta=meta,
            reuse_group=reuse_group,
        )

        if artifact_batch is not None and getattr(artifact_batch, "records", None):
            image_transport = SharedFSImageTransport(
                image_paths=[record.path for record in artifact_batch.records],
                artifact_manifest_path=artifact_batch.manifest_path,
            )
            if reuse_cache is not None and cache_key is not None:
                reuse_cache.setdefault(
                    cache_key,
                    ArtifactReuseEntry(
                        transport=_clone_shared_fs_transport(image_transport),
                        producer_task_id=task_id,
                        producer_meta=dict(meta),
                        reuse_group=reuse_group,
                    ),
                )
            log_event(
                logger,
                "artifact_extract_done",
                task_id=task_id,
                subset=str(job_meta.get("subset", "")),
                sample_id=str(job_meta.get("sample_id", "")),
                job_type=str(job_meta.get("job_type", "")),
                image_count=len(image_transport.image_paths),
                artifact_extract_ms=artifact_extract_ms,
                transport_mode="shared_fs",
                artifact_reuse=False,
            )
        else:
            image_transport = InlineImageTransport(images=images)
            log_event(
                logger,
                "artifact_extract_done",
                task_id=task_id,
                subset=str(job_meta.get("subset", "")),
                sample_id=str(job_meta.get("sample_id", "")),
                job_type=str(job_meta.get("job_type", "")),
                image_count=len(images),
                artifact_extract_ms=artifact_extract_ms,
                transport_mode="inline",
                artifact_reuse=False,
            )

        return JobEnvelope(task_id=task_id, meta=job_meta, image_transport=image_transport)

    def build_window_boundary_job(
        self,
        extractor: Any,
        *,
        task_id: str,
        subset: str,
        sample_id: str,
        window: Window,
        fps: float,
        nframes: int,
        repeat_index: int,
        repeat_count: int,
        window_pass: str = "coarse",
        reuse_cache: Optional[MutableMapping[Tuple[Any, ...], ArtifactReuseEntry]] = None,
    ) -> JobEnvelope:
        meta = {
            "subset": subset,
            "sample_id": sample_id,
            "job_type": "window_boundary",
            "window_pass": str(window_pass or "coarse"),
            "repeat_index": int(repeat_index),
            "window_repeat_count": max(1, int(repeat_count)),
            "logical_frame_count": len(window.frame_ids),
            **build_window_prompt_metadata(window, fps, nframes),
            **self._contact_sheet_meta(),
        }
        artifact_image_kind = (
            f"{window_pass}_contact_sheet" if self.use_contact_sheets and window_pass != "coarse"
            else "window_contact_sheet" if self.use_contact_sheets
            else f"{window_pass}_frame" if window_pass != "coarse"
            else "window_frame"
        )
        return self._build_job_payload(
            extractor,
            task_id=task_id,
            frame_ids=list(window.frame_ids),
            meta=meta,
            artifact_image_kind=artifact_image_kind,
            reuse_cache=reuse_cache,
        )

    def build_boundary_refinement_job(
        self,
        extractor: Any,
        *,
        task_id: str,
        subset: str,
        sample_id: str,
        boundary_window: BoundaryRefinementWindow,
    ) -> JobEnvelope:
        meta = {
            "subset": subset,
            "sample_id": sample_id,
            "job_type": "boundary_refinement",
            "boundary_id": int(boundary_window.boundary_id),
            "coarse_boundary_frame": int(boundary_window.coarse_boundary_frame),
            "frame_ids": [int(frame_id) for frame_id in boundary_window.frame_ids],
            "logical_frame_count": len(boundary_window.frame_ids),
            "window_start_frame": int(boundary_window.start_frame),
            "window_end_frame": int(boundary_window.end_frame),
            **self._contact_sheet_meta(),
        }
        artifact_image_kind = "boundary_refinement_contact_sheet" if self.use_contact_sheets else "boundary_refinement_frame"
        return self._build_job_payload(
            extractor,
            task_id=task_id,
            frame_ids=list(boundary_window.frame_ids),
            meta=meta,
            artifact_image_kind=artifact_image_kind,
        )

    def build_segment_label_job(
        self,
        extractor: Any,
        *,
        task_id: str,
        subset: str,
        sample_id: str,
        segment: Mapping[str, Any],
        frame_ids: List[int],
    ) -> JobEnvelope:
        meta = {
            "subset": subset,
            "sample_id": sample_id,
            "job_type": "segment_label",
            "segment_id": int(segment.get("seg_id", -1)),
            "segment_start_frame": int(segment.get("start_frame", 0)),
            "segment_end_frame": int(segment.get("end_frame", 0)),
            "frame_ids": [int(frame_id) for frame_id in frame_ids],
            "logical_frame_count": len(frame_ids),
            **self._contact_sheet_meta(),
        }
        artifact_image_kind = "segment_label_contact_sheet" if self.use_contact_sheets else "segment_label_frame"
        return self._build_job_payload(
            extractor,
            task_id=task_id,
            frame_ids=list(frame_ids),
            meta=meta,
            artifact_image_kind=artifact_image_kind,
        )


__all__ = ["ArtifactReuseEntry", "JobBuilder"]
