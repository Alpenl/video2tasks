"""Video reading, encoding, and artifact helpers for windowing."""

import base64
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..logging_utils import get_logger
from .task_artifacts import ArtifactBatch, TaskArtifactWriter, validate_image_payloads_or_raise


logger = get_logger(__name__)

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_WINDOWING_FACADE_MODULE = "video2tasks.server.windowing"


def _resolve_windowing_attr(name: str, fallback: Any) -> Any:
    facade = sys.modules.get(_WINDOWING_FACADE_MODULE)
    if facade is None:
        return fallback
    return getattr(facade, name, fallback)


def read_video_info(mp4_path: str) -> Tuple[float, int]:
    """Read video FPS and frame count."""
    cap = cv2.VideoCapture(mp4_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {mp4_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()

    try:
        fps_value = float(fps)
    except (TypeError, ValueError):
        fps_value = float("nan")
    if not math.isfinite(fps_value) or abs(fps_value) < 1e-6:
        fps_value = 30.0

    try:
        frame_count_value = float(frame_count)
    except (TypeError, ValueError):
        frame_count_value = float("nan")
    nframes = int(frame_count_value) if math.isfinite(frame_count_value) and frame_count_value > 0.0 else 0

    return float(fps_value), max(0, nframes)


def encode_image_720p_png_bytes(
    img_bgr: np.ndarray,
    target_w: int = 720,
    target_h: int = 480,
    compression: int = 0,
) -> bytes:
    """Encode image to PNG bytes, resizing if needed."""
    if img_bgr is None:
        return b""

    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return b""

    if (w != target_w) or (h != target_h):
        img_bgr = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(
        ".png",
        img_bgr,
        [int(cv2.IMWRITE_PNG_COMPRESSION), int(np.clip(compression, 0, 9))],
    )
    return buf.tobytes() if ok else b""


def encode_image_720p_png(
    img_bgr: np.ndarray,
    target_w: int = 720,
    target_h: int = 480,
    compression: int = 0,
) -> str:
    """Encode image to base64 PNG, resizing if needed."""
    payload = encode_image_720p_png_bytes(
        img_bgr,
        target_w=target_w,
        target_h=target_h,
        compression=compression,
    )
    return base64.b64encode(payload).decode("utf-8") if payload else ""


def chunk_frame_ids_for_contact_sheets(frame_ids: List[int], chunk_size: int) -> List[List[int]]:
    chunk_size = max(1, int(chunk_size))
    return [frame_ids[i : i + chunk_size] for i in range(0, len(frame_ids), chunk_size)]


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in _TRUTHY_ENV_VALUES


def _build_default_task_artifact_writer() -> Optional[TaskArtifactWriter]:
    if not _env_flag_enabled("VIDEO2TASKS_DUMP_INTERMEDIATE"):
        return None
    root_dir = os.getenv("VIDEO2TASKS_TMP_DIR", "tmp").strip() or "tmp"
    return TaskArtifactWriter(root_dir=root_dir)


class FrameExtractor:
    """Extract frames from a video file."""

    def __init__(
        self,
        mp4_path: str,
        artifact_writer: Optional[TaskArtifactWriter] = None,
    ):
        self.mp4_path = mp4_path
        self.cap = cv2.VideoCapture(mp4_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {mp4_path}")

        self.artifact_writer = artifact_writer or _build_default_task_artifact_writer()
        self.last_artifact_batch: Optional[ArtifactBatch] = None
        self._artifact_sequence = 0
        self._auto_sample_id = os.path.splitext(os.path.basename(self.mp4_path))[0] or "sample"

    def close(self) -> None:
        """Release video capture."""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _read_frame_bgr(self, fid: int) -> Optional[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None
        return bgr

    def _build_contact_sheet_png_bytes_via_ffmpeg(
        self,
        group_frame_ids: List[int],
        group_start_index: int,
        target_w: int,
        target_h: int,
        rows: int,
        cols: int,
    ) -> bytes:
        if not group_frame_ids:
            return b""

        cell_w = max(1, int(target_w) // max(1, int(cols)))
        cell_h = max(1, int(target_h) // max(1, int(rows)))
        font_size = max(12, min(cell_w, cell_h) // 6)
        select_expr = "+".join(f"eq(n\\,{int(fid)})" for fid in group_frame_ids)
        filter_chain = (
            f"select='{select_expr}',"
            f"scale={cell_w}:{cell_h},"
            f"drawtext=text='%{{eif\\:n+{int(group_start_index)}\\:d}}':"
            f"x=6:y=6:fontsize={font_size}:fontcolor=white:box=1:boxcolor=black@0.6,"
            f"tile={int(cols)}x{int(rows)}:padding=0:margin=0:color=black"
        )
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            self.mp4_path,
            "-vf",
            filter_chain,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            if stderr:
                logger.warning(f"[FrameExtractor] ffmpeg contact sheet failed: {stderr}")
            return b""
        return bytes(proc.stdout or b"")

    def _build_contact_sheet_b64_via_ffmpeg(
        self,
        group_frame_ids: List[int],
        group_start_index: int,
        target_w: int,
        target_h: int,
        rows: int,
        cols: int,
    ) -> str:
        payload = self._build_contact_sheet_png_bytes_via_ffmpeg(
            group_frame_ids,
            group_start_index,
            target_w,
            target_h,
            rows,
            cols,
        )
        return base64.b64encode(payload).decode("utf-8") if payload else ""

    def _build_contact_sheet_png_bytes_via_cv2(
        self,
        group_frame_ids: List[int],
        group_start_index: int,
        target_w: int,
        target_h: int,
        compression: int,
        rows: int,
        cols: int,
    ) -> bytes:
        cell_w = max(1, int(target_w) // max(1, int(cols)))
        cell_h = max(1, int(target_h) // max(1, int(rows)))
        sheet_h = cell_h * max(1, int(rows))
        sheet_w = cell_w * max(1, int(cols))
        sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
        font_scale = max(0.45, min(cell_w, cell_h) / 180.0)
        thickness = max(1, int(round(font_scale * 2)))

        for tile_idx, fid in enumerate(group_frame_ids):
            row, col = divmod(tile_idx, max(1, int(cols)))
            if row >= max(1, int(rows)):
                break
            bgr = self._read_frame_bgr(fid)
            if bgr is None:
                tile = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            else:
                tile = cv2.resize(bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            y0 = row * cell_h
            x0 = col * cell_w
            sheet[y0 : y0 + cell_h, x0 : x0 + cell_w] = tile

            label = str(int(group_start_index) + tile_idx)
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )
            cv2.rectangle(
                sheet,
                (x0 + 2, y0 + 2),
                (x0 + text_w + 12, y0 + text_h + baseline + 10),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.putText(
                sheet,
                label,
                (x0 + 6, y0 + text_h + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

        return encode_image_720p_png_bytes(sheet, target_w, target_h, compression)

    def _build_contact_sheet_b64_via_cv2(
        self,
        group_frame_ids: List[int],
        group_start_index: int,
        target_w: int,
        target_h: int,
        compression: int,
        rows: int,
        cols: int,
    ) -> str:
        payload = self._build_contact_sheet_png_bytes_via_cv2(
            group_frame_ids,
            group_start_index,
            target_w,
            target_h,
            compression,
            rows,
            cols,
        )
        return base64.b64encode(payload).decode("utf-8") if payload else ""

    def _persist_intermediate_artifacts(
        self,
        *,
        frame_groups: List[List[int]],
        source_tags: List[str],
        image_kind: str,
        metadata: Optional[Dict[str, Any]],
        image_payloads: Optional[List[bytes]] = None,
        images_b64: Optional[List[str]] = None,
    ) -> Optional[ArtifactBatch]:
        self.last_artifact_batch = None
        if getattr(self, "artifact_writer", None) is None:
            return None

        task_metadata: Dict[str, Any] = dict(metadata or {})
        task_metadata.setdefault("subset", "unspecified")
        task_metadata.setdefault("sample_id", self._auto_sample_id)
        task_metadata.setdefault("task_id", f"{self._auto_sample_id}_extract_{self._artifact_sequence:06d}")
        task_metadata.setdefault("mp4_path", self.mp4_path)
        task_metadata.setdefault("extract_sequence", self._artifact_sequence)

        if image_payloads is not None:
            batch = self.artifact_writer.write_images_bytes(
                metadata=task_metadata,
                images_bytes=image_payloads,
                image_kind=image_kind,
                frame_groups=frame_groups,
                source_tags=source_tags,
                extension="png",
            )
        else:
            batch = self.artifact_writer.write_images_b64(
                metadata=task_metadata,
                images_b64=images_b64 or [],
                image_kind=image_kind,
                frame_groups=frame_groups,
                source_tags=source_tags,
                extension="png",
            )
        self._artifact_sequence += 1
        self.last_artifact_batch = batch
        return batch

    def get_many_b64(
        self,
        frame_ids: List[int],
        target_w: int = 720,
        target_h: int = 480,
        compression: int = 0,
        use_contact_sheets: bool = False,
        contact_sheet_rows: int = 4,
        contact_sheet_cols: int = 4,
        artifact_metadata: Optional[Dict[str, Any]] = None,
        artifact_image_kind: Optional[str] = None,
        persist_artifacts: bool = True,
        return_images: bool = True,
    ) -> List[str]:
        """Extract multiple logical frames either directly or as contact sheets."""
        if not use_contact_sheets:
            sorted_indices = sorted(list(set(frame_ids)))
            frame_payload_map: Dict[int, bytes] = {}
            encode_png_bytes = _resolve_windowing_attr(
                "encode_image_720p_png_bytes",
                encode_image_720p_png_bytes,
            )

            for fid in sorted_indices:
                bgr = self._read_frame_bgr(fid)
                frame_payload_map[fid] = (
                    encode_png_bytes(bgr, target_w, target_h, compression)
                    if bgr is not None
                    else b""
                )

            image_payloads = [frame_payload_map.get(fid, b"") for fid in frame_ids]
            frame_groups = [[int(fid)] for fid in frame_ids]
            source_tags = ["cv2_frame" if frame_payload_map.get(fid, b"") else "missing" for fid in frame_ids]
            validate_image_payloads_or_raise(image_payloads, source_tags=source_tags)
            images = [
                base64.b64encode(payload).decode("utf-8")
                for payload in image_payloads
            ] if return_images else []
            if persist_artifacts:
                self._persist_intermediate_artifacts(
                    image_payloads=image_payloads,
                    frame_groups=frame_groups,
                    source_tags=source_tags,
                    image_kind=artifact_image_kind or "frame",
                    metadata=artifact_metadata,
                )
            return images

        rows = max(1, int(contact_sheet_rows))
        cols = max(1, int(contact_sheet_cols))
        chunk_size = rows * cols
        chunker = _resolve_windowing_attr(
            "chunk_frame_ids_for_contact_sheets",
            chunk_frame_ids_for_contact_sheets,
        )
        images: List[str] = []
        image_payloads: List[bytes] = []
        frame_groups = chunker(frame_ids, chunk_size)
        source_tags: List[str] = []

        for group_index, group in enumerate(frame_groups):
            group_start_index = group_index * chunk_size
            payload = self._build_contact_sheet_png_bytes_via_ffmpeg(
                group,
                group_start_index,
                target_w,
                target_h,
                rows,
                cols,
            )
            source = "ffmpeg"
            if not payload:
                payload = self._build_contact_sheet_png_bytes_via_cv2(
                    group,
                    group_start_index,
                    target_w,
                    target_h,
                    compression,
                    rows,
                    cols,
                )
                source = "cv2" if payload else "missing"
            image_payloads.append(payload)
            source_tags.append(source)

        validate_image_payloads_or_raise(image_payloads, source_tags=source_tags)
        if return_images:
            images = [base64.b64encode(payload).decode("utf-8") for payload in image_payloads]

        if persist_artifacts:
            self._persist_intermediate_artifacts(
                image_payloads=image_payloads,
                frame_groups=frame_groups,
                source_tags=source_tags,
                image_kind=artifact_image_kind or "contact_sheet",
                metadata=artifact_metadata,
            )
        return images

    def get_many_b64_with_artifacts(
        self,
        frame_ids: List[int],
        target_w: int = 720,
        target_h: int = 480,
        compression: int = 0,
        use_contact_sheets: bool = False,
        contact_sheet_rows: int = 4,
        contact_sheet_cols: int = 4,
        artifact_metadata: Optional[Dict[str, Any]] = None,
        artifact_image_kind: Optional[str] = None,
        return_images: bool = True,
    ) -> Tuple[List[str], Optional[ArtifactBatch]]:
        """Return encoded images plus persisted artifact batch metadata."""
        images = self.get_many_b64(
            frame_ids,
            target_w=target_w,
            target_h=target_h,
            compression=compression,
            use_contact_sheets=use_contact_sheets,
            contact_sheet_rows=contact_sheet_rows,
            contact_sheet_cols=contact_sheet_cols,
            artifact_metadata=artifact_metadata,
            artifact_image_kind=artifact_image_kind,
            persist_artifacts=True,
            return_images=return_images,
        )
        return images, self.last_artifact_batch


__all__ = [
    "FrameExtractor",
    "chunk_frame_ids_for_contact_sheets",
    "encode_image_720p_png",
    "encode_image_720p_png_bytes",
    "read_video_info",
]
