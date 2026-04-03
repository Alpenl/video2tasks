"""Video windowing and frame extraction utilities."""

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter
import numpy as np
import cv2
import base64
import subprocess

from .task_artifacts import ArtifactBatch, TaskArtifactWriter


@dataclass
class Window:
    """Video window definition."""
    window_id: int
    start_frame: int
    end_frame: int
    frame_ids: List[int]


@dataclass
class BoundaryRefinementWindow:
    """Short clip centered on a provisional boundary for local refinement."""
    boundary_id: int
    coarse_boundary_frame: int
    start_frame: int
    end_frame: int
    frame_ids: List[int]


_REFINEMENT_WINDOW_ID_BASE = 1_000_000


def read_video_info(mp4_path: str) -> Tuple[float, int]:
    """Read video FPS and frame count."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps is None or fps != fps or abs(fps) < 1e-6:
        fps = 30.0
    
    return float(fps), max(0, nframes)


def build_windows(
    fps: float,
    nframes: int,
    window_sec: float = 16.0,
    step_sec: float = 8.0,
    frames_per_window: int = 16
) -> List[Window]:
    """Build video windows with frame sampling."""
    if fps < 1e-6:
        fps = 30.0
    
    win_len = max(1, int(round(window_sec * fps)))
    step = max(1, int(round(step_sec * fps)))
    windows: List[Window] = []
    
    def get_frames(s: int, e: int, num: int) -> List[int]:
        idx = np.linspace(s, e, num=num).astype(int)
        return np.clip(idx, 0, nframes - 1).tolist()
    
    s = 0
    wid = 0
    while s < nframes:
        e = min(nframes - 1, s + win_len - 1)
        if (e - s < win_len // 2) and wid > 0:
            break
        windows.append(Window(wid, s, e, get_frames(s, e, frames_per_window)))
        wid += 1
        s += step
    
    return windows


def build_window_prompt_metadata(window: Window, fps: float, nframes: int) -> dict:
    """Build temporal metadata for prompt generation."""
    safe_fps = float(fps) if fps > 1e-6 else 30.0
    return {
        "window_id": int(window.window_id),
        "frame_ids": [int(frame_id) for frame_id in window.frame_ids],
        "fps": safe_fps,
        "window_start_frame": int(window.start_frame),
        "window_end_frame": int(window.end_frame),
        "nframes": max(0, int(nframes)),
    }


def _full_instruction_action_families(text: str) -> set[str]:
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    families = set()
    if tokens & {"add", "pour", "fill", "sprinkle"}:
        families.add("add")
    if tokens & {"season", "stir", "mix", "whisk", "toss"}:
        families.add("mix")
    if tokens & {"cook", "saute", "fry", "boil", "toast", "heat", "microwave", "bake"}:
        families.add("cook")
    if tokens & _PREP_INGREDIENT_ACTION_TOKENS:
        families.add("prep")
    if tokens & {"pick", "lift", "carry", "place", "transfer", "remove", "move", "stack", "insert", "put"}:
        families.add("transfer")
    return families


def _instruction_needs_refinement(text: str) -> bool:
    lower = text.lower()
    families = _full_instruction_action_families(text)
    non_transfer_families = families - {"transfer"}
    has_multi_clause_surface = (" and " in lower) or (" then " in lower) or ("," in lower)
    if len(non_transfer_families) >= 2 and has_multi_clause_surface:
        return True
    if _instruction_has_generic_phase_noun(text) and len(non_transfer_families) >= 1 and has_multi_clause_surface:
        return True
    return False


def _window_should_spawn_refinement(rec: dict) -> bool:
    vlm = rec.get("vlm_json", {})
    transitions = vlm.get("transitions", [])
    if transitions:
        return False

    instructions = [
        str(item).strip()
        for item in vlm.get("instructions", [])
        if str(item).strip()
    ]
    if not instructions or len(instructions) > 2:
        return False
    return any(_instruction_needs_refinement(text) for text in instructions)


def _sample_window_frame_ids(start_frame: int, end_frame: int, frames_per_window: int, nframes: int) -> List[int]:
    if nframes <= 0:
        return []
    idx = np.linspace(start_frame, end_frame, num=frames_per_window).astype(int)
    return np.clip(idx, 0, nframes - 1).tolist()


def _cluster_cut_votes(raw_cuts: List[Tuple[int, float]], fps: float) -> Tuple[List[int], Dict[int, float]]:
    """Cluster nearby cut votes while preserving distinct local peaks.

    The boundary detector is intentionally over-segmenting. Dense action regions can produce
    multiple candidate cuts inside one broader phase transition. For this pre-merge pass, it is
    safer to keep distinct local peaks than to collapse them into one representative point.

    The clustering rule is therefore intentionally narrow:
    - first aggregate votes landing on the exact same frame
    - then merge only short gaps into a micro-peak
    - emit one representative per micro-peak

    Inside one micro-peak, prefer the strongest frame (highest vote count, then highest support)
    and use the earliest frame only as a tie-breaker. This avoids a weak early singleton dragging
    the cluster representative away from a stronger nearby peak, while still keeping an onset bias
    inside tightly packed votes.
    """
    if not raw_cuts:
        return [], {}

    safe_fps = float(fps) if fps > 1e-6 else 30.0
    peak_gap_frames = max(2, min(5, int(round(0.2 * safe_fps))))

    aggregated_votes: Dict[int, Dict[str, float]] = {}
    for fid, weight in raw_cuts:
        frame = int(fid)
        bucket = aggregated_votes.setdefault(frame, {"count": 0.0, "support": 0.0})
        bucket["count"] += 1.0
        bucket["support"] += float(weight)

    sorted_frames = sorted(aggregated_votes)
    clustered_points: List[int] = []
    cut_support_by_point: Dict[int, float] = {}

    cur_frames: List[int] = []

    def flush_cluster() -> None:
        if not cur_frames:
            return
        point = min(
            cur_frames,
            key=lambda frame: (
                -aggregated_votes[frame]["count"],
                -aggregated_votes[frame]["support"],
                frame,
            ),
        )
        cluster_support = float(sum(aggregated_votes[frame]["support"] for frame in cur_frames))
        clustered_points.append(point)
        cut_support_by_point[point] = max(cut_support_by_point.get(point, 0.0), cluster_support)

    for fid in sorted_frames:
        if not cur_frames:
            cur_frames.append(int(fid))
            continue

        prev_fid = cur_frames[-1]
        if (fid - prev_fid) <= peak_gap_frames:
            cur_frames.append(int(fid))
            continue

        flush_cluster()
        cur_frames = [int(fid)]

    flush_cluster()
    return clustered_points, cut_support_by_point


def sample_segment_frame_ids(
    start_frame: int,
    end_frame: int,
    frames_per_window: int,
    nframes: int,
) -> List[int]:
    """Sample representative frames for a finalized segment.

    `end_frame` follows the segment convention used elsewhere in the pipeline:
    it is exclusive, so the last available frame is `end_frame - 1`.
    """
    if nframes <= 0:
        return []
    safe_start = max(0, int(start_frame))
    safe_end = max(safe_start, int(end_frame) - 1)
    return _sample_window_frame_ids(safe_start, safe_end, frames_per_window, nframes)


def chunk_frame_ids_for_contact_sheets(frame_ids: List[int], chunk_size: int) -> List[List[int]]:
    chunk_size = max(1, int(chunk_size))
    return [frame_ids[i : i + chunk_size] for i in range(0, len(frame_ids), chunk_size)]


def build_refinement_windows(
    base_windows: List[Window],
    by_wid: dict,
    fps: float,
    nframes: int,
    frames_per_window: int,
) -> List[Window]:
    """Build shorter refinement windows only for ambiguous first-pass windows."""
    refinement_windows: List[Window] = []
    safe_fps = float(fps) if fps > 1e-6 else 30.0

    for window in base_windows:
        rec = by_wid.get(window.window_id)
        if not rec or not _window_should_spawn_refinement(rec):
            continue

        base_len = max(1, int(window.end_frame - window.start_frame + 1))
        refine_len = min(
            base_len,
            max(int(round(base_len * 0.5)), int(round(3.0 * safe_fps))),
        )
        refine_step = max(1, refine_len // 2)
        max_start = max(window.start_frame, window.end_frame - refine_len + 1)

        starts = []
        cur_start = int(window.start_frame)
        while cur_start < max_start:
            starts.append(cur_start)
            cur_start += refine_step
        starts.append(int(max_start))

        for offset, sub_start in enumerate(sorted(set(starts))):
            sub_end = min(int(window.end_frame), sub_start + refine_len - 1)
            refinement_windows.append(
                Window(
                    window_id=_REFINEMENT_WINDOW_ID_BASE + int(window.window_id) * 10 + offset,
                    start_frame=sub_start,
                    end_frame=sub_end,
                    frame_ids=_sample_window_frame_ids(
                        sub_start,
                        sub_end,
                        frames_per_window,
                        nframes,
                    ),
                )
            )

    return refinement_windows


def build_boundary_refinement_windows(
    segments: List[dict],
    fps: float,
    nframes: int,
    window_sec: float,
    frames_per_window: int,
) -> List[BoundaryRefinementWindow]:
    """Build short local clips around provisional boundaries for position refinement."""
    if nframes <= 0 or len(segments) <= 1:
        return []

    safe_fps = float(fps) if fps > 1e-6 else 30.0
    window_len = max(1, int(round(window_sec * safe_fps)))
    refinement_windows: List[BoundaryRefinementWindow] = []

    for boundary_id, segment in enumerate(segments[:-1]):
        coarse_boundary = int(segment["end_frame"])
        half = window_len // 2
        start_frame = max(0, coarse_boundary - half)
        end_frame = min(nframes - 1, start_frame + window_len - 1)
        start_frame = max(0, end_frame - window_len + 1)
        frame_ids = _sample_window_frame_ids(
            start_frame,
            end_frame,
            frames_per_window,
            nframes,
        )
        refinement_windows.append(
            BoundaryRefinementWindow(
                boundary_id=boundary_id,
                coarse_boundary_frame=coarse_boundary,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_ids=frame_ids,
            )
        )

    return refinement_windows


def apply_boundary_refinement_results(
    segments: List[dict],
    refinement_results: dict,
    fps: float = 30.0,
    abstain_merge_max_support: float = -1.0,
) -> List[dict]:
    """Shift provisional boundaries using local boundary-refinement predictions.

    When `abstain_merge_max_support` is enabled, a local verifier can also veto a
    weak coarse boundary by returning no transition. This lets the second pass
    filter low-confidence false positives instead of only shifting accepted cuts.
    """
    if not segments:
        return []

    resolved_boundaries: dict[int, int] = {}
    rejected_boundaries: set[int] = set()

    for idx in range(len(segments) - 1):
        record = refinement_results.get(idx) or refinement_results.get(str(idx))
        if not isinstance(record, dict):
            continue

        frame_ids = record.get("frame_ids", [])
        vlm_json = record.get("vlm_json", {})
        transitions = vlm_json.get("transitions", []) if isinstance(vlm_json, dict) else []

        selected_frame: Optional[int] = None
        for transition in transitions:
            try:
                local_idx = int(transition)
            except (TypeError, ValueError):
                continue
            if 0 <= local_idx < len(frame_ids):
                selected_frame = int(frame_ids[local_idx])
                break

        if selected_frame is None:
            if abstain_merge_max_support < 0:
                continue

            support, _ = _boundary_support_between(segments[idx], segments[idx + 1])
            if support <= float(abstain_merge_max_support):
                rejected_boundaries.add(idx)
            continue

        left = segments[idx]
        right = segments[idx + 1]
        clamped_frame = max(int(left["start_frame"]) + 1, min(selected_frame, int(right["end_frame"]) - 1))
        if clamped_frame <= int(left["start_frame"]) or clamped_frame >= int(right["end_frame"]):
            continue
        resolved_boundaries[idx] = clamped_frame

    rebuilt: List[dict] = []
    current = dict(segments[0])

    for idx in range(len(segments) - 1):
        right = dict(segments[idx + 1])

        if idx in rejected_boundaries:
            current["end_frame"] = int(right["end_frame"])
            current["instruction"] = _choose_instruction(current, right, fps)
            if "boundary_support_after" in right:
                current["boundary_support_after"] = right["boundary_support_after"]
            continue

        boundary_frame = int(resolved_boundaries.get(idx, segments[idx]["end_frame"]))
        boundary_frame = max(int(current["start_frame"]) + 1, min(boundary_frame, int(right["end_frame"]) - 1))

        support, has_support = _boundary_support_between(segments[idx], segments[idx + 1])

        finalized = dict(current)
        finalized["end_frame"] = boundary_frame
        if has_support:
            finalized["boundary_support_after"] = support
        rebuilt.append(finalized)

        current = dict(right)
        current["start_frame"] = boundary_frame
        if has_support:
            current["boundary_support_before"] = support

    rebuilt.append(current)

    for idx, segment in enumerate(rebuilt):
        segment["seg_id"] = idx
    return rebuilt


def encode_image_720p_png_bytes(
    img_bgr: np.ndarray,
    target_w: int = 720,
    target_h: int = 480,
    compression: int = 0
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
        [int(cv2.IMWRITE_PNG_COMPRESSION), int(np.clip(compression, 0, 9))]
    )
    return buf.tobytes() if ok else b""


def encode_image_720p_png(
    img_bgr: np.ndarray,
    target_w: int = 720,
    target_h: int = 480,
    compression: int = 0
) -> str:
    """Encode image to base64 PNG, resizing if needed."""
    payload = encode_image_720p_png_bytes(
        img_bgr,
        target_w=target_w,
        target_h=target_h,
        compression=compression,
    )
    return base64.b64encode(payload).decode("utf-8") if payload else ""


_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in _TRUTHY_ENV_VALUES


def _build_default_task_artifact_writer() -> Optional[TaskArtifactWriter]:
    if not _env_flag_enabled("VIDEO2TASKS_DUMP_INTERMEDIATE"):
        return None
    root_dir = os.getenv("VIDEO2TASKS_TMP_DIR", "tmp").strip() or "tmp"
    return TaskArtifactWriter(root_dir=root_dir)


class FrameExtractor:
    """Extract frames from video file."""

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
            'ffmpeg',
            '-hide_banner',
            '-loglevel',
            'error',
            '-i',
            self.mp4_path,
            '-vf',
            filter_chain,
            '-frames:v',
            '1',
            '-f',
            'image2pipe',
            '-vcodec',
            'png',
            '-',
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            stderr = proc.stderr.decode('utf-8', errors='replace').strip()
            if stderr:
                print(f"[FrameExtractor] ffmpeg contact sheet failed: {stderr}")
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
        return base64.b64encode(payload).decode('utf-8') if payload else ""

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

            for fid in sorted_indices:
                bgr = self._read_frame_bgr(fid)
                frame_payload_map[fid] = encode_image_720p_png_bytes(
                    bgr, target_w, target_h, compression
                ) if bgr is not None else b""

            image_payloads = [frame_payload_map.get(fid, b"") for fid in frame_ids]
            images = [
                base64.b64encode(payload).decode("utf-8") if payload else ""
                for payload in image_payloads
            ] if return_images else []
            if persist_artifacts:
                frame_groups = [[int(fid)] for fid in frame_ids]
                source_tags = ["cv2_frame" if frame_payload_map.get(fid, b"") else "missing" for fid in frame_ids]
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
        images: List[str] = []
        image_payloads: List[bytes] = []
        frame_groups = chunk_frame_ids_for_contact_sheets(frame_ids, chunk_size)
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
            if return_images:
                images.append(base64.b64encode(payload).decode("utf-8") if payload else "")

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

_DESTINATION_SPLIT_RE = re.compile(
    r"\b(?:to|onto|into|over|inside|within|toward|towards|from|in|on)\b"
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "again", "all", "around", "at", "away", "body", "by",
    "central", "different", "for", "it", "its", "left", "of", "off", "out",
    "right", "same", "smallest", "the", "their", "them", "this", "those",
    "through", "to", "up", "using", "with",
}
_STRONG_ACTION_TOKENS = {
    "pick", "place", "stack", "lift", "move", "carry", "separate", "connect",
    "plug", "retrieve", "open", "close", "insert", "remove", "transfer",
    "pour", "fold", "nest",
}
_GENERIC_ACTION_TOKENS = {
    "adjust", "position", "reposition", "manipulate", "handle", "interact",
    "hold", "tilt", "push", "stabilize", "align", "support",
}
_PREP_ACTION_TOKENS = {"prepare", "begin", "start", "reach", "approach", "align", "hover"}
_ACTION_FILLERS = {
    "grasp", "release", "put", "set", "make", "keep", "continue", "moving",
    "placing", "picking", "lifting", "holding",
}
_ROBOT_MOTION_TOKENS = {"gripper", "robot", "arm", "workspace", "area", "work"}
_FILLER_TOKENS = {"wait", "explain", "describe", "narrate", "instruction", "gesture", "gesturing"}
_COOKING_CONTAINER_TOKENS = {"bowl", "pot", "pan", "processor", "salad"}
_PREP_INGREDIENT_ACTION_TOKENS = {
    "chop", "slice", "dice", "mince", "grate", "tear", "peel", "cut",
    "crush", "smash", "slit", "trim", "roll", "flatten",
}
_RECIPE_ACTION_TOKENS = {
    "add", "season", "stir", "mix", "whisk", "pour", "cook", "saute", "grind",
    "sprinkle", "toss", "fry", "boil", "toast",
    "fill", "fold", "roll",
}
_MIXING_ACTION_TOKENS = {"season", "stir", "mix", "whisk", "toss"}
_ACTION_FAMILY_SUPPORT_THRESHOLD = 1.2
_SAUCE_BASE_TOKENS = {"sauce", "tomato", "onion", "liquid", "mixture"}
_GENERIC_PHASE_NOUN_TOKENS = {"mixture", "content", "contents", "sauce", "base"}
_GENERIC_REFERENCE_TOKENS = {"component", "food", "ingredient", "ingredients"}
_DISH_FOCUS_TOKENS = {"salad"}
_FOCUS_NOISE_TOKENS = {
    "adding", "additional", "combined", "finish", "mash", "season", "simmering",
    "smooth", "sprinkle", "stir", "together", "use", "well", "tong",
}
_POT_LOADING_CONTEXT_TOKENS = {"butter", "mint", "nutmeg", "salt", "sugar"}
_HEATED_STAGE_TOKENS = {"boil", "cook", "fill", "filled", "fry", "rinse", "saut", "saute", "simmer", "water"}
_PROTEIN_TOKENS = {"bacon", "beef", "chicken", "fish", "ham", "hot", "meat", "pork", "sausage", "turkey"}
_MASH_CONTEXT_TOKENS = {"butter", "liquid", "mash", "masher", "milk", "potato"}
_GENERIC_ADDITIVE_TOKENS = {
    "butter", "liquid", "oil", "pepper", "salt", "seasoning", "spice",
    "spices", "sugar", "water",
}
_SEASONING_FOCUS_TOKENS = _GENERIC_ADDITIVE_TOKENS | {"five", "powder"}
_UTENSIL_TOKENS = {"fork", "knife", "ladle", "masher", "processor", "spoon", "tong", "tongs"}
_DESTINATION_FOCUS_SKIP_TOKENS = (
    _GENERIC_ADDITIVE_TOKENS
    | _UTENSIL_TOKENS
    | _GENERIC_PHASE_NOUN_TOKENS
    | _FOCUS_NOISE_TOKENS
    | {"dish", "plate", "side", "top"}
)


def _singularize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("ls"):
        return token[:-1]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _instruction_tokens(text: str) -> List[str]:
    return [_singularize_token(tok) for tok in _TOKEN_RE.findall(text.lower())]


def _instruction_action_head_tokens(text: str, limit: int = 2) -> set[str]:
    tokens = _instruction_tokens(text.replace("/", " "))
    return set(tokens[:limit])


def _destination_food_tokens(text: str) -> set[str]:
    parts = _DESTINATION_SPLIT_RE.split(text.lower(), maxsplit=1)
    if len(parts) < 2:
        return set()

    tokens = []
    for token in _instruction_tokens(parts[1].replace("/", " ")):
        if (
            token in _STOPWORDS
            or token in _STRONG_ACTION_TOKENS
            or token in _GENERIC_ACTION_TOKENS
            or token in _PREP_ACTION_TOKENS
            or token in _ACTION_FILLERS
            or token in _ROBOT_MOTION_TOKENS
            or token in _FILLER_TOKENS
            or token in _COOKING_CONTAINER_TOKENS
            or token in _DESTINATION_FOCUS_SKIP_TOKENS
            or token == "new"
        ):
            continue
        tokens.append(token)
    return set(tokens)


def _destination_focus_tokens(text: str) -> set[str]:
    tokens = _destination_tokens(text)
    tokens -= (_COOKING_CONTAINER_TOKENS - _DISH_FOCUS_TOKENS)
    tokens -= _STRONG_ACTION_TOKENS
    tokens -= _GENERIC_ACTION_TOKENS
    tokens -= _PREP_ACTION_TOKENS
    tokens -= _PREP_INGREDIENT_ACTION_TOKENS
    tokens -= _RECIPE_ACTION_TOKENS
    tokens -= _GENERIC_PHASE_NOUN_TOKENS
    tokens -= _GENERIC_REFERENCE_TOKENS
    tokens -= _FOCUS_NOISE_TOKENS
    tokens -= _UTENSIL_TOKENS
    return tokens - _SEASONING_FOCUS_TOKENS


def _primary_object_tokens(text: str) -> set[str]:
    head = _DESTINATION_SPLIT_RE.split(text.lower(), maxsplit=1)[0]
    tokens = []
    for token in _instruction_tokens(head.replace("/", " ")):
        if (
            token in _STOPWORDS
            or token in _STRONG_ACTION_TOKENS
            or token in _GENERIC_ACTION_TOKENS
            or token in _PREP_ACTION_TOKENS
            or token in _ACTION_FILLERS
            or token in _ROBOT_MOTION_TOKENS
            or token == "new"
        ):
            continue
        tokens.append(token)
    return set(tokens)


def _instruction_specificity(text: str) -> int:
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    action_tokens = _instruction_action_head_tokens(text)
    if tokens & _FILLER_TOKENS:
        return -3
    strong = len(action_tokens & _STRONG_ACTION_TOKENS)
    mixing = len(action_tokens & _MIXING_ACTION_TOKENS)
    generic = len(action_tokens & _GENERIC_ACTION_TOKENS)
    prep_ingredient = len(action_tokens & _PREP_INGREDIENT_ACTION_TOKENS)
    prep = len(action_tokens & _PREP_ACTION_TOKENS)
    if prep:
        return -2
    return strong * 2 + mixing * 2 - generic - prep_ingredient


def _instruction_is_generic(text: str) -> bool:
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    action_tokens = _instruction_action_head_tokens(text)
    if tokens & _FILLER_TOKENS:
        return True
    if action_tokens & _PREP_ACTION_TOKENS:
        return True
    strong = len(action_tokens & _STRONG_ACTION_TOKENS)
    generic = len(action_tokens & _GENERIC_ACTION_TOKENS)
    return generic >= max(1, strong)


def _instruction_is_prep_like(text: str) -> bool:
    return bool(_instruction_action_head_tokens(text) & _PREP_ACTION_TOKENS)


def _instruction_is_bridge_motion(text: str) -> bool:
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    object_tokens = _primary_object_tokens(text)
    if object_tokens:
        return False
    if tokens & _ROBOT_MOTION_TOKENS:
        return True
    return ("reposition" in tokens or "move" in tokens) and "gripper" in tokens


def _instruction_is_filler_segment(text: str) -> bool:
    lower = text.lower().strip()
    if any(lower.startswith(prefix) for prefix in ("wait", "explain", "describe", "narrate")):
        return True
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    return bool(tokens & _FILLER_TOKENS)


def _instruction_is_preparatory_segment(text: str) -> bool:
    lower = text.lower().strip()
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    completion_tokens = {
        "place", "stack", "insert", "remove", "retrieve", "connect", "plug",
        "fold", "pour", "nest",
    }
    if lower.startswith("reach "):
        return True
    if "align the gripper" in lower or "align gripper" in lower:
        return True
    if "position the gripper" in lower or "position gripper" in lower:
        return True
    if "reposition the gripper" in lower:
        return True
    if "prepare" in tokens and not (tokens & completion_tokens):
        return True
    if (tokens & {"align", "hover", "reach", "approach"}) and not (tokens & completion_tokens):
        return True
    return False


def _token_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(len(left | right))


def _destination_tokens(text: str) -> set[str]:
    parts = _DESTINATION_SPLIT_RE.split(text.lower(), maxsplit=1)
    if len(parts) < 2:
        return set()

    tokens = []
    for token in _instruction_tokens(parts[1].replace("/", " ")):
        if (
            token in _STOPWORDS
            or token in _STRONG_ACTION_TOKENS
            or token in _GENERIC_ACTION_TOKENS
            or token in _PREP_ACTION_TOKENS
            or token in _ACTION_FILLERS
            or token in _ROBOT_MOTION_TOKENS
            or token in _FILLER_TOKENS
            or token == "new"
        ):
            continue
        tokens.append(token)
    return set(tokens)


def _ingredient_tokens(text: str) -> set[str]:
    tokens = _primary_object_tokens(text) - _COOKING_CONTAINER_TOKENS
    tokens -= _PREP_INGREDIENT_ACTION_TOKENS
    tokens -= _RECIPE_ACTION_TOKENS
    tokens -= {"ingredient", "mixture", "content", "topping", "top"}
    if "scallion" in tokens:
        tokens |= {"green", "onion"}
    if "green" in tokens and "onion" in tokens:
        tokens.add("scallion")
    return tokens


def _instruction_has_recipe_action(text: str) -> bool:
    return bool(_instruction_action_head_tokens(text) & _RECIPE_ACTION_TOKENS)


def _instruction_has_prep_ingredient_action(text: str) -> bool:
    return bool(_instruction_action_head_tokens(text) & _PREP_INGREDIENT_ACTION_TOKENS)


def _instruction_has_mixing_action(text: str) -> bool:
    return bool(_instruction_action_head_tokens(text) & _MIXING_ACTION_TOKENS)


def _instruction_has_generic_phase_noun(text: str) -> bool:
    return bool(set(_instruction_tokens(text.replace("/", " "))) & _GENERIC_PHASE_NOUN_TOKENS)


def _subject_focus_tokens(text: str) -> set[str]:
    tokens = _primary_object_tokens(text)
    tokens -= _STRONG_ACTION_TOKENS
    tokens -= _GENERIC_ACTION_TOKENS
    tokens -= _PREP_ACTION_TOKENS
    tokens -= _PREP_INGREDIENT_ACTION_TOKENS
    tokens -= _RECIPE_ACTION_TOKENS
    tokens -= _GENERIC_PHASE_NOUN_TOKENS
    tokens -= _GENERIC_REFERENCE_TOKENS
    tokens -= _FOCUS_NOISE_TOKENS
    tokens -= _UTENSIL_TOKENS
    return tokens - _SEASONING_FOCUS_TOKENS


def _generic_reference_subject_tokens(text: str) -> set[str]:
    tokens = _primary_object_tokens(text)
    tokens -= _STRONG_ACTION_TOKENS
    tokens -= _GENERIC_ACTION_TOKENS
    tokens -= _PREP_ACTION_TOKENS
    tokens -= _PREP_INGREDIENT_ACTION_TOKENS
    tokens -= _RECIPE_ACTION_TOKENS
    tokens -= _GENERIC_PHASE_NOUN_TOKENS
    tokens -= _FOCUS_NOISE_TOKENS
    tokens -= _UTENSIL_TOKENS
    tokens -= _SEASONING_FOCUS_TOKENS
    return tokens & _GENERIC_REFERENCE_TOKENS


def _instruction_focus_tokens(text: str) -> set[str]:
    primary_tokens = (
        _ingredient_tokens(text)
        | (_primary_object_tokens(text) - _COOKING_CONTAINER_TOKENS)
    )
    primary_tokens -= _STRONG_ACTION_TOKENS
    primary_tokens -= _GENERIC_ACTION_TOKENS
    primary_tokens -= _PREP_ACTION_TOKENS
    primary_tokens -= _PREP_INGREDIENT_ACTION_TOKENS
    primary_tokens -= _RECIPE_ACTION_TOKENS
    primary_tokens -= _GENERIC_PHASE_NOUN_TOKENS
    primary_tokens -= _GENERIC_REFERENCE_TOKENS
    primary_tokens -= _FOCUS_NOISE_TOKENS
    primary_tokens -= _UTENSIL_TOKENS

    subject_tokens = _subject_focus_tokens(text)
    destination_tokens = _destination_food_tokens(text)
    destination_focus_tokens = _destination_focus_tokens(text)
    if destination_tokens and primary_tokens <= _GENERIC_ADDITIVE_TOKENS:
        return destination_tokens

    focus_tokens = (primary_tokens - _GENERIC_ADDITIVE_TOKENS) | destination_tokens
    if destination_focus_tokens and (not focus_tokens or focus_tokens <= _SEASONING_FOCUS_TOKENS):
        return (focus_tokens - _SEASONING_FOCUS_TOKENS) | destination_focus_tokens
    if subject_tokens and (not focus_tokens or focus_tokens <= _SEASONING_FOCUS_TOKENS):
        return (focus_tokens - _SEASONING_FOCUS_TOKENS) | subject_tokens

    return focus_tokens


def _is_pre_cook_pot_loading_pair(left_instruction: str, right_instruction: str) -> bool:
    shared_pot = _destination_tokens(left_instruction) & _destination_tokens(right_instruction) & {"pot"}
    if not shared_pot:
        return False

    left_tokens = set(_instruction_tokens(left_instruction.replace("/", " ")))
    right_tokens = set(_instruction_tokens(right_instruction.replace("/", " ")))
    combined_tokens = left_tokens | right_tokens
    if combined_tokens & _HEATED_STAGE_TOKENS:
        return False
    if combined_tokens & _PROTEIN_TOKENS:
        return False
    if not (combined_tokens & _POT_LOADING_CONTEXT_TOKENS):
        return False

    left_actions = _action_families(left_instruction)
    right_actions = _action_families(right_instruction)
    if left_actions and not left_actions <= {"add"}:
        return False
    if right_actions and not right_actions <= {"add"}:
        return False
    return True


def _is_same_pot_masher_continuation_pair(left_instruction: str, right_instruction: str) -> bool:
    shared_pot = _destination_tokens(left_instruction) & _destination_tokens(right_instruction) & {"pot"}
    if not shared_pot:
        return False

    left_tokens = set(_instruction_tokens(left_instruction.replace("/", " ")))
    right_tokens = set(_instruction_tokens(right_instruction.replace("/", " ")))
    combined_tokens = left_tokens | right_tokens
    if combined_tokens & _PROTEIN_TOKENS:
        return False
    if "pea" in combined_tokens and "potato" in combined_tokens:
        return False
    if not combined_tokens & _MASH_CONTEXT_TOKENS:
        return False

    left_has_mash = bool(left_tokens & {"mash", "masher", "potato"})
    right_has_mash = bool(right_tokens & {"mash", "masher", "potato"})
    left_has_liquid_context = bool(left_tokens & {"butter", "liquid", "milk"})
    right_has_liquid_context = bool(right_tokens & {"butter", "liquid", "milk"})
    return (
        (left_has_liquid_context and right_has_mash)
        or (right_has_liquid_context and left_has_mash)
        or (left_has_mash and right_has_mash)
    )


def _is_same_ingredient_finish_chain_pair(left_instruction: str, right_instruction: str) -> bool:
    left_pot = _destination_tokens(left_instruction) & {"pot"}
    right_pot = _destination_tokens(right_instruction) & {"pot"}
    if not (left_pot or right_pot):
        return False

    shared_ingredients = _ingredient_tokens(left_instruction) & _ingredient_tokens(right_instruction)
    if not shared_ingredients or shared_ingredients & _PROTEIN_TOKENS:
        return False

    left_tokens = set(_instruction_tokens(left_instruction.replace("/", " ")))
    right_tokens = set(_instruction_tokens(right_instruction.replace("/", " ")))
    combined_tokens = left_tokens | right_tokens
    if not combined_tokens & {"butter", "liquid", "mash", "masher", "milk"}:
        return False

    left_actions = _action_families(left_instruction)
    right_actions = _action_families(right_instruction)
    if left_actions and not left_actions <= {"mix"}:
        return False
    if right_actions and not right_actions <= {"mix"}:
        return False
    return True


def _is_plated_mashed_potato_finish_chain_pair(left_instruction: str, right_instruction: str) -> bool:
    left_tokens = set(_instruction_tokens(left_instruction.replace("/", " ")))
    right_tokens = set(_instruction_tokens(right_instruction.replace("/", " ")))
    combined_tokens = left_tokens | right_tokens
    if not {"mashed", "potato"} & combined_tokens:
        return False
    if not (combined_tokens & {"dish", "plate", "top"} or _destination_tokens(left_instruction) & {"plate"} or _destination_tokens(right_instruction) & {"plate"}):
        return False
    if not combined_tokens & {"garnish", "herb", "onion", "puree", "sauce", "sausage"}:
        return False
    return True


def _instruction_mentions_wrapper(text: str) -> bool:
    lower = text.lower()
    if "spring roll" in lower or "egg roll" in lower:
        return True
    tokens = set(_instruction_tokens(text.replace("/", " ")))
    return bool(tokens & {"wrap", "wrapper"})


def _is_wrapper_fill_to_roll_pair(left_instruction: str, right_instruction: str) -> bool:
    left_tokens = set(_instruction_tokens(left_instruction.replace("/", " ")))
    right_tokens = set(_instruction_tokens(right_instruction.replace("/", " ")))
    left_head_tokens = _instruction_action_head_tokens(left_instruction)
    right_head_tokens = _instruction_action_head_tokens(right_instruction)
    left_wrapper_or_dumpling_assembly = (
        _instruction_mentions_wrapper(left_instruction)
        or (
            bool(left_tokens & {"dumpling", "pierogi", "pierogy"})
            and bool(left_tokens & {"assemble", "fill", "fold"})
        )
    )
    return (
        left_wrapper_or_dumpling_assembly
        and _instruction_mentions_wrapper(right_instruction)
        and bool((left_head_tokens & {"place", "fill", "spread", "assemble", "fold"}) or (left_tokens & {"assemble"}))
        and bool((right_head_tokens & {"roll", "fold", "tuck"}) or (right_tokens & {"pleat"}))
    )


def _is_dough_roll_continuation_pair(left_instruction: str, right_instruction: str) -> bool:
    left_tokens = set(_instruction_tokens(left_instruction.replace("/", " ")))
    right_tokens = set(_instruction_tokens(right_instruction.replace("/", " ")))
    if "dough" not in left_tokens or "dough" not in right_tokens:
        return False
    if left_tokens & {"cut", "circle", "glass", "cutter"}:
        return False
    if right_tokens & {"cut", "circle", "glass", "cutter"}:
        return False

    left_head_tokens = _instruction_action_head_tokens(left_instruction)
    right_head_tokens = _instruction_action_head_tokens(right_instruction)
    return (
        bool(left_head_tokens & {"flatten", "roll"})
        and (
            bool(right_head_tokens & {"roll"})
            or ("rolling" in right_tokens and "pin" in right_tokens)
        )
    )


def _action_tokens(text: str) -> set[str]:
    tokens = _instruction_action_head_tokens(text)
    return tokens & (
        _RECIPE_ACTION_TOKENS
        | _PREP_INGREDIENT_ACTION_TOKENS
        | _MIXING_ACTION_TOKENS
        | _STRONG_ACTION_TOKENS
    )


def _action_families(text: str) -> set[str]:
    tokens = _action_tokens(text)
    families = set()
    if tokens & {"add", "pour", "fill", "sprinkle"}:
        families.add("add")
    if tokens & {"season", "stir", "mix", "whisk", "toss"}:
        families.add("mix")
    if tokens & {"cook", "saute", "fry", "boil", "toast"}:
        families.add("cook")
    if tokens & _PREP_INGREDIENT_ACTION_TOKENS:
        families.add("prep")
    if tokens & {"place", "transfer", "remove", "move"}:
        families.add("transfer")
    return families


def _segment_duration_sec(segment: dict, fps: float) -> float:
    if fps < 1e-6:
        fps = 30.0
    return max(0.0, (segment["end_frame"] - segment["start_frame"]) / fps)


def _boundary_support_between(left: dict, right: dict) -> Tuple[float, bool]:
    supports = []
    has_support = False
    for segment, key in ((left, "boundary_support_after"), (right, "boundary_support_before")):
        if key in segment:
            has_support = True
            try:
                supports.append(max(0.0, float(segment.get(key, 0.0))))
            except (TypeError, ValueError):
                supports.append(0.0)
    return (max(supports) if supports else 0.0), has_support


def _choose_instruction(left: dict, right: dict, fps: float) -> str:
    if _is_wrapper_fill_to_roll_pair(left["instruction"], right["instruction"]):
        return right["instruction"]
    if _is_dough_roll_continuation_pair(left["instruction"], right["instruction"]):
        return right["instruction"]
    if _is_same_ingredient_finish_chain_pair(left["instruction"], right["instruction"]):
        return right["instruction"]

    shared_ingredient_tokens = _ingredient_tokens(left["instruction"]) & _ingredient_tokens(right["instruction"])
    if (
        shared_ingredient_tokens
        and _instruction_has_prep_ingredient_action(left["instruction"])
        and _instruction_has_prep_ingredient_action(right["instruction"])
        and _action_tokens(left["instruction"]) != _action_tokens(right["instruction"])
    ):
        return right["instruction"]

    left_specificity = _instruction_specificity(left["instruction"])
    right_specificity = _instruction_specificity(right["instruction"])
    if right_specificity > left_specificity:
        return right["instruction"]
    if left_specificity > right_specificity:
        return left["instruction"]

    left_duration = _segment_duration_sec(left, fps)
    right_duration = _segment_duration_sec(right, fps)
    if right_duration >= left_duration * 0.75:
        return right["instruction"]
    return left["instruction"]


def _should_merge_segments(left: dict, right: dict, fps: float, boundary_support_threshold: float = 0.9) -> bool:
    left_tokens = _primary_object_tokens(left["instruction"])
    right_tokens = _primary_object_tokens(right["instruction"])
    similarity = _token_similarity(left_tokens, right_tokens)
    left_dest_tokens = _destination_tokens(left["instruction"])
    right_dest_tokens = _destination_tokens(right["instruction"])
    shared_dest_tokens = left_dest_tokens & right_dest_tokens
    left_ingredient_tokens = _ingredient_tokens(left["instruction"])
    right_ingredient_tokens = _ingredient_tokens(right["instruction"])
    shared_ingredient_tokens = left_ingredient_tokens & right_ingredient_tokens
    left_focus_tokens = _instruction_focus_tokens(left["instruction"])
    right_focus_tokens = _instruction_focus_tokens(right["instruction"])
    shared_focus_tokens = left_focus_tokens & right_focus_tokens
    shared_action_tokens = _action_tokens(left["instruction"]) & _action_tokens(right["instruction"])
    shared_action_families = _action_families(left["instruction"]) & _action_families(right["instruction"])
    boundary_support, has_boundary_support = _boundary_support_between(left, right)
    strong_boundary = has_boundary_support and boundary_support_threshold > 0.0 and boundary_support >= boundary_support_threshold

    left_recipe = _instruction_has_recipe_action(left["instruction"])
    right_recipe = _instruction_has_recipe_action(right["instruction"])
    left_prep_ingredient = _instruction_has_prep_ingredient_action(left["instruction"])
    right_prep_ingredient = _instruction_has_prep_ingredient_action(right["instruction"])
    left_mixing = _instruction_has_mixing_action(left["instruction"])
    right_mixing = _instruction_has_mixing_action(right["instruction"])
    left_generic = _instruction_is_generic(left["instruction"])
    right_generic = _instruction_is_generic(right["instruction"])
    left_prep = _instruction_is_prep_like(left["instruction"])
    right_prep = _instruction_is_prep_like(right["instruction"])
    left_filler = _instruction_is_filler_segment(left["instruction"])
    right_filler = _instruction_is_filler_segment(right["instruction"])
    left_generic_focus = bool(_generic_reference_subject_tokens(left["instruction"])) and not bool(
        left_focus_tokens - _GENERIC_REFERENCE_TOKENS
    )
    right_generic_focus = bool(_generic_reference_subject_tokens(right["instruction"])) and not bool(
        right_focus_tokens - _GENERIC_REFERENCE_TOKENS
    )
    left_destination_focus = _destination_focus_tokens(left["instruction"])
    right_destination_focus = _destination_focus_tokens(right["instruction"])
    left_additive_only = bool(left_ingredient_tokens) and left_ingredient_tokens <= _SEASONING_FOCUS_TOKENS
    right_additive_only = bool(right_ingredient_tokens) and right_ingredient_tokens <= _SEASONING_FOCUS_TOKENS
    same_ingredient_prep_to_transfer = bool(shared_ingredient_tokens) and (
        (left_prep_ingredient and right_recipe) or (right_prep_ingredient and left_recipe)
    )
    same_ingredient_same_action_prep = (
        bool(shared_ingredient_tokens)
        and bool(shared_action_tokens)
        and left_prep_ingredient
        and right_prep_ingredient
    )
    distinct_same_ingredient_prep_steps = (
        bool(shared_ingredient_tokens)
        and left_prep_ingredient
        and right_prep_ingredient
        and not shared_action_tokens
    )
    distinct_prepped_ingredient_steps = (
        bool(shared_dest_tokens & _COOKING_CONTAINER_TOKENS)
        and left_recipe
        and right_recipe
        and not (left_mixing or right_mixing)
        and bool(left_ingredient_tokens)
        and bool(right_ingredient_tokens)
        and not shared_ingredient_tokens
        and (left_prep_ingredient or right_prep_ingredient)
    )
    same_container_same_family_recipe_steps = (
        bool(shared_dest_tokens & _COOKING_CONTAINER_TOKENS)
        and left_recipe
        and right_recipe
        and bool(shared_action_families & {"add", "mix"})
    )
    pre_cook_pot_loading_sequence = _is_pre_cook_pot_loading_pair(
        left["instruction"],
        right["instruction"],
    )
    same_pot_masher_continuation = _is_same_pot_masher_continuation_pair(
        left["instruction"],
        right["instruction"],
    )
    same_ingredient_finish_chain = _is_same_ingredient_finish_chain_pair(
        left["instruction"],
        right["instruction"],
    )
    plated_mashed_potato_finish_chain = _is_plated_mashed_potato_finish_chain_pair(
        left["instruction"],
        right["instruction"],
    )
    generic_phase_to_explicit_ingredient_shift = (
        bool(shared_dest_tokens & {"pot", "pan"})
        and bool(shared_action_families & {"mix", "cook"})
        and has_boundary_support
        and boundary_support >= 1.0
        and not (left_focus_tokens & right_focus_tokens)
        and (
            (_instruction_has_generic_phase_noun(left["instruction"]) and bool(right_focus_tokens))
            or (_instruction_has_generic_phase_noun(right["instruction"]) and bool(left_focus_tokens))
        )
    )
    heated_distinct_add_sequence = (
        bool(shared_dest_tokens & {"pot", "pan"})
        and shared_action_families == {"add"}
        and bool(left_ingredient_tokens)
        and bool(right_ingredient_tokens)
        and not shared_ingredient_tokens
    )
    same_container_shared_focus_recipe_steps = (
        bool(shared_dest_tokens & _COOKING_CONTAINER_TOKENS)
        and bool(shared_focus_tokens)
        and not heated_distinct_add_sequence
        and (
            bool(shared_action_families)
            or bool(shared_action_tokens)
            or (left_recipe and right_recipe)
            or (left_mixing and right_recipe)
            or (right_mixing and left_recipe)
            or (left_prep and right_recipe)
            or (right_prep and left_recipe)
        )
    )
    same_container_generic_focus_bridge = (
        bool(shared_dest_tokens & _COOKING_CONTAINER_TOKENS)
        and (
            (left_generic_focus and bool(right_focus_tokens - _GENERIC_REFERENCE_TOKENS))
            or (right_generic_focus and bool(left_focus_tokens - _GENERIC_REFERENCE_TOKENS))
        )
        and (
            bool(shared_action_families)
            or (left_prep_ingredient and right_prep_ingredient)
            or (left_prep_ingredient and right_recipe)
            or (right_prep_ingredient and left_recipe)
            or (left_prep_ingredient and right_mixing)
            or (right_prep_ingredient and left_mixing)
        )
    )
    same_container_additive_mix_bridge = (
        bool(shared_dest_tokens & _COOKING_CONTAINER_TOKENS)
        and (
            (left_additive_only and bool(left_destination_focus) and right_mixing and bool(right_focus_tokens))
            or (right_additive_only and bool(right_destination_focus) and left_mixing and bool(left_focus_tokens))
        )
    )
    wrapper_fill_to_roll = _is_wrapper_fill_to_roll_pair(left["instruction"], right["instruction"])
    dough_roll_continuation = _is_dough_roll_continuation_pair(
        left["instruction"],
        right["instruction"],
    )

    if similarity < 0.34:
        # Adjacent cooking sub-steps often swap ingredients while staying in the same bowl/pot.
        if not (
            left_prep
            or right_prep
            or left_filler
            or right_filler
            or pre_cook_pot_loading_sequence
            or same_pot_masher_continuation
            or same_ingredient_finish_chain
            or plated_mashed_potato_finish_chain
            or (
                same_container_shared_focus_recipe_steps
                and (not has_boundary_support or boundary_support < _ACTION_FAMILY_SUPPORT_THRESHOLD)
            )
            or (
                same_container_generic_focus_bridge
                and (not has_boundary_support or boundary_support < _ACTION_FAMILY_SUPPORT_THRESHOLD)
            )
            or (
                same_container_additive_mix_bridge
                and (not has_boundary_support or boundary_support < _ACTION_FAMILY_SUPPORT_THRESHOLD)
            )
            or (
                shared_dest_tokens & _COOKING_CONTAINER_TOKENS
                and (left_recipe or right_recipe)
                and (shared_ingredient_tokens or shared_action_tokens)
            )
            or same_ingredient_prep_to_transfer
            or same_ingredient_same_action_prep
            or (distinct_same_ingredient_prep_steps and has_boundary_support and not strong_boundary)
            or (
                same_container_same_family_recipe_steps
                and not heated_distinct_add_sequence
                and has_boundary_support
                and boundary_support < _ACTION_FAMILY_SUPPORT_THRESHOLD
            )
            or wrapper_fill_to_roll
            or dough_roll_continuation
        ):
            return False

    left_duration = _segment_duration_sec(left, fps)
    right_duration = _segment_duration_sec(right, fps)

    if distinct_prepped_ingredient_steps:
        return False
    if generic_phase_to_explicit_ingredient_shift:
        return False
    if (
        same_container_shared_focus_recipe_steps
        and (not has_boundary_support or boundary_support < _ACTION_FAMILY_SUPPORT_THRESHOLD)
        and max(left_duration, right_duration) <= 18.0
    ):
        return True
    if (
        same_container_generic_focus_bridge
        and (not has_boundary_support or boundary_support < _ACTION_FAMILY_SUPPORT_THRESHOLD)
        and min(left_duration, right_duration) <= 4.5
        and max(left_duration, right_duration) <= 14.0
    ):
        return True
    if (
        same_container_additive_mix_bridge
        and (not has_boundary_support or boundary_support < _ACTION_FAMILY_SUPPORT_THRESHOLD)
        and min(left_duration, right_duration) <= 12.0
        and max(left_duration, right_duration) <= 18.0
    ):
        return True
    if (
        pre_cook_pot_loading_sequence
        and has_boundary_support
        and boundary_support <= 1.0
        and max(left_duration, right_duration) <= 14.0
    ):
        return True
    if (
        same_pot_masher_continuation
        and has_boundary_support
        and boundary_support <= 1.0
        and max(left_duration, right_duration) <= 35.0
    ):
        return True
    if same_ingredient_finish_chain and max(left_duration, right_duration) <= 20.0:
        return True
    if plated_mashed_potato_finish_chain and max(left_duration, right_duration) <= 22.0:
        return True
    if distinct_same_ingredient_prep_steps:
        if dough_roll_continuation:
            return True
        if not has_boundary_support:
            return False
        if strong_boundary:
            return False
        if min(left_duration, right_duration) <= 8.0:
            return True
        return False

    if left_generic and right_generic and not (left_prep or right_prep or left_filler or right_filler):
        return True
    if left_generic and left_duration <= 4.5 and not (left_prep or left_filler):
        return True
    if right_generic and right_duration <= 4.5 and not (right_prep or right_filler):
        return True
    if wrapper_fill_to_roll:
        return True
    if shared_dest_tokens & _COOKING_CONTAINER_TOKENS and (left_recipe or right_recipe):
        if shared_ingredient_tokens and min(left_duration, right_duration) <= 8.0:
            return True
        if shared_action_tokens and max(left_duration, right_duration) <= 8.0:
            return True
        if (left_mixing or right_mixing) and shared_ingredient_tokens and max(left_duration, right_duration) <= 20.0:
            return True
    if same_ingredient_prep_to_transfer:
        if min(left_duration, right_duration) <= 8.0:
            return True
    if same_ingredient_same_action_prep:
        if min(left_duration, right_duration) <= 8.0:
            return True
    if (
        same_container_same_family_recipe_steps
        and not heated_distinct_add_sequence
        and has_boundary_support
        and boundary_support < _ACTION_FAMILY_SUPPORT_THRESHOLD
    ):
        if min(left_duration, right_duration) <= 16.0:
            return True
    if similarity >= 0.6 and min(left_duration, right_duration) <= 6.5:
        return True
    return False


def merge_task_level_segments(
    segments: List[dict],
    fps: float,
    boundary_support_threshold: float = 0.9,
) -> List[dict]:
    """Merge over-segmented adjacent spans into task-level segments."""
    if not segments:
        return []

    merged: List[dict] = []

    for segment in segments:
        current = dict(segment)
        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        if _should_merge_segments(previous, current, fps, boundary_support_threshold=boundary_support_threshold):
            chosen_instruction = _choose_instruction(previous, current, fps)
            previous["end_frame"] = current["end_frame"]
            previous["instruction"] = chosen_instruction
            previous["confidence"] = max(
                float(previous.get("confidence", 0.0)),
                float(current.get("confidence", 0.0)),
            )
            previous["boundary_support_after"] = current.get(
                "boundary_support_after",
                previous.get("boundary_support_after", 0.0),
            )
        else:
            merged.append(current)

    return cleanup_auxiliary_segments(merged, fps)


def cleanup_auxiliary_segments(segments: List[dict], fps: float) -> List[dict]:
    """Absorb bridge/prep filler spans without semantic task merging."""
    if not segments:
        return []

    bridge_cleaned: List[dict] = []
    pending_bridge: Optional[dict] = None
    for segment in segments:
        current = dict(segment)
        if _instruction_is_bridge_motion(current["instruction"]):
            if pending_bridge is None:
                pending_bridge = current
            else:
                pending_bridge["end_frame"] = current["end_frame"]
                pending_bridge["confidence"] = max(
                    float(pending_bridge.get("confidence", 0.0)),
                    float(current.get("confidence", 0.0)),
                )
            continue
        if (
            _instruction_is_filler_segment(current["instruction"])
            and _segment_duration_sec(current, fps) <= 5.0
        ):
            if pending_bridge is None:
                pending_bridge = current
            else:
                pending_bridge["end_frame"] = current["end_frame"]
                pending_bridge["confidence"] = max(
                    float(pending_bridge.get("confidence", 0.0)),
                    float(current.get("confidence", 0.0)),
                )
            continue

        if pending_bridge is not None:
            current["start_frame"] = pending_bridge["start_frame"]
            current["confidence"] = max(
                float(current.get("confidence", 0.0)),
                float(pending_bridge.get("confidence", 0.0)),
            )
            pending_bridge = None

        bridge_cleaned.append(current)

    if pending_bridge is not None and bridge_cleaned:
        bridge_cleaned[-1]["end_frame"] = pending_bridge["end_frame"]
        bridge_cleaned[-1]["confidence"] = max(
            float(bridge_cleaned[-1].get("confidence", 0.0)),
            float(pending_bridge.get("confidence", 0.0)),
        )

    filler_adjusted: List[dict] = []
    for idx, segment in enumerate(bridge_cleaned):
        current = dict(segment)
        current_duration = _segment_duration_sec(current, fps)
        if (
            _instruction_is_filler_segment(current["instruction"])
            and 5.0 < current_duration <= 12.0
            and filler_adjusted
            and idx + 1 < len(bridge_cleaned)
        ):
            previous = filler_adjusted[-1]
            following = bridge_cleaned[idx + 1]
            previous_specific = _is_specific_instruction(previous["instruction"])
            following_specific = _is_specific_instruction(following["instruction"])
            previous_dest = _destination_tokens(previous["instruction"]) & _COOKING_CONTAINER_TOKENS
            following_dest = _destination_tokens(following["instruction"]) & _COOKING_CONTAINER_TOKENS
            workspace_shift = (
                previous_dest != following_dest
                or _token_similarity(
                    _primary_object_tokens(previous["instruction"]),
                    _primary_object_tokens(following["instruction"]),
                ) < 0.25
            )
            if previous_specific and following_specific and workspace_shift:
                previous["end_frame"] = current["end_frame"]
                previous["confidence"] = max(
                    float(previous.get("confidence", 0.0)),
                    float(current.get("confidence", 0.0)),
                )
                continue

        filler_adjusted.append(current)

    prep_cleaned: List[dict] = []
    pending_prep: Optional[dict] = None
    for segment in (filler_adjusted or bridge_cleaned or segments):
        current = dict(segment)
        if _instruction_is_preparatory_segment(current["instruction"]) and _segment_duration_sec(current, fps) <= 5.0:
            if pending_prep is None:
                pending_prep = current
            else:
                pending_prep["end_frame"] = current["end_frame"]
                pending_prep["confidence"] = max(
                    float(pending_prep.get("confidence", 0.0)),
                    float(current.get("confidence", 0.0)),
                )
            continue

        if pending_prep is not None:
            current["start_frame"] = pending_prep["start_frame"]
            current["confidence"] = max(
                float(current.get("confidence", 0.0)),
                float(pending_prep.get("confidence", 0.0)),
            )
            pending_prep = None

        prep_cleaned.append(current)

    if pending_prep is not None and prep_cleaned:
        if _segment_duration_sec(pending_prep, fps) > 3.0:
            prep_cleaned[-1]["end_frame"] = pending_prep["end_frame"]
            prep_cleaned[-1]["confidence"] = max(
                float(prep_cleaned[-1].get("confidence", 0.0)),
                float(pending_prep.get("confidence", 0.0)),
            )

    final_segments = prep_cleaned or filler_adjusted or bridge_cleaned or segments

    for idx, segment in enumerate(final_segments):
        segment["seg_id"] = idx

    return final_segments


def _segment_overlap_frames(left: dict, right: dict) -> int:
    return max(
        0,
        min(int(left["end_frame"]), int(right["end_frame"])) - max(int(left["start_frame"]), int(right["start_frame"]))
    )


def _contributors_for_segment(segment: dict, source_segments: List[dict]) -> List[dict]:
    contributors = []
    for candidate in source_segments:
        overlap = _segment_overlap_frames(segment, candidate)
        if overlap > 0:
            enriched = dict(candidate)
            enriched["_overlap_frames"] = overlap
            contributors.append(enriched)
    return contributors


def _is_specific_instruction(text: str) -> bool:
    return not (
        _instruction_is_generic(text)
        or _instruction_is_prep_like(text)
        or _instruction_is_filler_segment(text)
    )


def _merged_segment_looks_overcollapsed(
    segment: dict,
    source_segments: List[dict],
    fps: float,
    median_source_duration_sec: float,
) -> bool:
    contributors = _contributors_for_segment(segment, source_segments)
    if len(contributors) < 3:
        return False

    distinct_specific = {
        item["instruction"].strip().lower()
        for item in contributors
        if _is_specific_instruction(item["instruction"])
    }
    if len(distinct_specific) < 2:
        return False

    duration_sec = _segment_duration_sec(segment, fps)
    return duration_sec >= max(10.0, median_source_duration_sec * 1.8)


def _should_fallback_to_light_cleanup(
    light_segments: List[dict],
    merged_segments: List[dict],
    fps: float,
    min_segments: int,
    collapse_ratio: float,
) -> bool:
    if len(light_segments) < min_segments:
        return False
    if not merged_segments:
        return False
    if len(merged_segments) >= max(1, int(np.ceil(len(light_segments) * collapse_ratio))):
        return False

    source_durations = [_segment_duration_sec(segment, fps) for segment in light_segments]
    median_source_duration_sec = float(np.median(source_durations)) if source_durations else 0.0
    return any(
        _merged_segment_looks_overcollapsed(segment, light_segments, fps, median_source_duration_sec)
        for segment in merged_segments
    )


def _score_instruction_candidate(segment: dict, candidate: dict, current_instruction: str) -> float:
    overlap = float(candidate.get("_overlap_frames", 0))
    text = candidate["instruction"]
    score = overlap
    score += float(_instruction_specificity(text)) * 24.0
    if text == current_instruction:
        score += 18.0
    if _instruction_is_generic(text):
        score -= 12.0
    if _instruction_is_prep_like(text):
        score -= 16.0
    if _instruction_is_filler_segment(text):
        score -= 24.0
    return score


def refine_segment_instructions(final_segments: List[dict], source_segments: List[dict]) -> List[dict]:
    """Refine final labels from contributing pre-merge segment labels."""
    refined: List[dict] = []
    for segment in final_segments:
        current = dict(segment)
        contributors = _contributors_for_segment(current, source_segments)
        if not contributors:
            refined.append(current)
            continue

        best = max(
            contributors,
            key=lambda candidate: _score_instruction_candidate(current, candidate, current["instruction"]),
        )
        current_is_weak = not _is_specific_instruction(current["instruction"])
        best_is_specific = _is_specific_instruction(best["instruction"])
        overlap_ratio = (
            float(best["_overlap_frames"]) / max(1.0, float(current["end_frame"] - current["start_frame"]))
        )

        if best_is_specific and (current_is_weak or overlap_ratio >= 0.45):
            current["instruction"] = best["instruction"]

        refined.append(current)

    for idx, segment in enumerate(refined):
        segment["seg_id"] = idx
    return refined


def apply_deferred_segment_labels(final_segments: List[dict], label_results: dict) -> List[dict]:
    """Override final segment instructions with a second-pass segment labeling result."""
    labeled: List[dict] = []
    for segment in final_segments:
        current = dict(segment)
        seg_id = int(current.get("seg_id", len(labeled)))
        label_payload = label_results.get(seg_id) or label_results.get(str(seg_id))
        if isinstance(label_payload, dict):
            instructions = label_payload.get("instructions", [])
            if isinstance(instructions, list) and instructions:
                text = str(instructions[0]).strip()
                if text and text.lower() != "unknown":
                    current["instruction"] = text
        labeled.append(current)

    for idx, segment in enumerate(labeled):
        segment["seg_id"] = idx
    return labeled


def _dominant_instruction_from_candidates(candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None

    counts = Counter(candidates)
    return max(
        counts.items(),
        key=lambda item: (
            item[1],
            _instruction_specificity(item[0]),
            len(_ingredient_tokens(item[0])),
            len(_action_families(item[0])),
        ),
    )[0]


def _phase_container_tokens(text: str, fallback_containers: Optional[set[str]] = None) -> set[str]:
    explicit_containers = _destination_tokens(text) & _COOKING_CONTAINER_TOKENS
    if explicit_containers:
        return explicit_containers

    fallback = set(fallback_containers or ())
    if not fallback:
        return set()

    tokens = set(_instruction_tokens(text.replace("/", " ")))
    if tokens & (_SAUCE_BASE_TOKENS | {"sausage"}):
        return fallback
    return set()


def _is_sauce_base_phase(text: str, fallback_containers: Optional[set[str]] = None) -> bool:
    containers = _phase_container_tokens(text, fallback_containers) & {"pot", "pan"}
    if not containers:
        return False

    action_families = _action_families(text)
    if not action_families or not action_families <= {"add", "mix", "cook"}:
        return False

    ingredients = _ingredient_tokens(text)
    if not ingredients:
        return False

    return bool(ingredients & _SAUCE_BASE_TOKENS) and ingredients <= (_SAUCE_BASE_TOKENS | {"sauteed"})


def _instruction_phase_signature(text: str, fallback_containers: Optional[set[str]] = None) -> tuple:
    action_families = tuple(sorted(_action_families(text)))
    if not action_families:
        action_families = tuple(sorted(_action_tokens(text)))

    dest_tokens = tuple(sorted(_phase_container_tokens(text, fallback_containers)))
    if _is_sauce_base_phase(text, fallback_containers):
        return (("sauce_base",), dest_tokens[:1], ("sauce_base",))

    focus_tokens = list(sorted(_ingredient_tokens(text)))
    if not focus_tokens:
        focus_tokens = list(sorted(_primary_object_tokens(text) - _COOKING_CONTAINER_TOKENS))

    return (
        action_families,
        dest_tokens[:1],
        tuple(focus_tokens[:2]),
    )


def _should_split_on_instruction_drift(
    left_instruction: str,
    right_instruction: str,
    fallback_containers: Optional[set[str]] = None,
) -> bool:
    left_actions = _action_families(left_instruction)
    right_actions = _action_families(right_instruction)

    shared_containers = (
        _phase_container_tokens(left_instruction, fallback_containers)
        & _phase_container_tokens(right_instruction, fallback_containers)
        & _COOKING_CONTAINER_TOKENS
    )
    if not shared_containers:
        return False

    if shared_containers & {"bowl", "processor", "salad"}:
        if left_actions <= {"add", "mix"} and right_actions <= {"add", "mix"}:
            return False

    if left_actions != right_actions and (left_actions or right_actions):
        return True

    left_ingredients = _ingredient_tokens(left_instruction)
    right_ingredients = _ingredient_tokens(right_instruction)
    if (
        left_actions == right_actions == {"add"}
        and left_ingredients
        and right_ingredients
        and not (left_ingredients & right_ingredients)
    ):
        return True

    return False


def _instruction_runs_for_segment(
    segment: dict,
    instruction_timeline: List[List[str]],
    fps: float,
    bin_sec: float = 2.0,
) -> List[dict]:
    bin_frames = max(1, int(round(max(0.5, bin_sec) * max(fps, 1.0))))
    runs: List[dict] = []
    context_containers = _destination_tokens(segment["instruction"]) & _COOKING_CONTAINER_TOKENS

    for start in range(int(segment["start_frame"]), int(segment["end_frame"]), bin_frames):
        end = min(int(segment["end_frame"]), start + bin_frames)
        candidates: List[str] = []
        for fid in range(start, end):
            if 0 <= fid < len(instruction_timeline):
                candidates.extend(instruction_timeline[fid])

        instruction = _dominant_instruction_from_candidates(candidates)
        if not instruction:
            continue

        signature = _instruction_phase_signature(instruction, fallback_containers=context_containers)
        if runs and runs[-1]["signature"] == signature:
            runs[-1]["end_frame"] = end
            runs[-1]["candidates"].append(instruction)
            runs[-1]["instruction"] = _dominant_instruction_from_candidates(runs[-1]["candidates"]) or runs[-1]["instruction"]
            continue

        runs.append({
            "start_frame": start,
            "end_frame": end,
            "instruction": instruction,
            "signature": signature,
            "candidates": [instruction],
        })

    return runs


def _split_segment_on_instruction_drift(
    segment: dict,
    instruction_timeline: List[List[str]],
    fps: float,
    min_segment_frames: int,
    min_phase_frames: int,
    depth: int,
) -> List[dict]:
    current = dict(segment)
    if depth <= 0 or (int(current["end_frame"]) - int(current["start_frame"])) < min_segment_frames:
        return [current]

    context_containers = _destination_tokens(current["instruction"]) & _COOKING_CONTAINER_TOKENS
    runs = _instruction_runs_for_segment(current, instruction_timeline, fps)
    if len(runs) < 2:
        return [current]

    candidates = []
    for idx in range(len(runs) - 1):
        left_run = runs[idx]
        right_run = runs[idx + 1]
        left_duration = int(left_run["end_frame"]) - int(left_run["start_frame"])
        right_duration = int(right_run["end_frame"]) - int(right_run["start_frame"])
        if left_duration < min_phase_frames or right_duration < min_phase_frames:
            continue
        if not _should_split_on_instruction_drift(
            left_run["instruction"],
            right_run["instruction"],
            fallback_containers=context_containers,
        ):
            continue

        boundary = int((int(left_run["end_frame"]) + int(right_run["start_frame"])) / 2)
        if boundary - int(current["start_frame"]) < min_phase_frames:
            continue
        if int(current["end_frame"]) - boundary < min_phase_frames:
            continue

        score = float(min(left_duration, right_duration))
        if _action_families(left_run["instruction"]) != _action_families(right_run["instruction"]):
            score += float(min_phase_frames)
        if _is_specific_instruction(left_run["instruction"]):
            score += 0.25 * min_phase_frames
        if _is_specific_instruction(right_run["instruction"]):
            score += 0.25 * min_phase_frames

        candidates.append((score, boundary, left_run, right_run))

    if not candidates:
        return [current]

    _, boundary, left_run, right_run = max(candidates, key=lambda item: item[0])
    left_segment = dict(current)
    left_segment["end_frame"] = boundary
    left_segment["instruction"] = left_run["instruction"]
    left_segment["boundary_support_after"] = 0.0

    right_segment = dict(current)
    right_segment["start_frame"] = boundary
    right_segment["instruction"] = right_run["instruction"]
    right_segment["boundary_support_before"] = 0.0

    return (
        _split_segment_on_instruction_drift(
            left_segment,
            instruction_timeline,
            fps,
            min_segment_frames,
            min_phase_frames,
            depth - 1,
        )
        + _split_segment_on_instruction_drift(
            right_segment,
            instruction_timeline,
            fps,
            min_segment_frames,
            min_phase_frames,
            depth - 1,
        )
    )


def split_long_raw_segments_on_instruction_drift(
    raw_segments: List[dict],
    instruction_timeline: List[List[str]],
    fps: float,
    min_segment_sec: float = 30.0,
    min_phase_sec: float = 5.0,
) -> List[dict]:
    if not raw_segments:
        return []

    if fps < 1e-6:
        fps = 30.0

    min_segment_frames = max(1, int(round(min_segment_sec * fps)))
    min_phase_frames = max(1, int(round(min_phase_sec * fps)))
    split_segments: List[dict] = []

    for segment in raw_segments:
        split_segments.extend(
            _split_segment_on_instruction_drift(
                segment,
                instruction_timeline,
                fps,
                min_segment_frames,
                min_phase_frames,
                depth=3,
            )
        )

    for idx, segment in enumerate(split_segments):
        segment["seg_id"] = idx

    return split_segments


def build_segments_via_cuts(
    sample_id: str,
    windows: List[Window],
    by_wid: dict,
    fps: float,
    nframes: int,
    frames_per_window: int = 16,
    boundary_prompt_mode: str = "freeform",
    adaptive_merge_guard: bool = True,
    adaptive_merge_min_segments: int = 8,
    adaptive_merge_collapse_ratio: float = 0.6,
    boundary_support_threshold: float = 0.9,
    refine_final_instructions: bool = True,
) -> dict:
    """Build final segments from window results."""
    if nframes == 0:
        return {}
    
    if fps < 1e-6:
        fps = 30.0
    
    raw_cuts = []
    instruction_timeline = [[] for _ in range(nframes)]
    center_weights = np.hanning(frames_per_window + 2)[1:-1]
    
    for w in windows:
        rec = by_wid.get(w.window_id)
        if not rec:
            continue
        
        vlm = rec.get("vlm_json", {})
        transitions = vlm.get("transitions", [])
        instructions = vlm.get("instructions", [])
        f_ids = w.frame_ids
        cur_len = len(f_ids)
        
        if cur_len == 0:
            continue
        
        # Collect cut points
        for t_idx in transitions:
            try:
                idx = int(t_idx)
                if 0 <= idx < cur_len:
                    global_fid = f_ids[idx]
                    if cur_len == frames_per_window:
                        w_val = center_weights[idx]
                    else:
                        w_val = 1.0 if min(idx, cur_len - 1 - idx) > 2 else 0.5
                    raw_cuts.append((global_fid, float(w_val)))
            except (ValueError, IndexError):
                pass
        
        # Collect instructions
        try:
            boundaries = [0] + [int(t) for t in transitions if 0 <= int(t) < cur_len] + [cur_len]
            boundaries = sorted(list(set(boundaries)))
            
            for i in range(len(boundaries) - 1):
                if i < len(instructions):
                    inst = str(instructions[i]).strip()
                    if inst and inst.lower() != "unknown":
                        s_local, e_local = boundaries[i], boundaries[i + 1]
                        for k in range(s_local, e_local):
                            if k < cur_len:
                                global_fid = f_ids[k]
                                if global_fid < nframes:
                                    instruction_timeline[global_fid].append(inst)
        except (ValueError, IndexError):
            pass
    
    # Cluster cuts
    final_cut_points = [0]
    cut_support_by_point: dict[int, float] = {}
    
    if raw_cuts:
        clustered_points, clustered_support = _cluster_cut_votes(raw_cuts, fps)
        final_cut_points.extend(clustered_points)
        cut_support_by_point.update(clustered_support)
    
    final_cut_points.append(nframes)
    final_cut_points = sorted(list(set(final_cut_points)))

    if boundary_prompt_mode in {"center_scan", "multi_probe_scan"}:
        # Local probe-style prompting is much less reliable when only one window votes
        # for a boundary. Require corroboration from overlapping windows before
        # promoting an interior cut into the final segmentation.
        min_cluster_support = max(1.0, float(np.max(center_weights))) + 1e-6
        corroboration_gap_frames = max(1, int(round(fps)))
        interior_points = sorted(int(point) for point in final_cut_points[1:-1])
        corroborated_support_by_point: dict[int, float] = {}
        corroborated_points: List[int] = []

        for point in interior_points:
            neighborhood_support = float(
                sum(
                    float(cut_support_by_point.get(other, 0.0))
                    for other in interior_points
                    if abs(other - point) <= corroboration_gap_frames
                )
            )
            if neighborhood_support >= min_cluster_support:
                corroborated_points.append(point)
                corroborated_support_by_point[point] = neighborhood_support

        merged_probe_points: List[int] = []
        if corroborated_points:
            cur_probe_cluster: List[int] = []

            def flush_probe_cluster() -> None:
                if not cur_probe_cluster:
                    return
                representative = min(
                    cur_probe_cluster,
                    key=lambda point: (
                        -corroborated_support_by_point.get(point, 0.0),
                        point,
                    ),
                )
                merged_probe_points.append(representative)

            for point in sorted(set(corroborated_points)):
                if not cur_probe_cluster or (point - cur_probe_cluster[-1]) <= corroboration_gap_frames:
                    cur_probe_cluster.append(point)
                    continue
                flush_probe_cluster()
                cur_probe_cluster = [point]

            flush_probe_cluster()

        final_cut_points = [0] + merged_probe_points + [nframes]
        cut_support_by_point = {
            point: float(corroborated_support_by_point.get(point, cut_support_by_point.get(point, 0.0)))
            for point in merged_probe_points
        }
    
    # Build segments
    raw_segments = []
    seg_id = 0
    
    for i in range(len(final_cut_points) - 1):
        s, e = int(final_cut_points[i]), int(final_cut_points[i + 1])
        min_frames = max(1, int(0.8 * fps))
        
        if (e - s) < min_frames:
            continue
        
        margin = int((e - s) * 0.2) if e > s else 0
        mid_s, mid_e = s + margin, e - margin
        
        candidates = []
        for f in range(mid_s, mid_e + 1):
            if f < nframes:
                candidates.extend(instruction_timeline[f])
        
        if not candidates:
            for f in range(s, e):
                if f < nframes:
                    candidates.extend(instruction_timeline[f])
        
        if candidates:
            best_inst = Counter(candidates).most_common(1)[0][0]
            raw_segments.append({
                "seg_id": seg_id,
                "start_frame": s,
                "end_frame": e,
                "instruction": best_inst,
                "confidence": 1.0,
                "boundary_support_before": float(cut_support_by_point.get(s, 0.0)),
                "boundary_support_after": float(cut_support_by_point.get(e, 0.0)),
            })
            seg_id += 1

    raw_segments = split_long_raw_segments_on_instruction_drift(
        raw_segments,
        instruction_timeline,
        fps,
    )
    light_segments = cleanup_auxiliary_segments(raw_segments, fps)
    merged_segments = merge_task_level_segments(
        raw_segments,
        fps,
        boundary_support_threshold=boundary_support_threshold,
    )
    use_light_fallback = adaptive_merge_guard and _should_fallback_to_light_cleanup(
        light_segments,
        merged_segments,
        fps,
        min_segments=adaptive_merge_min_segments,
        collapse_ratio=adaptive_merge_collapse_ratio,
    )
    final_output = light_segments if use_light_fallback else merged_segments

    if refine_final_instructions:
        final_output = refine_segment_instructions(final_output, light_segments)
    
    return {
        "sample_id": sample_id,
        "nframes": nframes,
        "segments": final_output,
        "diagnostics": {
            "light_segment_count": len(light_segments),
            "merged_segment_count": len(merged_segments),
            "selected_segment_count": len(final_output),
            "selection_policy": "light_cleanup_fallback" if use_light_fallback else "semantic_merge",
        },
    }
