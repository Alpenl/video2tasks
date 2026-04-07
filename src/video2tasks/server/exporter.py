"""Optional exported video artifacts for finalized segmentation results."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

from ..config import ExportConfig


def _ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def _pick_default_cjk_font_file() -> str:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return ""


def _escape_filter_value(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(",", "\\,")
    )


def _subtitle_xy(position: str, *, margin_x: int, margin_y: int) -> Tuple[str, str]:
    pos = str(position).strip().lower()
    if pos == "top_right":
        return (f"w-tw-{int(margin_x)}", str(int(margin_y)))
    if pos == "top_left":
        return (str(int(margin_x)), str(int(margin_y)))
    if pos == "bottom_right":
        return (f"w-tw-{int(margin_x)}", f"h-th-{int(margin_y)}")
    if pos == "bottom_left":
        return (str(int(margin_x)), f"h-th-{int(margin_y)}")
    raise ValueError(f"Unsupported subtitle position: {position}")


def _wrap_caption(text: str, *, max_chars_per_line: int = 18) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if " " in raw:
        words = raw.split()
        lines: List[str] = []
        current: List[str] = []
        current_len = 0
        for word in words:
            next_len = current_len + (1 if current else 0) + len(word)
            if current and next_len > max_chars_per_line:
                lines.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len = next_len
        if current:
            lines.append(" ".join(current))
        return "\n".join(lines)
    lines = [raw[i : i + max_chars_per_line] for i in range(0, len(raw), max_chars_per_line)]
    return "\n".join(lines)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return slug[:48] or "segment"


def _segment_instruction(segment: Dict[str, Any]) -> str:
    return str(segment.get("instruction", "")).strip() or "Unknown task step"


def _segment_subtitle(segment: Dict[str, Any]) -> str:
    subtitle = str(segment.get("export_subtitle", "")).strip()
    if subtitle:
        return subtitle
    return _segment_instruction(segment)


class ClipExporter:
    def __init__(self, export_config: ExportConfig) -> None:
        self.export_config = export_config

    def export(
        self,
        *,
        run_dir: str,
        sample_id: str,
        video_path: str,
        fps: float,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        sample_dir = Path(run_dir) / self.export_config.clips_dirname / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = sample_dir / self.export_config.manifest_name

        clip_paths: List[str] = []
        manifest_records: List[Dict[str, Any]] = []
        used_subtitle_fallback = False

        for index, segment in enumerate(segments):
            start_frame = int(segment.get("start_frame", 0))
            end_frame = int(segment.get("end_frame", start_frame))
            if end_frame <= start_frame:
                continue

            instruction = _segment_instruction(segment)
            clip_name = f"seg_{index:02d}_{_slugify(instruction)}.mp4"
            clip_path = sample_dir / clip_name
            subtitle_text = _segment_subtitle(segment) if self.export_config.subtitles.enabled else ""
            wrote_with_subtitles = self._write_clip(
                video_path,
                clip_path,
                start_frame,
                end_frame,
                fps,
                subtitle_text=subtitle_text,
            )
            if subtitle_text and not wrote_with_subtitles:
                used_subtitle_fallback = True

            clip_paths.append(str(clip_path))
            manifest_records.append(
                {
                    "segment_id": int(segment.get("seg_id", index)),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "instruction": instruction,
                    "subtitle": _segment_subtitle(segment),
                    "file": clip_name,
                }
            )

        manifest_path.write_text(
            json.dumps(manifest_records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "manifest_path": str(manifest_path),
            "clip_paths": clip_paths,
            "clip_count": len(clip_paths),
            "used_subtitle_fallback": used_subtitle_fallback,
        }

    def _write_clip(
        self,
        video_path: str,
        output_path: Path,
        start_frame: int,
        end_frame: int,
        fps: float,
        *,
        subtitle_text: str = "",
    ) -> bool:
        subtitle_text = str(subtitle_text or "").strip()
        if subtitle_text and _ffmpeg_exists():
            try:
                self._write_clip_with_ffmpeg(
                    video_path,
                    output_path,
                    start_frame,
                    end_frame,
                    fps,
                    subtitle_text=subtitle_text,
                )
                return True
            except Exception:
                pass

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video for clip export: {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                raise RuntimeError(f"Cannot read video frame for clip export: {video_path}")
            height, width = frame.shape[:2]
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else 30.0, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot create output video: {output_path}")

        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        for _ in range(int(start_frame), int(end_frame)):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            writer.write(frame)

        writer.release()
        capture.release()
        return False

    def _write_clip_with_ffmpeg(
        self,
        video_path: str,
        output_path: Path,
        start_frame: int,
        end_frame: int,
        fps: float,
        *,
        subtitle_text: str,
    ) -> None:
        safe_fps = fps if fps and fps > 1e-6 else 30.0
        start_sec = float(start_frame) / float(safe_fps)
        duration_sec = max(0.0, float(end_frame - start_frame) / float(safe_fps))
        if duration_sec <= 1e-3:
            raise ValueError("clip duration too short")

        caption_path = output_path.with_suffix(".caption.txt")
        caption_path.write_text(_wrap_caption(subtitle_text) + "\n", encoding="utf-8")

        font_file = str(self.export_config.subtitles.font_file).strip() or _pick_default_cjk_font_file()
        x_expr, y_expr = _subtitle_xy(self.export_config.subtitles.position, margin_x=24, margin_y=24)
        drawtext_parts = [
            f"textfile='{_escape_filter_value(str(caption_path))}'",
            f"fontsize={int(self.export_config.subtitles.font_size)}",
            "fontcolor=white",
            f"x={x_expr}",
            f"y={y_expr}",
            "box=1",
            "boxcolor=black@0.45",
            "boxborderw=8",
        ]
        if font_file:
            drawtext_parts.append(f"fontfile='{_escape_filter_value(font_file)}'")

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            f"{duration_sec:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            "drawtext=" + ":".join(drawtext_parts),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)


class AnnotatedVideoExporter:
    def __init__(self, export_config: ExportConfig) -> None:
        self.export_config = export_config

    def export(
        self,
        *,
        run_dir: str,
        sample_id: str,
        video_path: str,
        fps: float,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        sample_dir = Path(run_dir) / self.export_config.annotated_dirname / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_path = sample_dir / self.export_config.annotated_name

        if not self.export_config.subtitles.enabled:
            shutil.copyfile(video_path, output_path)
            return {
                "annotated_path": str(output_path),
                "copied_without_subtitles": True,
                "used_subtitle_fallback": False,
            }

        if not _ffmpeg_exists():
            shutil.copyfile(video_path, output_path)
            return {
                "annotated_path": str(output_path),
                "copied_without_subtitles": True,
                "used_subtitle_fallback": True,
            }

        safe_fps = fps if fps and fps > 1e-6 else 30.0
        font_file = str(self.export_config.subtitles.font_file).strip() or _pick_default_cjk_font_file()
        x_expr, y_expr = _subtitle_xy(self.export_config.subtitles.position, margin_x=24, margin_y=24)

        filters: List[str] = []
        caption_paths: List[str] = []
        for index, segment in enumerate(segments):
            start_frame = int(segment.get("start_frame", 0))
            end_frame = int(segment.get("end_frame", start_frame))
            if end_frame <= start_frame:
                continue

            caption_text = _wrap_caption(_segment_subtitle(segment))
            if not caption_text:
                continue

            caption_path = sample_dir / f"seg_{index:02d}.caption.txt"
            caption_path.write_text(caption_text + "\n", encoding="utf-8")
            caption_paths.append(str(caption_path))

            start_sec = float(start_frame) / float(safe_fps)
            end_sec = float(end_frame) / float(safe_fps)
            if end_sec <= start_sec + 1e-4:
                continue
            enable_expr = f"gte(t,{start_sec:.3f})*lt(t,{end_sec:.3f})"
            drawtext_parts = [
                f"textfile='{_escape_filter_value(str(caption_path))}'",
                f"fontsize={int(self.export_config.subtitles.font_size)}",
                "fontcolor=white",
                f"x={x_expr}",
                f"y={y_expr}",
                "box=1",
                "boxcolor=black@0.45",
                "boxborderw=8",
                f"enable='{enable_expr}'",
            ]
            if font_file:
                drawtext_parts.append(f"fontfile='{_escape_filter_value(font_file)}'")
            filters.append("drawtext=" + ":".join(drawtext_parts))

        if not filters:
            shutil.copyfile(video_path, output_path)
            return {
                "annotated_path": str(output_path),
                "copied_without_subtitles": True,
                "used_subtitle_fallback": True,
                "caption_paths": caption_paths,
            }

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        return {
            "annotated_path": str(output_path),
            "copied_without_subtitles": False,
            "used_subtitle_fallback": False,
            "caption_paths": caption_paths,
        }


def export_sample_outputs(
    *,
    run_dir: str,
    sample_id: str,
    video_path: str,
    fps: float,
    segments: List[Dict[str, Any]],
    export_config: ExportConfig,
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "export_enabled": bool(export_config.enabled),
        "export_attempted": False,
        "export_mode": str(export_config.mode),
        "export_subtitles_enabled": bool(export_config.subtitles.enabled),
    }
    if not export_config.enabled:
        diagnostics["export_reason"] = "disabled"
        return diagnostics

    normalized_segments = [dict(segment) for segment in segments if isinstance(segment, dict)]
    if not normalized_segments:
        diagnostics["export_reason"] = "empty_segments"
        return diagnostics

    diagnostics["export_attempted"] = True
    errors: List[str] = []
    successes = 0

    mode = str(export_config.mode).strip().lower()
    if mode in {"annotated", "both"}:
        try:
            annotated_result = AnnotatedVideoExporter(export_config).export(
                run_dir=run_dir,
                sample_id=sample_id,
                video_path=video_path,
                fps=fps,
                segments=normalized_segments,
            )
            diagnostics["export_annotated_path"] = annotated_result["annotated_path"]
            diagnostics["export_annotated_copied_without_subtitles"] = bool(
                annotated_result.get("copied_without_subtitles", False)
            )
            diagnostics["export_annotated_used_subtitle_fallback"] = bool(
                annotated_result.get("used_subtitle_fallback", False)
            )
            successes += 1
        except Exception as exc:
            errors.append(f"annotated:{type(exc).__name__}")
            diagnostics["export_annotated_error"] = str(exc).strip() or type(exc).__name__

    if mode in {"clips", "both"}:
        try:
            clips_result = ClipExporter(export_config).export(
                run_dir=run_dir,
                sample_id=sample_id,
                video_path=video_path,
                fps=fps,
                segments=normalized_segments,
            )
            diagnostics["export_clip_manifest_path"] = clips_result["manifest_path"]
            diagnostics["export_clip_count"] = int(clips_result["clip_count"])
            diagnostics["export_clips_used_subtitle_fallback"] = bool(
                clips_result.get("used_subtitle_fallback", False)
            )
            successes += 1
        except Exception as exc:
            errors.append(f"clips:{type(exc).__name__}")
            diagnostics["export_clips_error"] = str(exc).strip() or type(exc).__name__

    if errors and successes:
        diagnostics["export_reason"] = "partial_failure"
        diagnostics["export_errors"] = errors
    elif errors:
        diagnostics["export_reason"] = "failed"
        diagnostics["export_errors"] = errors
    else:
        diagnostics["export_reason"] = "applied"

    return diagnostics
