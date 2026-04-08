import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from video2tasks.config import ExportConfig
from video2tasks.server.exporter import AnnotatedVideoExporter, ClipExporter, export_sample_outputs


def _make_test_video(path: Path, *, fps: float = 10.0, frame_count: int = 30) -> None:
    width, height = 96, 64
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (idx * 7) % 255
        frame[:, :, 1] = (idx * 11) % 255
        frame[:, :, 2] = (idx * 13) % 255
        writer.write(frame)
    writer.release()


def _require_ffmpeg_tools() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")


def _make_test_video_with_audio(path: Path, *, fps: float = 10.0, frame_count: int = 30) -> None:
    _require_ffmpeg_tools()
    video_only_path = path.with_name(path.stem + ".video_only.mp4")
    _make_test_video(video_only_path, fps=fps, frame_count=frame_count)
    duration_sec = frame_count / fps if fps > 0 else frame_count / 30.0
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_only_path),
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration_sec:.3f}",
        "-shortest",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True)


def test_annotated_export_copies_video_when_subtitles_disabled(tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path)
    config = ExportConfig(enabled=True, mode="annotated", subtitles={"enabled": False})

    result = AnnotatedVideoExporter(config).export(
        run_dir=str(tmp_path / "run"),
        sample_id="sample_a",
        video_path=str(video_path),
        fps=10.0,
        segments=[{"start_frame": 0, "end_frame": 10, "instruction": "open lid"}],
    )

    annotated_path = Path(result["annotated_path"])
    assert annotated_path.exists()
    assert result["copied_without_subtitles"] is True
    assert result["used_subtitle_fallback"] is False
    assert result["contract_success"] is True
    assert annotated_path.read_bytes() == video_path.read_bytes()


def test_annotated_export_uses_ffmpeg_when_subtitles_enabled(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path)
    config = ExportConfig(enabled=True, mode="annotated", subtitles={"enabled": True, "font_size": 20})

    calls = []

    def fake_run(cmd, check):
        del check
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"annotated")

    monkeypatch.setattr("video2tasks.server.exporter._ffmpeg_exists", lambda: True)
    monkeypatch.setattr("video2tasks.server.exporter.subprocess.run", fake_run)

    result = AnnotatedVideoExporter(config).export(
        run_dir=str(tmp_path / "run"),
        sample_id="sample_b",
        video_path=str(video_path),
        fps=10.0,
        segments=[{"start_frame": 0, "end_frame": 10, "instruction": "pick up bowl", "export_subtitle": "拿起碗"}],
    )

    assert calls
    assert any("drawtext=" in part for part in calls[0])
    assert Path(result["annotated_path"]).read_bytes() == b"annotated"
    assert result["copied_without_subtitles"] is False
    assert result["used_subtitle_fallback"] is False
    assert result["contract_success"] is True
    assert result["caption_paths"]
    assert Path(result["caption_paths"][0]).read_text(encoding="utf-8").strip() == "拿起碗"


def test_clips_export_writes_manifest_and_clips(tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video_with_audio(video_path, frame_count=40)
    config = ExportConfig(enabled=True, mode="clips", subtitles={"enabled": False})
    segments = [
        {"seg_id": 0, "start_frame": 0, "end_frame": 10, "instruction": "move cup", "export_subtitle": "移动杯子"},
        {"seg_id": 1, "start_frame": 10, "end_frame": 20, "instruction": "place cup", "export_subtitle": "放下杯子"},
    ]

    result = ClipExporter(config).export(
        run_dir=str(tmp_path / "run"),
        sample_id="sample_c",
        video_path=str(video_path),
        fps=10.0,
        segments=segments,
    )

    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()
    assert result["clip_count"] == 2
    assert result["used_subtitle_fallback"] is False
    assert result["contract_success"] is True
    assert len(result["clip_paths"]) == 2
    assert all(Path(path).exists() for path in result["clip_paths"])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [record["segment_id"] for record in manifest] == [0, 1]
    assert [record["instruction"] for record in manifest] == ["move cup", "place cup"]
    assert [record["subtitle"] for record in manifest] == ["移动杯子", "放下杯子"]
    assert all(record["audio_preserved"] is True for record in manifest)
    assert all(record["render_complete"] is True for record in manifest)
    assert [record["rendered_frame_count"] for record in manifest] == [10, 10]
    assert all(record["export_status"] == "success" for record in manifest)


def test_clips_subtitle_enabled_export_marks_subtitle_truth_as_unverified(tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video_with_audio(video_path, frame_count=30)
    config = ExportConfig(enabled=True, mode="clips", subtitles={"enabled": True, "font_size": 20})

    result = ClipExporter(config).export(
        run_dir=str(tmp_path / "run"),
        sample_id="sample_subtitle_truth",
        video_path=str(video_path),
        fps=10.0,
        segments=[{"seg_id": 0, "start_frame": 0, "end_frame": 10, "instruction": "pick up bowl", "export_subtitle": "pick up bowl"}],
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert result["contract_success"] is False
    assert result["contract_status"] == "degraded"
    assert "subtitle_render_unverified" in result["contract_errors"]
    assert manifest[0]["subtitle_requested"] is True
    assert manifest[0]["subtitle_rendered"] is None
    assert manifest[0]["audio_preserved"] is True
    assert manifest[0]["render_complete"] is True
    assert manifest[0]["rendered_frame_count"] == 10
    assert manifest[0]["export_status"] == "degraded"
    assert "subtitle_render_unverified" in manifest[0]["contract_errors"]


def test_clips_manifest_records_actual_render_facts(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, frame_count=20)
    config = ExportConfig(enabled=True, mode="clips", subtitles={"enabled": True})

    def fake_write_clip(self, video_path, output_path, start_frame, end_frame, fps, *, subtitle_text=""):
        del self, video_path, fps, subtitle_text
        output_path.write_bytes(b"clip")
        return {
            "render_backend": "opencv",
            "subtitle_rendered": False,
            "audio_preserved": False,
            "render_complete": False,
            "rendered_frame_count": 3,
            "requested_frame_count": int(end_frame - start_frame),
        }

    monkeypatch.setattr(ClipExporter, "_write_clip", fake_write_clip)

    result = ClipExporter(config).export(
        run_dir=str(tmp_path / "run"),
        sample_id="sample_manifest_truth",
        video_path=str(video_path),
        fps=10.0,
        segments=[
            {
                "seg_id": 7,
                "start_frame": 4,
                "end_frame": 14,
                "instruction": "move cup",
                "export_subtitle": "移动杯子",
            }
        ],
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert result["contract_success"] is False
    assert result["contract_status"] == "degraded"
    assert manifest[0]["segment_id"] == 7
    assert manifest[0]["subtitle_requested"] is True
    assert manifest[0]["subtitle_rendered"] is False
    assert manifest[0]["audio_preserved"] is False
    assert manifest[0]["render_complete"] is False
    assert manifest[0]["render_backend"] == "opencv"
    assert manifest[0]["rendered_frame_count"] == 3
    assert manifest[0]["requested_frame_count"] == 10
    assert manifest[0]["export_status"] == "degraded"


def test_clips_export_refuses_guessed_rendered_frame_count(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, frame_count=20)
    config = ExportConfig(enabled=True, mode="clips", subtitles={"enabled": True})

    def fake_write_clip(self, video_path, output_path, start_frame, end_frame, fps, *, subtitle_text=""):
        del self, video_path, output_path, start_frame, end_frame, fps, subtitle_text
        return {
            "render_backend": "ffmpeg",
            "subtitle_rendered": True,
            "audio_preserved": True,
            "render_complete": True,
        }

    monkeypatch.setattr(ClipExporter, "_write_clip", fake_write_clip)

    with pytest.raises(RuntimeError, match="missing_render_fact:rendered_frame_count"):
        ClipExporter(config).export(
            run_dir=str(tmp_path / "run"),
            sample_id="sample_missing_rendered_frame_count",
            video_path=str(video_path),
            fps=10.0,
            segments=[{"start_frame": 0, "end_frame": 10, "instruction": "demo", "export_subtitle": "字幕"}],
        )


def test_clips_export_rejects_dirty_render_fact_types(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, frame_count=20)
    config = ExportConfig(enabled=True, mode="clips", subtitles={"enabled": True})

    def fake_write_clip(self, video_path, output_path, start_frame, end_frame, fps, *, subtitle_text=""):
        del self, video_path, output_path, start_frame, end_frame, fps, subtitle_text
        return {
            "render_backend": "ffmpeg",
            "subtitle_rendered": 0,
            "audio_preserved": "false",
            "render_complete": "false",
            "rendered_frame_count": "10",
        }

    monkeypatch.setattr(ClipExporter, "_write_clip", fake_write_clip)

    with pytest.raises(RuntimeError, match="invalid_render_fact:subtitle_rendered"):
        ClipExporter(config).export(
            run_dir=str(tmp_path / "run"),
            sample_id="sample_dirty_render_facts",
            video_path=str(video_path),
            fps=10.0,
            segments=[{"start_frame": 0, "end_frame": 10, "instruction": "demo", "export_subtitle": "字幕"}],
        )


def test_clips_export_rejects_dirty_audio_preserved_fact(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, frame_count=20)
    config = ExportConfig(enabled=True, mode="clips", subtitles={"enabled": True})

    def fake_write_clip(self, video_path, output_path, start_frame, end_frame, fps, *, subtitle_text=""):
        del self, video_path, output_path, start_frame, end_frame, fps, subtitle_text
        return {
            "render_backend": "ffmpeg",
            "subtitle_rendered": None,
            "audio_preserved": "false",
            "render_complete": True,
            "rendered_frame_count": 10,
        }

    monkeypatch.setattr(ClipExporter, "_write_clip", fake_write_clip)

    with pytest.raises(RuntimeError, match="invalid_render_fact:audio_preserved"):
        ClipExporter(config).export(
            run_dir=str(tmp_path / "run"),
            sample_id="sample_dirty_audio_preserved",
            video_path=str(video_path),
            fps=10.0,
            segments=[{"start_frame": 0, "end_frame": 10, "instruction": "demo", "export_subtitle": "字幕"}],
        )


def test_clips_export_rejects_dirty_render_complete_fact(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, frame_count=20)
    config = ExportConfig(enabled=True, mode="clips", subtitles={"enabled": True})

    def fake_write_clip(self, video_path, output_path, start_frame, end_frame, fps, *, subtitle_text=""):
        del self, video_path, output_path, start_frame, end_frame, fps, subtitle_text
        return {
            "render_backend": "ffmpeg",
            "subtitle_rendered": None,
            "audio_preserved": True,
            "render_complete": "false",
            "rendered_frame_count": 10,
        }

    monkeypatch.setattr(ClipExporter, "_write_clip", fake_write_clip)

    with pytest.raises(RuntimeError, match="invalid_render_fact:render_complete"):
        ClipExporter(config).export(
            run_dir=str(tmp_path / "run"),
            sample_id="sample_dirty_render_complete",
            video_path=str(video_path),
            fps=10.0,
            segments=[{"start_frame": 0, "end_frame": 10, "instruction": "demo", "export_subtitle": "字幕"}],
        )


def test_clips_export_rejects_dirty_rendered_frame_count_type(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, frame_count=20)
    config = ExportConfig(enabled=True, mode="clips", subtitles={"enabled": True})

    def fake_write_clip(self, video_path, output_path, start_frame, end_frame, fps, *, subtitle_text=""):
        del self, video_path, output_path, start_frame, end_frame, fps, subtitle_text
        return {
            "render_backend": "ffmpeg",
            "subtitle_rendered": None,
            "audio_preserved": True,
            "render_complete": True,
            "rendered_frame_count": "10",
        }

    monkeypatch.setattr(ClipExporter, "_write_clip", fake_write_clip)

    with pytest.raises(RuntimeError, match="invalid_render_fact:rendered_frame_count"):
        ClipExporter(config).export(
            run_dir=str(tmp_path / "run"),
            sample_id="sample_dirty_rendered_frame_count_type",
            video_path=str(video_path),
            fps=10.0,
            segments=[{"start_frame": 0, "end_frame": 10, "instruction": "demo", "export_subtitle": "字幕"}],
        )


def test_clips_export_rejects_negative_rendered_frame_count(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, frame_count=20)
    config = ExportConfig(enabled=True, mode="clips", subtitles={"enabled": True})

    def fake_write_clip(self, video_path, output_path, start_frame, end_frame, fps, *, subtitle_text=""):
        del self, video_path, output_path, start_frame, end_frame, fps, subtitle_text
        return {
            "render_backend": "ffmpeg",
            "subtitle_rendered": None,
            "audio_preserved": True,
            "render_complete": True,
            "rendered_frame_count": -1,
        }

    monkeypatch.setattr(ClipExporter, "_write_clip", fake_write_clip)

    with pytest.raises(RuntimeError, match="invalid_render_fact:rendered_frame_count"):
        ClipExporter(config).export(
            run_dir=str(tmp_path / "run"),
            sample_id="sample_negative_rendered_frame_count",
            video_path=str(video_path),
            fps=10.0,
            segments=[{"start_frame": 0, "end_frame": 10, "instruction": "demo", "export_subtitle": "字幕"}],
        )


def test_clips_ffmpeg_nominal_path_validates_actual_output_file(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, frame_count=20)
    config = ExportConfig(enabled=True, mode="clips", subtitles={"enabled": True})

    def fake_write_clip_with_ffmpeg(self, video_path, output_path, start_frame, end_frame, fps, *, subtitle_text):
        del self, video_path, start_frame, end_frame, fps, subtitle_text
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
        assert writer.isOpened()
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, :, 1] = 255
        writer.write(frame)
        writer.release()

    monkeypatch.setattr("video2tasks.server.exporter._ffmpeg_exists", lambda: True)
    monkeypatch.setattr(ClipExporter, "_write_clip_with_ffmpeg", fake_write_clip_with_ffmpeg)

    result = ClipExporter(config).export(
        run_dir=str(tmp_path / "run"),
        sample_id="sample_ffmpeg_nominal_needs_probe",
        video_path=str(video_path),
        fps=10.0,
        segments=[{"start_frame": 0, "end_frame": 10, "instruction": "demo", "export_subtitle": "字幕"}],
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert result["contract_success"] is False
    assert result["contract_status"] == "degraded"
    assert "audio_not_preserved" in result["contract_errors"]
    assert "render_incomplete" in result["contract_errors"]
    assert "subtitle_render_unverified" in result["contract_errors"]
    assert manifest[0]["subtitle_rendered"] is None
    assert manifest[0]["audio_preserved"] is False
    assert manifest[0]["render_complete"] is False
    assert manifest[0]["rendered_frame_count"] == 1
    assert manifest[0]["requested_frame_count"] == 10
    assert manifest[0]["export_status"] == "degraded"


def test_clips_real_ffmpeg_short_source_does_not_overclaim(tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video_with_audio(video_path, frame_count=30)
    config = ExportConfig(enabled=True, mode="clips", subtitles={"enabled": False})

    result = ClipExporter(config).export(
        run_dir=str(tmp_path / "run"),
        sample_id="sample_real_ffmpeg_short_source",
        video_path=str(video_path),
        fps=10.0,
        segments=[{"seg_id": 0, "start_frame": 0, "end_frame": 60, "instruction": "demo"}],
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert result["clip_count"] == 1
    assert result["contract_success"] is False
    assert result["contract_status"] == "degraded"
    assert result["contract_errors"] == ["render_incomplete"]
    assert manifest[0]["subtitle_requested"] is False
    assert manifest[0]["subtitle_rendered"] is False
    assert manifest[0]["audio_preserved"] is True
    assert manifest[0]["render_complete"] is False
    assert manifest[0]["rendered_frame_count"] == 30
    assert manifest[0]["requested_frame_count"] == 60
    assert manifest[0]["export_status"] == "degraded"


def test_clips_export_zero_output_is_not_success(tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, frame_count=12)
    config = ExportConfig(enabled=True, mode="clips", subtitles={"enabled": False})

    result = ClipExporter(config).export(
        run_dir=str(tmp_path / "run"),
        sample_id="sample_zero_output",
        video_path=str(video_path),
        fps=10.0,
        segments=[{"start_frame": 5, "end_frame": 5, "instruction": "skip me"}],
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert result["clip_count"] == 0
    assert result["contract_success"] is False
    assert result["contract_status"] == "degraded"
    assert "no_clips_exported" in result["contract_errors"]
    assert manifest == []


def test_export_sample_outputs_zero_clips_is_failed_not_applied(tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, frame_count=12)

    diagnostics = export_sample_outputs(
        run_dir=str(tmp_path / "run"),
        sample_id="sample_zero_output_top_level",
        video_path=str(video_path),
        fps=10.0,
        segments=[{"start_frame": 5, "end_frame": 5, "instruction": "skip me"}],
        export_config=ExportConfig(enabled=True, mode="clips", subtitles={"enabled": False}),
    )

    assert diagnostics["export_attempted"] is True
    assert diagnostics["export_reason"] == "failed"
    assert diagnostics["export_clip_count"] == 0
    assert diagnostics["export_clips_contract_status"] == "degraded"
    assert "no_clips_exported" in diagnostics["export_clips_contract_errors"]
    assert "clips:degraded" in diagnostics["export_errors"]


def test_clips_export_marks_audio_losing_path_as_failed(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, frame_count=24)
    monkeypatch.setattr("video2tasks.server.exporter._ffmpeg_exists", lambda: False)

    diagnostics = export_sample_outputs(
        run_dir=str(tmp_path / "run"),
        sample_id="sample_clips_audio_contract",
        video_path=str(video_path),
        fps=10.0,
        segments=[{"start_frame": 0, "end_frame": 12, "instruction": "rotate handle"}],
        export_config=ExportConfig(enabled=True, mode="clips", subtitles={"enabled": False}),
    )

    assert diagnostics["export_attempted"] is True
    assert diagnostics["export_reason"] == "failed"
    assert diagnostics["export_clips_contract_status"] == "degraded"
    assert "audio_not_preserved" in diagnostics["export_clips_contract_errors"]
    assert "clips:degraded" in diagnostics["export_errors"]


def test_annotated_subtitle_fallback_is_not_reported_as_applied(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, frame_count=18)
    monkeypatch.setattr("video2tasks.server.exporter._ffmpeg_exists", lambda: False)

    diagnostics = export_sample_outputs(
        run_dir=str(tmp_path / "run"),
        sample_id="sample_annotated_fallback",
        video_path=str(video_path),
        fps=10.0,
        segments=[{"start_frame": 0, "end_frame": 9, "instruction": "pick up bowl", "export_subtitle": "拿起碗"}],
        export_config=ExportConfig(enabled=True, mode="annotated", subtitles={"enabled": True}),
    )

    assert diagnostics["export_attempted"] is True
    assert diagnostics["export_annotated_used_subtitle_fallback"] is True
    assert diagnostics["export_annotated_contract_status"] == "degraded"
    assert diagnostics["export_reason"] == "failed"
    assert "annotated:degraded" in diagnostics["export_errors"]


def test_export_sample_outputs_returns_disabled_diagnostics(tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path)

    diagnostics = export_sample_outputs(
        run_dir=str(tmp_path / "run"),
        sample_id="sample_d",
        video_path=str(video_path),
        fps=10.0,
        segments=[{"start_frame": 0, "end_frame": 10, "instruction": "rotate handle"}],
        export_config=ExportConfig(enabled=False),
    )

    assert diagnostics["export_enabled"] is False
    assert diagnostics["export_attempted"] is False
    assert diagnostics["export_reason"] == "disabled"
