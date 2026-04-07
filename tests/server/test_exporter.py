import json
from pathlib import Path

import cv2
import numpy as np

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
    assert result["caption_paths"]
    assert Path(result["caption_paths"][0]).read_text(encoding="utf-8").strip() == "拿起碗"


def test_clips_export_writes_manifest_and_clips(tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, frame_count=40)
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
    assert len(result["clip_paths"]) == 2
    assert all(Path(path).exists() for path in result["clip_paths"])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [record["segment_id"] for record in manifest] == [0, 1]
    assert [record["instruction"] for record in manifest] == ["move cup", "place cup"]
    assert [record["subtitle"] for record in manifest] == ["移动杯子", "放下杯子"]


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
