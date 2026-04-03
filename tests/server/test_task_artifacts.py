import base64
import json
from pathlib import Path

from video2tasks.server.task_artifacts import TaskArtifactWriter
from video2tasks.server.windowing import FrameExtractor


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("utf-8")


def test_task_artifact_writer_writes_task_scoped_images_and_manifest(tmp_path) -> None:
    root_dir = tmp_path / "tmp"
    writer = TaskArtifactWriter(root_dir=str(root_dir))

    batch = writer.write_images_b64(
        metadata={"subset": "subsetA", "sample_id": "sample42", "task_id": "window_0001"},
        images_b64=[_b64(b"img0"), _b64(b"img1")],
        image_kind="contact_sheet",
        frame_groups=[[10, 11], [12, 13]],
        source_tags=["ffmpeg", "cv2"],
    )

    task_dir = Path(batch.task_dir)
    assert task_dir.exists()
    assert task_dir.is_dir()

    image0 = task_dir / "images" / "contact_sheet_0000.png"
    image1 = task_dir / "images" / "contact_sheet_0001.png"
    assert image0.read_bytes() == b"img0"
    assert image1.read_bytes() == b"img1"

    manifest = json.loads(Path(batch.manifest_path).read_text(encoding="utf-8"))
    assert manifest["metadata"]["subset"] == "subsetA"
    assert manifest["metadata"]["sample_id"] == "sample42"
    assert manifest["metadata"]["task_id"] == "window_0001"
    assert manifest["image_count"] == 2
    assert manifest["records"][0]["frame_ids"] == [10, 11]
    assert manifest["records"][0]["source"] == "ffmpeg"
    assert manifest["records"][1]["source"] == "cv2"


def test_task_artifact_writer_writes_raw_bytes_without_base64_decode(tmp_path) -> None:
    root_dir = tmp_path / "tmp"
    writer = TaskArtifactWriter(root_dir=str(root_dir))

    batch = writer.write_images_bytes(
        metadata={"subset": "subsetA", "sample_id": "sample42", "task_id": "window_0002"},
        images_bytes=[b"raw0", b"raw1"],
        image_kind="contact_sheet",
        frame_groups=[[1], [2]],
        source_tags=["ffmpeg", "cv2"],
    )

    assert Path(batch.records[0].path).read_bytes() == b"raw0"
    assert Path(batch.records[1].path).read_bytes() == b"raw1"
    manifest = json.loads(Path(batch.manifest_path).read_text(encoding="utf-8"))
    assert [record["byte_size"] for record in manifest["records"]] == [4, 4]


def test_frame_extractor_with_artifacts_records_contact_sheet_sources(
    monkeypatch,
    tmp_path,
) -> None:
    extractor = FrameExtractor.__new__(FrameExtractor)
    extractor.mp4_path = "demo.mp4"
    extractor.artifact_writer = TaskArtifactWriter(root_dir=str(tmp_path / "tmp"))
    extractor.last_artifact_batch = None
    extractor._artifact_sequence = 0
    extractor._auto_sample_id = "demo"

    def fake_ffmpeg(self, group, group_start_index, target_w, target_h, rows, cols):
        if group_start_index == 0:
            return b""
        return f"ffmpeg-{group_start_index}".encode("utf-8")

    def fake_cv2(self, group, group_start_index, target_w, target_h, compression, rows, cols):
        return f"cv2-{group_start_index}".encode("utf-8")

    monkeypatch.setattr(FrameExtractor, "_build_contact_sheet_png_bytes_via_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(FrameExtractor, "_build_contact_sheet_png_bytes_via_cv2", fake_cv2)

    images, batch = extractor.get_many_b64_with_artifacts(
        [0, 1, 2, 3, 4],
        use_contact_sheets=True,
        contact_sheet_rows=2,
        contact_sheet_cols=2,
        artifact_metadata={
            "subset": "youcook2",
            "sample_id": "yc_demo",
            "task_id": "yc_demo_w0",
        },
        return_images=False,
    )

    assert images == []
    assert batch is not None

    manifest = json.loads(Path(batch.manifest_path).read_text(encoding="utf-8"))
    assert manifest["metadata"]["task_id"] == "yc_demo_w0"
    assert [record["source"] for record in manifest["records"]] == ["cv2", "ffmpeg"]
    assert manifest["records"][0]["frame_ids"] == [0, 1, 2, 3]
    assert manifest["records"][1]["frame_ids"] == [4]
    assert Path(manifest["records"][0]["path"]).read_bytes() == b"cv2-0"
    assert Path(manifest["records"][1]["path"]).read_bytes() == b"ffmpeg-4"
