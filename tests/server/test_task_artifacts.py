import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from video2tasks.server.task_artifacts import ArtifactPayloadValidationError, TaskArtifactWriter
from video2tasks.server.windowing import FrameExtractor


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("utf-8")


def _png_bytes(value: int = 64) -> bytes:
    image = np.full((4, 4, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_task_artifact_writer_writes_task_scoped_images_and_manifest(tmp_path) -> None:
    root_dir = tmp_path / "tmp"
    writer = TaskArtifactWriter(root_dir=str(root_dir))
    payload0 = _png_bytes(32)
    payload1 = _png_bytes(96)

    batch = writer.write_images_b64(
        metadata={"subset": "subsetA", "sample_id": "sample42", "task_id": "window_0001"},
        images_b64=[_b64(payload0), _b64(payload1)],
        image_kind="contact_sheet",
        frame_groups=[[10, 11], [12, 13]],
        source_tags=["ffmpeg", "cv2"],
    )

    task_dir = Path(batch.task_dir)
    assert task_dir.exists()
    assert task_dir.is_dir()

    image0 = task_dir / "images" / "contact_sheet_0000.png"
    image1 = task_dir / "images" / "contact_sheet_0001.png"
    assert image0.read_bytes() == payload0
    assert image1.read_bytes() == payload1

    manifest = json.loads(Path(batch.manifest_path).read_text(encoding="utf-8"))
    assert manifest["metadata"]["subset"] == "subsetA"
    assert manifest["metadata"]["sample_id"] == "sample42"
    assert manifest["metadata"]["task_id"] == "window_0001"
    assert manifest["image_count"] == 2
    assert manifest["records"][0]["frame_ids"] == [10, 11]
    assert manifest["records"][0]["source"] == "ffmpeg"
    assert manifest["records"][1]["source"] == "cv2"


def test_task_artifact_writer_writes_raw_png_bytes_without_base64_decode(tmp_path) -> None:
    root_dir = tmp_path / "tmp"
    writer = TaskArtifactWriter(root_dir=str(root_dir))
    payload0 = _png_bytes(10)
    payload1 = _png_bytes(20)

    batch = writer.write_images_bytes(
        metadata={"subset": "subsetA", "sample_id": "sample42", "task_id": "window_0002"},
        images_bytes=[payload0, payload1],
        image_kind="contact_sheet",
        frame_groups=[[1], [2]],
        source_tags=["ffmpeg", "cv2"],
    )

    assert Path(batch.records[0].path).read_bytes() == payload0
    assert Path(batch.records[1].path).read_bytes() == payload1
    manifest = json.loads(Path(batch.manifest_path).read_text(encoding="utf-8"))
    assert [record["byte_size"] for record in manifest["records"]] == [len(payload0), len(payload1)]


@pytest.mark.parametrize(
    ("images_b64", "expected_reason"),
    [
        ([""], "empty_payload"),
        (["%%%not-base64%%%"], "base64_decode_failed"),
        ([_b64(b"not-a-real-image")], "image_decode_failed"),
    ],
)
def test_task_artifact_writer_rejects_bad_payloads(tmp_path, images_b64, expected_reason) -> None:
    root_dir = tmp_path / "tmp"
    writer = TaskArtifactWriter(root_dir=str(root_dir))

    with pytest.raises(ArtifactPayloadValidationError, match=expected_reason):
        writer.write_images_b64(
            metadata={"subset": "subsetA", "sample_id": "sample42", "task_id": "window_bad"},
            images_b64=images_b64,
            image_kind="contact_sheet",
            frame_groups=[[10]],
            source_tags=["ffmpeg"],
        )

    assert not any(root_dir.rglob("*.png"))
    assert not any(root_dir.rglob("manifest.json"))


def test_task_artifact_writer_rejects_bad_raw_bytes_with_specific_exception(tmp_path) -> None:
    root_dir = tmp_path / "tmp"
    writer = TaskArtifactWriter(root_dir=str(root_dir))

    with pytest.raises(ArtifactPayloadValidationError, match="image_decode_failed"):
        writer.write_images_bytes(
            metadata={"subset": "subsetA", "sample_id": "sample42", "task_id": "window_bad_raw"},
            images_bytes=[b"not-a-real-image"],
            image_kind="contact_sheet",
            frame_groups=[[10]],
            source_tags=["ffmpeg"],
        )

    assert not any(root_dir.rglob("*.png"))
    assert not any(root_dir.rglob("manifest.json"))


def test_task_artifact_writer_normalizes_non_string_b64_input_to_validation_error(tmp_path) -> None:
    root_dir = tmp_path / "tmp"
    writer = TaskArtifactWriter(root_dir=str(root_dir))

    with pytest.raises(ArtifactPayloadValidationError, match="invalid_payload_type"):
        writer.write_images_b64(
            metadata={"subset": "subsetA", "sample_id": "sample42", "task_id": "window_bad_type"},
            images_b64=[123],
            image_kind="contact_sheet",
            frame_groups=[[10]],
            source_tags=["ffmpeg"],
        )

    task_dir = root_dir / "subsetA" / "sample42" / "window_bad_type"
    assert not any(root_dir.rglob("*.png"))
    assert not any(root_dir.rglob("manifest.json"))
    assert not task_dir.exists()


def test_task_artifact_writer_cleans_up_partial_files_when_image_write_fails(tmp_path, monkeypatch) -> None:
    root_dir = tmp_path / "tmp"
    writer = TaskArtifactWriter(root_dir=str(root_dir))
    payload = _png_bytes(42)
    original_write_bytes = Path.write_bytes
    failed = False

    def flaky_write_bytes(self: Path, data: bytes) -> int:
        nonlocal failed
        written = original_write_bytes(self, data)
        if self.suffix == ".png" and not failed:
            failed = True
            raise OSError("simulated image write failure")
        return written

    monkeypatch.setattr(Path, "write_bytes", flaky_write_bytes)

    with pytest.raises(OSError, match="simulated image write failure"):
        writer.write_images_bytes(
            metadata={"subset": "subsetA", "sample_id": "sample42", "task_id": "window_partial_image_write"},
            images_bytes=[payload],
            image_kind="contact_sheet",
            frame_groups=[[10]],
            source_tags=["ffmpeg"],
        )

    task_dir = root_dir / "subsetA" / "sample42" / "window_partial_image_write"
    assert not any(root_dir.rglob("*.png"))
    assert not any(root_dir.rglob("manifest.json"))
    assert not task_dir.exists()


def test_task_artifact_writer_cleans_up_partial_files_when_manifest_write_fails(tmp_path, monkeypatch) -> None:
    root_dir = tmp_path / "tmp"
    writer = TaskArtifactWriter(root_dir=str(root_dir))
    payload = _png_bytes(42)
    original_write_text = Path.write_text

    def flaky_write_text(self: Path, data: str, *args, **kwargs) -> int:
        written = original_write_text(self, data, *args, **kwargs)
        if self.name == "manifest.json":
            raise OSError("simulated manifest write failure")
        return written

    monkeypatch.setattr(Path, "write_text", flaky_write_text)

    with pytest.raises(OSError, match="simulated manifest write failure"):
        writer.write_images_bytes(
            metadata={"subset": "subsetA", "sample_id": "sample42", "task_id": "window_partial_manifest_write"},
            images_bytes=[payload],
            image_kind="contact_sheet",
            frame_groups=[[10]],
            source_tags=["ffmpeg"],
        )

    task_dir = root_dir / "subsetA" / "sample42" / "window_partial_manifest_write"
    assert not any(root_dir.rglob("*.png"))
    assert not any(root_dir.rglob("manifest.json"))
    assert not task_dir.exists()


def test_task_artifact_writer_failed_retry_preserves_last_committed_artifacts(tmp_path, monkeypatch) -> None:
    root_dir = tmp_path / "tmp"
    writer = TaskArtifactWriter(root_dir=str(root_dir))
    old_payload = _png_bytes(11)
    new_payload = _png_bytes(22)

    batch = writer.write_images_bytes(
        metadata={"subset": "subsetA", "sample_id": "sample42", "task_id": "window_retry"},
        images_bytes=[old_payload],
        image_kind="contact_sheet",
        frame_groups=[[10]],
        source_tags=["ffmpeg"],
    )

    old_manifest = Path(batch.manifest_path).read_text(encoding="utf-8")
    old_image_path = Path(batch.records[0].path)
    original_write_bytes = Path.write_bytes
    failed = False

    def flaky_write_bytes(self: Path, data: bytes) -> int:
        nonlocal failed
        written = original_write_bytes(self, data)
        if self.suffix == ".png" and ".staging-" in str(self) and not failed:
            failed = True
            raise OSError("simulated staged image write failure")
        return written

    monkeypatch.setattr(Path, "write_bytes", flaky_write_bytes)

    with pytest.raises(OSError, match="simulated staged image write failure"):
        writer.write_images_bytes(
            metadata={"subset": "subsetA", "sample_id": "sample42", "task_id": "window_retry"},
            images_bytes=[new_payload],
            image_kind="contact_sheet",
            frame_groups=[[20]],
            source_tags=["cv2"],
        )

    assert old_image_path.read_bytes() == old_payload
    assert Path(batch.manifest_path).read_text(encoding="utf-8") == old_manifest


def test_task_artifact_writer_manifest_serialization_failure_leaves_no_partial_outputs(tmp_path) -> None:
    root_dir = tmp_path / "tmp"
    writer = TaskArtifactWriter(root_dir=str(root_dir))

    with pytest.raises(TypeError):
        writer.write_images_bytes(
            metadata={
                "subset": "subsetA",
                "sample_id": "sample42",
                "task_id": "window_bad_manifest",
                "bad": object(),
            },
            images_bytes=[_png_bytes(55)],
            image_kind="contact_sheet",
            frame_groups=[[10]],
            source_tags=["ffmpeg"],
        )

    task_dir = root_dir / "subsetA" / "sample42" / "window_bad_manifest"
    assert not any(root_dir.rglob("*.png"))
    assert not any(root_dir.rglob("manifest.json"))
    assert not task_dir.exists()


def test_task_artifact_writer_sanitizes_path_tokens_and_keeps_paths_under_root(tmp_path) -> None:
    root_dir = tmp_path / "tmp"
    writer = TaskArtifactWriter(root_dir=str(root_dir))
    payload = _png_bytes(77)

    batch = writer.write_images_bytes(
        metadata={
            "subset": "../../../../outside",
            "sample_id": "/tmp/hack",
            "task_id": "../..",
        },
        images_bytes=[payload],
        image_kind="../../contact sheet",
        frame_groups=[[7]],
        source_tags=["ffmpeg"],
        extension="../png",
    )

    task_dir = Path(batch.task_dir)
    assert task_dir == root_dir / "outside" / "tmp_hack" / "task"
    task_dir.resolve().relative_to(root_dir.resolve())

    image_path = Path(batch.records[0].path)
    assert image_path == task_dir / "images" / "contact_sheet_0000.png"
    image_path.resolve().relative_to(root_dir.resolve())
    assert image_path.read_bytes() == payload

    manifest = json.loads(Path(batch.manifest_path).read_text(encoding="utf-8"))
    assert manifest["records"][0]["path"] == str(image_path)


def test_task_artifact_writer_uses_fallback_tokens_for_fully_invalid_path_parts(tmp_path) -> None:
    root_dir = tmp_path / "tmp"
    writer = TaskArtifactWriter(root_dir=str(root_dir))

    batch = writer.write_images_bytes(
        metadata={
            "subset": "___",
            "sample_id": "---",
            "task_id": "...",
        },
        images_bytes=[_png_bytes(88)],
        image_kind="***",
        frame_groups=[[3]],
        source_tags=["cv2"],
        extension="???",
    )

    task_dir = Path(batch.task_dir)
    assert task_dir == root_dir / "subset" / "sample" / "task"
    image_path = Path(batch.records[0].path)
    assert image_path == task_dir / "images" / "artifact_0000.png"
    assert image_path.exists()


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
    cv2_payload = _png_bytes(200)
    ffmpeg_payload = _png_bytes(150)

    def fake_ffmpeg(self, group, group_start_index, target_w, target_h, rows, cols):
        if group_start_index == 0:
            return b""
        return ffmpeg_payload

    def fake_cv2(self, group, group_start_index, target_w, target_h, compression, rows, cols):
        return cv2_payload

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
    assert Path(manifest["records"][0]["path"]).read_bytes() == cv2_payload
    assert Path(manifest["records"][1]["path"]).read_bytes() == ffmpeg_payload
