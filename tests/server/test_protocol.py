import pytest

from video2tasks.server.protocol import (
    InlineImageTransport,
    JobEnvelope,
    ProtocolValidationError,
    ResultEnvelope,
    SharedFSImageTransport,
)


def test_job_envelope_parses_shared_fs_transport_and_serializes_to_typed_schema() -> None:
    envelope = JobEnvelope.parse_payload(
        {
            "task_id": "demo::sample_w0",
            "dispatch_id": "d7",
            "meta": {"subset": "demo", "sample_id": "sample"},
            "image_transport": {
                "mode": "shared_fs",
                "image_paths": ["/tmp/frame_000.png", "/tmp/frame_001.png"],
                "artifact_manifest_path": "/tmp/manifest.json",
            },
        }
    )

    assert isinstance(envelope.image_transport, SharedFSImageTransport)
    assert envelope.image_transport.image_paths == [
        "/tmp/frame_000.png",
        "/tmp/frame_001.png",
    ]
    assert envelope.image_transport.artifact_manifest_path == "/tmp/manifest.json"
    assert envelope.model_dump_payload() == {
        "task_id": "demo::sample_w0",
        "dispatch_id": "d7",
        "meta": {"subset": "demo", "sample_id": "sample"},
        "image_transport": {
            "mode": "shared_fs",
            "image_paths": ["/tmp/frame_000.png", "/tmp/frame_001.png"],
            "artifact_manifest_path": "/tmp/manifest.json",
        },
    }


def test_job_envelope_parses_legacy_shared_fs_payload() -> None:
    envelope = JobEnvelope.parse_payload(
        {
            "task_id": "demo::sample_w0",
            "meta": {"subset": "demo"},
            "image_paths": ["/tmp/frame_000.png"],
            "artifact_manifest_path": "/tmp/manifest.json",
        }
    )

    assert isinstance(envelope.image_transport, SharedFSImageTransport)
    assert envelope.image_transport.image_paths == ["/tmp/frame_000.png"]
    assert envelope.model_dump_payload()["image_transport"]["mode"] == "shared_fs"


def test_job_envelope_parses_inline_transport() -> None:
    envelope = JobEnvelope.parse_payload(
        {
            "task_id": "demo::sample_w0",
            "meta": {"subset": "demo"},
            "image_transport": {
                "mode": "inline",
                "images": [
                    "data:image/png;base64,Zm9v",
                    "data:image/png;base64,YmFy",
                ],
            },
        }
    )

    assert isinstance(envelope.image_transport, InlineImageTransport)
    assert envelope.image_transport.images == [
        "data:image/png;base64,Zm9v",
        "data:image/png;base64,YmFy",
    ]


def test_result_envelope_parses_and_resolves_dispatch_id_from_meta() -> None:
    envelope = ResultEnvelope.parse_payload(
        {
            "task_id": "demo::sample_w0",
            "vlm_json": {"transitions": [], "instructions": ["Add potatoes"]},
            "meta": {
                "subset": "demo",
                "dispatch_id": "d3",
            },
        }
    )

    assert envelope.dispatch_id == ""
    assert envelope.resolved_dispatch_id == "d3"
    assert envelope.model_dump_payload()["vlm_json"] == {
        "transitions": [],
        "instructions": ["Add potatoes"],
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "task_id": "demo::sample_w0",
                "meta": {},
            },
            "missing image transport",
        ),
        (
            {
                "task_id": "demo::sample_w0",
                "meta": {},
                "image_paths": ["/tmp/frame_000.png"],
                "images": ["data:image/png;base64,Zm9v"],
            },
            "multiple image transport payloads",
        ),
        (
            {
                "task_id": "demo::sample_w0",
                "meta": {},
                "image_transport": {
                    "mode": "inline",
                    "images": ["data:image/png;base64,Zm9v"],
                },
                "image_paths": ["/tmp/frame_000.png"],
            },
            "mixed typed and legacy image transport fields",
        ),
        (
            {
                "task_id": "demo::sample_w0",
                "meta": {},
                "image_transport": {"mode": "shared_fs", "images": ["oops"]},
            },
            "images",
        ),
        (
            {
                "task_id": "demo::sample_w0",
                "meta": {},
                "image_transport": {"mode": "mystery", "image_paths": []},
            },
            "image transport",
        ),
        (
            {
                "task_id": "demo::sample_w0",
                "meta": {},
                "images": [123],
            },
            "images.0",
        ),
    ],
)
def test_job_envelope_rejects_bad_payloads(payload: dict, message: str) -> None:
    with pytest.raises(ProtocolValidationError, match=message):
        JobEnvelope.parse_payload(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "task_id": "demo::sample_w0",
                "dispatch_id": 7,
                "vlm_json": {},
                "meta": {},
            },
            "dispatch_id",
        ),
        (
            {
                "task_id": "demo::sample_w0",
                "vlm_json": [],
                "meta": {},
            },
            "vlm_json",
        ),
        (
            {
                "task_id": "demo::sample_w0",
                "vlm_json": {},
                "meta": [],
            },
            "meta",
        ),
    ],
)
def test_result_envelope_rejects_bad_payloads(payload: dict, message: str) -> None:
    with pytest.raises(ProtocolValidationError, match=message):
        ResultEnvelope.parse_payload(payload)
