"""Typed server-worker transport protocol."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictStr, TypeAdapter, ValidationError, field_validator


class ProtocolValidationError(ValueError):
    """Raised when a transport payload fails schema validation."""


class _ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SharedFSImageTransport(_ProtocolModel):
    """Shared-filesystem image transport using on-disk paths."""

    mode: Literal["shared_fs"] = "shared_fs"
    image_paths: List[StrictStr] = Field(default_factory=list)
    artifact_manifest_path: StrictStr | None = None

    @field_validator("image_paths")
    @classmethod
    def _validate_image_paths(cls, value: List[str]) -> List[str]:
        cleaned: List[str] = []
        for idx, path in enumerate(value):
            text = path.strip()
            if not text:
                raise ValueError(f"image_paths.{idx} must be non-empty")
            cleaned.append(text)
        return cleaned

    @field_validator("artifact_manifest_path")
    @classmethod
    def _validate_manifest_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None


class InlineImageTransport(_ProtocolModel):
    """Inline image transport using data URLs or raw base64 strings."""

    mode: Literal["inline"] = "inline"
    images: List[StrictStr] = Field(default_factory=list)

    @field_validator("images")
    @classmethod
    def _validate_images(cls, value: List[str]) -> List[str]:
        cleaned: List[str] = []
        for idx, image in enumerate(value):
            text = image.strip()
            if not text:
                raise ValueError(f"images.{idx} must be non-empty")
            cleaned.append(text)
        return cleaned


ImageTransport = Union[SharedFSImageTransport, InlineImageTransport]
_IMAGE_TRANSPORT_ADAPTER = TypeAdapter(ImageTransport)


class JobEnvelope(_ProtocolModel):
    """Server -> worker job dispatch envelope."""

    task_id: StrictStr
    dispatch_id: StrictStr = ""
    meta: Dict[str, Any] = Field(default_factory=dict)
    image_transport: ImageTransport

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("task_id must be non-empty")
        return text

    @field_validator("dispatch_id")
    @classmethod
    def _validate_dispatch_id(cls, value: str) -> str:
        return value.strip()

    @classmethod
    def parse_payload(cls, payload: Any) -> "JobEnvelope":
        normalized = _normalize_job_payload(payload)
        try:
            return cls.model_validate(normalized)
        except ValidationError as exc:
            raise ProtocolValidationError(f"invalid job envelope: {exc}") from exc

    def with_dispatch(self, dispatch_id: str) -> "JobEnvelope":
        dispatch_text = dispatch_id.strip()
        meta = dict(self.meta)
        meta["dispatch_id"] = dispatch_text
        return self.model_copy(update={"dispatch_id": dispatch_text, "meta": meta})

    def model_dump_payload(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def source_count(self) -> int:
        if isinstance(self.image_transport, SharedFSImageTransport):
            return len(self.image_transport.image_paths)
        return len(self.image_transport.images)


class ResultEnvelope(_ProtocolModel):
    """Worker -> server result submission envelope."""

    task_id: StrictStr
    dispatch_id: StrictStr = ""
    vlm_output: StrictStr = ""
    vlm_json: Dict[str, Any] = Field(default_factory=dict)
    latency_s: float = 0.0
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("task_id must be non-empty")
        return text

    @field_validator("dispatch_id", "vlm_output")
    @classmethod
    def _validate_string_field(cls, value: str) -> str:
        return value.strip()

    @classmethod
    def parse_payload(cls, payload: Any) -> "ResultEnvelope":
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ProtocolValidationError(f"invalid result envelope: {exc}") from exc

    @property
    def resolved_dispatch_id(self) -> str:
        if self.dispatch_id:
            return self.dispatch_id
        dispatch_id = self.meta.get("dispatch_id")
        if isinstance(dispatch_id, str):
            return dispatch_id.strip()
        return ""

    def model_dump_payload(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


def _normalize_job_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProtocolValidationError("job envelope must be a dict")

    normalized = dict(payload)
    has_typed_transport = "image_transport" in normalized
    has_legacy_paths = "image_paths" in normalized or "artifact_manifest_path" in normalized
    has_legacy_inline = "images" in normalized

    if has_typed_transport:
        if has_legacy_paths or has_legacy_inline:
            raise ProtocolValidationError("mixed typed and legacy image transport fields")
        normalized["image_transport"] = _parse_image_transport(normalized["image_transport"])
        return normalized

    if has_legacy_paths and has_legacy_inline:
        raise ProtocolValidationError("multiple image transport payloads are not allowed")

    if has_legacy_paths:
        if "image_paths" not in normalized:
            raise ProtocolValidationError("shared_fs transport requires image_paths")
        normalized["image_transport"] = _parse_image_transport(
            {
                "mode": "shared_fs",
                "image_paths": normalized.pop("image_paths"),
                "artifact_manifest_path": normalized.pop("artifact_manifest_path", None),
            }
        )
        return normalized

    if has_legacy_inline:
        normalized["image_transport"] = _parse_image_transport(
            {
                "mode": "inline",
                "images": normalized.pop("images"),
            }
        )
        return normalized

    raise ProtocolValidationError("missing image transport")


def _parse_image_transport(payload: Any) -> ImageTransport:
    if not isinstance(payload, dict):
        raise ProtocolValidationError("image transport must be a dict")

    mode = payload.get("mode")
    model: type[SharedFSImageTransport] | type[InlineImageTransport]
    if mode == "shared_fs":
        model = SharedFSImageTransport
    elif mode == "inline":
        model = InlineImageTransport
    else:
        raise ProtocolValidationError(f"invalid image transport mode: {mode!r}")

    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise ProtocolValidationError(f"invalid image transport: {exc}") from exc


__all__ = [
    "ImageTransport",
    "InlineImageTransport",
    "JobEnvelope",
    "ProtocolValidationError",
    "ResultEnvelope",
    "SharedFSImageTransport",
]
