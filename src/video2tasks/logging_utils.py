"""Shared logging utilities for the video2tasks package."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping


PACKAGE_LOGGER_NAME = "video2tasks"
_MESSAGE_FORMAT = logging.Formatter("%(message)s")


@dataclass(frozen=True)
class EventSchema:
    """Frozen structured-event contract for operator-facing logs."""

    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()


FROZEN_EVENT_SCHEMAS: Dict[str, EventSchema] = {
    "artifact_extract_done": EventSchema(
        required_fields=(
            "task_id",
            "subset",
            "sample_id",
            "job_type",
            "image_count",
            "artifact_extract_ms",
            "transport_mode",
            "artifact_reuse",
        ),
    ),
    "artifact_reuse_hit": EventSchema(
        required_fields=(
            "task_id",
            "subset",
            "sample_id",
            "job_type",
            "artifact_reuse",
            "artifact_reuse_group",
            "artifact_producer_task_id",
            "artifact_consumer_task_id",
        ),
    ),
    "job_dispatched": EventSchema(
        required_fields=(
            "task_id",
            "dispatch_id",
            "subset",
            "sample_id",
            "job_type",
            "source_count",
            "transport_mode",
            "artifact_reuse",
        ),
    ),
    "infer_attempt": EventSchema(
        required_fields=(
            "task_id",
            "dispatch_id",
            "subset",
            "sample_id",
            "job_type",
            "infer_ms",
        ),
    ),
    "job_done": EventSchema(
        required_fields=(
            "task_id",
            "dispatch_id",
            "subset",
            "sample_id",
            "job_type",
            "infer_ms",
            "submit_ms",
        ),
    ),
    "result_empty_retry": EventSchema(
        required_fields=(
            "task_id",
            "dispatch_id",
            "subset",
            "sample_id",
            "job_type",
            "attempt",
            "retry_limit",
            "infer_ms",
            "submit_ms",
        ),
    ),
    "result_timeout_retry": EventSchema(
        required_fields=(
            "task_id",
            "dispatch_id",
            "subset",
            "sample_id",
            "job_type",
            "attempt",
            "retry_limit",
        ),
    ),
    "sample_stage_start": EventSchema(
        required_fields=(
            "subset",
            "sample_id",
            "stage",
        ),
    ),
    "sample_stage_done": EventSchema(
        required_fields=(
            "subset",
            "sample_id",
            "stage",
            "elapsed_ms",
        ),
    ),
    "fallback_applied": EventSchema(
        required_fields=(
            "subset",
            "sample_id",
        ),
        optional_fields=(
            "selection_policy",
            "fallback_reason",
        ),
    ),
    "sample_failed": EventSchema(
        required_fields=(
            "subset",
            "sample_id",
            "reason",
            "details",
        ),
    ),
    "finalize_done": EventSchema(
        required_fields=(
            "subset",
            "sample_id",
            "finalize_ms",
            "segment_count",
        ),
    ),
}

FROZEN_EVENT_NAMES: tuple[str, ...] = tuple(FROZEN_EVENT_SCHEMAS.keys())
_IDENTIFIER_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "subset",
        "sample_id",
        "stage",
        "job_type",
        "task_id",
        "dispatch_id",
    }
)


def _required_field_violations(event: str, fields: Mapping[str, Any]) -> tuple[str, ...]:
    schema = FROZEN_EVENT_SCHEMAS.get(event)
    if schema is None:
        return ()

    missing: list[str] = []
    for field_name in schema.required_fields:
        if field_name not in fields:
            missing.append(field_name)
            continue

        value = fields[field_name]
        if value is None:
            missing.append(field_name)
            continue

        if field_name in _IDENTIFIER_REQUIRED_FIELDS:
            if not isinstance(value, str) or not value.strip():
                missing.append(field_name)

    return tuple(missing)


def configure_logging(level: str) -> logging.Logger:
    """Configure the shared package logger to emit plain messages to stdout."""
    normalized_level = str(level or "INFO").upper()
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    package_logger.setLevel(getattr(logging, normalized_level, logging.INFO))
    package_logger.propagate = False

    for handler in list(package_logger.handlers):
        package_logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(_MESSAGE_FORMAT)
    package_logger.addHandler(handler)
    return package_logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger rooted under the shared video2tasks namespace."""
    logger_name = str(name or PACKAGE_LOGGER_NAME)
    if logger_name != PACKAGE_LOGGER_NAME and not logger_name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        logger_name = f"{PACKAGE_LOGGER_NAME}.{logger_name}"
    return logging.getLogger(logger_name)


def log_event(logger: logging.Logger, event: str, level: str = "info", **fields: Any) -> None:
    """Emit a single-line structured event through the existing logger tree."""
    event_name = str(event)
    normalized_fields: Dict[str, Any] = {str(key): value for key, value in fields.items()}
    missing_required_fields = _required_field_violations(event_name, normalized_fields)
    if missing_required_fields:
        missing_rendered = ", ".join(missing_required_fields)
        raise ValueError(f"structured event '{event_name}' missing required fields: {missing_rendered}")

    payload: Dict[str, Any] = {"event": event_name}
    for key, value in normalized_fields.items():
        if value is None:
            continue
        payload[key] = value

    log_method = getattr(logger, str(level).lower(), logger.info)
    log_method(json.dumps(payload, ensure_ascii=False, sort_keys=True))
