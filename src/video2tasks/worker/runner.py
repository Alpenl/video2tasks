"""Worker runner implementation."""

import os
import time
import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any

import requests
import numpy as np
from PIL import Image

from ..config import Config
from ..vlm import create_backend
from ..prompt import (
    prompt_boundary_refinement,
    prompt_segment_instruction,
    prompt_switch_detection,
)

MAX_LOCAL_RETRIES = 4
MAX_CONNECTION_RETRIES = 30
MAX_SHUTDOWN_RETRIES = 3
MAX_SUBMIT_RETRIES = 3
_ACCEPTED_SUBMIT_STATUSES = {
    "received",
    "retry_triggered",
    "already_received",
    "stale_ignored",
    "empty_retry_exhausted",
    "ok",
}
_IMAGE_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


@dataclass
class LoadedImage:
    raw_bytes: bytes
    mime_type: str
    bgr: Optional[np.ndarray] = None


def _is_empty_vlm_json(vlm_json: Optional[Dict[str, Any]]) -> bool:
    return (not isinstance(vlm_json, dict)) or (not vlm_json)


def _decode_image_bytes_to_numpy(img_bytes: bytes) -> Optional[np.ndarray]:
    if not img_bytes:
        return None

    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        rgb_array = np.array(img)
        return rgb_array[:, :, ::-1]
    except Exception:
        return None


def _decode_b64_to_bytes_and_mime(b64_str: str) -> tuple[bytes, str]:
    if not b64_str:
        return b"", "image/png"

    payload = b64_str
    mime_type = "image/png"
    if b64_str.startswith("data:") and "," in b64_str:
        header, payload = b64_str.split(",", 1)
        if ";" in header:
            mime_type = header[5:].split(";", 1)[0] or mime_type
        else:
            mime_type = header[5:] or mime_type

    try:
        return base64.b64decode(payload), mime_type
    except Exception:
        return b"", mime_type


def decode_b64_to_numpy(b64_str: str) -> Optional[np.ndarray]:
    """Decode base64 string to numpy BGR array."""
    img_bytes, _mime_type = _decode_b64_to_bytes_and_mime(b64_str)
    return _decode_image_bytes_to_numpy(img_bytes)


def decode_path_to_numpy(path: str) -> Optional[np.ndarray]:
    """Decode an on-disk image into a numpy BGR array."""
    if not path:
        return None
    try:
        img_bytes = Path(path).read_bytes()
    except Exception:
        return None
    return _decode_image_bytes_to_numpy(img_bytes)


def _mime_type_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return _IMAGE_MIME_BY_SUFFIX.get(suffix, "image/png")


def load_job_image_records(job: Dict[str, Any], decode_arrays: bool = True) -> list[LoadedImage]:
    """Load job images while optionally preserving raw bytes for direct upload."""
    image_paths = [str(path) for path in (job.get("image_paths") or []) if str(path)]
    if image_paths:
        records: list[LoadedImage] = []
        failed_paths: list[str] = []
        for path in image_paths:
            try:
                raw_bytes = Path(path).read_bytes()
            except Exception:
                failed_paths.append(path)
                continue
            if not raw_bytes:
                failed_paths.append(path)
                continue
            bgr = _decode_image_bytes_to_numpy(raw_bytes) if decode_arrays else None
            if decode_arrays and bgr is None:
                failed_paths.append(path)
                continue
            records.append(LoadedImage(raw_bytes=raw_bytes, mime_type=_mime_type_from_path(path), bgr=bgr))
        if failed_paths:
            preview = ", ".join(failed_paths[:3])
            raise RuntimeError(f"failed to load image_paths: {preview}")
        return records

    images_b64 = [str(item) for item in (job.get("images") or [])]
    records: list[LoadedImage] = []
    failed_indices: list[int] = []
    for idx, b64 in enumerate(images_b64):
        raw_bytes, mime_type = _decode_b64_to_bytes_and_mime(b64)
        if not raw_bytes:
            failed_indices.append(idx)
            continue
        bgr = _decode_image_bytes_to_numpy(raw_bytes) if decode_arrays else None
        if decode_arrays and bgr is None:
            failed_indices.append(idx)
            continue
        records.append(LoadedImage(raw_bytes=raw_bytes, mime_type=mime_type, bgr=bgr))
    if failed_indices:
        raise RuntimeError(f"failed to decode inline images at indices {failed_indices}")
    return records


def load_job_images(job: Dict[str, Any]) -> list[np.ndarray]:
    """Backward-compatible wrapper returning decoded numpy arrays."""
    records = load_job_image_records(job, decode_arrays=True)
    return [record.bgr for record in records if record.bgr is not None]


def build_backend_kwargs(config: Config) -> Dict[str, Any]:
    """Build backend kwargs from worker configuration."""
    if config.worker.backend == "qwen3vl":
        return {
            "model_path": config.worker.qwen3vl.model_path,
            "device_map": config.worker.qwen3vl.device_map,
        }

    if config.worker.backend == "remote_api":
        return {
            "url": config.worker.remote_api.api_url,
            "api_key": config.worker.remote_api.api_key,
            "headers": config.worker.remote_api.headers,
            "timeout_sec": config.worker.remote_api.timeout_sec,
        }

    if config.worker.backend == "openai":
        return {
            "api_key": config.worker.openai.api_key or os.getenv("OPENAI_API_KEY", ""),
            "model": config.worker.openai.model,
            "base_url": config.worker.openai.base_url,
            "timeout_sec": config.worker.openai.timeout_sec,
            "organization": config.worker.openai.organization,
            "project": config.worker.openai.project,
            "reasoning_effort": config.worker.openai.reasoning_effort,
            "max_output_tokens": config.worker.openai.max_output_tokens,
            "jpeg_quality": config.worker.openai.jpeg_quality,
        }

    if config.worker.backend == "gemini":
        return {
            "api_key": config.worker.gemini.api_key or os.getenv("GEMINI_API_KEY", ""),
            "model": config.worker.gemini.model,
            "api_mode": config.worker.gemini.api_mode,
            "base_url": config.worker.gemini.base_url,
            "timeout_sec": config.worker.gemini.timeout_sec,
            "max_output_tokens": config.worker.gemini.max_output_tokens,
            "jpeg_quality": config.worker.gemini.jpeg_quality,
        }

    return {}


def submit_result_with_retries(server_url: str, payload: Dict[str, Any], task_id: str) -> str:
    """Submit a finished job result and require an explicit server ack."""
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_SUBMIT_RETRIES + 1):
        try:
            response = requests.post(
                f"{server_url}/submit_result",
                json=payload,
                timeout=30,
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")

            body = response.json()
            status = str(body.get("status", ""))
            if status in _ACCEPTED_SUBMIT_STATUSES:
                return status
            raise RuntimeError(f"unexpected status {status or '<missing>'}")
        except Exception as exc:
            last_error = exc
            print(
                f"[Warn] Submit failed for {task_id} "
                f"(attempt {attempt}/{MAX_SUBMIT_RETRIES}): {exc}"
            )
            if attempt < MAX_SUBMIT_RETRIES:
                delay_s = float(min(2 * attempt, 8))
                time.sleep(delay_s)

    raise RuntimeError(f"submit_result failed for {task_id}: {last_error}")


def run_worker(config: Config) -> None:
    """Run the worker loop."""
    server_url = config.worker.server_url

    backend_kwargs = build_backend_kwargs(config)
    backend = create_backend(config.worker.backend, **backend_kwargs)
    print(f"[Worker] Using backend: {backend.name}")
    backend.warmup()

    print(f"[Worker] Connecting to {server_url}")

    connection_retry_count = 0
    had_prior_connection = False

    try:
        while True:
            try:
                try:
                    r = requests.get(f"{server_url}/get_job", timeout=60)
                    had_prior_connection = True
                    connection_retry_count = 0
                except requests.exceptions.RequestException:
                    connection_retry_count += 1
                    max_retries = (
                        MAX_SHUTDOWN_RETRIES if had_prior_connection else MAX_CONNECTION_RETRIES
                    )
                    if connection_retry_count >= max_retries:
                        if had_prior_connection:
                            print("[Worker] Server became unavailable after prior connection. Exiting.")
                        else:
                            print(f"[Worker] Failed to connect after {MAX_CONNECTION_RETRIES} retries. Exiting.")
                        break
                    print(
                        f"[Worker] Waiting for server at {server_url}... "
                        f"(attempt {connection_retry_count}/{max_retries})"
                    )
                    time.sleep(2)
                    continue

                if r.status_code != 200:
                    time.sleep(0.5)
                    continue

                resp = r.json()
                if resp.get("status") == "empty":
                    time.sleep(0.5)
                    continue

                job = resp.get("data")
                if job is None:
                    print("[Worker] Invalid job data received")
                    time.sleep(1)
                    continue

                task_id = job.get("task_id", "unknown")
                dispatch_id = str(job.get("dispatch_id") or "")
                meta = job.get("meta", {})
                job_type = str(meta.get("job_type", "window_boundary"))
                contact_sheet_rows = int(meta.get("contact_sheet_rows", 0) or 0)
                contact_sheet_cols = int(meta.get("contact_sheet_cols", 0) or 0)

                logical_frame_count = int(meta.get("logical_frame_count") or 0)
                if logical_frame_count <= 0:
                    frame_ids = meta.get("frame_ids", [])
                    if isinstance(frame_ids, list) and frame_ids:
                        logical_frame_count = len(frame_ids)
                    else:
                        source_count = len(job.get("image_paths") or job.get("images") or [])
                        if contact_sheet_rows > 0 and contact_sheet_cols > 0:
                            logical_frame_count = source_count * contact_sheet_rows * contact_sheet_cols
                        else:
                            logical_frame_count = source_count

                use_raw_image_payloads = config.worker.backend == "gemini"
                image_records = load_job_image_records(job, decode_arrays=not use_raw_image_payloads)
                images = [
                    {"raw_bytes": record.raw_bytes, "mime_type": record.mime_type}
                    for record in image_records
                ] if use_raw_image_payloads else [record.bgr for record in image_records if record.bgr is not None]

                if job_type == "segment_label":
                    prompt = prompt_segment_instruction(
                        logical_frame_count,
                        contact_sheet_rows=contact_sheet_rows,
                        contact_sheet_cols=contact_sheet_cols,
                        sheet_count=(len(image_records) if contact_sheet_rows > 0 and contact_sheet_cols > 0 else 0),
                    )
                elif job_type == "boundary_refinement":
                    prompt = prompt_boundary_refinement(
                        logical_frame_count,
                        contact_sheet_rows=contact_sheet_rows,
                        contact_sheet_cols=contact_sheet_cols,
                        sheet_count=(len(image_records) if contact_sheet_rows > 0 and contact_sheet_cols > 0 else 0),
                    )
                else:
                    prompt = prompt_switch_detection(
                        logical_frame_count,
                        mode=config.windowing.boundary_prompt_mode,
                        contact_sheet_rows=contact_sheet_rows,
                        contact_sheet_cols=contact_sheet_cols,
                        sheet_count=(len(image_records) if contact_sheet_rows > 0 and contact_sheet_cols > 0 else 0),
                    )
                vlm_json: Dict[str, Any] = {}

                for attempt in range(MAX_LOCAL_RETRIES):
                    try:
                        vlm_json = backend.infer(images, prompt)
                    except Exception as exc:
                        print(f"[Err] Inference failed: {exc}")
                        vlm_json = {}

                    if not _is_empty_vlm_json(vlm_json):
                        break

                    print(
                        f"[Warn] {task_id} Empty VLM JSON "
                        f"(attempt {attempt + 1}/{MAX_LOCAL_RETRIES})"
                    )
                    if attempt + 1 < MAX_LOCAL_RETRIES:
                        delay_s = float(min(2 * (attempt + 1), 8))
                        print(f"[Worker] Sleeping {delay_s:.1f}s before local retry")
                        time.sleep(delay_s)

                if _is_empty_vlm_json(vlm_json):
                    print(f"[Fail] {task_id} Returning empty to trigger server retry")
                else:
                    print(
                        f"[Done] {task_id} ({logical_frame_count}logical/{len(image_records)}img) -> Cuts: {vlm_json.get('transitions', [])}"
                    )

                submit_result_with_retries(
                    server_url,
                    {
                        "task_id": task_id,
                        "dispatch_id": dispatch_id,
                        "vlm_json": vlm_json,
                        "meta": meta,
                    },
                    task_id,
                )

            except KeyboardInterrupt:
                print("[Worker] Stopping...")
                break
            except Exception as exc:
                print(f"[Error] Loop crashed: {exc}")
                time.sleep(1)

    finally:
        backend.cleanup()
