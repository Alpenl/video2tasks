"""FastAPI server for job queue management."""

import os
import json
import time
import glob
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from ..config import Config, DatasetConfig
from .windowing import (
    read_video_info, build_windows, FrameExtractor,
    apply_boundary_refinement_results, apply_deferred_segment_labels,
    build_boundary_refinement_windows, build_segments_via_cuts,
    build_refinement_windows, build_window_prompt_metadata, Window,
    sample_segment_frame_ids,
)
from .task_artifacts import TaskArtifactWriter


class SubmitModel(BaseModel):
    """Model for job result submission."""
    task_id: str
    dispatch_id: str = ""
    vlm_output: str = ""
    vlm_json: Dict[str, Any] = Field(default_factory=dict)
    latency_s: float = 0.0
    meta: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class DatasetCtx:
    """Dataset context for processing."""
    data_root: str
    subset: str
    data_dir: str
    run_dir: str
    samples_dir: str
    sample_ids: List[str]


def parse_datasets(config: Config) -> List[DatasetCtx]:
    """Parse dataset configurations into contexts."""
    ctxs = []
    for ds in config.datasets:
        data_dir = Path(ds.root) / ds.subset
        run_dir = Path(config.run.base_dir) / ds.subset / config.run.run_id
        samples_dir = run_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # List sample IDs
        if data_dir.exists():
            sample_ids = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
        else:
            sample_ids = []
        
        ctxs.append(DatasetCtx(
            data_root=ds.root,
            subset=ds.subset,
            data_dir=str(data_dir),
            run_dir=str(run_dir),
            samples_dir=str(samples_dir),
            sample_ids=sample_ids
        ))
    return ctxs


def _requeue_empty_result(
    job_queue: List[Dict[str, Any]],
    retry_counts: Dict[str, int],
    task_id: str,
    job: Optional[Dict[str, Any]],
    max_retries_per_job: int = 0,
) -> tuple[int, bool]:
    if not job:
        return 0, False

    retry_counts[task_id] = retry_counts.get(task_id, 0) + 1
    attempt = retry_counts[task_id]
    if max_retries_per_job > 0 and attempt > max_retries_per_job:
        return attempt, False

    job_queue.append(job)
    return attempt, True


def create_app(config: Config) -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(title="Video2Tasks Server")
    
    # Initialize dataset contexts
    dataset_ctxs = parse_datasets(config)
    samples_dir_by_subset = {ctx.subset: ctx.samples_dir for ctx in dataset_ctxs}
    data_dir_by_subset = {ctx.subset: ctx.data_dir for ctx in dataset_ctxs}
    
    # Thread-safe job management
    queue_lock = threading.Lock()
    job_queue: List[Dict[str, Any]] = []
    inflight: Dict[str, Dict[str, Any]] = {}
    timeout_retry_counts: Dict[str, int] = {}
    empty_retry_counts: Dict[str, int] = {}
    dispatch_counts: Dict[str, int] = {}
    completed_dispatch_ids: Dict[str, str] = {}
    artifact_root_dir = os.getenv("VIDEO2TASKS_TMP_DIR", "tmp").strip() or "tmp"
    task_artifact_writer = TaskArtifactWriter(root_dir=artifact_root_dir)
    
    # Per-sample locks
    _sample_locks: Dict[str, threading.Lock] = {}
    _sample_locks_lock = threading.Lock()
    
    def get_sample_lock(sample_key: str) -> threading.Lock:
        with _sample_locks_lock:
            if sample_key not in _sample_locks:
                _sample_locks[sample_key] = threading.Lock()
            return _sample_locks[sample_key]
    
    def sample_out_dir(samples_dir: str, sample_id: str) -> str:
        p = Path(samples_dir) / sample_id
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    
    def windows_jsonl_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "windows.jsonl")
    
    def segments_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "segments.json")

    def segment_labels_jsonl_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "segment_labels.jsonl")

    def boundary_refinements_jsonl_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "boundary_refinements.jsonl")
    
    def done_marker_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / ".DONE")

    def load_segment_label_results(samples_dir: str, sample_id: str) -> Dict[int, Dict[str, Any]]:
        path = Path(segment_labels_jsonl_path(samples_dir, sample_id))
        results: Dict[int, Dict[str, Any]] = {}
        if not path.exists():
            return results

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    segment_id = int(record["segment_id"])
                    vlm_json = record["vlm_json"]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                if isinstance(vlm_json, dict):
                    results[segment_id] = vlm_json
        return results

    def load_boundary_refinement_results(samples_dir: str, sample_id: str) -> Dict[int, Dict[str, Any]]:
        path = Path(boundary_refinements_jsonl_path(samples_dir, sample_id))
        results: Dict[int, Dict[str, Any]] = {}
        if not path.exists():
            return results

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    boundary_id = int(record["boundary_id"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                results[boundary_id] = record
        return results

    def _resolve_samples_dir(subset: str) -> str:
        samples_dir = samples_dir_by_subset.get(subset)
        if not samples_dir:
            samples_dir = str(Path(config.run.base_dir) / subset / config.run.run_id / "samples")
            Path(samples_dir).mkdir(parents=True, exist_ok=True)
            samples_dir_by_subset[subset] = samples_dir
        return samples_dir

    def _append_jsonl_record(path: str, record: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _persist_result_record(
        task_id: str,
        dispatch_id: str,
        vlm_json: Dict[str, Any],
        meta: Dict[str, Any],
        terminal_error: str = "",
    ) -> None:
        subset = str(meta.get("subset", dataset_ctxs[0].subset if dataset_ctxs else "default"))
        sid = str(meta.get("sample_id", "unknown"))
        samples_dir = _resolve_samples_dir(subset)
        job_type = str(meta.get("job_type", "window_boundary"))
        common_fields: Dict[str, Any] = {
            "task_id": task_id,
            "dispatch_id": dispatch_id,
            "vlm_json": vlm_json,
        }
        if terminal_error:
            common_fields["terminal_error"] = terminal_error

        sample_key = f"{subset}::{sid}"
        with get_sample_lock(sample_key):
            if job_type == "segment_label":
                rec = {
                    **common_fields,
                    "segment_id": int(meta.get("segment_id", -1)),
                }
                _append_jsonl_record(segment_labels_jsonl_path(samples_dir, sid), rec)
                return

            if job_type == "boundary_refinement":
                rec = {
                    **common_fields,
                    "boundary_id": int(meta.get("boundary_id", -1)),
                    "coarse_boundary_frame": int(meta.get("coarse_boundary_frame", -1)),
                    "frame_ids": [int(frame_id) for frame_id in meta.get("frame_ids", [])],
                }
                _append_jsonl_record(boundary_refinements_jsonl_path(samples_dir, sid), rec)
                return

            rec = {
                **common_fields,
                "window_id": meta.get("window_id"),
            }
            _append_jsonl_record(windows_jsonl_path(samples_dir, sid), rec)

    def _build_job_payload(
        extractor: FrameExtractor,
        *,
        task_id: str,
        frame_ids: List[int],
        meta: Dict[str, Any],
        artifact_image_kind: str,
    ) -> Dict[str, Any]:
        images, artifact_batch = extractor.get_many_b64_with_artifacts(
            frame_ids,
            config.windowing.target_width,
            config.windowing.target_height,
            config.windowing.png_compression,
            use_contact_sheets=config.windowing.use_contact_sheets,
            contact_sheet_rows=config.windowing.contact_sheet_rows,
            contact_sheet_cols=config.windowing.contact_sheet_cols,
            artifact_metadata={**meta, "task_id": task_id},
            artifact_image_kind=artifact_image_kind,
            return_images=getattr(extractor, "artifact_writer", None) is None,
        )
        job = {
            "task_id": task_id,
            "meta": dict(meta),
        }
        if artifact_batch is not None and artifact_batch.records:
            job["image_paths"] = [record.path for record in artifact_batch.records]
            job["artifact_manifest_path"] = artifact_batch.manifest_path
        else:
            job["images"] = images
        return job

    app.state.job_queue = job_queue
    app.state.inflight = inflight
    app.state.timeout_retry_counts = timeout_retry_counts
    app.state.empty_retry_counts = empty_retry_counts
    app.state.dispatch_counts = dispatch_counts
    app.state.completed_dispatch_ids = completed_dispatch_ids
    app.state.task_artifact_writer = task_artifact_writer
    app.state.samples_dir_by_subset = samples_dir_by_subset

    @app.get("/get_job")
    def get_job() -> Dict[str, Any]:
        with queue_lock:
            if not job_queue:
                return {"status": "empty"}
            base_job = job_queue.pop(0)
            task_id = str(base_job["task_id"])
            dispatch_counts[task_id] = dispatch_counts.get(task_id, 0) + 1
            dispatch_id = f"d{dispatch_counts[task_id]}"
            job_meta = dict(base_job.get("meta", {}))
            job_meta["dispatch_id"] = dispatch_id
            dispatched_job = dict(base_job)
            dispatched_job["dispatch_id"] = dispatch_id
            dispatched_job["meta"] = job_meta
            inflight[task_id] = {"ts": time.time(), "job": base_job, "dispatch_id": dispatch_id}
            return {"status": "ok", "data": dispatched_job}
    
    @app.post("/submit_result")
    def submit_result(res: SubmitModel) -> Dict[str, str]:
        tid = res.task_id
        dispatch_id = str(res.dispatch_id or res.meta.get("dispatch_id", "")).strip()

        with queue_lock:
            accepted_dispatch_id = completed_dispatch_ids.get(tid)
            inflight_info = inflight.get(tid)

            if dispatch_id and accepted_dispatch_id == dispatch_id:
                return {"status": "already_received"}

            if not inflight_info:
                return {"status": "stale_ignored"}

            expected_dispatch_id = str(inflight_info.get("dispatch_id", ""))
            if not dispatch_id or dispatch_id != expected_dispatch_id:
                return {"status": "stale_ignored"}

            job_info = inflight.pop(tid)

        result_meta = dict(job_info["job"].get("meta", {}))
        result_meta.update(dict(res.meta))
        result_meta["dispatch_id"] = dispatch_id

        if not res.vlm_json:
            with queue_lock:
                attempt, requeued = _requeue_empty_result(
                    job_queue,
                    empty_retry_counts,
                    tid,
                    job_info["job"],
                    config.server.max_empty_retries_per_job,
                )
                limit = config.server.max_empty_retries_per_job
                limit_label = "inf" if limit <= 0 else str(limit)
                if requeued:
                    print(
                        f"[Warn] Task {tid} empty, re-queueing to tail "
                        f"(empty attempt {attempt}/{limit_label})"
                    )
                    return {"status": "retry_triggered"}

                completed_dispatch_ids[tid] = dispatch_id
                timeout_retry_counts.pop(tid, None)
                empty_retry_counts.pop(tid, None)

            print(
                f"[Err] Task {tid} empty retry budget exhausted "
                f"(empty attempt {attempt}/{limit_label}); recording terminal empty result"
            )
            _persist_result_record(
                tid,
                dispatch_id,
                {},
                result_meta,
                terminal_error="empty_retry_exhausted",
            )
            return {"status": "empty_retry_exhausted"}

        with queue_lock:
            completed_dispatch_ids[tid] = dispatch_id
            timeout_retry_counts.pop(tid, None)
            empty_retry_counts.pop(tid, None)

        _persist_result_record(tid, dispatch_id, res.vlm_json, result_meta)
        return {"status": "received"}
    
    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}
    
    # Producer loop
    def producer_loop():
        # Compute progress totals
        total = sum(len(ctx.sample_ids) for ctx in dataset_ctxs)
        progress_total = config.progress.total_override if config.progress.total_override > 0 else total
        
        done = 0
        for ctx in dataset_ctxs:
            for sid in ctx.sample_ids:
                if Path(done_marker_path(ctx.samples_dir, sid)).exists():
                    done += 1
        
        print(
            f"[Server] Started. IMG=PNG, "
            f"FIXED={config.windowing.target_width}x{config.windowing.target_height}, "
            f"FRAMES_PER_WINDOW={config.windowing.frames_per_window}\n"
            f"[Plan] DATASETS={[(c.data_dir, c.subset) for c in dataset_ctxs]}\n"
            f"[Resume] Already done: {done}/{progress_total} (computed_total={total})"
        )
        
        # Initialize states
        states = {}
        for ctx in dataset_ctxs:
            states[ctx.subset] = {
                "cur_idx": 0,
                "sample_status": {sid: 0 for sid in ctx.sample_ids},
            }
        
        dataset_idx = 0
        global_done = done
        
        while True:
            # Check inflight timeouts
            now = time.time()
            with queue_lock:
                expired = [
                    tid for tid, info in inflight.items()
                    if now - info["ts"] > config.server.inflight_timeout_sec
                ]
                for tid in expired:
                    job = inflight.pop(tid)["job"]
                    timeout_retry_counts[tid] = timeout_retry_counts.get(tid, 0) + 1
                    attempt = timeout_retry_counts[tid]
                    limit = config.server.max_retries_per_job
                    limit_label = "inf" if limit <= 0 else str(limit)
                    if limit <= 0 or attempt <= limit:
                        job_queue.append(job)
                        print(
                            f"[Warn] Task {tid} timed out, re-queueing to tail "
                            f"(timeout attempt {attempt}/{limit_label})"
                        )
                    else:
                        print(
                            f"[Err] Task {tid} timed out and exhausted retry budget "
                            f"(timeout attempt {attempt}/{limit_label})"
                        )
            
            # All datasets done
            if dataset_idx >= len(dataset_ctxs):
                if config.server.auto_exit_after_all_done:
                    print(f"[All Done] {global_done}/{progress_total}. Exiting.")
                    os._exit(0)
                time.sleep(1.0)
                continue
            
            ctx = dataset_ctxs[dataset_idx]
            st = states[ctx.subset]
            cur_idx = st["cur_idx"]
            sample_status = st["sample_status"]
            sample_ids = ctx.sample_ids
            
            # Current dataset done, wait for queue to clear
            if cur_idx >= len(sample_ids):
                with queue_lock:
                    if not job_queue and not inflight:
                        print(f"[Dataset] Completed {ctx.subset}. Switching to next...")
                        dataset_idx += 1
                time.sleep(0.2)
                continue
            
            # Produce jobs if queue not full
            with queue_lock:
                q_len = len(job_queue)
            
            if q_len < config.server.max_queue:
                sid = sample_ids[cur_idx]
                s_dir = Path(ctx.data_dir) / sid
                
                # Skip if already done
                if Path(done_marker_path(ctx.samples_dir, sid)).exists():
                    sample_status[sid] = 3
                    st["cur_idx"] += 1
                    time.sleep(0.01)
                    continue
                
                # Find video
                mp4s = list(s_dir.glob("Frame_*.mp4"))
                if not mp4s:
                    st["cur_idx"] += 1
                    time.sleep(0.01)
                    continue
                mp4 = str(mp4s[0])
                
                w_path = windows_jsonl_path(ctx.samples_dir, sid)
                
                # Step A: Generate window tasks
                if sample_status[sid] == 0:
                    try:
                        fps, nframes = read_video_info(mp4)
                        windows = build_windows(
                            fps, nframes,
                            config.windowing.window_sec,
                            config.windowing.step_sec,
                            config.windowing.frames_per_window
                        )
                        
                        # Load completed windows
                        done_wids = set()
                        if Path(w_path).exists():
                            with open(w_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    try:
                                        done_wids.add(json.loads(line)["window_id"])
                                    except (json.JSONDecodeError, KeyError) as e:
                                        print(f"[Warn] Corrupted line in {w_path}: {e}")
                        
                        with FrameExtractor(mp4, artifact_writer=task_artifact_writer) as extractor:
                            cnt = 0
                            
                            for w in windows:
                                if w.window_id in done_wids:
                                    continue
                                
                                tid = f"{ctx.subset}::{sid}_w{w.window_id}"
                                
                                # Check if already active
                                active = False
                                with queue_lock:
                                    if any(j["task_id"] == tid for j in job_queue) or tid in inflight:
                                        active = True
                                
                                if active:
                                    continue
                                
                                meta = {
                                    "subset": ctx.subset,
                                    "sample_id": sid,
                                    "job_type": "window_boundary",
                                    "logical_frame_count": len(w.frame_ids),
                                    **build_window_prompt_metadata(w, fps, nframes),
                                    "use_contact_sheets": config.windowing.use_contact_sheets,
                                    "contact_sheet_rows": (
                                        config.windowing.contact_sheet_rows
                                        if config.windowing.use_contact_sheets else 0
                                    ),
                                    "contact_sheet_cols": (
                                        config.windowing.contact_sheet_cols
                                        if config.windowing.use_contact_sheets else 0
                                    ),
                                }
                                job = _build_job_payload(
                                    extractor,
                                    task_id=tid,
                                    frame_ids=w.frame_ids,
                                    meta=meta,
                                    artifact_image_kind=(
                                        "window_contact_sheet"
                                        if config.windowing.use_contact_sheets else "window_frame"
                                    ),
                                )
                                
                                with queue_lock:
                                    job_queue.append(job)
                                
                                cnt += 1
                                if cnt > 20:
                                    break
                        
                        if cnt == 0:
                            sample_status[sid] = 2
                    
                    except Exception as e:
                        print(f"[Err] {ctx.subset}/{sid}: {e}")
                        import traceback
                        traceback.print_exc()
                        st["cur_idx"] += 1
                
                # Step B: Finalize
                if sample_status[sid] == 2:
                    try:
                        fps, nframes = read_video_info(mp4)
                        windows = build_windows(
                            fps, nframes,
                            config.windowing.window_sec,
                            config.windowing.step_sec,
                            config.windowing.frames_per_window
                        )
                        
                        by_wid = {}
                        if Path(w_path).exists():
                            with open(w_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    try:
                                        d = json.loads(line)
                                        by_wid[d["window_id"]] = d
                                    except (json.JSONDecodeError, KeyError):
                                        pass
                        
                        if len(by_wid) >= len(windows):
                            refinement_windows: List[Window] = []
                            if config.windowing.enable_refinement_pass:
                                refinement_frames = (
                                    config.windowing.refinement_frames_per_window
                                    or config.windowing.frames_per_window
                                )
                                refinement_windows = build_refinement_windows(
                                    windows,
                                    by_wid,
                                    fps,
                                    nframes,
                                    refinement_frames,
                                )
                                if refinement_windows:
                                    missing_refinement = [
                                        refinement_window
                                        for refinement_window in refinement_windows
                                        if refinement_window.window_id not in by_wid
                                    ]
                                    if missing_refinement:
                                        with FrameExtractor(mp4, artifact_writer=task_artifact_writer) as extractor:
                                            cnt = 0

                                            for refinement_window in missing_refinement:
                                                tid = f"{ctx.subset}::{sid}_rw{refinement_window.window_id}"
                                                active = False
                                                with queue_lock:
                                                    if any(j["task_id"] == tid for j in job_queue) or tid in inflight:
                                                        active = True

                                                if active:
                                                    continue

                                                meta = {
                                                    "subset": ctx.subset,
                                                    "sample_id": sid,
                                                    "job_type": "window_boundary",
                                                    "window_pass": "refinement",
                                                    "logical_frame_count": len(refinement_window.frame_ids),
                                                    **build_window_prompt_metadata(refinement_window, fps, nframes),
                                                    "use_contact_sheets": config.windowing.use_contact_sheets,
                                                    "contact_sheet_rows": (
                                                        config.windowing.contact_sheet_rows
                                                        if config.windowing.use_contact_sheets else 0
                                                    ),
                                                    "contact_sheet_cols": (
                                                        config.windowing.contact_sheet_cols
                                                        if config.windowing.use_contact_sheets else 0
                                                    ),
                                                }
                                                job = _build_job_payload(
                                                    extractor,
                                                    task_id=tid,
                                                    frame_ids=refinement_window.frame_ids,
                                                    meta=meta,
                                                    artifact_image_kind=(
                                                        "refinement_contact_sheet"
                                                        if config.windowing.use_contact_sheets else "refinement_frame"
                                                    ),
                                                )

                                                with queue_lock:
                                                    job_queue.append(job)

                                                cnt += 1
                                                if cnt > 20:
                                                    break

                                        if cnt > 0:
                                            time.sleep(0.05)
                                            continue

                                    if len(by_wid) < len(windows) + len(refinement_windows):
                                        time.sleep(0.05)
                                        continue

                            print(f"[Finalize] {ctx.subset}/{sid}...")

                            provisional_res = build_segments_via_cuts(
                                sid, windows + refinement_windows, by_wid, fps, nframes,
                                config.windowing.frames_per_window,
                                boundary_prompt_mode=config.windowing.boundary_prompt_mode,
                                adaptive_merge_guard=config.windowing.adaptive_merge_guard,
                                adaptive_merge_min_segments=config.windowing.adaptive_merge_min_segments,
                                adaptive_merge_collapse_ratio=config.windowing.adaptive_merge_collapse_ratio,
                                boundary_support_threshold=config.windowing.boundary_support_threshold,
                                refine_final_instructions=config.windowing.refine_final_instructions,
                            )

                            final_res = provisional_res
                            if config.windowing.enable_boundary_refinement:
                                boundary_results = load_boundary_refinement_results(ctx.samples_dir, sid)
                                boundary_refinement_frames = (
                                    config.windowing.boundary_refinement_frames_per_window
                                    or config.windowing.frames_per_window
                                )
                                boundary_refinement_windows = build_boundary_refinement_windows(
                                    provisional_res.get("segments", []),
                                    fps,
                                    nframes,
                                    config.windowing.boundary_refinement_window_sec,
                                    boundary_refinement_frames,
                                )
                                missing_boundaries = [
                                    boundary_window
                                    for boundary_window in boundary_refinement_windows
                                    if boundary_window.boundary_id not in boundary_results
                                ]

                                if missing_boundaries:
                                    with FrameExtractor(mp4, artifact_writer=task_artifact_writer) as extractor:
                                        cnt = 0
                                        for boundary_window in missing_boundaries:
                                            boundary_id = int(boundary_window.boundary_id)
                                            tid = f"{ctx.subset}::{sid}_b{boundary_id}"
                                            active = False
                                            with queue_lock:
                                                if any(j["task_id"] == tid for j in job_queue) or tid in inflight:
                                                    active = True

                                            if active:
                                                continue

                                            meta = {
                                                "subset": ctx.subset,
                                                "sample_id": sid,
                                                "job_type": "boundary_refinement",
                                                "boundary_id": boundary_id,
                                                "coarse_boundary_frame": int(boundary_window.coarse_boundary_frame),
                                                "frame_ids": [int(frame_id) for frame_id in boundary_window.frame_ids],
                                                "logical_frame_count": len(boundary_window.frame_ids),
                                                "window_start_frame": int(boundary_window.start_frame),
                                                "window_end_frame": int(boundary_window.end_frame),
                                                "use_contact_sheets": config.windowing.use_contact_sheets,
                                                "contact_sheet_rows": (
                                                    config.windowing.contact_sheet_rows
                                                    if config.windowing.use_contact_sheets else 0
                                                ),
                                                "contact_sheet_cols": (
                                                    config.windowing.contact_sheet_cols
                                                    if config.windowing.use_contact_sheets else 0
                                                ),
                                            }
                                            job = _build_job_payload(
                                                extractor,
                                                task_id=tid,
                                                frame_ids=boundary_window.frame_ids,
                                                meta=meta,
                                                artifact_image_kind=(
                                                    "boundary_refinement_contact_sheet"
                                                    if config.windowing.use_contact_sheets else "boundary_refinement_frame"
                                                ),
                                            )

                                            with queue_lock:
                                                job_queue.append(job)

                                            cnt += 1
                                            if cnt > 20:
                                                break

                                    if cnt > 0:
                                        time.sleep(0.05)
                                        continue

                                    time.sleep(0.05)
                                    continue

                                final_res = dict(provisional_res)
                                final_res["segments"] = apply_boundary_refinement_results(
                                    provisional_res.get("segments", []),
                                    boundary_results,
                                    fps=fps,
                                    abstain_merge_max_support=(
                                        config.windowing.boundary_refinement_abstain_merge_max_support
                                    ),
                                )
                                diagnostics = dict(final_res.get("diagnostics", {}))
                                diagnostics["boundary_refinement_enabled"] = True
                                diagnostics["boundary_refinement_count"] = len(boundary_results)
                                final_res["diagnostics"] = diagnostics

                            if config.windowing.segment_labeling_mode == "deferred":
                                label_results = load_segment_label_results(ctx.samples_dir, sid)
                                missing_segments = [
                                    segment
                                    for segment in final_res.get("segments", [])
                                    if int(segment.get("seg_id", -1)) not in label_results
                                ]

                                if missing_segments:
                                    with FrameExtractor(mp4, artifact_writer=task_artifact_writer) as extractor:
                                        cnt = 0
                                        for segment in missing_segments:
                                            seg_id = int(segment.get("seg_id", -1))
                                            tid = f"{ctx.subset}::{sid}_seg{seg_id}"
                                            active = False
                                            with queue_lock:
                                                if any(j["task_id"] == tid for j in job_queue) or tid in inflight:
                                                    active = True

                                            if active:
                                                continue

                                            frame_ids = sample_segment_frame_ids(
                                                int(segment["start_frame"]),
                                                int(segment["end_frame"]),
                                                config.windowing.frames_per_window,
                                                nframes,
                                            )
                                            meta = {
                                                "subset": ctx.subset,
                                                "sample_id": sid,
                                                "job_type": "segment_label",
                                                "segment_id": seg_id,
                                                "segment_start_frame": int(segment["start_frame"]),
                                                "segment_end_frame": int(segment["end_frame"]),
                                                "frame_ids": [int(frame_id) for frame_id in frame_ids],
                                                "logical_frame_count": len(frame_ids),
                                                "use_contact_sheets": config.windowing.use_contact_sheets,
                                                "contact_sheet_rows": (
                                                    config.windowing.contact_sheet_rows
                                                    if config.windowing.use_contact_sheets else 0
                                                ),
                                                "contact_sheet_cols": (
                                                    config.windowing.contact_sheet_cols
                                                    if config.windowing.use_contact_sheets else 0
                                                ),
                                            }
                                            job = _build_job_payload(
                                                extractor,
                                                task_id=tid,
                                                frame_ids=frame_ids,
                                                meta=meta,
                                                artifact_image_kind=(
                                                    "segment_label_contact_sheet"
                                                    if config.windowing.use_contact_sheets else "segment_label_frame"
                                                ),
                                            )

                                            with queue_lock:
                                                job_queue.append(job)

                                            cnt += 1
                                            if cnt > 20:
                                                break

                                    if cnt > 0:
                                        time.sleep(0.05)
                                        continue

                                    time.sleep(0.05)
                                    continue

                                final_res = dict(final_res)
                                final_res["segments"] = apply_deferred_segment_labels(
                                    final_res.get("segments", []),
                                    label_results,
                                )
                                diagnostics = dict(final_res.get("diagnostics", {}))
                                diagnostics["segment_labeling_mode"] = "deferred"
                                diagnostics["segment_label_count"] = len(label_results)
                                final_res["diagnostics"] = diagnostics
                            
                            with open(segments_path(ctx.samples_dir, sid), "w", encoding="utf-8") as f:
                                json.dump(final_res, f, indent=2, ensure_ascii=False)
                            
                            done_path = done_marker_path(ctx.samples_dir, sid)
                            already_done = Path(done_path).exists()
                            Path(done_path).touch()
                            
                            sample_status[sid] = 3
                            st["cur_idx"] += 1
                            
                            if not already_done:
                                global_done += 1
                            print(f"[Progress] {global_done}/{progress_total} (finished: {ctx.subset}/{sid})")
                    
                    except Exception as e:
                        print(f"[Err-Finalize] {ctx.subset}/{sid}: {e}")
            
            time.sleep(0.1)
    
    # Start producer thread
    producer_thread = threading.Thread(target=producer_loop, daemon=True)
    producer_thread.start()
    
    return app


def run_server(config: Config) -> None:
    """Run the server with given configuration."""
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower()
    )
