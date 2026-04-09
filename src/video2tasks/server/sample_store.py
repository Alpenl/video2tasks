"""Sample result persistence helpers extracted from the server app."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


RecordNormalizer = Callable[[Dict[str, Any]], Dict[str, Any]]
PayloadNormalizer = Callable[[Dict[str, Any]], Dict[str, Any]]


def _normalize_stage_names(stages: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for stage in stages:
        name = str(stage).strip()
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return tuple(normalized)


class SampleStore:
    """Resolve sample output paths and persist/reload sample artifacts."""

    def __init__(
        self,
        *,
        base_dir: str,
        run_id: str,
        initial_samples_dir_by_subset: Dict[str, str],
        default_subset: str,
        window_repeat_count: int,
        normalize_window_record: RecordNormalizer,
        normalize_boundary_refinement_record: RecordNormalizer,
        normalize_segment_label_payload: PayloadNormalizer,
    ) -> None:
        self.base_dir = str(base_dir)
        self.run_id = str(run_id)
        self._samples_dir_by_subset = {
            str(subset): str(samples_dir)
            for subset, samples_dir in dict(initial_samples_dir_by_subset).items()
        }
        self.default_subset = str(default_subset or "default")
        self.window_repeat_count = max(1, int(window_repeat_count))
        self.normalize_window_record = normalize_window_record
        self.normalize_boundary_refinement_record = normalize_boundary_refinement_record
        self.normalize_segment_label_payload = normalize_segment_label_payload
        self._sample_locks: Dict[str, threading.Lock] = {}
        self._sample_locks_lock = threading.Lock()

    def _sample_key(self, subset: str, sample_id: str) -> str:
        return f"{subset}::{sample_id}"

    def _get_sample_lock(self, subset: str, sample_id: str) -> threading.Lock:
        sample_key = self._sample_key(subset, sample_id)
        with self._sample_locks_lock:
            if sample_key not in self._sample_locks:
                self._sample_locks[sample_key] = threading.Lock()
            return self._sample_locks[sample_key]

    def resolve_samples_dir(self, subset: str) -> str:
        samples_dir = self._samples_dir_by_subset.get(subset)
        if not samples_dir:
            samples_dir = str(Path(self.base_dir) / subset / self.run_id / "samples")
            Path(samples_dir).mkdir(parents=True, exist_ok=True)
            self._samples_dir_by_subset[subset] = samples_dir
        return samples_dir

    def sample_out_dir(self, subset: str, sample_id: str) -> str:
        path = Path(self.resolve_samples_dir(subset)) / sample_id
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def windows_jsonl_path(self, subset: str, sample_id: str) -> str:
        return str(Path(self.sample_out_dir(subset, sample_id)) / "windows.jsonl")

    def segments_path(self, subset: str, sample_id: str) -> str:
        return str(Path(self.sample_out_dir(subset, sample_id)) / "segments.json")

    def segment_labels_jsonl_path(self, subset: str, sample_id: str) -> str:
        return str(Path(self.sample_out_dir(subset, sample_id)) / "segment_labels.jsonl")

    def boundary_refinements_jsonl_path(self, subset: str, sample_id: str) -> str:
        return str(Path(self.sample_out_dir(subset, sample_id)) / "boundary_refinements.jsonl")

    def done_marker_path(self, subset: str, sample_id: str) -> str:
        return str(Path(self.sample_out_dir(subset, sample_id)) / ".DONE")

    def failed_marker_path(self, subset: str, sample_id: str) -> str:
        return str(Path(self.sample_out_dir(subset, sample_id)) / ".FAILED")

    def failure_report_path(self, subset: str, sample_id: str) -> str:
        return str(Path(self.sample_out_dir(subset, sample_id)) / "failure.json")

    def sample_runtime_path(self, subset: str, sample_id: str) -> str:
        return str(Path(self.sample_out_dir(subset, sample_id)) / "sample_runtime.json")

    def _write_json(self, path: str, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=False)

    def _append_jsonl_record(self, path: str, record: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_indexed_result_records(
        self,
        path: Path,
        index_key: str,
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        results: Dict[int, Dict[str, Any]] = {}
        failures: Dict[int, Dict[str, Any]] = {}
        if not path.exists():
            return results, failures

        with open(path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                try:
                    record = json.loads(line)
                    index = int(record[index_key])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue

                terminal_error = str(record.get("terminal_error", "")).strip()
                if terminal_error:
                    failures[index] = record
                    results.pop(index, None)
                    continue

                results[index] = record
                failures.pop(index, None)

        return results, failures

    def load_window_results(self, subset: str, sample_id: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        path = Path(self.windows_jsonl_path(subset, sample_id))
        results: Dict[int, Dict[str, Any]] = {}
        failures: Dict[int, Dict[str, Any]] = {}
        if not path.exists():
            return results, failures

        grouped_records: Dict[int, Dict[int, Dict[str, Any]]] = {}
        with open(path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                try:
                    record = json.loads(line)
                    window_id = int(record["window_id"])
                    repeat_index = int(record.get("repeat_index", 0) or 0)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                grouped_records.setdefault(window_id, {})[repeat_index] = record

        for window_id, repeat_records in grouped_records.items():
            normalized_repeats = []
            failed_repeats: Dict[int, Dict[str, Any]] = {}

            for repeat_index, record in sorted(repeat_records.items()):
                terminal_error = str(record.get("terminal_error", "")).strip()
                if terminal_error:
                    failed_repeats[repeat_index] = record
                    continue

                vlm_json = self.normalize_window_record(record)
                if not vlm_json:
                    failed_repeats[repeat_index] = {**record, "terminal_error": "invalid_vlm_json"}
                    continue

                normalized_record = dict(record)
                normalized_record["repeat_index"] = repeat_index
                normalized_record["vlm_json"] = vlm_json
                normalized_repeats.append(normalized_record)

            if normalized_repeats:
                representative = max(
                    normalized_repeats,
                    key=lambda record: (
                        len(record.get("vlm_json", {}).get("transitions", [])),
                        -int(record.get("repeat_index", 0)),
                    ),
                )
                aggregated_record = dict(representative)
                aggregated_record["repeat_records"] = normalized_repeats
                aggregated_record["repeat_success_count"] = len(normalized_repeats)
                aggregated_record["repeat_target_count"] = self.window_repeat_count
                aggregated_record["repeat_indices"] = [
                    int(record.get("repeat_index", 0))
                    for record in normalized_repeats
                ]
                results[window_id] = aggregated_record

            if failed_repeats:
                first_failed_index = min(failed_repeats)
                failure_record = dict(failed_repeats[first_failed_index])
                failure_record["repeat_indices_failed"] = sorted(int(idx) for idx in failed_repeats)
                failures[window_id] = failure_record

        return results, failures

    def load_segment_label_results(self, subset: str, sample_id: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        records, failures = self._load_indexed_result_records(
            Path(self.segment_labels_jsonl_path(subset, sample_id)),
            "segment_id",
        )
        results: Dict[int, Dict[str, Any]] = {}
        for segment_id, record in records.items():
            vlm_json = self.normalize_segment_label_payload(record.get("vlm_json", {}))
            if not vlm_json:
                failures[segment_id] = {**record, "terminal_error": "invalid_vlm_json"}
                continue
            results[segment_id] = vlm_json
        return results, failures

    def load_boundary_refinement_results(self, subset: str, sample_id: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        records, failures = self._load_indexed_result_records(
            Path(self.boundary_refinements_jsonl_path(subset, sample_id)),
            "boundary_id",
        )
        results: Dict[int, Dict[str, Any]] = {}
        for boundary_id, record in records.items():
            vlm_json = self.normalize_boundary_refinement_record(record)
            if not vlm_json:
                failures[boundary_id] = {**record, "terminal_error": "invalid_vlm_json"}
                continue
            normalized_record = dict(record)
            normalized_record["vlm_json"] = vlm_json
            results[boundary_id] = normalized_record
        return results, failures

    def persist_result_record(
        self,
        task_id: str,
        dispatch_id: str,
        vlm_json: Dict[str, Any],
        meta: Dict[str, Any],
        terminal_error: str = "",
    ) -> None:
        subset = str(meta.get("subset", self.default_subset))
        sample_id = str(meta.get("sample_id", "unknown"))
        job_type = str(meta.get("job_type", "window_boundary"))
        common_fields: Dict[str, Any] = {
            "task_id": task_id,
            "dispatch_id": dispatch_id,
            "vlm_json": vlm_json,
        }
        logical_frame_count = int(meta.get("logical_frame_count", 0) or 0)
        if logical_frame_count > 0:
            common_fields["logical_frame_count"] = logical_frame_count
        if terminal_error:
            common_fields["terminal_error"] = terminal_error

        with self._get_sample_lock(subset, sample_id):
            if job_type == "segment_label":
                self._append_jsonl_record(
                    self.segment_labels_jsonl_path(subset, sample_id),
                    {
                        **common_fields,
                        "segment_id": int(meta.get("segment_id", -1)),
                    },
                )
                return

            if job_type == "boundary_refinement":
                self._append_jsonl_record(
                    self.boundary_refinements_jsonl_path(subset, sample_id),
                    {
                        **common_fields,
                        "boundary_id": int(meta.get("boundary_id", -1)),
                        "coarse_boundary_frame": int(meta.get("coarse_boundary_frame", -1)),
                        "frame_ids": [int(frame_id) for frame_id in meta.get("frame_ids", [])],
                    },
                )
                return

            self._append_jsonl_record(
                self.windows_jsonl_path(subset, sample_id),
                {
                    **common_fields,
                    "window_id": meta.get("window_id"),
                    "repeat_index": int(meta.get("repeat_index", 0) or 0),
                },
            )

    def load_sample_runtime(self, subset: str, sample_id: str) -> Optional[Dict[str, Any]]:
        path = Path(self.sample_runtime_path(subset, sample_id))
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def persist_sample_runtime(self, subset: str, sample_id: str, payload: Dict[str, Any]) -> None:
        with self._get_sample_lock(subset, sample_id):
            self._write_json(self.sample_runtime_path(subset, sample_id), payload)

    def persist_sample_failure(
        self,
        subset: str,
        sample_id: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
        *,
        sample_runtime: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "subset": subset,
            "sample_id": sample_id,
            "reason": reason,
            "details": details or {},
        }

        with self._get_sample_lock(subset, sample_id):
            for stale_path in (self.segments_path(subset, sample_id), self.done_marker_path(subset, sample_id)):
                try:
                    Path(stale_path).unlink()
                except FileNotFoundError:
                    pass
            self._write_json(self.failure_report_path(subset, sample_id), payload)
            if sample_runtime is not None:
                self._write_json(self.sample_runtime_path(subset, sample_id), sample_runtime)
            Path(self.failed_marker_path(subset, sample_id)).touch()

    def persist_sample_payload(self, subset: str, sample_id: str, payload: Dict[str, Any]) -> None:
        with self._get_sample_lock(subset, sample_id):
            self._write_json(self.segments_path(subset, sample_id), payload)

    def finalize_sample_success(
        self,
        subset: str,
        sample_id: str,
        payload: Dict[str, Any],
        *,
        required_stages: Iterable[str],
        completed_stages: Iterable[str],
        sample_runtime: Optional[Dict[str, Any]] = None,
    ) -> bool:
        normalized_required_stages = _normalize_stage_names(required_stages)
        normalized_completed_stages = _normalize_stage_names(completed_stages)
        missing_required_stages = [
            stage
            for stage in normalized_required_stages
            if stage not in normalized_completed_stages
        ]
        if missing_required_stages:
            missing_rendered = ", ".join(missing_required_stages)
            raise ValueError(f"missing required stages: {missing_rendered}")

        payload_to_persist = dict(payload)
        diagnostics = dict(payload_to_persist.get("diagnostics", {}))
        diagnostics["required_stages"] = list(normalized_required_stages)
        diagnostics["completed_stages"] = list(normalized_completed_stages)
        payload_to_persist["diagnostics"] = diagnostics

        with self._get_sample_lock(subset, sample_id):
            self._write_json(self.segments_path(subset, sample_id), payload_to_persist)
            if sample_runtime is not None:
                self._write_json(self.sample_runtime_path(subset, sample_id), sample_runtime)

            done_path = Path(self.done_marker_path(subset, sample_id))
            already_done = done_path.exists()
            for stale_failure_path in (self.failed_marker_path(subset, sample_id), self.failure_report_path(subset, sample_id)):
                try:
                    Path(stale_failure_path).unlink()
                except FileNotFoundError:
                    pass
            done_path.touch()
        return already_done


__all__ = ["SampleStore"]
