import json
from pathlib import Path

import pytest

from video2tasks.server.sample_store import SampleStore


def _make_store(tmp_path: Path, *, repeat_target: int = 2) -> SampleStore:
    return SampleStore(
        base_dir=str(tmp_path),
        run_id="testrun",
        initial_samples_dir_by_subset={},
        default_subset="demo",
        window_repeat_count=repeat_target,
        normalize_window_record=lambda record: record.get("vlm_json", {}),
        normalize_boundary_refinement_record=lambda record: record.get("vlm_json", {}),
        normalize_segment_label_payload=lambda payload: payload,
    )


def test_sample_store_persists_and_loads_window_results(tmp_path: Path) -> None:
    store = _make_store(tmp_path, repeat_target=2)

    store.persist_result_record(
        task_id="demo::sample_w0_r0",
        dispatch_id="d1",
        vlm_json={"transitions": [1], "instructions": ["Add potatoes", "Stir the pot"]},
        meta={
            "subset": "demo",
            "sample_id": "sample",
            "job_type": "window_boundary",
            "window_id": 0,
            "repeat_index": 0,
            "logical_frame_count": 4,
        },
    )
    store.persist_result_record(
        task_id="demo::sample_w0_r1",
        dispatch_id="d2",
        vlm_json={"transitions": [1], "instructions": ["Add potatoes", "Stir the pot"]},
        meta={
            "subset": "demo",
            "sample_id": "sample",
            "job_type": "window_boundary",
            "window_id": 0,
            "repeat_index": 1,
            "logical_frame_count": 4,
        },
    )

    results, failures = store.load_window_results("demo", "sample")

    assert failures == {}
    assert results[0]["repeat_success_count"] == 2
    assert results[0]["repeat_indices"] == [0, 1]


def test_sample_store_failure_overwrites_done_and_segments(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    sample_dir = Path(store.sample_out_dir("demo", "sample"))
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "segments.json").write_text(json.dumps({"segments": [{"seg_id": 0}]}), encoding="utf-8")
    (sample_dir / ".DONE").write_text("", encoding="utf-8")

    store.persist_sample_failure("demo", "sample", "step_a_exception", {"error": "boom"})

    assert not (sample_dir / "segments.json").exists()
    assert not (sample_dir / ".DONE").exists()
    assert (sample_dir / ".FAILED").exists()
    report = json.loads((sample_dir / "failure.json").read_text(encoding="utf-8"))
    assert report["reason"] == "step_a_exception"
    assert report["details"]["error"] == "boom"


def test_sample_store_finalize_success_writes_done_and_clears_failure(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    sample_dir = Path(store.sample_out_dir("demo", "sample"))
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / ".FAILED").write_text("", encoding="utf-8")
    (sample_dir / "failure.json").write_text(json.dumps({"reason": "stale"}), encoding="utf-8")

    already_done = store.finalize_sample_success(
        "demo",
        "sample",
        {"segments": [{"seg_id": 0, "instruction": "Add potatoes"}]},
        required_stages=["stage1_segments", "stage2_text"],
        completed_stages=["stage1_segments", "stage2_text"],
    )

    assert already_done is False
    assert (sample_dir / ".DONE").exists()
    assert not (sample_dir / ".FAILED").exists()
    assert not (sample_dir / "failure.json").exists()
    payload = json.loads((sample_dir / "segments.json").read_text(encoding="utf-8"))
    assert payload["segments"][0]["instruction"] == "Add potatoes"


def test_sample_store_finalize_success_rejects_missing_required_stage_completion(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    sample_dir = Path(store.sample_out_dir("demo", "sample"))
    sample_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="missing required stages"):
        store.finalize_sample_success(
            "demo",
            "sample",
            {"segments": [{"seg_id": 0, "instruction": "Add potatoes"}]},
            required_stages=["stage1_segments", "stage2_text"],
            completed_stages=["stage1_segments"],
        )

    assert not (sample_dir / "segments.json").exists()
    assert not (sample_dir / ".DONE").exists()
    assert not (sample_dir / ".FAILED").exists()
    assert not (sample_dir / "failure.json").exists()


def test_sample_store_finalize_success_allows_stage1_only_terminal_contract(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    sample_dir = Path(store.sample_out_dir("demo", "sample"))
    sample_dir.mkdir(parents=True, exist_ok=True)

    store.finalize_sample_success(
        "demo",
        "sample",
        {"segments": [{"seg_id": 0, "instruction": "Add potatoes"}]},
        required_stages=["stage1_segments"],
        completed_stages=["stage1_segments"],
    )

    assert (sample_dir / "segments.json").exists()
    assert (sample_dir / ".DONE").exists()
    assert not (sample_dir / ".FAILED").exists()
    assert not (sample_dir / "failure.json").exists()


def test_sample_store_persist_sample_payload_does_not_touch_terminal_markers(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    sample_dir = Path(store.sample_out_dir("demo", "sample"))
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / ".DONE").write_text("", encoding="utf-8")
    (sample_dir / ".FAILED").write_text("", encoding="utf-8")
    (sample_dir / "failure.json").write_text(json.dumps({"reason": "stale"}), encoding="utf-8")

    store.persist_sample_payload(
        "demo",
        "sample",
        {"segments": [{"seg_id": 0, "instruction": "Add potatoes"}], "task_hierarchy": {"roots": []}},
    )

    payload = json.loads((sample_dir / "segments.json").read_text(encoding="utf-8"))
    assert payload["segments"][0]["instruction"] == "Add potatoes"
    assert payload["task_hierarchy"] == {"roots": []}
    assert (sample_dir / ".DONE").exists()
    assert (sample_dir / ".FAILED").exists()
    assert json.loads((sample_dir / "failure.json").read_text(encoding="utf-8")) == {"reason": "stale"}


def test_sample_store_copies_initial_subset_dir_mapping(tmp_path: Path) -> None:
    initial_mapping = {"demo": str(tmp_path / "seeded" / "samples")}
    store = SampleStore(
        base_dir=str(tmp_path),
        run_id="testrun",
        initial_samples_dir_by_subset=initial_mapping,
        default_subset="demo",
        window_repeat_count=2,
        normalize_window_record=lambda record: record.get("vlm_json", {}),
        normalize_boundary_refinement_record=lambda record: record.get("vlm_json", {}),
        normalize_segment_label_payload=lambda payload: payload,
    )

    initial_mapping["demo"] = str(tmp_path / "mutated" / "samples")

    assert store.resolve_samples_dir("demo") == str(tmp_path / "seeded" / "samples")
