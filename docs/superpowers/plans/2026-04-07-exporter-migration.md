# Exporter Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate optional subtitle-rendered annotated video and per-segment clip export into `video2tasks` without changing the default segmentation behavior.

**Architecture:** Keep export as an optional post-finalize stage behind `config.export.enabled`. Reuse the new local `server/exporter.py` module, call it after segment postprocessing, and record export failures in diagnostics instead of failing the sample.

**Tech Stack:** Python, FastAPI server flow, OpenCV, ffmpeg, pytest

---

### Task 1: Wire Export Into Finalize

**Files:**
- Modify: `src/video2tasks/server/app.py`
- Modify: `src/video2tasks/server/exporter.py`

- [ ] Review the finalize path and confirm `run_dir`, `sample_id`, `video_path`, `fps`, and final `segments` are all available after postprocess.
- [ ] Import `export_sample_outputs` into `app.py`.
- [ ] Call exporter after `run_llm_postprocess_pass(...)` and before writing `segments.json`.
- [ ] Merge exporter diagnostics into `final_res["diagnostics"]`.
- [ ] Ensure exporter exceptions are caught so sample completion still succeeds.

### Task 2: Expose Config

**Files:**
- Modify: `config.example.yaml`
- Modify: `tests/test_config.py`

- [ ] Add an `export:` example block with defaults matching `ExportConfig`.
- [ ] Add assertions for default export config values.
- [ ] Add at least one environment override test for export-related env vars.

### Task 3: Test Exporters

**Files:**
- Create: `tests/server/test_exporter.py`
- Modify: `src/video2tasks/server/exporter.py` if fixes are required by tests

- [ ] Add a small synthetic MP4 helper using OpenCV.
- [ ] Test annotated export copy path when subtitles are disabled.
- [ ] Test annotated export ffmpeg path by monkeypatching `_ffmpeg_exists` and `subprocess.run`.
- [ ] Test clips export writes clips and manifest.

### Task 4: Verify

**Files:**
- No additional code files expected unless verification finds issues

- [ ] Run `python3 -m py_compile` on modified Python files.
- [ ] Run targeted pytest for config and exporter tests.
- [ ] Fix any failures before stopping.
