# Pipeline Hardening And Boundary Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate silent pipeline failure modes and protect real boundary recall so the system does not miss true task switches during generalized video segmentation.

**Architecture:** Keep the current windowing plus VLM pipeline, but harden two weak layers: terminal failure accounting in the server and semantic over-merging in final segmentation. Structured VLM outputs will be validated before they enter the pipeline, and final boundary construction will separate recall-first cuts from optional semantic cleanup.

**Tech Stack:** Python, FastAPI, Pydantic, NumPy, OpenCV, pytest

---

## Scope And Constraints

- Only make generalized improvements that apply to all videos.
- Do not add scene-specific or dataset-specific heuristics.
- Keep the current default model family and contact-sheet upload path intact.
- Prioritize boundary recall over elegance of final segments.
- Empty JSON and timeout exhaustion must no longer be treated as successful completion.

## File Map

**Primary code**

- Modify: `src/video2tasks/server/app.py`
- Modify: `src/video2tasks/server/windowing.py`
- Modify: `src/video2tasks/vlm/gemini_api.py`
- Modify: `src/video2tasks/vlm/openai_api.py`
- Modify: `src/video2tasks/config.py`
- Modify: `src/video2tasks/cli/server.py`
- Modify: `src/video2tasks/cli/worker.py`
- Modify: `src/video2tasks/cli/cluster.py`

**Primary tests**

- Modify: `tests/server/test_app_retry.py`
- Modify: `tests/server/test_windowing.py`
- Modify: `tests/vlm/test_gemini.py`
- Modify: `tests/vlm/test_openai.py`
- Modify: `tests/test_config.py`
- Add if needed: `tests/cli/test_server.py`
- Add if needed: `tests/cli/test_worker.py`

## Task 1: Harden Server Failure Semantics

**Files:**

- Modify: `src/video2tasks/server/app.py`
- Modify: `tests/server/test_app_retry.py`

- [ ] Add explicit terminal failure recording for timeout exhaustion.
- [ ] Make empty-result exhaustion produce a terminal failure state that finalization can detect.
- [ ] Ensure failed windows do not count toward normal completion gates.
- [ ] Prevent `.DONE` and final `segments.json` from being written when required window jobs terminal-fail.
- [ ] Add regression tests for timeout exhaustion, empty exhaustion, and finalize gating.

## Task 2: Validate Structured VLM Output Before It Enters The Pipeline

**Files:**

- Modify: `src/video2tasks/vlm/gemini_api.py`
- Modify: `src/video2tasks/vlm/openai_api.py`
- Modify: `src/video2tasks/server/windowing.py`
- Modify: `tests/vlm/test_gemini.py`
- Modify: `tests/vlm/test_openai.py`

- [ ] Centralize semantic validation of `thought`, `transitions`, and `instructions`.
- [ ] Reject empty `instructions`.
- [ ] Reject mismatched cardinality where `len(instructions) != len(transitions) + 1`.
- [ ] Reject non-monotonic or duplicate transitions.
- [ ] Keep backend parsing permissive enough to recover valid JSON, but do not allow structurally valid yet unusable payloads through.
- [ ] Add regression tests for empty instructions, mismatched counts, and duplicate or malformed transitions.

## Task 3: Protect Recall In Boundary Post-Processing

**Files:**

- Modify: `src/video2tasks/server/windowing.py`
- Modify: `tests/server/test_windowing.py`

- [ ] Split recall-first boundary construction from optional semantic merge output.
- [ ] Remove or relax raw short-segment dropping so strong boundaries stay visible in final `end_frame` output.
- [ ] Make strong boundary support a hard no-merge guard.
- [ ] Make fallback selection depend on recall preservation, not only collapse ratio heuristics.
- [ ] Preserve strong cut points in diagnostics so misses can be traced.
- [ ] Add regression tests proving strong nearby cuts survive final output.

## Task 4: Fix Config And CLI Gaps

**Files:**

- Modify: `src/video2tasks/config.py`
- Modify: `src/video2tasks/cli/server.py`
- Modify: `src/video2tasks/cli/worker.py`
- Modify: `src/video2tasks/cli/cluster.py`
- Modify: `tests/test_config.py`
- Add if needed: `tests/cli/test_cluster.py`

- [ ] Re-validate environment overrides through Pydantic instead of mutating nested fields unsafely.
- [ ] Support env-only configuration paths consistently across server, worker, and cluster CLIs.
- [ ] Keep `WORKER_COUNT` override behavior intact while ensuring validation errors surface clearly.
- [ ] Add regression tests for env-only startup and invalid env override rejection.

## Task 5: Verification

**Files:**

- No required code changes, but may update tests above

- [ ] Run targeted pytest coverage for config, VLM backends, server retry handling, and windowing.
- [ ] If any failing behavior remains, fix it before review.
- [ ] After local verification, dispatch a review subagent focused on code and behavior risk.
- [ ] Review subagent findings, implement necessary fixes, and re-verify targeted tests.

## Verification Commands

- `pytest tests/server/test_app_retry.py -q`
- `pytest tests/server/test_windowing.py -q`
- `pytest tests/vlm/test_gemini.py tests/vlm/test_openai.py -q`
- `pytest tests/test_config.py tests/cli/test_cluster.py -q`
- `pytest -q`

## Expected Outcome

- Timeout and empty-result exhaustion are terminal failures, not silent successes.
- Invalid VLM JSON never reaches the segmentation stage.
- Strong boundaries remain visible through final output unless explicitly rejected by generalized logic.
- CLI and env configuration behavior is consistent and validated.
- The pipeline is safer to iterate on without hiding boundary-recall regressions.
