# Full Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete all agreed remediation work from the 2026-04-08 comprehensive review while preserving the confirmed product and deployment decisions.

**Architecture:** Execute in waves. Fix correctness and contract issues first, freeze run/output semantics second, then refactor runtime/protocol boundaries, and only then do profile-driven performance cleanup. Keep `server/app.py` and runtime contract files under tight ownership because they are the main conflict hotspot.

**Tech Stack:** Python, FastAPI, Pydantic, ffmpeg, OpenCV, pytest

---

## Confirmed Decisions

These are already fixed and should not be re-debated during implementation:

- Deployment mode is `single-machine shared-fs`.
- `.DONE` means: all required stages enabled by the current config have completed.
- Subtitle localization is a formal Stage 2 artifact.
- `segments.json` should hold segmentation + Stage 2 text results only; run/export/fallback state belongs in a separate manifest.
- `source instruction` is always English.
- `clips` export must preserve audio.
- Resume is rejected across config/prompt/backend changes unless explicitly forced.
- `config.example.yaml` is a minimal runnable template, not the full source of truth for every tuned setting.

## Parallelism Rules

### Safe To Run In Parallel

These areas have mostly disjoint write sets and can be given to separate subagents in the same wave:

1. Config and secret cleanup
2. Eval correctness fixes
3. Export contract fixes
4. Stage 2 logic fixes inside `llm_merge.py`
5. Docs and config-template cleanup once deployment/config contracts are already frozen

### Should Stay Linear Or Under Single Ownership

These areas touch the same core control files or define contracts that downstream tasks depend on:

1. `server/app.py` lifecycle and failure-closure work
2. Run manifest and resume validation
3. `JobEnvelope` / `ResultEnvelope` / image transport schema
4. `create_app()` side-effect extraction

### Practical Rule

If a task edits any of these files, it should either run alone or be assigned to one owner for the whole wave:

- `src/video2tasks/server/app.py`
- `src/video2tasks/config.py`
- any new manifest/protocol module that becomes the source of truth

## Wave Plan

### Wave 0: Immediate Safety And Truth-Source Cleanup

**Can run in parallel:** yes

### Task 1: Remove Secrets And Fix Config Truth Source

**Files:**
- Modify: `config.g3flash.yaml`
- Modify: `config.example.yaml`
- Modify: `README.md`
- Modify: `README_CN.md`
- Modify: `src/video2tasks/config.py`
- Test: `tests/test_config.py`

- [ ] Remove real secrets from tracked config files.
- [ ] Decide how `config.yaml` auto-discovery is restricted or explicitly documented.
- [ ] Update `config.example.yaml` to be a minimal runnable template.
- [ ] Document precedence as `env > yaml > defaults`.
- [ ] Add config tests that lock the new loading behavior.

**Parallel lane:** Safe to give to one subagent immediately.

### Task 1B: Stop Infinite Empty-Result Retry As An Immediate Bleed Fix

**Files:**
- Modify: `src/video2tasks/config.py`
- Modify: `config.example.yaml`
- Modify: docs mentioning retry defaults
- Test: `tests/test_config.py`

- [ ] Write a failing config-level test that locks a finite default for empty-result retry handling.
- [ ] Change the default away from effectively unbounded empty-result retry behavior.
- [ ] Document the new default and any override path.

**Parallel lane:** Can run with Task 1, but should be owned by the same config/ops worker to avoid drift.

### Task 2: Fix Boundary Eval Correctness

**Files:**
- Modify: `src/video2tasks/eval/*`
- Test: `tests/*eval*`

- [ ] Write a failing test for one prediction matching multiple GT boundaries.
- [ ] Fix the matching logic so one prediction can only satisfy one GT.
- [ ] Re-run the focused eval tests.

**Parallel lane:** Safe to give to a second subagent immediately.

### Wave 1: Correctness Stop-Bleed In Stage 1

**Can run partly in parallel:** limited

This wave is the first hotspot. `app.py` and `windowing.py` must not be edited by multiple workers without strict ownership.

### Task 3: Sample Failure Closure And `.DONE` Semantics

**Files:**
- Modify: `src/video2tasks/server/app.py`
- Test: `tests/server/test_app_retry.py`
- Test: add/modify app-level failure closure tests

- [ ] Write failing tests for missing `Frame_*.mp4`, Step A exception, and finalize exception.
- [ ] Route all three paths into explicit `failed` terminal handling.
- [ ] Make `.DONE` only appear after all required configured stages finish.
- [ ] Verify samples cannot remain in gray states.

**Execution:** Keep under one owner. This is linear and blocks later manifest/output work.

### Task 4: Reject Bad Artifacts Before Job Enqueue

**Files:**
- Modify: `src/video2tasks/server/windowing.py`
- Modify: `src/video2tasks/server/task_artifacts.py`
- Modify: `src/video2tasks/server/app.py`
- Test: `tests/server/test_task_artifacts.py`
- Test: add bad-image-path negative tests

- [ ] Add a failing test where empty or undecodable image payloads would previously reach `image_paths`.
- [ ] Make server-side artifact/build logic reject bad payloads before they become jobs.
- [ ] Ensure failure is attributed to extraction/preparation, not worker timeout.

**Execution:** Same wave as Task 3, but preferably a separate worker owns `windowing.py` and the main integrator owns `app.py` merge.

### Task 5: Fix Stage 1 Final Output Selection

**Files:**
- Modify: `src/video2tasks/server/windowing.py`
- Test: `tests/server/test_windowing.py`

- [ ] Add a failing test that demonstrates diagnostics and final `segments` diverge.
- [ ] Make the selected final segment set explicit and testable.
- [ ] Ensure diagnostics match what is actually emitted.

**Parallel lane:** Can run in parallel with Task 3 if it stays inside `windowing.py` only.

### Task 5B: Fix `logical_frame_count` Reload Validation Mismatch

**Files:**
- Modify: `src/video2tasks/server/app.py`
- Modify: any resume/reload helper used by window result ingestion
- Test: app/window reload tests

- [ ] Add a failing test for window-result reload using the real persisted field layout.
- [ ] Fix the second-pass validation to read the correct location and semantics for `logical_frame_count`.
- [ ] Ensure reload validation agrees with current on-disk format.

**Execution:** Same owner as Task 3, because this belongs to the app/reload path.

### Wave 2: Freeze Run Identity And Output Contracts

**Can run in parallel:** mostly no; one core owner should control `app.py`, manifest, and output contracts

### Task 6: Add Run Manifest And Resume Validation

**Files:**
- Create: `src/video2tasks/server/run_manifest.py` or equivalent
- Modify: `src/video2tasks/server/app.py`
- Modify: `src/video2tasks/config.py`
- Test: app/resume tests

- [ ] Define manifest schema: config hash, git/version marker, schema version, backend summary, forced-resume marker.
- [ ] Write manifest at run start.
- [ ] Reject resume when key identity fields differ.
- [ ] Add explicit `--force-resume` path and record it in manifest.

**Execution:** Linear. This is a foundation contract.

### Task 7: Make Stage 2 A Formal Output Layer

**Files:**
- Modify: `src/video2tasks/server/llm_merge.py`
- Modify: `src/video2tasks/server/app.py`
- Test: `tests/server/test_llm_merge.py`
- Test: `tests/server/test_llm_summary.py`

- [ ] Write a failing test that proves localized subtitles do not persist as Stage 2 results.
- [ ] Decouple Stage 2 subtitle generation from `export.enabled`.
- [ ] Write Stage 2 text outputs back into final result files.
- [ ] Allow summary to proceed independently when merge fails, if inputs are still usable.

**Parallel lane:** Can be owned by one worker, but final app integration should be reviewed carefully because it overlaps Task 6.

### Task 8: Split Output Responsibilities Between `segments.json` And Manifest

**Files:**
- Modify: `src/video2tasks/server/app.py`
- Modify: `src/video2tasks/server/exporter.py`
- Modify: any manifest model/file from Task 6
- Test: app/export integration tests

- [ ] Keep `segments.json` focused on segments and Stage 2 text results.
- [ ] Move export/fallback/run-state facts into a separate manifest.
- [ ] Ensure `.DONE` is tied to the required-stage contract, not merely directory existence.

**Execution:** Linear with Task 6. Same owner recommended.

### Task 8B: Freeze Deployment/Operator Truth Early

**Files:**
- Modify: `README.md`
- Modify: `README_CN.md`
- Modify: `config.example.yaml`
- Modify: operator-facing docs that describe worker deployment

- [ ] Update docs to explicitly state the supported deployment mode is `single-machine shared-fs`.
- [ ] Remove or rewrite wording that suggests unsupported multi-machine path semantics.
- [ ] Align config-template commentary with the confirmed deployment assumption.

**Execution:** Do this in Wave 2, not at the end, because later protocol/runtime work depends on the operator truth already being frozen.

### Wave 3: Export Contract And Stage 2/Export Integration

**Can run in parallel:** yes, with disjoint ownership

### Task 9: Export Manifest Truthfulness And Audio Contract

**Files:**
- Modify: `src/video2tasks/server/exporter.py`
- Test: `tests/server/test_exporter.py`

- [ ] Add failing tests for manifest claiming subtitle/audio facts that were not actually rendered.
- [ ] Make export manifest record actual render facts.
- [ ] Enforce that `clips` success means audio is preserved.
- [ ] Mark non-compliant fallback paths as failure or downgraded non-success.

**Parallel lane:** Safe to give to a dedicated export worker.

### Task 10: Stage 2 Naming And Data Shape Cleanup

**Files:**
- Modify: `src/video2tasks/server/llm_merge.py`
- Modify: `src/video2tasks/config.py`
- Test: Stage 2 tests

- [ ] Keep `source instruction` explicitly English.
- [ ] Make `language=en` / `language=zh` semantics explicit in config and docs.
- [ ] Replace positional `summary_levels` handling with a clearer named representation if feasible without breaking external callers.

**Parallel lane:** Safe to give to a separate Stage 2 worker if it does not touch exporter files.

### Wave 4: Protocol And Runtime Refactor

**Can run in parallel:** mostly no

### Task 11: Formalize Job/Result/Image Transport Schema

**Files:**
- Create: `src/video2tasks/server/protocol.py` or equivalent
- Modify: `src/video2tasks/server/app.py`
- Modify: `src/video2tasks/worker/runner.py`
- Modify: `src/video2tasks/vlm/base.py`
- Test: server/worker protocol tests

- [ ] Define typed envelope objects for job dispatch and result submission.
- [ ] Stop relying on loose dict keys as the only contract.
- [ ] Move backend-specific image representation knowledge out of generic worker flow.

**Execution:** Linear. High conflict risk across app/worker/backend.

### Task 12: Extract Runtime Lifecycle From `create_app()`

**Files:**
- Create: `src/video2tasks/server/runtime.py` or equivalent
- Modify: `src/video2tasks/server/app.py`
- Modify: CLI startup code
- Test: app/runtime tests

- [ ] Separate app construction from producer lifecycle startup.
- [ ] Introduce explicit start/stop/join handles.
- [ ] Remove hidden side effects from `create_app()`.

**Execution:** Linear after Task 11 or by the same owner in one continuous refactor branch.

### Wave 5: Logging, Docs, And Profile-Driven Cleanup

**Can run in parallel:** yes, after contracts stabilize

### Task 13: Logging Unification

**Files:**
- Modify: `src/video2tasks/server/*`
- Modify: `src/video2tasks/worker/*`
- Modify: `src/video2tasks/vlm/*`

- [ ] Replace raw `print`-style operational logging with a shared logger path.
- [ ] Make configured log level actually control app/server/worker output.

### Task 14: Docs And Operator Surface Cleanup

**Files:**
- Modify: `README.md`
- Modify: `README_CN.md`
- Modify: `config.example.yaml`
- Modify: principles/review docs as needed

- [ ] Align docs with the confirmed single-machine shared-fs deployment model.
- [ ] Document `.DONE`, Stage 2, manifest, resume, and config-template semantics.

### Task 15: Profile-Driven Performance Pass

**Files:**
- Likely modify: queue/timeline/encoding/retry related files after measurement
- Add/update: profiling notes or benchmark docs

- [ ] Measure real hotspots before changing queue structures or encoding paths.
- [ ] Only optimize items that remain hot after correctness and contract cleanup.

### Wave 3.5: Contract Test Guardrails Before Runtime Refactor

**Can run in parallel:** yes

### Task 11B: Missing Contract Tests

**Files:**
- Modify/Create: tests around failure closure, artifact path sanitation, Stage 2 app contract, manifest semantics

- [ ] Add direct tests for `_persist_sample_failure(...)` side effects.
- [ ] Add tests for artifact path sanitation and bad path handling.
- [ ] Add app-level Stage 2 writeback tests.
- [ ] Add manifest and `.DONE` contract tests that lock the newly confirmed semantics.

**Execution:** This wave must happen before Wave 4 runtime/protocol refactors. These tests are the guardrails for the later structural work.

## Recommended Subagent Layout

### Wave 0

- Agent A: config/secret cleanup + empty-result retry default
- Agent B: eval fix

### Wave 1

- Main owner: `server/app.py` failure closure + `logical_frame_count` reload fix
- Agent C: `windowing.py` bad-artifact rejection + final output selection

### Wave 2

- Main owner: run manifest + output manifest integration + Stage 2 contract app wiring
- Agent D: Stage 2 subtitle/writeback + summary independence inside `llm_merge.py`
- Agent E: early docs/config-template sync for confirmed deployment mode

### Wave 3

- Agent F: export contract and tests
- Agent G: Stage 2 naming/data-shape cleanup

### Wave 3.5

- Agent H: missing contract tests

### Wave 4

- Main owner only, or one highly trusted worker per task in strict sequence:
  - protocol formalization
  - runtime lifecycle extraction

### Wave 5

- Agent I: logging unification
- Agent J: docs cleanup
- Main owner or profiling worker: performance pass

## What Must Be Linear

These should not be parallelized across multiple writers:

- `server/app.py` failure closure and `.DONE` semantics
- manifest schema definition and resume validation
- output manifest integration and Stage 2 app wiring
- protocol formalization across app/worker/backend
- `create_app()` lifecycle extraction

## What Can Be Parallelized Safely

- secret/config cleanup
- eval fix
- `windowing.py` output-selection fix
- Stage 2 internal logic cleanup
- exporter contract cleanup
- contract test additions after output semantics settle
- docs synchronization after deployment/config semantics are frozen

## Completion Criteria

The remediation is not complete until all of the following are true:

- No tracked file contains real credentials.
- Samples terminate only in explicit success/failure states.
- Resume is identity-checked by manifest.
- Stage 2 subtitle output exists independently of export.
- `segments.json` and manifest responsibilities are separated.
- Export success semantics match actual rendered facts.
- Eval recall bug is fixed with regression coverage.
- App/runtime side effects are explicit.
- Core contracts have direct tests.
- Performance work is based on profiling, not structural guesswork.
