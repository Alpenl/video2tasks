# OpenAI Backend Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native OpenAI GPT-5.2 backend that the worker can call directly for multi-image task segmentation.

**Architecture:** Extend the existing backend factory and worker config with an `openai` option, then implement a dedicated OpenAI backend that sends image windows to Responses API and returns the same `vlm_json` shape used by the current pipeline. Keep server behavior unchanged and reuse existing retry logic on empty backend output.

**Tech Stack:** Python 3.10+, requests, pydantic, pytest, OpenAI Responses API

---

## Chunk 1: Tests First

### Task 1: Add failing config tests

**Files:**
- Create: `tests/test_config.py`
- Modify: `src/video2tasks/config.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run `uv run --extra dev pytest tests/test_config.py -v` and verify failure**
- [ ] **Step 3: Add minimal config support for `backend: openai`**
- [ ] **Step 4: Re-run the same test and verify pass**

### Task 2: Add failing backend tests

**Files:**
- Create: `tests/vlm/test_openai.py`
- Modify: `src/video2tasks/vlm/factory.py`
- Modify: `src/video2tasks/vlm/openai_api.py`

- [ ] **Step 1: Write failing tests for request construction and response parsing**
- [ ] **Step 2: Run `uv run --extra dev pytest tests/vlm/test_openai.py -v` and verify failure**
- [ ] **Step 3: Add minimal implementation to pass**
- [ ] **Step 4: Re-run the same test file and verify pass**

## Chunk 2: Runtime Integration

### Task 3: Wire worker configuration into backend creation

**Files:**
- Modify: `src/video2tasks/worker/runner.py`

- [ ] **Step 1: Add failing integration-oriented tests if needed**
- [ ] **Step 2: Implement OpenAI config handoff in worker runner**
- [ ] **Step 3: Run focused tests and verify pass**

## Chunk 3: Docs

### Task 4: Update docs and examples

**Files:**
- Modify: `config.example.yaml`
- Modify: `README.md`
- Modify: `README_CN.md`

- [ ] **Step 1: Document install and config changes**
- [ ] **Step 2: Verify snippets match actual config fields**

## Chunk 4: Verification

### Task 5: Run final verification

**Files:**
- Modify: `pyproject.toml` if dependencies need adjustment

- [ ] **Step 1: Run `uv run --extra dev pytest`**
- [ ] **Step 2: Run `uv run --extra dev python -m compileall src tests`**
- [ ] **Step 3: Inspect `git diff --stat` before closeout**
