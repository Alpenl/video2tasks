# Endpoint Volatility Runbook

This runbook freezes the operator guidance for endpoint volatility triage.

It is documentation-only and assumes the existing deployment contract: `single-machine shared-fs`.

## 1) Frozen Contracts You Must Hold Constant

- Config layering is **only** `env > yaml > defaults`.
  `yaml` means the file loaded via `--config` or `VIDEO2TASKS_CONFIG`.
- `config.example.yaml` is a **minimal runnable template**, not a full tuning catalog.
- External endpoint instability is **not automatically** a code failure.
  You must classify with runtime evidence before blaming code.

## 2) Observation Entry Points (In This Order)

Use the same order every time to avoid false attribution.

### A. Structured log events (first signal)

Use the frozen event names from [`docs/observability/event-schema.md`](../observability/event-schema.md):

- Retry pressure: `result_empty_retry`, `result_timeout_retry`
- Normal completion evidence: `job_done`, `finalize_done`
- Fallback evidence: `fallback_applied`
- Terminal failure evidence: `sample_failed` (contains `reason` and `details`)

If retries spike across many `sample_id` values while some samples still converge to `job_done`/`finalize_done`, treat that as endpoint volatility first, not an immediate code regression.

### B. `<run_dir>/samples/<sample_id>/sample_runtime.json`

This is the sample-level canonical runtime artifact for triage:

- `terminal_state`
- `stages.required/completed/pending`
- `retry` counters (`total_retries`, `empty_result_retries`, `timeout_retries`, `dispatch_count`)
- `fallback`
- `failure` (if failed)

### C. `<run_dir>/samples/<sample_id>/failure.json`

When `.FAILED` exists, `failure.json` must exist. Use it to read the operator-facing terminal `reason` and `details` for that sample.

### D. `<run_dir>/run_summary.json`

Use run-level aggregates to confirm scope/blast radius:

- `sample_counts`
- `retry` totals
- `fallback.reason_counts`
- `failure_reasons`

## 3) Attribution: Endpoint Volatility vs Code/Data Failure

External endpoint volatility is an infrastructure/input condition. It is not equivalent to a product code defect.

Use this matrix:

| Evidence pattern | Primary attribution | Why |
| --- | --- | --- |
| Many `result_empty_retry` / `result_timeout_retry` events across multiple samples; `run_summary.retry` totals spike; some samples still finish | External endpoint volatility | Endpoint quality is unstable but pipeline control flow still works |
| `sample_failed.reason` is `window_boundary_failed` / `segment_label_failed` / `boundary_refinement_failed`, with high retry pressure around the same period | Usually external endpoint volatility first | Required-stage failures can be downstream effects of unstable external responses |
| `failure.json.reason` is `artifact_extraction_failed` or `artifact_preparation_failed` | Data/artifact problem (or local environment) | Failure happened before endpoint inference became the root cause |
| `failure.json.reason` is `finalize_exception` or `finalize_empty_segments`, reproducible with stable endpoint conditions | Code/data issue | Finalization logic or upstream data contract likely violated |
| Same sample fails deterministically with same reason/details even after endpoint health recovers | Code/data issue | Deterministic repeatability usually indicates non-endpoint root cause |

## 4) Minimal Escalation Packet

When escalating, attach all four evidence layers:

1. Structured event snippets (`result_empty_retry` / `result_timeout_retry` / `sample_failed` / `fallback_applied`)
2. Sample artifact: `sample_runtime.json`
3. Sample terminal report: `failure.json` (if failed)
4. Run aggregate: `run_summary.json`

Without this packet, do not claim a code regression or endpoint-only incident.
