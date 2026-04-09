# Performance Baseline Runbook

This runbook fixes the P2-3 measurement procedure so optimization decisions are based on evidence instead of intuition.

## Scope

- Goal: establish before/after evidence for one bounded hotspot only.
- Do not change result contracts while measuring.
- Keep `sample_runtime.json`, `run_summary.json`, manifest layout, image layout, event names, and job/result envelopes unchanged.

## Baselines

### Official Smoke Baseline

Canonical command:

```bash
v2t-cluster --config config.smoke.yaml
```

- Use this as the repo-wide fixed smoke baseline.
- Save the full stdout/stderr log.
- Record total wall time.
- Aggregate structured events from the log.

### Representative Real Run

- Start from `config.g3flash.yaml`.
- Derive a temporary one-sample config in `tmp/` or `/tmp/`.
- Do not commit the temporary config.
- Keep the representative backend aligned with the current repo baseline. Do not switch to an unrelated endpoint family just because another historical config exists.
- For Gemini representative runs derived from `config.g3flash.yaml`, keep the same backend family and endpoint path as the current main config path:
  - backend family: `gemini`
  - model family: `gemini-3-flash-preview`
  - endpoint family: `https://api.duckcoding.ai`
- If credentials must be sourced from a local untracked config such as `config.yaml`, only copy the credential into the temporary config. Do not treat a different historical endpoint as representative evidence for `config.g3flash.yaml`.
- Keep the same pipeline shape as the source config unless a temporary change is strictly required to make a single-sample measurement:
  - one dataset entry
  - same backend family
  - same Stage 2 backend family / model / endpoint family
  - same windowing / Stage 2 behavior
  - same logging format
- Save the full stdout/stderr log.
- Record total wall time.

Recommended real-run shape:

- `datasets`: one subset from `config.g3flash.yaml`
- one sample directory inside that subset
- dedicated `run.run_id`
- dedicated `run.base_dir` so before/after runs do not collide

## Evidence To Preserve

For each run, preserve:

- config path used
- full process log
- wall-clock start/end and total wall time in ms
- `run_manifest.json`
- `sample_runtime.json`
- `run_summary.json`

Aggregate these structured-event fields from the log:

- `artifact_extract_done.artifact_extract_ms`
- `infer_attempt.infer_ms`
- `job_done.submit_ms`
- `finalize_done.finalize_ms`
- retry event counts:
  - `result_empty_retry`
  - `result_timeout_retry`
- Stage 2 request / fallback evidence from:
  - structured `fallback_applied` events
  - `sample_runtime.json.fallback`
  - `run_summary.json.fallback`
- Remote instability evidence from raw log lines when completed-event metrics are incomplete:
  - read timeout / connection timeout / request failed lines
  - empty structured payload retries
  - non-200 remote status churn

## Suggested Log Aggregation

Parse newline-delimited JSON events from the run log and compute:

- total count
- sum
- p50
- p95
- max

Required metric groups:

- `artifact_extract_ms`
- `infer_ms`
- `submit_ms`
- `finalize_ms`

Required counts:

- `result_empty_retry`
- `result_timeout_retry`
- `fallback_applied`

## Blame Rules

Use the representative real run to decide whether to optimize code.

### External API Dominated

Stop after documentation + evidence if most runtime is clearly remote-call dominated. Treat it as external API dominated when the evidence shows both of these patterns:

- completed-event metrics favor remote cost, or the real run spends most of its time waiting on remote completion rather than local finalize
- retries, fallback evidence, or raw-log request failures point at request volatility, empty results, timeout churn, or Stage 2 remote variance

Operationally, this means the largest delays live in worker inference or Stage 2 requests, not in local artifact extraction.

### Representative Baseline Blocked By External API

Treat the real baseline as blocked, and stop local optimization, when the representative run does not form a valid comparable baseline because remote instability prevents normal completion. Typical evidence includes one or more of:

- repeated read timeout / connection timeout / request failed lines in the raw log
- repeated empty structured payload retries
- run remains incomplete, with `run_summary.json` still showing pending work or `sample_runtime.json` not reaching a terminal success/failure state
- only partial `infer_attempt` / `job_done` coverage relative to the number of dispatched jobs

In this situation, the correct conclusion is not “local hotspot proven” but “real baseline inconclusive due to remote instability”. Stop before code optimization and keep the evidence pack.

### Local Artifact Extraction Dominated

Proceed to one bounded hotspot optimization only if the representative real run shows at least one of:

- `artifact_extract_ms` is a meaningful share of total wall time
- `artifact_extract_ms` sum is comparable to other non-remote local stages
- p95 `artifact_extract_ms` is consistently high on the same workload
- the hotspot matches an already identified repeated local cost in the extraction path

Proceed only when the real run is otherwise stable enough to be a meaningful before/after comparison.

### I/O Dominated

Treat as local I/O dominated when:

- `artifact_extract_ms` is high
- remote evidence is not the main blocker
- and the cost is mostly correlated with artifact writes or filesystem transport rather than model latency

Do not expand scope into unrelated I/O tuning during P2-3. Only continue if the chosen hotspot directly explains that cost.

## Decision Gate

1. Run smoke baseline.
2. Run representative real baseline.
3. Aggregate the evidence.
4. Decide:
   - representative baseline blocked by external API: stop at evidence + runbook + summary
   - external API dominated: stop at evidence + runbook + summary
   - otherwise: optimize one hotspot only

## Before/After Comparison

If optimization is justified, compare before vs after on both smoke and representative real run:

- total wall time
- `artifact_extract_ms` sum
- `artifact_extract_ms` p95
- retry counts
- `infer_ms` distribution for regression checks
- `submit_ms` and `finalize_ms` for regression checks

The acceptance rule is narrow:

- improvement should show up primarily in `artifact_extract_ms`
- no contract changes
- no anomalous regression in retry behavior or remote inference timing
