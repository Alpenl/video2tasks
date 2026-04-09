# Official Smoke Demo Runbook

This runbook defines the **official first-run path** for this repository.

- No external API key required.
- No user-provided video required.
- Uses `dummy` backend and a built-in MP4 fixture.

## 1) Prerequisites

From repo root:

```bash
pip install -e .
```

## 2) Run The Official Smoke Command

```bash
v2t-cluster --config config.smoke.yaml
```

The command should exit automatically after processing the built-in sample.

## 3) Fixed Input / Output Paths

- Input video:
  `./tests/fixtures/smoke_dataset/demo_smoke/sample_001/Frame_demo.mp4`
- Run dir:
  `./tmp/smoke_runs/demo_smoke/official_smoke_demo`
- Sample dir:
  `./tmp/smoke_runs/demo_smoke/official_smoke_demo/samples/sample_001`

## 4) Result Check Order (Operator Contract)

Check results in this exact order:

1. `./tmp/smoke_runs/demo_smoke/official_smoke_demo/samples/sample_001/.DONE` or `.FAILED`
2. `./tmp/smoke_runs/demo_smoke/official_smoke_demo/samples/sample_001/segments.json`
3. `./tmp/smoke_runs/demo_smoke/official_smoke_demo/run_manifest.json`

Notes:

- `.DONE` + `segments.json` + `run_manifest.json` are the current guaranteed smoke outputs and are covered by `tests/integration/test_official_smoke_demo.py`.
- Future runtime-evidence rollout may add `sample_runtime.json` and `run_summary.json`; treat them as optional at this stage.

## 5) Re-run Cleanly

```bash
rm -rf ./tmp/smoke_runs/demo_smoke/official_smoke_demo
v2t-cluster --config config.smoke.yaml
```
