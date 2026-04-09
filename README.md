<div align="center">

# 🎬 Video2Tasks

**Split Multi-Task Robot Videos into Single-Task Segments with Auto-Generated Instructions for VLA Training**

[![Python 3.8+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

[English](README.md) | [中文文档](README_CN.md)

</div>

---

## 📖 Overview

### 🎯 What Problem Does This Solve?

When training **VLA (Vision-Language-Action) models** like [π₀ (pi-zero)](https://www.physicalintelligence.company/blog/pi0), you need **single-task video segments with instruction labels**. However, real-world robot demonstration videos often contain **multiple consecutive tasks** without any annotation:

```
Input:  Long video with multiple tasks, NO labels
           ┃
           ▼
     ┌─────────────────────────────────────────────────────────────┐
     │  🎬 Video2Tasks                                             │
     │  • VLM-powered task boundary detection                      │
     │  • Auto-generate natural language instructions              │
     │  • Parallel processing on a single machine                  │
     └─────────────────────────────────────────────────────────────┘
           ┃
           ▼
Output: Single-task segments + instruction labels, READY for VLA training

  segment_001.mp4         segment_002.mp4         segment_003.mp4
  "Pick up the fork"      "Place the fork"        "Pick up the spoon"
```

**Video2Tasks = Task Segmentation + Instruction Labeling → VLA Training Data Pipeline**

### 🔧 How It Works

This tool uses a **server/worker architecture** (FastAPI server + worker processes) with VLMs (like Qwen3-VL) to analyze video frames, detect task boundaries, and generate natural language instructions for each segment.

**Supported deployment mode:** `single-machine shared-fs`.

That means the server and all workers must run on the same machine (or the same container host volume) and see the **same filesystem paths** for runtime artifacts (for example `./runs` and `./tmp`). Running workers on different machines or different mountpoints is not supported by the current transport semantics.

| Component | Description |
|-----------|-------------|
| **Server** | Manages job queues, video frame extraction, and result aggregation |
| **Worker** | Runs VLM inference to detect task transitions and generate instructions |

---

## 📊 Output Example

### VLM Window-by-Window Reasoning

The VLM analyzes each overlapping frame window and provides detailed reasoning about task transitions:

<details>
<summary>🔍 Click to see VLM reasoning for multiple windows</summary>

**Window 0** - Detecting bag → mask transition:
```json
{
  "task_id": "LongData601-1189::1765279974654_w0",
  "window_id": 0,
  "vlm_json": {
    "thought": "Frames 0-2: The person is standing, hands open, wearing gloves, facing the room. No object interaction yet. Frame 3: The person reaches toward a white tote bag on the sofa. Frame 4: The person grasps the tote bag and begins lifting it. Frames 5-11: The person continues manipulating the tote bag, opening it, adjusting its straps, and handling its contents. This is a continuous interaction with the same object (the tote bag). Frame 12: The person reaches into the bag and pulls out a white object with a black strap (likely a mask or headgear). The interaction from frame 12 onward shifts from the tote bag to the white object (mask). Therefore, a switch occurs at frame 12.",
    "transitions": [12],
    "instructions": ["Pick up and manipulate the tote bag", "Retrieve and adjust the white face mask"]
  }
}
```

**Window 3** - Detecting multiple object switches:
```json
{
  "task_id": "LongData601-1189::1765279974654_w3",
  "window_id": 3,
  "vlm_json": {
    "thought": "Frames 0-2: The robot's left hand reaches for and grasps a small black object from the left table. The right hand holds a white tote bag. Frames 3-5: The left hand places the black object into the tote bag. Frames 6-7: The left hand releases the black object into the bag and then reaches back to pick up another small black object. This is a clear switch: the robot completes interaction with the first black object and starts interacting with a second, distinct black object. Frame 15: The robot reaches for the white kettle on the left table. This marks a new interaction with a different object (the kettle). Therefore, switches are detected at frame 6 (first→second black object) and frame 15 (bag→kettle).",
    "transitions": [6, 15],
    "instructions": ["Place the first black object into the tote bag", "Place the second black object into the tote bag", "Pick up the white kettle"]
  }
}
```

**Window 8** - No switch detected (continuous task):
```json
{
  "task_id": "LongData601-1189::1765279974654_w8",
  "window_id": 8,
  "vlm_json": {
    "thought": "Frames 0-2: The robot is holding a small white object (possibly a charger or adapter) in both hands and appears to be manipulating or connecting it to a cable. Frames 3-15: The robot continues to hold the same object and cable, moving it toward an electrical outlet near the TV stand. There is no clear release of the object or grasping of a new distinct object. The entire sequence involves manipulating a single object. No switch occurs.",
    "transitions": [],
    "instructions": ["Move the charger toward the power outlet"]
  }
}
```

**Window 14** - Complex multi-object sequence:
```json
{
  "task_id": "LongData601-1189::1765279974654_w14",
  "window_id": 14,
  "vlm_json": {
    "thought": "Frames 0-2: The person is interacting with objects on the left table — moving a small white cup and adjusting a black rectangular object. Frame 3: The right hand moves away from the black object and reaches toward the ashtray. This marks the end of interaction with the black object. Frame 4: The right hand grasps the ashtray. This is a clear switch to a new object. Frames 5-7: The person moves the ashtray toward the trash can. Frame 11: The right hand reaches down to pick up a pair of white slippers from the floor. This is a clear switch from ashtray to slippers. Switches occur at frame 3 (black object→ashtray) and frame 11 (ashtray→slippers).",
    "transitions": [3, 11],
    "instructions": ["Move the black rectangular object and cup", "Pick up the ashtray", "Pick up the white slippers", "Place the slippers on the rack"]
  }
}
```

</details>

### Final Segmentation Output

A 4501-frame video automatically split into 16 single-task segments:

```json
{
  "video_id": "1765279974654",
  "nframes": 4501,
  "segments": [
    {"seg_id": 0,  "start_frame": 0,    "end_frame": 373,  "instruction": "Pick up and manipulate the tote bag"},
    {"seg_id": 1,  "start_frame": 373,  "end_frame": 542,  "instruction": "Retrieve and adjust the white face mask"},
    {"seg_id": 2,  "start_frame": 542,  "end_frame": 703,  "instruction": "Open and place items into the bag"},
    {"seg_id": 3,  "start_frame": 703,  "end_frame": 912,  "instruction": "Place the first black object into the tote bag"},
    {"seg_id": 4,  "start_frame": 912,  "end_frame": 1214, "instruction": "Place the second black object into the tote bag"},
    {"seg_id": 5,  "start_frame": 1214, "end_frame": 1375, "instruction": "Place the white cup on the table"},
    {"seg_id": 6,  "start_frame": 1375, "end_frame": 1524, "instruction": "Move the cup to the right table"},
    {"seg_id": 7,  "start_frame": 1524, "end_frame": 1784, "instruction": "Connect the power adapter to the cable"},
    {"seg_id": 8,  "start_frame": 1784, "end_frame": 2991, "instruction": "Plug the device into the power strip"},
    {"seg_id": 9,  "start_frame": 2991, "end_frame": 3135, "instruction": "Interact with black object on coffee table"},
    {"seg_id": 10, "start_frame": 3135, "end_frame": 3238, "instruction": "Adjust the ashtray"},
    {"seg_id": 11, "start_frame": 3238, "end_frame": 3359, "instruction": "Interact with the white mug"},
    {"seg_id": 12, "start_frame": 3359, "end_frame": 3478, "instruction": "Move the black rectangular object and cup"},
    {"seg_id": 13, "start_frame": 3478, "end_frame": 3711, "instruction": "Pick up the ashtray"},
    {"seg_id": 14, "start_frame": 3711, "end_frame": 4095, "instruction": "Move the white slippers from the shoe rack"},
    {"seg_id": 15, "start_frame": 4095, "end_frame": 4501, "instruction": "Raise the window blind"}
  ]
}
```

> 🎯 Each segment contains exactly ONE task with auto-generated natural language instruction — ready for VLA training!

---

## 💡 Why This Architecture?

<table>
<tr>
<td width="50%">

### 🧠 Single-Machine Server/Worker

Not just a single script. FastAPI acts as the orchestrator; workers handle inference only.

**Run the server and multiple worker processes on the same machine, sharing the same filesystem.**

</td>
<td width="50%">

### 🛡️ Production-Ready Resilience

- ⏱️ Inflight timeout & re-dispatch
- 🔄 Configurable retry limits
- 📍 `.DONE` completion markers (see semantics below)

Critical mechanisms for running large-scale tasks to completion.

</td>
</tr>
<tr>
<td width="50%">

### 🎯 Smart Segmentation Algorithm

Not just throwing images at a model. `build_segments_via_cuts` performs **weighted voting** across overlapping windows with **Hanning Window** edge weighting.

Solves the classic "unstable edge detection" problem.

</td>
<td width="50%">

### ✍️ Domain-Specific Prompts

`prompt_switch_detection` explicitly distinguishes:
- **True Switch**: Transition to a new object
- **False Switch**: Different operation on the same object

Tailored for manipulation datasets, **significantly reducing over-segmentation**.

</td>
</tr>
</table>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Video Windowing** | Configurable video window sampling parameters |
| 🤖 **Pluggable Backends** | Support for Qwen3-VL, Remote API, or custom VLM implementations |
| 📊 **Smart Aggregation** | Automatic segment generation with weighted voting & Hanning window |
| 🔄 **Parallel Workers** | Run multiple worker processes on one machine (shared filesystem) |
| ⚙️ **YAML Config** | Simple, declarative configuration management |
| 🖥️ **Cross-Platform** | Linux/GPU recommended; Windows/CPU with dummy backend |

---

## 🏗️ Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │         │                 │
│     Server      │────────▶│   Job Queue     │◀────────│     Worker      │
│    (FastAPI)    │         │                 │         │     (VLM)       │
│                 │         │                 │         │                 │
└────────┬────────┘         └─────────────────┘         └────────┬────────┘
         │                                                       │
         ▼                                                       ▼
┌─────────────────┐                                     ┌─────────────────┐
│   Video Files   │                                     │    VLM Model    │
└─────────────────┘                                     └─────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ly-geming/video2tasks.git
cd video2tasks

# Install with core dependencies
pip install -e .

# Or install with Qwen3-VL support (requires GPU)
pip install -e ".[qwen3vl]"
```

### Official Smoke Demo (First Run)

Run the official smoke demo first. It does not require your own video or any external API key.

```bash
v2t-cluster --config config.smoke.yaml
```

Smoke output paths are fixed and test-covered:
- Run dir: `./tmp/smoke_runs/demo_smoke/official_smoke_demo`
- Sample dir: `./tmp/smoke_runs/demo_smoke/official_smoke_demo/samples/sample_001`

After the command exits, inspect results in this order:
1. `samples/sample_001/.DONE` or `.FAILED`
2. `samples/sample_001/segments.json`
3. `samples/sample_001/sample_runtime.json`
4. `run_manifest.json`
5. `run_summary.json`

`segments.json` remains the result-layer truth. `sample_runtime.json` and `run_summary.json` are the operator-evidence layer for runtime/export/fallback/retry state.

For full commands and expected outputs, see [Official Smoke Demo Runbook](docs/runbooks/official-smoke-demo.md).

### Move To Your Own Data (After Smoke)

```bash
# Copy the minimal runnable template
cp config.example.yaml config.yaml

# Edit dataset path and non-secret settings
vim config.yaml  # or your preferred editor

# Start server + configured workers
v2t-cluster --config config.yaml
```

Use environment variables for secrets such as `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `LLM_MERGE_API_KEY`.

The code default for `worker.count` is `7`, but [`config.example.yaml`](config.example.yaml) intentionally sets `worker.count: 1` for a conservative first custom run. If you copy the template unchanged, you will start with `1`, not `7`.

The CLI no longer auto-discovers `./config.yaml` from the current working directory. Use `--config config.yaml` explicitly, or export `VIDEO2TASKS_CONFIG=/absolute/path/to/config.yaml`.
The frozen config-layering contract is exactly: `env > yaml > defaults` (environment variables override YAML, YAML overrides built-in defaults). If no YAML is supplied, only `env > defaults` remain active.

**Deployment contract (`single-machine shared-fs`):**
- Run the server and workers on the same machine, and ensure they see the same on-disk paths.
- If you containerize, mount the same host directories into the same in-container paths for both server and workers (do not rely on different mountpoints or path rewrites).

**Terminal 1 - Start the Server:**
```bash
v2t-server --config config.yaml
```

**Terminal 2 - Start a Worker:**
```bash
v2t-worker --config config.yaml
```

---

## ⚙️ Configuration

[`config.example.yaml`](config.example.yaml) is the minimal runnable template, not the full source of truth for every tuning knob or production tuning. Omitted values fall back to the defaults defined in [`src/video2tasks/config.py`](src/video2tasks/config.py).

Configuration precedence is:

1. Environment variables
2. YAML loaded via `--config` or `VIDEO2TASKS_CONFIG`
3. Built-in defaults

This is the only supported layering model for operator triage and incident analysis.

For secrets, prefer environment variables over YAML. Keep tracked config files credential-free.

The server now defaults `server.max_empty_retries_per_job` to `3`. Set it to `0` only if you explicitly want unlimited retries after empty model outputs.

Common sections in the full config model:

| Section | Description |
|---------|-------------|
| `datasets` | Video dataset paths and subsets |
| `run` | Output directory configuration |
| `server` | Host, port, and queue settings |
| `worker` | Worker count, VLM backend selection and model paths |
| `windowing` | Frame sampling parameters |

### Output Artifacts (Operator Contract)

`<run_dir>` below means `<run.base_dir>/<subset>/<run_id>`; by default that is `./runs/<subset>/<run_id>`.

- `<run_dir>/samples/<sample_id>/windows.jsonl`: Stage 1 window-level raw results (append-only).
- `<run_dir>/samples/<sample_id>/segments.json`: sample result-layer output. It only carries segmentation + Stage 2 text artifacts (merge/summary/subtitle-localization results). Runtime/export/fallback/retry state is not segmentation truth.
  Source instructions are always English. Subtitle localization changes subtitle text only.
- `<run_dir>/samples/<sample_id>/sample_runtime.json`: sample-level operator evidence. It is the canonical runtime artifact for terminal state, required-stage completion, fallback summary, retry summary, export summary, and failure reference.
- `<run_dir>/samples/<sample_id>/.DONE` / `<run_dir>/samples/<sample_id>/.FAILED`: sample terminal markers.
  `.DONE` means all stages listed in `<run_dir>/run_manifest.json.required_stages` completed for that sample.
- `failure.json`: required whenever `.FAILED` exists. It carries the operator-facing terminal reason/details for that sample.
- `<run_dir>/run_summary.json`: run-level operator evidence aggregated from `run_manifest.json` + per-sample `sample_runtime.json` records.

Terminal-state matrix:

| Runtime outcome | `.DONE` | `.FAILED` | `failure.json` |
| --- | --- | --- | --- |
| All required stages completed | present | absent | absent |
| Any required stage fails (`window_boundary_failed`, `segment_label_failed`, `boundary_refinement_failed`, `export_failed`) | absent | present | present |
| Known-bad artifacts are rejected before dispatch (`artifact_extraction_failed`, `artifact_preparation_failed`) | absent | present | present |
| Finalize crashes or empties a required-stage result (`finalize_exception`, `finalize_empty_segments`) | absent | present | present |

Additional rules:

- Writing `.FAILED` always removes stale `.DONE`. Writing `.DONE` always removes stale `.FAILED` and `failure.json`.
- `failure.json.reason` is the sample-level terminal reason. Raw per-job causes such as `empty_retry_exhausted` and `timeout_retry_exhausted` remain in `windows.jsonl` / `segment_labels.jsonl` / `boundary_refinements.jsonl` as `terminal_error`, then the sample converges to `.FAILED` when the corresponding required stage is observed as failed.
- `server.max_empty_retries_per_job` defaults to `3`. Set it to `0` only if you explicitly want unlimited empty-result retries. Once the budget is exhausted, the raw job record is terminal and the sample must eventually close as `.FAILED`; it must not remain half-finished.
- `<run_dir>/exports/<sample_id>/annotated.mp4`: expected when `export.mode=annotated|both` and annotated export succeeds.
- `<run_dir>/clips/<sample_id>/...`: expected when `export.mode=clips|both` and clip export succeeds.
  `<run_dir>/clips/<sample_id>/manifest.json` is the clip export contract record. Clips must preserve audio (`audio_preserved=true`).
- `<run_dir>/run_manifest.json` (run-level): records run identity (config/prompt hashes, backend summary, `required_stages`) and resume validation metadata.
  Resume is strict by default: cross identity continuation (config/prompt/backend/required stage mismatch) is rejected unless you explicitly set `run.force_resume=true` or `RUN_FORCE_RESUME=true`.
- `sample_runtime.json` + `run_summary.json` are the operator runtime-evidence layer. They exist for operator decisions and auditing, not as a replacement for the final segmentation result.
- `segments.json.diagnostics` is still dual-written during the P0 compatibility window, but it is compatibility shadow data, not the canonical runtime evidence location.

For endpoint volatility triage (and how to separate endpoint volatility from code/data failures), see [Endpoint Volatility Runbook](docs/runbooks/endpoint-volatility.md).

---

## 🔌 VLM Backends

### Dummy Backend (Default)

Lightweight backend for testing and Windows/CPU environments. Returns mock results without loading heavy models.

```yaml
worker:
  count: 7
  backend: dummy
```

### Qwen3-VL Backend

Full inference using Qwen3-VL-32B-Instruct (or other variants).

**Requirements:**
- 🐧 Linux with NVIDIA GPU
- 💾 24GB+ VRAM (for 32B model)
- 🔥 PyTorch with CUDA support

```yaml
worker:
  backend: qwen3vl
  qwen3vl:
    model_path: /path/to/model
```

### Remote API Backend

Use an external API endpoint for inference:

```yaml
worker:
  backend: remote_api
  remote_api:
    api_url: http://your-api-server/infer
```

<details>
<summary>📡 API Request/Response Format</summary>

**Request:**
```json
{
  "prompt": "...",
  "images_b64_png": ["...", "..."]
}
```

**Response (either format is accepted):**
```json
{
  "transitions": [6],
  "instructions": ["Place the fork", "Place the spoon"],
  "thought": "..."
}
```

or:
```json
{
  "vlm_json": {
    "transitions": [6],
    "instructions": ["Place the fork", "Place the spoon"],
    "thought": "..."
  }
}
```

</details>

### OpenAI Backend

Call OpenAI Responses API directly from the worker. `gpt-5.2` is the default target model for this backend.

```yaml
worker:
  backend: openai
  openai:
    api_key: ""  # Optional when OPENAI_API_KEY is set
    model: gpt-5.2
    reasoning_effort: low
```

You can provide the API key either in a local untracked `config.yaml` or through the environment. Environment variables are the recommended path for real credentials so they do not end up in tracked YAML files.

```bash
export OPENAI_API_KEY=your_api_key
```

### Custom Backend

Implement the `VLMBackend` interface to add your own:

```python
from video2tasks.vlm.base import VLMBackend

class MyBackend(VLMBackend):
    def infer(self, images, prompt):
        # Your inference logic
        return {"transitions": [], "instructions": []}
```

---

## 📁 Project Structure

```
video2tasks/
├── 📂 src/video2tasks/
│   ├── config.py              # Configuration models
│   ├── prompt.py              # Prompt templates
│   ├── 📂 server/             # FastAPI server
│   │   ├── app.py
│   │   └── windowing.py
│   ├── 📂 worker/             # Worker implementation
│   │   └── runner.py
│   ├── 📂 vlm/                # VLM backends
│   │   ├── dummy.py
│   │   ├── qwen3vl.py
│   │   └── remote_api.py
│   └── 📂 cli/                # CLI entrypoints
│       ├── server.py
│       └── worker.py
├── 📄 config.example.yaml
├── 📄 pyproject.toml
├── 📄 README.md
├── 📄 README_CN.md
└── 📄 LICENSE
```

---

## 🧪 Testing

```bash
# Validate configuration
v2t-validate --config config.yaml

# Run tests
pytest
```

---

## 💻 Requirements

<table>
<tr>
<th>Minimum (Dummy Backend)</th>
<th>Recommended (Qwen3-VL)</th>
</tr>
<tr>
<td>

- Python 3.8+
- 4GB RAM
- Any OS

</td>
<td>

- Python 3.8+
- Linux + NVIDIA GPU
- 24GB+ VRAM
- CUDA 11.8+ / 12.x

</td>
</tr>
</table>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- VLM support via [Transformers](https://huggingface.co/docs/transformers/)
- Inspired by robotic video analysis research

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

WARNING!!
thanks for the great using tips from YuanJingYi (Sun Yat-sen University）
PLEASE name your video like this:
<img width="386" height="143" alt="348e206ad4948edee65c82d8c12ae671" src="https://github.com/user-attachments/assets/272bad75-872d-4321-9e24-e59f211ae880" />
and put each video in each folder like this
<img width="355" height="139" alt="1be04121f3312610400b559daa5bd7b3" src="https://github.com/user-attachments/assets/c65e841f-893e-411d-8e33-3a52cef95a1b" />



</div>
