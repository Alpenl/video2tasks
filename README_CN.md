<div align="center">

# 🎬 Video2Tasks

**多任务机器人视频 → 单任务片段 + 自动指令标注 → VLA 训练数据**

[![Python 3.8+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

[English](README.md) | [中文文档](README_CN.md)

</div>

---

## 📖 概览

### 🎯 解决什么问题？

训练 **VLA（Vision-Language-Action）模型**（如 [π₀ (pi-zero)](https://www.physicalintelligence.company/blog/pi0)）时，你需要的是**带指令标注的单任务视频片段**。然而，真实的机器人演示视频往往包含**多个连续任务**且**没有任何标注**：

```
输入:  包含多个任务的长视频，无标注
           ┃
           ▼
     ┌─────────────────────────────────────────────────────────────┐
     │  🎬 Video2Tasks                                             │
     │  • VLM 驱动的任务边界检测                                     │
     │  • 自动生成自然语言指令标注                                   │
     │  • 单机并行处理（共享文件系统）                               │
     └─────────────────────────────────────────────────────────────┘
           ┃
           ▼
输出: 单任务片段 + 指令标注，可直接用于 VLA 训练

  segment_001.mp4         segment_002.mp4         segment_003.mp4
  "Pick up the fork"      "Place the fork"        "Pick up the spoon"
```

**Video2Tasks = 任务切分 + 指令标注 → VLA 训练数据生产线**

### 🔧 工作原理

本工具采用 **Server/Worker 架构**（FastAPI Server + 多个 Worker 进程），使用视觉语言模型（如 Qwen3-VL）分析视频帧，检测任务边界，并为每个片段生成自然语言指令。

**支持的部署模式：** `single-machine shared-fs`。

也就是说：Server 与所有 Worker 必须运行在同一台机器（或共享同一套容器宿主机卷），并且看到的运行产物路径必须一致（例如 `./runs`、`./tmp`）。当前的传输语义不支持把 Worker 跑在不同机器或不同挂载点路径下。

| 组件 | 描述 |
|------|------|
| **Server** | 读取视频、分窗抽帧、管理任务队列并聚合结果 |
| **Worker** | 调用 VLM 推理，检测任务切换点并生成指令标注 |

---

## 📊 输出示例

### VLM 逐窗口推理过程

VLM 会分析每个重叠的帧窗口，并提供详细的任务切换推理：

<details>
<summary>🔍 点击查看多个窗口的 VLM 推理过程</summary>

**Window 0** - 检测 bag → mask 切换:
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

**Window 3** - 检测多物体切换:
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

**Window 8** - 未检测到切换（连续任务）:
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

**Window 14** - 复杂多物体序列:
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

### 最终切分结果

一个 4501 帧的视频自动切分成 16 个单任务片段：

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

> 🎯 每个片段只包含**一个任务**，并自动生成自然语言指令 —— 直接用于 VLA 训练！

---

## 💡 为什么选择这套架构？

<table>
<tr>
<td width="50%">

### 🧠 单机 Server/Worker（共享文件系统）

这不是一个死循环脚本。FastAPI 作为调度中心，Worker 只负责推理。

**当前仅支持在同一台机器上运行 Server 与多个 Worker 进程，并共享同一套文件系统路径。**

</td>
<td width="50%">

### 🛡️ 工程化容错

- ⏱️ Inflight 超时重发
- 🔄 失败重试上限
- 📍 `.DONE` 完成标记（见下文语义）

这些机制是大规模任务稳定跑完的关键。

</td>
</tr>
<tr>
<td width="50%">

### 🎯 智能切分算法

不是简单把图片丢给模型。`build_segments_via_cuts` 对多窗口结果做**加权投票**，并引入 **Hanning Window** 处理窗口边缘权重。

解决了"窗口边缘识别不稳"的经典问题。

</td>
<td width="50%">

### ✍️ 专业 Prompt 设计

`prompt_switch_detection` 明确区分：
- **True Switch**：切换到新物体
- **False Switch**：同一物体不同操作

贴合 Manipulation 数据集的痛点，**显著降低过切**。

</td>
</tr>
</table>

---

## ✨ 特性

| 特性 | 描述 |
|------|------|
| 🎥 **视频分窗** | 可配置的视频窗口抽样参数 |
| 🧩 **可插拔后端** | 支持 Qwen3-VL / 远程 API / 自定义 VLM |
| 📊 **智能聚合** | 加权投票 + Hanning Window 自动聚合分段结果 |
| 🔄 **并行 Worker** | 同机多进程并行（共享文件系统） |
| ⚙️ **YAML 配置** | 简洁的声明式配置管理 |
| 🧪 **跨平台** | 推荐 Linux + GPU；Windows/CPU 可用 dummy 后端 |

---

## 🏗️ 架构图

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

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/ly-geming/video2tasks.git
cd video2tasks

# 安装核心依赖
pip install -e .

# 如果使用 Qwen3-VL（需要 GPU）
pip install -e ".[qwen3vl]"
```

### 官方 Smoke Demo（首次运行）

建议先跑官方 smoke demo。它不需要你准备自己的视频，也不依赖任何外部 API Key。

```bash
v2t-cluster --config config.smoke.yaml
```

smoke 输出路径是固定且有测试覆盖的：
- Run 目录：`./tmp/smoke_runs/demo_smoke/official_smoke_demo`
- Sample 目录：`./tmp/smoke_runs/demo_smoke/official_smoke_demo/samples/sample_001`

命令退出后按以下顺序查看结果：
1. `samples/sample_001/.DONE` 或 `.FAILED`
2. `samples/sample_001/segments.json`
3. `samples/sample_001/sample_runtime.json`
4. `run_manifest.json`
5. `run_summary.json`

`segments.json` 仍然是结果层真相；`sample_runtime.json` 和 `run_summary.json` 是承载 runtime/export/fallback/retry 状态的 operator 证据层。

完整命令与预期结果请参考：[Official Smoke Demo Runbook](docs/runbooks/official-smoke-demo.md)

### 切换到你的真实数据（Smoke 通过后）

```bash
# 复制最小可运行模板
cp config.example.yaml config.yaml

# 修改数据路径和非敏感配置
vim config.yaml  # 或使用你喜欢的编辑器

# 一条命令启动 Server + Worker
v2t-cluster --config config.yaml
```

像 `OPENAI_API_KEY`、`GEMINI_API_KEY`、`LLM_MERGE_API_KEY` 这类敏感信息应通过环境变量提供。

代码中的 `worker.count` 默认值是 `7`，但 [`config.example.yaml`](config.example.yaml) 这个最小模板为了保守起跑，显式写的是 `worker.count: 1`。如果你直接复制模板不改，实际会以 `1` 启动，而不是 `7`。

CLI 现在不会再从当前工作目录隐式扫描 `./config.yaml`。请显式传 `--config config.yaml`，或者导出 `VIDEO2TASKS_CONFIG=/absolute/path/to/config.yaml`。
冻结后的官方配置层次只有：`env > yaml > defaults`（环境变量覆盖 YAML，YAML 覆盖代码默认值）。如果没有提供 YAML，则只剩 `env > defaults` 两层。

**部署契约（`single-machine shared-fs`）：**
- Server 与 Worker 必须在同一台机器上运行，并且看到的本地路径必须一致。
- 如果容器化，请把同一组宿主机目录以相同的容器内路径同时挂载给 Server 与 Worker（不要依赖不同挂载点或路径重写）。

**终端 1 - 启动服务器：**
```bash
v2t-server --config config.yaml
```

**终端 2 - 启动 Worker：**
```bash
v2t-worker --config config.yaml
```

---

## ⚙️ 配置说明

[`config.example.yaml`](config.example.yaml) 现在是最小可运行模板，不是完整调优真相，也不再试图列出所有调优项。未写出的字段会回落到 [`src/video2tasks/config.py`](src/video2tasks/config.py) 中定义的默认值。

配置优先级为：

1. 环境变量
2. 通过 `--config` 或 `VIDEO2TASKS_CONFIG` 指定加载的 YAML
3. 代码内置默认值

这也是运维排障与事故归因时唯一认可的分层模型。

敏感信息优先放环境变量，不要提交到受版本控制的 YAML 中。

`server.max_empty_retries_per_job` 的默认值现在是 `3`。只有在你明确要允许空结果无限重试时，才把它设成 `0`。

完整配置模型中的常见分组：

| 配置项 | 描述 |
|--------|------|
| `datasets` | 视频数据集路径和子集 |
| `run` | 输出目录配置 |
| `server` | 主机、端口和队列设置 |
| `worker` | VLM 后端选择和模型路径 |
| `windowing` | 帧采样参数 |

### 输出产物与状态标记（运维契约）

下文中的 `<run_dir>` 统一表示 `<run.base_dir>/<subset>/<run_id>`；默认示例是 `./runs/<subset>/<run_id>`。

- `<run_dir>/samples/<sample_id>/windows.jsonl`：Stage 1 每个窗口的原始结果（append-only）。
- `<run_dir>/samples/<sample_id>/segments.json`：样本结果层产物。只承载 segmentation + Stage 2 文本产物（merge/summary/字幕本地化结果）。runtime/export/fallback/retry 状态不属于最终切分真相。
  source instruction 永远是英文；字幕本地化只改变字幕文本。
- `<run_dir>/samples/<sample_id>/sample_runtime.json`：样本级 operator 证据。它是 canonical runtime artifact，承载终态、required-stage 完成情况、fallback 概况、retry 概况、export 概况，以及 failure 引用。
- `<run_dir>/samples/<sample_id>/.DONE` / `<run_dir>/samples/<sample_id>/.FAILED`：样本终态标记。
  `.DONE` 的语义是：该样本已完成 `<run_dir>/run_manifest.json.required_stages` 里定义的全部必需阶段。
- `failure.json`：只要存在 `.FAILED` 就必须存在，用来记录该样本面向 operator 的终态 reason/details。
- `<run_dir>/run_summary.json`：run 级 operator 证据，由 `run_manifest.json` 与各样本的 `sample_runtime.json` 聚合得到。

终态矩阵：

| 运行时结果 | `.DONE` | `.FAILED` | `failure.json` |
| --- | --- | --- | --- |
| 全部 required stages 完成 | 存在 | 不存在 | 不存在 |
| 任一 required stage 失败（`window_boundary_failed`、`segment_label_failed`、`boundary_refinement_failed`、`export_failed`） | 不存在 | 存在 | 存在 |
| 已知坏产物在 dispatch 前被拒绝（`artifact_extraction_failed`、`artifact_preparation_failed`） | 不存在 | 存在 | 存在 |
| finalize 崩溃，或把 required-stage 结果清空（`finalize_exception`、`finalize_empty_segments`） | 不存在 | 存在 | 存在 |

额外规则：

- 写入 `.FAILED` 时必须移除陈旧 `.DONE`；写入 `.DONE` 时必须移除陈旧 `.FAILED` 和 `failure.json`。
- `failure.json.reason` 是样本级终态原因。像 `empty_retry_exhausted`、`timeout_retry_exhausted` 这类 job 级原始原因会保留在 `windows.jsonl` / `segment_labels.jsonl` / `boundary_refinements.jsonl` 的 `terminal_error` 字段里；当对应 required stage 被识别为失败时，样本再收敛到 `.FAILED`。
- `server.max_empty_retries_per_job` 默认是 `3`。只有在你明确要允许空结果无限重试时才设成 `0`。一旦预算耗尽，原始 job 记录就进入终态，样本最终必须收敛到 `.FAILED`，不能停在半完成目录状态。
- `<run_dir>/exports/<sample_id>/annotated.mp4`：当 `export.mode=annotated|both` 且 annotated 导出成功时可见。
- `<run_dir>/clips/<sample_id>/...`：当 `export.mode=clips|both` 且 clips 导出成功时可见。
  `<run_dir>/clips/<sample_id>/manifest.json` 是 clips 导出契约记录。clips 导出必须保留音频（`audio_preserved=true`）。
- `<run_dir>/run_manifest.json`（run 级）：记录 run 身份（config/prompt hash、backend 摘要、`required_stages`）与 resume 校验元数据。
  resume 默认严格拒绝跨 identity 续跑（config/prompt/backend/required-stages 不一致）。只有显式设置 `run.force_resume=true` 或 `RUN_FORCE_RESUME=true` 才会放行。
- `sample_runtime.json` + `run_summary.json` 是 operator runtime-evidence 层，用于判断与审计，不替代最终切分结果本体。
- `segments.json.diagnostics` 在 P0 兼容窗口内仍会双写，但它只是兼容影子数据，不再是 canonical runtime evidence 的位置。

端点波动排障（以及如何区分端点不稳与代码/数据失败）请看 [Endpoint Volatility Runbook](docs/runbooks/endpoint-volatility.md)。

---

## 🔌 VLM 后端

### Dummy 后端（默认）

轻量级后端，用于测试和 Windows/CPU 环境。返回模拟结果，不加载重型模型。

```yaml
worker:
  backend: dummy
```

### Qwen3-VL 后端

使用 Qwen3-VL-32B-Instruct（或其他变体）进行完整推理。

**要求：**
- 🐧 Linux + NVIDIA GPU
- 💾 24GB+ 显存（32B 模型）
- 🔥 PyTorch + CUDA 支持

```yaml
worker:
  backend: qwen3vl
  qwen3vl:
    model_path: /path/to/model
```

### 远程 API 后端

如不想本地部署模型，可配置远程 API：

```yaml
worker:
  backend: remote_api
  remote_api:
    api_url: http://your-api-server/infer
```

<details>
<summary>📡 API 请求/响应格式</summary>

**请求体：**
```json
{
  "prompt": "...",
  "images_b64_png": ["...", "..."]
}
```

**响应格式（两种皆可）：**
```json
{
  "transitions": [6],
  "instructions": ["Place the fork", "Place the spoon"],
  "thought": "..."
}
```

或者：
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

### OpenAI 后端

Worker 可直接调用 OpenAI Responses API。这个后端默认面向 `gpt-5.2`。

```yaml
worker:
  backend: openai
  openai:
    api_key: ""  # 如果设置了 OPENAI_API_KEY，这里可以留空
    model: gpt-5.2
    reasoning_effort: low
```

API Key 可以写在本地未跟踪的 `config.yaml` 中，也可以通过环境变量提供；真实凭证更推荐放环境变量：

```bash
export OPENAI_API_KEY=your_api_key
```

### 自定义后端

实现 `VLMBackend` 接口来添加你自己的 VLM：

```python
from video2tasks.vlm.base import VLMBackend

class MyBackend(VLMBackend):
    def infer(self, images, prompt):
        # 你的推理逻辑
        return {"transitions": [], "instructions": []}
```

---

## 📁 项目结构

```
video2tasks/
├── 📂 src/video2tasks/
│   ├── config.py              # 配置模型
│   ├── prompt.py              # Prompt 模板
│   ├── 📂 server/             # FastAPI 服务端
│   │   ├── app.py
│   │   └── windowing.py
│   ├── 📂 worker/             # Worker 实现
│   │   └── runner.py
│   ├── 📂 vlm/                # VLM 后端
│   │   ├── dummy.py
│   │   ├── qwen3vl.py
│   │   └── remote_api.py
│   └── 📂 cli/                # CLI 入口
│       ├── server.py
│       └── worker.py
├── 📄 config.example.yaml
├── 📄 pyproject.toml
├── 📄 README.md
├── 📄 README_CN.md
└── 📄 LICENSE
```

---

## 🧪 测试与验证

```bash
# 验证配置文件
v2t-validate --config config.yaml

# 运行测试
pytest
```

---

## 💻 系统要求

<table>
<tr>
<th>最低配置（Dummy 后端）</th>
<th>推荐配置（Qwen3-VL）</th>
</tr>
<tr>
<td>

- Python 3.8+
- 4GB 内存
- 任意操作系统

</td>
<td>

- Python 3.8+
- Linux + NVIDIA GPU
- 24GB+ 显存
- CUDA 11.8+ / 12.x

</td>
</tr>
</table>

---

## 🤝 贡献

欢迎贡献代码！请随时提交 Pull Request。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 基于 [FastAPI](https://fastapi.tiangolo.com/) 构建
- VLM 支持来自 [Transformers](https://huggingface.co/docs/transformers/)
- 灵感来源于机器人视频分析研究

---

<div align="center">

**⭐ 如果觉得有用，请给个 Star！⭐**

</div>
