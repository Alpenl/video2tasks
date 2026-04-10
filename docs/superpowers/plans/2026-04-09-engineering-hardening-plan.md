# Engineering Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `video2tasks` feel like a dependable engineering system on a single machine with a shared filesystem: easy to start correctly, easy to resume safely, easy to inspect when external APIs wobble, and cheap to maintain without large-bang refactors.

**Architecture:** Do not treat this as a greenfield cleanup. The repo already has meaningful contract work in place: `run_manifest.py`, `sample_store.py`, `job_builder.py`, `segment_semantics.py`, explicit config precedence, and a minimal structured logging helper. The next plan should build on those anchors, close the remaining product/contract gaps, and postpone deep module splits until the canonical runtime artifacts and Stage 2 app integration are stable.

**Tech Stack:** Python, FastAPI, Pydantic, OpenCV, ffmpeg, pytest

---

## Current-State Calibration

This proposal is directionally right, but the repo is not starting from zero.

Already landed in the current tree:

- `single-machine shared-fs` is already the documented deployment contract in `README.md` and `README_CN.md`.
- `run_manifest.py` already exists and is wired into `create_app()`, with resume validation covered by `tests/server/test_run_manifest.py`.
- `sample_store.py`, `job_builder.py`, and `segment_semantics.py` already exist, so “continue splitting `app.py` / `windowing.py` / `llm_merge.py`” should be treated as later-stage follow-up, not first-wave work.
- Config precedence is already `env > yaml > defaults`, and `Config.load()` no longer implicitly scans `./config.yaml`.
- `config.g3flash.yaml` is already scrubbed of real secrets.
- `logging_utils.py` already provides `log_event(...)`, and `job_builder.py` / `app.py` already emit a first batch of structured events.
- `llm_merge.py` already has a narrow module-side Stage 2 envelope via `run_llm_stage2_pass(...)`.

Still materially incomplete:

- There is no official smoke dataset + smoke config + smoke test path that a new operator can run without inventing local inputs.
- There is no run-level summary artifact. Operators still have to infer outcome from logs and sample directories.
- The structured event helper exists, but there is no frozen event schema document or test-backed field contract.
- `app.py` still uses the legacy Stage 2 app integration path: `run_llm_postprocess_pass(...)` plus `run_export_subtitle_localization_pass(...)`, instead of the canonical `run_llm_stage2_pass(...)`.
- `segments.json` still carries runtime diagnostics written during finalize, which conflicts with the README’s stated result-layer contract.
- `/health` is still a trivial `{"status": "ok"}` and should not be mistaken for real operator observability.

## Review Conclusion

### Proposal Points To Keep

- “好用工程”的目标定义成可预期、可复现、可定位、可维护，这个方向正确。
- 先做产品面可用性，再做契约和可观测性，再做结构拆分和性能，这是正确顺序。
- 行为测试优先于结构测试，这个建议值得直接采纳。
- 性能优化必须后置到可观测性和 profile 之后，这一点必须保留。
- “不要为多机分布式预埋复杂设计”是正确约束；短期目标仍是单机共享盘。

### Proposal Points To Adjust

- `run manifest` 不该再列为“待建设核心项”。它已经存在；当前任务应该从“新建 run manifest”改成“把它纳入官方 operator 路径，并在其旁边补齐 sample-level runtime evidence 和 run summary”。
- “固定结构化事件 schema”不应理解成引入新 telemetry 基础设施。当前应该做的是冻结现有事件名与字段、补文档和测试，不要新上 tracing/metrics 栈。
- “固化官方配置层次”也不再是配置系统重写任务。当前主要是补文档、补 smoke 配置、补测试锁定，避免 README 与默认值再次漂移。
- “继续拆 `app.py`”方向对，但优先级应下调到 P2。当前 `app.py` 仍是 1318 行热点文件，过早拆分会和 contract/observability 收口互相打架。
- “Stage 2 做窄接口”建议应上调到 P1，因为窄接口其实已经在 `llm_merge.py` 里存在，但 `app.py` 还没有切过去。这是低于大拆分、但高于继续堆 README 的真实缺口。
- “端点波动不算代码失败”的排障文档值得做，但最好排在 smoke demo 和 run summary 之后，让文档能引用真实 artifact、事件名和 reason 字段，而不是泛泛而谈。

### Proposal Missing One Important Addition

需要补一项：**sample-level runtime evidence artifact**。

原因：

- 仅有 `run_summary.json` 不足以替代当前 `segments.json.diagnostics`。
- 仅有 `run_manifest.json` 也不足以解释单个 sample 为何 fallback、为何失败、为何导出缺失。
- 现有 README 已经把 `segments.json` 定义为“result-layer output”，所以还需要一个并排的 sample 运行态文件来承接 operator 证据。

建议新增的正式 artifact：

- `<run_dir>/samples/<sample_id>/sample_runtime.json`
- `<run_dir>/run_summary.json`

其中 `sample_runtime.json` 记录单 sample 的运行态、fallback、导出结果、失败原因；`run_summary.json` 只做 run 级聚合，不取代 sample 级细节。

## What “Good Engineering” Means Here

在 `video2tasks` 里，“好用工程”不是指模块数量变多，也不是指先支持多机；验收标准应围绕下面四类能力。

### 1. 可预期

验收标准：

- 一个新用户在干净 checkout 上，有一条官方 smoke 命令可以成功跑通。
- README 的唯一“首次运行推荐路径”与仓库内的 smoke 测试使用同一套命令、同一套配置、同一份 fixture。
- 结果查看路径明确，至少包括 `.DONE` / `.FAILED`、`segments.json`、`sample_runtime.json`、`run_summary.json`。

### 2. 可复现

验收标准：

- `run_manifest.json` 持续作为 resume identity 的唯一 run-level 身份锚点。
- 配置优先级只允许 `env > yaml > defaults`，README、示例配置、测试全部锁定这一点。
- smoke 路径和 dummy backend 输出具备稳定断言，不依赖外部 API。

### 3. 可定位

验收标准：

- 关键事件有固定事件名和字段约定，至少能回答：抽帧是否完成、job 是否下发、infer 是否重试、结果为何 fallback、sample 为何失败、finalize 花了多久。
- run 结束后可以直接查看 `run_summary.json`，无需通读完整日志就知道成功数、失败数、fallback 数、empty retry/timeout retry 数、Stage 2/导出概况。
- sample 失败时，`failure.json` 与 `sample_runtime.json` 能把“代码问题 / 数据问题 / 外部端点波动”区分开。

### 4. 可维护

验收标准：

- 高价值测试围绕 artifact contract 和行为路径，而不是围绕 helper 私有实现。
- `app.py`、`windowing.py`、`llm_merge.py` 的后续拆分建立在稳定 facade 上，不改变结果 contract。
- 性能优化以 profile 为前提，且能从 run summary / structured events 看见改善前后的差异。

## Serial Gates And Parallelism Rules

### Serial Gates

#### Gate 0: Freeze Artifact Names And Smoke Surface

必须先定死以下名称，再允许多个子代理并行写文档和测试：

- `config.smoke.yaml`
- `tests/fixtures/smoke_dataset/...`
- `sample_runtime.json`
- `run_summary.json`
- 官方 smoke 命令

#### Gate 1: Freeze Canonical Runtime Evidence Destination

必须明确：

- `segments.json` 是结果层真相
- `sample_runtime.json` 是 sample 运行态证据
- `run_summary.json` 是 run 级聚合
- `segments.json.diagnostics` 的兼容窗口和移除时点在本 gate 一次定死，不允许不同 lane 各自决定

在这个 gate 之前，不要并行推进 README 结果查看说明、行为测试重写、Stage 2 app 切换。

#### Gate 2: Freeze Stage 2 App Facade

在 `app.py` 切到 `run_llm_stage2_pass(...)` 之前，不要并行推进 `llm_merge.py` 深拆和 `app.py` 大拆。

### Single-Owner Hotspot Files

以下文件在同一 wave 内应保持单 owner，避免多子代理互相踩：

- `src/video2tasks/server/app.py`
- `src/video2tasks/server/sample_store.py`
- `README.md`
- `README_CN.md`

### Good Parallel Lanes

下面这些工作流适合并行派子代理，但前提是通过对应 serial gate：

- smoke demo 资产 + smoke 测试 + README 最小运行路径
- structured event schema 文档 + logging 合约测试
- endpoint volatility runbook + operator FAQ
- 行为测试迁移，按 `windowing` / `llm_merge` / app artifact contract 分 lane 切
- P2 之后的 `windowing.py` 和 `llm_merge.py` 责任拆分

## Wave Plan

## P0: Product Surface And Operator Evidence

### Task P0-1: Official Smoke Demo

**Goal:** 让新用户在仓库 checkout 后，不需要准备自己的视频，也不需要外部 API，就能跑出一套被官方文档和自动化测试共同认可的最小闭环。

**Files:**

- Create: `config.smoke.yaml`
- Create: `tests/fixtures/smoke_dataset/demo_smoke/sample_001/Frame_demo.mp4`
- Create: `tests/integration/test_official_smoke_demo.py`
- Create: `docs/runbooks/official-smoke-demo.md`
- Modify: `README.md`
- Modify: `README_CN.md`

**Key Risks:**

- fixture 太大，导致仓库膨胀。
- fixture 太假，跑不出稳定 `segments.json` 断言。
- README 出现“真实运行推荐命令”和“smoke 命令”两套互相竞争的话术。

**Test / Verification:**

- `pytest tests/integration/test_official_smoke_demo.py -q`
- 手工验证官方命令在 dummy backend 下可跑通。
- README 中的路径、命令、结果位置与测试断言完全一致。

**Parallelism:** 通过 Gate 0 后可并行执行，但 `README.md` / `README_CN.md` 应由同一个 owner 落稿。

- [ ] 选定一个小体积、可提交的真实 MP4 smoke fixture，优先控制在“足够真实但不显著增大仓库”。
- [ ] 新增 `config.smoke.yaml`，强制使用 dummy backend、单机共享盘、本地 run 目录。
- [ ] 增加 smoke integration test，断言 `.DONE`、`segments.json`、`run_manifest.json` 存在且字段最小完整。
- [ ] 在 README / README_CN 中把官方首次运行路径改成 smoke demo，而不是让用户先手改 `config.yaml` 再试。
- [ ] 在 smoke 文档中明确结果查看顺序：`.DONE` / `.FAILED` -> `segments.json` -> `sample_runtime.json` -> `run_summary.json`。

### Task P0-2: Freeze Structured Event Schema

**Goal:** 在不引入新 observability 基础设施的前提下，把现有事件名与关键字段冻结下来，变成 operator 可依赖的契约。

**Files:**

- Create: `docs/observability/event-schema.md`
- Modify: `src/video2tasks/logging_utils.py`
- Modify: `tests/test_logging.py`
- Modify: `src/video2tasks/server/job_builder.py`
- Modify: `src/video2tasks/server/app.py`

**Key Risks:**

- 试图把 logging 做成新框架，导致范围膨胀。
- 先写 schema 后改实现时字段频繁 churn。
- `app.py` 与 `job_builder.py` 同时改名导致测试大面积失效。

**Test / Verification:**

- `pytest tests/test_logging.py -q`
- 针对 dummy run 收集日志，确认关键事件存在且字段齐全。
- 文档中的字段名与代码里实际事件完全一致。

**Parallelism:** 事件 schema 文档和 `tests/test_logging.py` 可以先并行；`app.py` 字段补齐必须单 owner。

- [ ] 冻结首批正式事件名：`artifact_extract_done`、`artifact_reuse_hit`、`job_dispatched`、`infer_attempt`、`job_done`、`result_empty_retry`、`result_timeout_retry`、`fallback_applied`、`sample_failed`、`finalize_done`。
- [ ] 为每个事件写出“必有字段”和“可选字段”，尤其是 `subset`、`sample_id`、`job_type`、`task_id`、`dispatch_id`、耗时字段、fallback reason。
- [ ] 只在必要处为 `logging_utils.py` 增加轻量 helper；不要引入新的 telemetry stack。
- [ ] 用测试锁定 event 名和核心字段，避免后续文档和代码再漂移。

### Task P0-3: Runtime Closure And Terminal-State Semantics

**Goal:** 把最核心的运行时正确性闭环在早期定死：sample 生命周期只能收敛到 `done` 或 `failed`；`.DONE` / `.FAILED` / `failure.json` 的交互由单一状态机负责；空结果重试和坏产物拒绝都变成有界且可解释的行为。

**Files:**

- Modify: `src/video2tasks/server/app.py`
- Modify: `src/video2tasks/server/sample_store.py`
- Modify: `src/video2tasks/server/task_artifacts.py`
- Modify: `src/video2tasks/server/windowing.py`
- Modify: `tests/server/test_app_retry.py`
- Modify: `tests/server/test_sample_store.py`
- Modify: `README.md`
- Modify: `README_CN.md`

**Key Risks:**

- 只把 `.DONE` / `.FAILED` 写进文档和 smoke 断言，而没有真正收口运行时写入条件。
- `app.py` 和 `sample_store.py` 被多人同时改动，破坏终态一致性。
- 坏图拒绝、空结果重试、finalize 异常三个止血点分散实现，最后仍留下灰色状态。

**Test / Verification:**

- `pytest tests/server/test_app_retry.py tests/server/test_sample_store.py -q`
- 增加 config-matrix 验证，至少覆盖：
  - 仅 Stage 1 必需时，Stage 1 完成即可写 `.DONE`
  - Stage 2 必需且 export 关闭时，Stage 2 写回成功后才写 `.DONE`
  - export 启用时，按当前 config 中 `required_stages` 的定义决定 `.DONE` 是否等待 export
  - optional stage 被禁用时，不阻止 `.DONE`
  - 任一 required stage 失败时，写 `.FAILED` 和 `failure.json`，且不得残留 `.DONE`
  - finalize 异常、已知坏产物、超过上限的空结果重试都必须收敛到 `failed`
- 在 operator 文档中补一张终态矩阵，明确 `.DONE` / `.FAILED` 与 `failure.json` 的联动规则。

**Parallelism:** 必须串行，`src/video2tasks/server/app.py` 与 `src/video2tasks/server/sample_store.py` 由单 owner 负责；`windowing.py` / `task_artifacts.py` 的坏产物拒绝逻辑可由协作者准备，但最终只能并到同一个 runtime owner 分支。

- [ ] 明确唯一终态：sample 只能进入 `done` 或 `failed`，不允许静默跳过、stuck finalize 或半完成目录状态。
- [ ] 重写 `.DONE` 写入条件：仅当当前 config 所要求的全部 required stages 完成时才写入。
- [ ] 重写 `.FAILED` / `failure.json` 交互：required stage 失败时必须落 `.FAILED` 和 `failure.json`，并清理/阻止 `.DONE`。
- [ ] 为 `.DONE` / `.FAILED` 增加 config-matrix 测试，覆盖 disabled optional stages、required Stage 2、export 组合和失败交互。
- [ ] 在 dispatch 前拒绝已知坏产物，把空字节/不可解码图片归因到 extraction 或 preparation，而不是 worker timeout。
- [ ] 保持 `server.max_empty_retries_per_job` 为有界默认值，并把“达到上限后的终态”和 reason 字段写成 operator 可读 contract。

### Task P0-4: Add `sample_runtime.json` And `run_summary.json`

**Goal:** 把 operator 证据层从 `segments.json` 中剥离出来，先建立新 artifact，即使 P0 阶段仍保留一段兼容期。

**Files:**

- Create: `src/video2tasks/server/run_summary.py`
- Modify: `src/video2tasks/server/sample_store.py`
- Modify: `src/video2tasks/server/app.py`
- Create: `tests/server/test_run_summary.py`
- Modify: `tests/server/test_sample_store.py`
- Modify: `README.md`
- Modify: `README_CN.md`

**Key Risks:**

- 只做 run summary、不做 sample-level 证据，最终仍要依赖 `segments.json.diagnostics`。
- 在 `app.py` 里重复拼装 summary，导致和 sample 实际产物不一致。
- 过早删除旧 `segments.json.diagnostics`，打断现有测试和使用者。

**Test / Verification:**

- `pytest tests/server/test_run_summary.py tests/server/test_sample_store.py -q`
- dummy smoke run 结束后，检查 `<run_dir>/run_summary.json` 与 sample 目录中的 `sample_runtime.json`。
- 验证成功 run、失败 run、fallback run、export disabled run 都能产出稳定摘要。

**Parallelism:** `run_summary.py` 与测试文件可以并行准备；`app.py` 与 `sample_store.py` 的落地必须单 owner。

- [ ] 在 `SampleStore` 中增加 `sample_runtime.json` 路径与持久化 helper。
- [ ] 定义 `sample_runtime.json` 的最小字段：sample 终态、required stages 完成情况、fallback 概况、retry 概况、export 概况、失败原因引用。
- [ ] 新增 `run_summary.py`，从 sample runtime 记录和 run manifest 聚合 run 级统计，不把 summary 逻辑散落在 `app.py` 里。
- [ ] P0 允许兼容期双写：保留旧 `segments.json.diagnostics`，同时把 canonical 运行态写进 `sample_runtime.json`。
- [ ] README 明确 `sample_runtime.json` / `run_summary.json` 是 operator evidence，不是 segmentation truth。

### Task P0-5: Freeze Operator Docs For Config Layering And Endpoint Volatility

**Goal:** 把当前已经基本正确的配置层次、单机共享盘限制、端点波动排障路径固化成官方说法。

**Files:**

- Create: `docs/runbooks/endpoint-volatility.md`
- Modify: `README.md`
- Modify: `README_CN.md`
- Modify: `config.example.yaml`

**Key Risks:**

- 在 P0-2 / P0-4 命名没定之前就写文档，导致再次返工。
- 把“外部 API 抖动”写成泛泛建议，而不是指向真实 artifact 和 event reason。

**Test / Verification:**

- README 和 runbook 交叉检查，无相互矛盾描述。
- smoke 文档、配置模板、排障文档使用统一术语。

**Parallelism:** 通过 Gate 0 和 Gate 1 后可并行执行。

- [ ] 明确官方配置层次只认 `env > yaml > defaults`。
- [ ] 明确 `config.example.yaml` 是最小模板，不是完整调优真相。
- [ ] 写清楚端点波动的观测入口：日志事件、`sample_runtime.json`、`failure.json`、`run_summary.json`。
- [ ] 明确“外部端点不稳定 != 代码失败”，但也要列出何时应归因于代码或数据。

## P1: Canonical Contracts And Behavior Tests

### Task P1-1: Switch App Integration To Canonical Stage 2 Interface

**Goal:** 让 `app.py` 以 `run_llm_stage2_pass(...)` 作为 Stage 2 唯一 app-side contract，停止继续扩展 legacy orchestration。

**Files:**

- Modify: `src/video2tasks/server/app.py`
- Modify: `src/video2tasks/server/llm_merge.py`
- Modify: `tests/server/test_llm_summary.py`
- Modify: `tests/server/test_app_retry.py`

**Key Risks:**

- app 侧 Stage 2 切换与 P0 的 runtime artifact 双写逻辑互相冲突。
- 现有测试大量围绕 legacy subtitle/export shape，切换时回归面较大。
- 如果同时推动 `llm_merge.py` 深拆，会放大冲突。
- 如果只换函数入口、不同时切断 export-gated Stage 2 语义，仍会保留旧的 contract 错误。

**Test / Verification:**

- `pytest tests/server/test_llm_summary.py tests/server/test_app_retry.py -q`
- dummy backend 下验证 merge / summary / subtitle 的成功、fallback、disabled 三条主路径。
- 验证 Stage 2 结果仍写回 `segments.json`，但 runtime evidence 转由 `sample_runtime.json` 承载。
- 增加 app-level config matrix，至少覆盖：
  - Stage 2 开启、export 关闭时，Stage 2 仍被调用并写回正式结果产物
  - Stage 2 关闭、export 开启时，export 只消费现有 Stage 2 结果或 source instruction fallback，不再拥有 subtitle-generation 语义
  - subtitle/localization 写回由 Stage 2 config 控制，而不是 `export.enabled`
  - export 读取 Stage 2 输出，而不是在导出阶段再次决定是否生成 Stage 2 字幕

**Parallelism:** 必须串行，单 owner 负责。

- [ ] 让 `app.py` 直接消费 `run_llm_stage2_pass(...)` 的 envelope，而不是分别拼 `run_llm_postprocess_pass(...)` 与 `run_export_subtitle_localization_pass(...)`。
- [ ] 明确 Stage 2 是官方 artifact layer：其 invocation、writeback、fallback 和 diagnostics 由 Stage 2 config 控制，不由 `export.enabled` 控制。
- [ ] 要求 Stage 2 在 export 关闭时仍然写出正式结果产物；export 只是消费 Stage 2 输出，不再拥有 subtitle-generation 语义。
- [ ] 保留 module-side legacy facade 作为兼容层，但不再继续增加 app 依赖。
- [ ] 把 Stage 2 diagnostics 与 export diagnostics 统一接入 `sample_runtime.json`。
- [ ] 在 README 和测试中把 `run_llm_stage2_pass(...)` 明确为 canonical app contract。

### Task P1-2: Stop Treating `segments.json` As A Runtime Dump

**Goal:** 完成从兼容期双写到正式 contract 的切换，让 `segments.json` 只承载 segmentation + Stage 2 text results。

**Files:**

- Modify: `src/video2tasks/server/app.py`
- Modify: `src/video2tasks/server/sample_store.py`
- Modify: `tests/server/test_llm_summary.py`
- Modify: `tests/server/test_sample_store.py`
- Modify: `README.md`
- Modify: `README_CN.md`

**Key Risks:**

- 现有下游或测试可能还在读取 `segments.json.diagnostics`。
- 过快删除旧字段会让排障入口突然消失。

**Test / Verification:**

- 针对 `segments.json` 增加 contract test，确认没有运行态噪声字段。
- 验证所有运行态信息仍可在 `sample_runtime.json` / `failure.json` / `run_summary.json` 找到。

**Parallelism:** 与 P1-1 同 owner；这是同一条 contract 收口线。

- [ ] 在 Gate 1 对外冻结 `segments.json.diagnostics` 的 deprecation policy，明确兼容窗口、迁移目标和移除条件。
- [ ] 在兼容期结束后移除 `segments.json` 中的 runtime diagnostics 写入。
- [ ] 将 README 的“result-layer truth”说法与代码真实行为对齐。
- [ ] 如果需要兼容过渡，为旧字段保留一个明确 deprecation 窗口，而不是长期双写。

### Task P1-3: Convert High-Value Tests From Structure-Oriented To Behavior-Oriented

**Goal:** 把最容易随重构漂移、却对用户价值不高的结构测试，逐步换成围绕 artifact contract、resume、安全失败闭环和 Stage 2 行为的测试。

**Files:**

- Modify: `tests/server/test_app_retry.py`
- Modify: `tests/server/test_llm_summary.py`
- Modify: `tests/server/test_run_manifest.py`
- Modify: `tests/server/test_job_builder.py`
- Modify: `tests/server/test_sample_store.py`
- Modify: `tests/eval/test_official_boundaries.py`
- Create: `tests/integration/test_pipeline_contracts.py`

**Key Risks:**

- 误删能抓住真实 bug 的结构测试。
- 一次性重写太多测试，导致 CI 噪声过大。

**Test / Verification:**

- 按文件分批迁移，每一批都先新增行为测试，再删除多余结构断言。
- 保证 P0/P1 定义的 artifact contract 全部有测试覆盖。

**Parallelism:** 可以并行拆成多个 test lane，但前提是 Gate 1 和 P1-1 contract 已稳定。

- [ ] 先列出“必须保留的行为断言”清单：resume identity、`.DONE`/`.FAILED`、Stage 2 writeback、failure closure、smoke artifact contract、boundary eval correctness。
- [ ] 对 `windowing`、`llm_merge`、app finalize 分别新增行为测试，再逐步删除只断言内部 helper 形状的测试。
- [ ] 保留少量 characterization tests 作为迁移保护网，不追求彻底“纯黑盒”。

## P2: Controlled Refactors And Profile-Led Performance

### Task P2-1: Continue Splitting `app.py` By Runtime Responsibility

**Goal:** 在 contracts 稳定后，把 `create_app()` 收缩回应用装配层，而不是运行时总控层。

**Files:**

- Modify: `src/video2tasks/server/app.py`
- Create: `src/video2tasks/server/producer.py`
- Create: `src/video2tasks/server/routes.py`
- Create: `src/video2tasks/server/runtime_state.py`
- Modify: `tests/server/test_app_retry.py`

**Key Risks:**

- 在 contract 尚未稳定时拆分会放大 merge conflict。
- 把“文件变小”误当成完成标准，导致职责仍然混乱。

**Test / Verification:**

- 现有 app/runtime tests 全通过。
- `create_app()` 只保留装配、state 注入、route 注册、runtime 启停。

**Parallelism:** 以 `app.py` owner 为主，必须串行推进。

- [ ] 先抽 producer loop 和 sample progression。
- [ ] 再抽 route handlers 和 health surface。
- [ ] 保留一个稳定 facade，避免一次把 import 路径打散。

### Task P2-2: Split `llm_merge.py` And `windowing.py` On Stable Facades

**Goal:** 继续按责任拆分大文件，但以 stable facade 为边界，而不是为了缩行数。

**Files:**

- Modify: `src/video2tasks/server/llm_merge.py`
- Create: `src/video2tasks/server/stage2_merge.py`
- Create: `src/video2tasks/server/stage2_summary.py`
- Create: `src/video2tasks/server/stage2_subtitles.py`
- Modify: `src/video2tasks/server/windowing.py`
- Create: `src/video2tasks/server/window_media.py`
- Create: `src/video2tasks/server/segmentation.py`
- Modify: corresponding tests under `tests/server/`

**Key Risks:**

- facade 未定就拆，最后只是把耦合搬家。
- 多子代理同时改 `llm_merge.py` 或 `windowing.py` 导致冲突。

**Test / Verification:**

- facade 级行为测试全部保留不变。
- 新模块拆出后，`app.py` 不需要理解内部 helper 细节。

**Parallelism:** 通过 Gate 2 后可并行。`windowing` lane 与 `llm_merge` lane 可以分别给不同子代理。

- [ ] 先定义 facade 仍然是谁：例如 app 只依赖 `run_llm_stage2_pass(...)` 和 window/build/finalize 入口。
- [ ] Stage 2 内部拆 `merge` / `summary` / `subtitles`。
- [ ] Stage 1 内部拆媒体处理、窗口计划、segment 组装。

### Task P2-3: Profile-Led Performance Work

**Goal:** 在 observability 足够的前提下，针对真实热点做优化，而不是凭感觉重写。

**Files:**

- Modify: `src/video2tasks/server/app.py`
- Modify: `src/video2tasks/server/windowing.py`
- Modify: `src/video2tasks/server/llm_merge.py`
- Modify: any hotspot-specific module identified by profile
- Create: `docs/runbooks/performance-baseline.md`

**Key Risks:**

- 把端点抖动误判成代码热点。
- 在没有 run summary / sample runtime / events 的情况下做错误优化。

**Test / Verification:**

- 先留 profile 证据和 baseline 文档。
- 再做单热点优化，并对比 smoke run 与 representative real run 的前后差异。

**Parallelism:** 在热点明确后，可按热点分 lane；在此之前不要提前并行“优化”。

- [ ] 先用 `run_summary.json` 和结构化事件锁定 CPU、I/O、外部 API 三类耗时来源。
- [ ] 每次只优化一个热点，并留下前后证据。
- [ ] 不改变 contract 与 artifact shape，除非先单独立项。

## Suggested Subagent Bundles

最适合并行派发的任务组：

### Bundle A: Smoke Demo Surface

- `config.smoke.yaml`
- `tests/fixtures/smoke_dataset/...`
- `tests/integration/test_official_smoke_demo.py`
- `README.md`
- `README_CN.md`
- `docs/runbooks/official-smoke-demo.md`

前提：通过 Gate 0，artifact 名称已冻结。

### Bundle B: Observability Contract

- `docs/observability/event-schema.md`
- `src/video2tasks/logging_utils.py`
- `src/video2tasks/server/run_summary.py`
- `src/video2tasks/server/sample_store.py`
- `tests/test_logging.py`
- `tests/server/test_run_summary.py`

前提：`app.py` 部分仍需单 owner 合并；`src/video2tasks/server/sample_store.py` 继续保持单 owner；其他文件可以先由子代理并行准备。

### Bundle C: Behavior-Test Migration

- `tests/server/test_app_retry.py`
- `tests/server/test_llm_summary.py`
- `tests/server/test_run_manifest.py`
- `tests/eval/test_official_boundaries.py`
- `tests/integration/test_pipeline_contracts.py`

前提：Gate 1 和 P1-1 contract 已稳定。

## Not Doing List

为防止范围膨胀，本计划明确不做以下事项：

- 不把多机分布式 worker、跨机 transport、异构挂载路径支持设为短期目标。
- 不引入 Prometheus、OpenTelemetry、tracing 平台等新 observability 基础设施。
- 不重写配置系统；当前只做官方配置层次文档和测试锁定。
- 不为了“抽象统一”而引入大而全 backend 抽象层；Stage 2 只需要窄接口。
- 不在 P0/P1 阶段做大规模 `app.py` / `windowing.py` / `llm_merge.py` 搬家式重构。
- 不在没有 profile 之前做性能优化。
- 不让 README、runbook、config 模板、测试各说各话。

## Final Priority Order

按当前真实约束，推荐优先级是：

1. P0-1 官方 smoke demo
2. P0-2 结构化事件 schema 冻结
3. P0-3 runtime closure 与 `.DONE` / `.FAILED` 语义收口
4. P0-4 `sample_runtime.json` + `run_summary.json`
5. P0-5 operator 文档与排障文档
6. P1-1 Stage 2 canonical app integration
7. P1-2 `segments.json` 去运行态化
8. P1-3 行为测试迁移
9. P2-1 / P2-2 继续拆大文件
10. P2-3 profile-led performance

这个顺序的核心原则只有一句话：

先让系统更容易跑对、看懂、续跑和排障，再去让它更“漂亮”。
