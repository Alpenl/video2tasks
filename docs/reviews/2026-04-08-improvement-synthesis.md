# 2026-04-08 改进方案收敛文档

## 执行摘要

三份并行方案的共识非常明确：当前最值得做的，不是继续容忍大文件，也不是发起一次性的大重构，而是先建立几个稳定、可验证、低风险的中间边界，同时补齐最小但真正有用的可观测性与回归护栏。

本次收敛后的推荐结论如下：

- 第一批改进应聚焦“边界收紧 + provenance 澄清 + 测试缝 + 最小结构化观测”，不改主业务算法，不改 HTTP 协议，不改现有产物主格式。
- 第一批的目标不是把 `app.py`、`windowing.py`、`llm_merge.py` 一次拆干净，而是先建立后续拆分的前置边界，让后续工作可以安全并行。
- Stage 2 的正式契约化方向是正确的，但不应挤进第一批核心范围。第一批最多只为它清理依赖前提，不做主调用链切换、不做配置大迁移。
- 可观测性现在就值得做，但只做最小闭环：稳定字段、关键事件、关键耗时、显式 empty/fallback 状态，不上新的 telemetry 基础设施。

推荐的第一批落地范围控制在 5 项：

1. 抽出 `segment_semantics.py`，正式承载 Stage 1 / Stage 2 共享的 segment 语义规则。
2. 抽出 `sample_store.py`，把样本路径与结果持久化从 `app.py` 中剥离。
3. 抽出 `job_builder.py`，把四类 job 的 payload 与 metadata 组装集中起来。
4. 澄清 repeat artifact reuse 的 producer / consumer provenance，并重写跨 batch reuse 测试。
5. 增加结构化日志 helper 与第一批关键事件/耗时打点。

这 5 项都满足同一标准：低风险、高收益、可验证，而且直接为第二批的 `routes.py` / `producer.py` / Stage 2 契约化改造铺路。

## 三路方案收敛结论

### 1. 先立稳定边界，再做更深层拆分

架构方案和 Stage 2 方案都指出了同一个问题：当前最危险的不是文件大，而是边界不正式。

- `app.py` 是装配、持久化、job 构建、路由和 producer 状态机的混合入口。
- `windowing.py` 与 `llm_merge.py` 共享一套没有正式建模的 segment 语义。
- `llm_merge.py` 内部已经出现更好的 Stage 2 包络方向，但应用层和配置层还停留在旧形状。

因此，本轮收敛后的原则是：

- 先把共享语义和编排依赖收口成公共边界。
- 再逐步把大文件内部职责迁出。
- 不把“模块搬家”和“行为变化”混成同一批提交。

### 2. 第一优先级应是“降低耦合并提升可诊断性”，而不是“追求最终形态”

三份方案都反对大爆炸重构，原因一致：

- 一次同时改 app 编排、Stage 2 契约、producer 行为、artifact 协议和日志体系，回归面过大。
- 当前已有的 runtime、protocol、Stage 2 包络、repeat reuse 优化都不是方向错误，问题主要在边界和语义不够干净。
- 很多问题本质上是“没有正式接口”和“没有稳定字段”，不是“算法必须重写”。

因此，第一批应优先做：

- 稳定公共模块边界。
- 稳定 provenance 字段语义。
- 稳定测试缝和 characterization tests。
- 稳定日志字段与关键事件。

### 3. Stage 2 应后置到“边界已稳”之后，而不是抢在第一批核心范围里

Stage 2 方案判断是成立的：

- Stage 2 不应继续把 `OpenAIBackend` 当成事实契约。
- `run_llm_stage2_pass()` 比旧接口更接近正式输出包络。
- `provider`、`summary.enabled_levels`、`subtitles.target_language` 这类归一化方向值得做。

但收敛后的优先级判断是：

- Stage 2 正式契约化应作为第二批主线，而不是第一批核心范围。
- 第一批不切 Stage 2 主集成路径，不扶正完整新包络，不推进配置迁移。
- 第一批最多只做为后续 Stage 2 收敛铺路的依赖清理，不做大范围接口切换。

原因很简单：Stage 2 的契约化虽然正确，但它比 `segment_semantics.py`、`sample_store.py`、`job_builder.py`、provenance 和测试缝更容易牵动 finalize、export、配置兼容和下游 JSON 消费方。

### 4. 可观测性现在就值得做，但只做最小闭环

可观测性方案里的判断应直接采纳：

- 现在不需要上 Prometheus、Tracing 或新的监控系统。
- 但必须让日志回答三类问题：慢在哪里、空返发生在哪里、fallback 是哪一层触发的。
- artifact provenance 语义必须先澄清，否则后续日志和测试都建立在模糊字段上。

本轮收敛后的可观测性原则是：

- 先统一字段，不先扩基础设施。
- 先打关键路径，不追求全覆盖。
- 先覆盖 repeat reuse、worker infer、result retry、finalize、fallback 这几条主排障路径。

## 推荐第一批改进

### 第一批的范围标准

第一批必须同时满足以下四个条件：

- 不改变主业务行为，只建立边界、语义和护栏。
- 对现有测试和对外契约的冲击可控。
- 每一项都能通过现有测试或新增 characterization tests 独立验证。
- 每一项都能直接降低后续拆分成本或排障成本。

### 第一批改进列表

#### 1. 新增 `segment_semantics.py`

目标：

- 把当前由 `windowing.py` 私有持有、但已被 `llm_merge.py` 依赖的 segment 语义规则正式抽成公共模块。

首批范围：

- 只迁出当前跨 Stage 1 / Stage 2 共享的最小公共语义函数及其必要依赖。
- 保持 `windowing.py` 和 `llm_merge.py` 的业务逻辑不变，只改依赖落点。

收益：

- 消除跨文件 import 私有函数。
- 为后续 `windowing.py` 和 `llm_merge.py` 并行拆分建立前置边界。

#### 2. 新增 `sample_store.py`

目标：

- 把样本输出路径、JSONL 读写、结果加载、`.DONE` / `.FAILED` / `failure.json` 等持久化职责从 `app.py` 中拿出来。

首批范围：

- 只迁移路径解析、结果读写、失败落盘和相关 helper。
- 不改变文件格式，不改变 finalize 业务语义。

收益：

- 让 `app.py` 开始从“系统总装层”收缩回“应用装配层”。
- 为后续 `routes.py` 和 `producer.py` 提取提供稳定持久化依赖。

#### 3. 新增 `job_builder.py`

目标：

- 收口四类 job 的 payload 和 metadata 组装逻辑。

首批范围：

- 迁出 `_build_job_payload()`。
- 收口 contact sheet / inline / shared_fs transport 选择逻辑。
- 收口 window、refinement、boundary refinement、segment label 四类 job 的公共 metadata 拼装。

收益：

- 消除 `app.py` 中重复的 metadata 片段。
- 为后续 producer 拆分和 provenance 打点提供统一出口。

#### 4. 澄清 artifact provenance，并重写跨 batch reuse 测试

目标：

- 把“谁生产了 artifact”和“谁消费了 artifact”从语义上拆开。
- 把“跨 producer 周期仍可复用”从隐式常量驱动测试改成显式契约测试。

首批范围：

- 保留 `artifact_manifest_path` 和共享 artifact 模型。
- 在 artifact manifest 中增加 producer 语义字段。
- 在 job meta / 日志中增加 consumer 语义字段。
- 统一 `artifact_reuse` 与 `artifact_reuse_group` 命名。
- 为 Step A 单次批次预算建立显式测试缝或命名常量。
- 重写跨 batch reuse 测试，不再依赖 `25 > 20` 这种隐式门槛。

收益：

- 直接提升排障可读性。
- 锁定 repeat reuse 的真实业务契约，降低未来重构误报率。

#### 5. 增加结构化日志 helper 与第一批关键打点

目标：

- 让关键路径具备稳定字段和可检索的事件名。

首批范围：

- 在 `logging_utils.py` 增加统一事件输出 helper。
- 优先落地以下事件：
  - `artifact_extract_done`
  - `artifact_reuse_hit`
  - `job_dispatched`
  - `infer_attempt`
  - `job_done`
  - `result_empty_retry`
  - `result_timeout_retry`
  - `fallback_applied`
  - `sample_failed`
  - `finalize_done`
- 优先落地以下耗时：
  - `artifact_extract_ms`
  - `infer_ms`
  - `submit_ms`
  - `finalize_ms`

收益：

- 基本覆盖“慢点 / 空返 / fallback / repeat reuse”四条主要排障路径。
- 为第二批 `/health` 动态增强与更细粒度观测提供稳定字段基础。

### 推荐实施顺序

推荐按以下顺序落地：

1. `segment_semantics.py`
2. `sample_store.py`
3. `job_builder.py`
4. artifact provenance 澄清
5. Step A 显式批次测试缝与跨 batch reuse 测试重写
6. 结构化日志 helper
7. 第一批事件与耗时打点

顺序理由：

- `segment_semantics.py` 是 `windowing.py` / `llm_merge.py` 后续拆分的前提。
- `sample_store.py` 和 `job_builder.py` 是 `app.py` 缩边界的两个最低风险切口。
- provenance 字段应建立在相对稳定的 job meta 与持久化落点之上。
- 日志字段和事件名应在 provenance 语义确定后再统一，否则容易重复 churn。

## 后续批次建议

### 第二批

第二批建议聚焦“正式拆编排层”，包括：

- 提取 `routes.py`
- 提取 `producer.py`
- 收敛 `app.py` 到“应用装配 + 运行入口 + 兼容 facade”
- 视第一批稳定情况，最小化增强 `/health` 的动态状态字段

进入第二批的前提是：

- `sample_store.py` 与 `job_builder.py` 已稳定。
- repeat reuse 的 provenance 和测试已固定。
- 第一批日志字段已落地，能辅助排障。

### 第三批

第三批建议聚焦“Stage 2 正式契约化”，包括：

- 引入 `Stage2TextBackend` 或等价窄接口
- 集中 Stage 2 backend 创建
- 扶正 `run_llm_stage2_pass()` 为 canonical API
- 保留 `run_llm_postprocess_pass()` 作为兼容 wrapper
- 增加 `provider`、`summary.enabled_levels`、`subtitles.target_language` 等兼容别名和归一化读法
- 明确 `subtitles.items` 是 canonical 结果，`segments[].export_subtitle` 只是兼容镜像

这部分建议后置的原因不是“不重要”，而是它更依赖稳定的 app / finalize 边界和更强的兼容验证。

### 第四批

第四批才考虑更深层的模块整理和可观测性扩展，包括：

- `window_media.py` / `window_plan.py` / `segmentation.py` 等 Stage 1 内部再拆分
- `stage2_merge.py` / `stage2_summary.py` / `stage2_subtitles.py` 等 Stage 2 内部再拆分
- `/health` 更丰富的 live 指标
- 是否需要 metrics / tracing 的再评估

## 并行/串行关系

### 必须串行的部分

1. `segment_semantics.py` 必须先于 `windowing.py` 和 `llm_merge.py` 的进一步拆分。
2. `sample_store.py` 必须先于 `producer.py` 提取。
3. `job_builder.py` 必须先于更深的 app 路由与 dispatch 拆分。
4. provenance 字段语义必须先于结构化日志字段定稿。
5. 显式批次测试缝必须先于跨 batch reuse 测试重写。

### 可以并行的部分

1. `sample_store.py` 与 `segment_semantics.py` 可以由不同 owner 并行推进，但合并顺序仍应先确认 `segment_semantics.py` 的公共 API。
2. `job_builder.py` 可以在 `sample_store.py` 落地后并行推进，不需要等待 `routes.py` / `producer.py`。
3. provenance 字段设计稳定后，测试重写和日志 helper 可以并行。
4. 日志 helper 落地后，server 侧和 worker 侧事件打点可以并行。

### 推荐 ownership

建议按三条工作流拆 ownership：

- 架构边界线：
  - `segment_semantics.py`
  - `sample_store.py`
  - `job_builder.py`
  - 目标是建立稳定依赖边界，不改业务行为

- 质量与验证线：
  - Step A 显式批次测试缝
  - 跨 batch reuse 测试重写
  - `sample_store` / `job_builder` characterization tests
  - 目标是把第一批改动的回归风险锁死

- 可观测性线：
  - provenance 字段接线
  - 结构化日志 helper
  - 第一批关键事件与耗时打点
  - 目标是让新边界和 repeat reuse 语义可诊断

其中需要明确串行交接的点有两个：

- 架构边界线先定字段和 API，质量线与可观测性线再接入。
- provenance 字段命名先定稿，日志与测试断言再落地。

## 验证计划

### 验证原则

- 先做 characterization，再做搬迁。
- 先按边界验证，再跑主路径回归。
- 每个模块拆分都应优先证明“行为没变”，而不是只证明“代码搬完了”。

### 现有测试回归清单

第一批改动完成后，建议至少执行以下测试：

```bash
pytest tests/server/test_runtime.py
pytest tests/server/test_run_manifest.py
pytest tests/server/test_app_retry.py
pytest tests/server/test_windowing.py
pytest tests/server/test_llm_summary.py
pytest tests/test_logging.py
```

验证重点：

- `create_app()` / `run_server()` 的装配与生命周期契约未变。
- `/get_job` / `/submit_result`、retry、timeout、finalize 主路径未变。
- Stage 1 segmentation 语义未漂移。
- Stage 2 现有 facade 与 summary / subtitle 行为未被第一批边界工作误伤。
- 日志基础行为和字段契约未回归。

### 建议新增测试

建议把新增验证集中在 4 类 characterization / contract tests：

1. `sample_store` golden tests
   - 验证 `windows.jsonl`、`boundary_refinements.jsonl`、`failure.json` 等读写往返一致。

2. `job_builder` metadata snapshot tests
   - 分别锁定四类 job 的 `JobEnvelope.meta` 字段集合和关键字段值。

3. provenance contract tests
   - 验证 `artifact_reuse=seed` / `hit` / `none`
   - 验证 producer 字段与 consumer 字段语义分离
   - 验证共享 `artifact_manifest_path` 但 consumer `task_id` / `dispatch_id` 仍独立

4. structured logging contract tests
   - 锁定关键事件名和字段名
   - 不要求比对整条日志字符串，但必须验证核心字段存在且值正确

### 人工验证场景

建议额外做 3 个最小人工验证场景：

1. Step A repeat reuse 场景
   - 预期只有一次 `artifact_extract_done`
   - 后续 repeat 出现多个 `artifact_reuse_hit`

2. 空返重试场景
   - 预期 worker 侧出现 `infer_empty_local_retry`
   - server 侧出现 `result_empty_retry`
   - 预算耗尽时出现 terminal 事件

3. Stage 2 / export fallback 场景
   - 预期出现 `fallback_applied`
   - 可明确定位触发阶段和原因

## 本轮不做事项

为防止范围膨胀，本轮明确不做以下事项：

- 不修改 `producer_loop()` 的核心状态推进语义。
- 不修改 `/get_job`、`/submit_result` 的 HTTP contract。
- 不修改 `windows.jsonl`、`segments.json`、`failure.json` 等现有产物主格式。
- 不修改 `build_segments_via_cuts()`、`run_llm_postprocess_pass()` 等核心业务逻辑。
- 不把 Stage 2 并入 `VLMBackend`，也不在本轮扩展多 provider 支持。
- 不在本轮扶正完整 `run_llm_stage2_pass()` 主调用链，不做 finalize 主路径切换。
- 不做配置大迁移，不强推 `llm_merge` 到 `stage2` 的整块改名。
- 不重写 exporter 契约，不取消 `export_subtitle` 兼容字段。
- 不引入新的异步模型、事件总线、repository framework 或 telemetry 基础设施。
- 不上 Prometheus、Tracing 或全量监控系统。
- 不把 artifact transport、manifest、result envelope、run/export manifest 一次性统一成大一统 provenance 对象。

## 最终推荐

如果本轮只能批准一批“低风险、高收益、可验证”的改进，建议批准以下范围：

1. `segment_semantics.py`
2. `sample_store.py`
3. `job_builder.py`
4. artifact provenance 澄清
5. Step A 显式批次测试缝与跨 batch reuse 测试重写
6. 结构化日志 helper 与第一批关键事件/耗时打点

这组范围的共同特点是：

- 不直接改业务算法。
- 不触碰对外协议和现有主产物格式。
- 能立即降低耦合与排障成本。
- 能为第二批 `routes.py` / `producer.py` 拆分，以及第三批 Stage 2 契约化改造建立稳定前提。
