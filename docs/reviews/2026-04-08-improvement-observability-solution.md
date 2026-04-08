# 2026-04-08 Improvement: Observability, Artifact Provenance, and Test Robustness

## 结论

这三个建议都值得做，但不应该一起升级成“大重构”。推荐的取舍是：

- `artifact_manifest_path` / provenance 值得现在就改，但只做窄改，不做协议大改。
  当前 Step A repeat reuse 的核心收益已经成立，问题不在“能不能复用”，而在“复用后 provenance 语义不够干净”。应把“谁生产了 artifact”与“谁消费了 artifact”拆开，而不是继续让复用任务共享一个语义上像“当前任务自产”的 manifest。
- “跨 producer batch reuse” 测试需要补强，而且要从“依赖 `window_repeat_count=25` 恰好跨过 `cnt > 20`”改成“显式制造两批或多批生产”的测试缝。
- 可观测性应先上最小但有用的一层：结构化事件日志 + 关键阶段耗时 + 明确的 empty/fallback 状态字段。现在不需要上 Prometheus 或 tracing 系统，但必须让日志能回答“慢在哪里、空返发生在哪一层、fallback 是哪一层触发的”。

推荐优先级：

1. 窄改 artifact provenance 语义。
2. 把跨 batch reuse 测试改成显式批次测试。
3. 给 server/worker/finalize 增加最小结构化观测点。
4. 再决定是否需要更重的协议或指标系统。

## 当前问题定位

### 1. repeat artifact reuse 已经有效，但 provenance 语义仍然偏“首个 repeat 绑定”

当前 Step A reuse 的事实是清楚的：

- `src/video2tasks/server/app.py` 的 `_build_job_payload(...)` 在首次产出 contact-sheet artifact 时，把 `artifact_metadata={**meta, "task_id": task_id}` 传给 `TaskArtifactWriter`。
- `TaskArtifactWriter` 会把这份 metadata 原样写进 `manifest.json`。
- 后续 repeat 命中 sample 级内存 cache 时，server 只是 clone 同一个 `SharedFSImageTransport`，也就是继续复用同一个 `artifact_manifest_path`。

这带来一个语义问题：

- 对首个 repeat 来说，manifest 里的 `metadata.task_id` 同时扮演“artifact producer task id”和“artifact consumer task id”。
- 对后续复用 repeat 来说，`artifact_manifest_path` 指向的 manifest 仍然带着首个 repeat 的 `task_id`。

这不一定是 correctness bug，但它会让排障时出现两个歧义：

- 看到某个任务携带 `artifact_manifest_path=A`，无法直接判断这是“该任务自产 artifact”还是“复用了别的 repeat 的 artifact”。
- 看到 manifest 里的 `task_id`，无法区分它表示“producer”还是“当前 consumer”。

换句话说，当前实现的 provenance 更像“producer-first provenance”，但字段命名还在暗示“task-scoped provenance”。

### 2. 现有跨 batch reuse 测试能测到行为，但过度依赖隐式常量

`tests/server/test_app_retry.py` 已经有 4 类重要护栏：

- 同一 logical window 的 repeat 复用。
- 不同 logical window 不复用。
- 跨 producer batch 仍复用。
- refinement pass 不复用 Step A 的 cache。

问题主要集中在“跨 producer batch”这一条：

- 当前测试通过 `window_repeat_count=25` 来跨过 `producer_loop` 内部的 `cnt > 20` 分批门槛。
- 测试并不知道“20”到底是产品约束、实现细节，还是临时批次预算。
- 如果以后把 20 改成 16、32、配置项或 helper 返回值，这条测试会因为错误原因失效，或者更糟糕的是继续通过但不再真正覆盖“跨批次”。

这说明当前测试验证的是“某个隐式常量驱动下的现状”，而不是“cache 在不同 producer 周期之间仍可生效”这个真正要锁定的契约。

### 3. 当前日志不足以定位慢点、空返和 fallback

`src/video2tasks/logging_utils.py` 目前只提供 plain message logger。`app.py` 和 `worker/runner.py` 里的日志大多是人类可读字符串，缺少稳定字段。

当前缺口主要有三类：

- 慢点定位不足：
  没有 `extract_ms`、`infer_ms`、`submit_ms`、`finalize_ms`、`export_ms` 这类阶段耗时。
- 空返定位不足：
  能看到 empty retry，但很难区分是“artifact 加载失败导致空返”“backend infer 空返”“normalize 后被判空”“server retry budget 耗尽”。
- fallback 定位不足：
  Stage 2 / export 的 fallback 主要存在于 `diagnostics` 或局部字符串里，不能在日志侧被稳定筛选。

`/health` 目前也只返回静态 `{"status": "ok"}`，对 live debugging 基本没有帮助。

## 推荐方案

### 1. 对 artifact provenance 做窄改，现在就值得做

推荐判断：`值得现在做，但只做 producer/consumer 语义澄清，不做 transport 大重构。`

原因：

- 当前问题已经影响排障质量。
- 这类改动不需要破坏 reuse 本身，也不需要复制 artifact。
- 如果继续拖，后续日志、验收测试和 run/export manifest 都会建立在模糊语义上。

#### 推荐做法

保留现有 `SharedFSImageTransport.artifact_manifest_path`，但明确它代表：

- `artifact producer manifest path`
- 而不是“当前 consumer 任务专属 manifest path”

然后把 provenance 拆成两层。

第一层：producer provenance，写入 artifact manifest

- `producer_task_id`
- `producer_job_type`
- `producer_window_pass`
- `producer_repeat_index`
- `artifact_reuse_scope`
- `artifact_reuse_group`

第二层：consumer provenance，写入 job meta / 日志

- `task_id`
- `dispatch_id`
- `artifact_manifest_path`
- `artifact_producer_task_id`
- `artifact_reuse`，取值建议为 `seed` / `hit` / `none`
- `artifact_reuse_group`

其中：

- `artifact_reuse_group` 应是稳定的逻辑组标识，建议来自 `_repeat_artifact_reuse_key(...)` 的哈希或可序列化摘要，而不是直接复用某个 task id。
- 首次生成 artifact 的任务记为 `artifact_reuse=seed`。
- 后续命中 cache 的 repeat 记为 `artifact_reuse=hit`。
- 未走 repeat artifact reuse 的任务记为 `artifact_reuse=none`。

#### 当前不建议做的事

- 不建议现在把 `SharedFSImageTransport` 改成一整套嵌套 `artifact_provenance` 对象。
- 不建议给每个 consumer repeat 复制一份 manifest。
- 不建议为了“provenance 看起来更独立”而放弃复用共享 manifest path。

这些做法要么协议 churn 太大，要么直接损伤当前优化收益。

#### 推荐取舍

最合理的取舍是：

- 保留“共享 artifact，manifest 只生成一次”的模型。
- 通过 producer/consumer 字段拆分，把 provenance 语义补干净。

这是一笔低风险、高收益的收敛，不是大改。

### 2. 把“跨 producer batch”测试改成显式批次测试

推荐目标不是“继续覆盖 20 这个数字”，而是固定下面这条契约：

> 当同一个 sample 的同一个 logical window 在不同 producer 周期内继续入队 repeat job 时，已经生成的 Step A artifact 仍可被后续周期复用。

#### 推荐测试策略

首选方案：给 Step A 入队逻辑一个显式批次缝

- 把 Step A 的“单次 producer 周期最多 enqueue 多少个 job”提成显式 helper 参数或内部常量封装。
- 测试通过这个显式缝制造两批或三批生产，而不是靠 `25 > 20` 间接跨批次。

具体建议二选一：

方案 A，优先推荐：

- 提取一个小 helper，例如“Step A enqueue planner/executor”。
- helper 显式接受 `enqueue_budget`。
- 生产逻辑默认传现有预算值。
- 测试传 `enqueue_budget=1` 或 `2`，用 3 个 repeat 就能稳定制造多批次。

方案 B，次优：

- 把当前 20 提成命名明确的内部常量或配置，如 `producer_enqueue_budget_per_cycle`。
- 测试里显式设成 `1`。

#### 推荐验收测试矩阵

第一组，正向契约：

- `window_repeat_count=3`
- `enqueue_budget=1`
- 第 1 个 producer 周期生成 `r0`，触发一次 artifact extraction
- 第 2 个周期生成 `r1`，不再 extraction
- 第 3 个周期生成 `r2`，不再 extraction
- 断言所有 repeat 的 `artifact_manifest_path` 相同
- 断言 `artifact_producer_task_id` 固定为首个 producer repeat，而当前 `task_id` 各自独立

第二组，负向边界：

- 不同 logical window，即使 `frame_ids` 相同，也不复用
- 不同 `window_pass` 不复用
- sample 进入 `.DONE` / `.FAILED` 或被 `_fail_sample(...)` 清理后，cache 不泄漏到后续样本处理
- refinement / boundary refinement / segment label 不复用 Step A cache

第三组，语义契约：

- 对 `seed` repeat，manifest 内 producer 字段与当前任务一致
- 对 `hit` repeat，manifest 仍指向 producer，但 job meta / 日志里明确标出 consumer 自身 `task_id`

#### 为什么这种方案更稳

因为它验证的是“跨周期复用”这个业务契约，而不是“碰巧超过某个内部循环阈值”这个实现细节。

### 3. 上最小但有用的可观测性方案

推荐目标：

- 不引入新的基础设施依赖
- 只靠现有 logger 就能让慢点、空返、fallback 可检索、可聚合、可对账

最小方案包含三部分。

#### 3.1 结构化事件日志

建议在 `logging_utils.py` 上增加一个极轻量 helper，统一输出稳定字段，例如：

- `event`
- `subset`
- `sample_id`
- `task_id`
- `dispatch_id`
- `job_type`
- `window_pass`
- `repeat_index`
- `window_id`
- `boundary_id`
- `segment_id`
- `artifact_transport`
- `artifact_manifest_path`
- `artifact_producer_task_id`
- `artifact_reuse`
- `artifact_reuse_group`
- `attempt`
- `terminal_error`
- `fallback_stage`
- `fallback_reason`

日志格式不要求 JSON-only，但必须保证字段名稳定、便于 grep 和后续日志平台提取。

#### 3.2 关键阶段耗时

建议优先打下面这些时延：

server 侧：

- `artifact_extract_ms`
- `job_queue_wait_ms`
- `finalize_ms`
- `boundary_refinement_finalize_ms`
- `segment_label_finalize_ms`
- `postprocess_ms`
- `subtitle_localize_ms`
- `export_ms`

worker 侧：

- `image_load_ms`
- `infer_ms`
- `submit_ms`
- `local_retry_sleep_ms`

其中最关键的是前三个：

- `artifact_extract_ms`
- `infer_ms`
- `finalize_ms`

这三项先有了，慢点的大头就基本能定位。

#### 3.3 显式 empty / fallback 事件

现在最难查的不是“任务失败”，而是“任务为什么慢失败、空返了几轮、最后是哪个 fallback 生效”。

建议固定以下事件：

server 侧：

- `result_empty_retry`
- `result_empty_terminal`
- `result_timeout_retry`
- `result_timeout_terminal`
- `sample_failed`

worker 侧：

- `job_start`
- `infer_attempt`
- `infer_empty_local_retry`
- `job_done`
- `submit_result`
- `submit_result_failed`

Stage 2 / export 侧：

- `fallback_applied`
- `export_done`
- `export_failed`

`fallback_applied` 至少要带：

- `fallback_stage`
- `fallback_reason`
- `sample_id`
- `run_id`

这样才能快速回答“这个样本为什么没有看到预期 Stage 2 产物，是 skip、fallback 还是 hard failure”。

### 4. `/health` 做一层最小动态增强

不建议把 `/health` 做成全量监控接口，但它至少应返回当前正在发生的几个 live 指标：

- `queue_depth`
- `inflight_count`
- `completed_dispatch_count`
- `empty_retry_task_count`
- `timeout_retry_task_count`
- `artifact_reuse_cache_sample_count`
- `artifact_reuse_cache_entry_count`

如果成本可控，再补：

- `current_subset`
- `current_sample_id`
- `global_done`
- `failed_sample_count`

这对现场判断“系统是卡在 queue、卡在 worker、还是卡在 finalize”已经足够有用。

## 第一批落地范围

这一批以低风险、高收益为原则，建议只做下面 5 项。

### 1. 澄清 artifact provenance 语义

范围：

- 给 artifact manifest 增加 producer 语义字段
- 给 job meta / 日志增加 consumer 语义字段
- 统一 `artifact_reuse` / `artifact_reuse_group` 命名

不做：

- 不重写 transport 协议
- 不复制 artifact
- 不引入新的持久化层

收益：

- 直接提升排障可读性
- 给后续日志和测试奠定稳定字段

### 2. 把 Step A 单次批次预算提成显式测试缝

范围：

- 提 helper 或内部命名常量
- 让测试能显式构造两批以上 producer 周期

收益：

- 让跨 batch reuse 验收测试不再绑定隐式常量
- 降低未来重构 producer_loop 时的误报率

### 3. 重写跨 batch reuse 验收测试

范围：

- 显式多批次
- 覆盖 producer/consumer provenance 断言
- 增加 cache 清理后的负向测试

收益：

- 真正锁定“跨 producer 周期仍复用”的行为
- 顺手把 provenance 语义一起固化

### 4. 增加结构化日志 helper

范围：

- `logging_utils.py` 增加统一事件输出 helper
- 不改日志基础设施，只改字段表达

收益：

- 后续所有 server/worker/finalize 观测点都能共用同一套格式

### 5. 给 server/worker/finalize 打第一批事件和耗时

最先值得打点的事件：

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

最先值得记录的耗时：

- `artifact_extract_ms`
- `infer_ms`
- `submit_ms`
- `finalize_ms`

收益：

- 基本能覆盖“慢点/空返/fallback”三类主要排障路径

## 风险与取舍

### 1. 为什么不建议现在就做“大一统 provenance 对象”

如果现在把 `SharedFSImageTransport`、artifact manifest、result envelope、run/export manifest 一次性全部改成统一 provenance 模型，收益当然更完整，但风险明显更高：

- server / worker / tests / docs 同时要改
- 容易把“观测性增强”升级成“协议重构”
- 这会拖慢真正高收益的第一批落地

因此当前更合理的是先把语义补清楚，而不是把所有协议一步到位。

### 2. 为什么测试必须去掉对隐式批次常量的依赖

继续依赖 `25 > 20` 的测试不是完全没价值，但它把契约和实现细节绑死了。长期看，这类测试会出现两种坏结果：

- 改了批次预算，测试误报
- 没改契约，但重构了 producer 结构，测试失去覆盖价值

所以这里最值得补的是“显式批次缝”，不是继续堆更大的 repeat_count。

### 3. 为什么最小可观测性先选结构化日志，而不是 metrics/tracing

当前问题还没有到必须上完整 telemetry 基础设施的程度。更现实的约束是：

- 先让每个关键路径都有稳定字段
- 先让慢点和空返能被 grep 到
- 先让 fallback 能被机器识别，而不是埋在自然语言字符串里

结构化日志是当前风险最低、见效最快的方案。

### 4. 日志量会上升，但这是可接受的

新增事件和耗时后，日志量一定会上升。建议的控制方式不是少打点，而是分层：

- `INFO` 保留关键生命周期事件
- `DEBUG` 再承载更细粒度的 attempt 明细

这样既能维持默认可读性，又不阻碍深度排障。

## 验证计划

### 1. provenance 验证

新增测试应明确验证：

- 首个 repeat 生成的 manifest 带有 producer 字段
- 后续 repeat 命中 reuse 时，共享同一个 `artifact_manifest_path`
- 但 consumer 自己的 `task_id`、`dispatch_id` 仍然独立
- `artifact_reuse=seed` 与 `artifact_reuse=hit` 的语义可区分

### 2. 跨 batch 验收验证

新增测试应显式制造至少两个 producer 周期，并验证：

- 第一个周期只触发一次 extraction
- 后续周期不重复 extraction
- cache 能跨周期保留
- cache 在 sample 终态或失败清理后不会泄漏

### 3. 可观测性验证

建议至少补一轮日志契约测试或集成测试，验证以下事件会出现且字段齐全：

- `artifact_extract_done`
- `artifact_reuse_hit`
- `result_empty_retry`
- `result_timeout_retry`
- `fallback_applied`
- `finalize_done`

不要求完整比对整条日志字符串，但要锁定关键字段名和字段值。

### 4. 人工验证场景

建议跑 3 个最小人工场景，确认日志和行为对齐：

1. Step A repeat reuse 场景  
   预期：只出现一次 `artifact_extract_done`，随后出现多个 `artifact_reuse_hit`

2. 空返重试场景  
   预期：worker 侧出现 `infer_empty_local_retry`，server 侧出现 `result_empty_retry`，最终若耗尽预算则出现 `result_empty_terminal`

3. export / Stage 2 fallback 场景  
   预期：日志中出现 `fallback_applied`，并能明确看出是哪个阶段触发、因为什么触发

## 最终建议

这次不需要把 artifact reuse、测试框架和 observability 一口气升级成“大系统治理项目”。更优先、也更稳的路线是：

- 先把 artifact 的 producer/consumer provenance 补清楚
- 再把跨 producer batch 的测试从隐式常量依赖改成显式批次契约
- 同时用最小结构化日志把慢点、空返、fallback 变成可检索事实

如果第一批只做这三件事，风险低，收益已经足够明显，而且不会和现有业务逻辑发生高耦合冲突。
