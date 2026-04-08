# 2026-04-08 架构改进方案：主控层拆分与模块边界收紧

## 结论

结论先说：

1. “继续拆大文件，收紧模块边界”这个建议是有效的，而且在当前架构治理项里优先级合理。  
   但它不是全仓库的绝对第一优先级，不能压过综合审计里已经确认的 correctness 止血项，例如样本失败闭环、run/resume 身份、正式协议与产物契约。换句话说，它应当作为“止血完成后的第一批结构治理”，而不是替代止血。

2. 三个重点文件里，`app.py` 应排架构拆分第一位，`windowing.py` 与 `llm_merge.py` 应按“先切公共语义边界，再拆内部实现”的顺序推进。  
   原因不是单纯行数大，而是：
   - `app.py` 现在是 HTTP、队列状态、文件持久化、producer/finalize 状态机、运行时生命周期的耦合点。
   - `windowing.py` 同时承载媒体 I/O、切窗规划、文本启发式和 segment 装配逻辑。
   - `llm_merge.py` 同时承载 Stage 2 provider 初始化、structured request、merge 共识、summary、subtitle 和 legacy facade。

3. 不建议做一次性“大爆炸重构”。  
   推荐路线是先建立 2 到 3 个真正稳定的中间边界，再把主流程逐步迁出。最小可落地第一批应控制在：
   - 抽出 `app.py` 的持久化与 job 构建边界
   - 抽出 `windowing.py` / `llm_merge.py` 共享的公共 segment 语义模块
   - 暂时不改 producer 主循环算法，不改 HTTP 协议，不改 Stage 1/Stage 2 业务行为

4. 如果只能选一个最先动手的文件，选 `src/video2tasks/server/app.py`；如果只能选一个最先建立的公共模块，选“segment 语义公共模块”。  
   这两处分别解决当前最重的编排耦合点，以及 `llm_merge.py` 直接 import `windowing.py` 私有函数的边界破坏。

## 当前问题定位

### 1. `app.py` 不是“应用工厂”，而是整个 server 运行面的混合入口

当前 `src/video2tasks/server/app.py` 有 1572 行。`create_app()` 从 `app.py:281` 开始，到 `app.py:1555` 才结束。它在一个函数里混合了至少六类职责：

- 数据集与 run manifest 准备：`app.py:286-323`
- 队列、inflight、重试计数、artifact cache 等内存状态：`app.py:291-307`
- 样本输出路径、JSONL 读写、失败落盘：`app.py:335-647`
- job payload 与 image transport 构建：`app.py:649-703`
- FastAPI 路由：`app.py:716-806`
- producer / finalize 状态机与运行循环：`app.py:809-1548`
- runtime 装配与 server 入口：`app.py:1549-1572`

这带来三个直接问题：

- `create_app()` 很难成为稳定的测试装配入口，因为它必须理解过多内部状态。
- 任何新增 job 类型、落盘格式、重试规则、阶段分支，都会继续向这个函数堆 if/else。
- `app.state` 当前兼作 service locator，路由、运行时和测试都在透传内部可变状态，进一步放大了耦合面。

从测试面看，这个问题已经非常具体。`tests/server/test_runtime.py:25-67` 明确把 “`create_app()` 本身不应偷偷运行 producer” 当作契约；`tests/server/test_app_retry.py` 和 `tests/server/test_run_manifest.py` 又直接依赖 `create_app()` 提供 manifest、queue、postprocess 等状态与函数拼装结果。  
这说明 `app.py` 不只是大，而是已经成为系统行为的事实总装层。

### 2. `windowing.py` 把媒体 I/O、规则语义和 Stage 1 装配挤在一起

当前 `src/video2tasks/server/windowing.py` 有 2600 行，至少混合了四层内容：

- 切窗与 frame 采样模型：`windowing.py:21-102`、`windowing.py:396-483`
- 图片编码、contact sheet、artifact 写盘和 `FrameExtractor`：`windowing.py:590-860`
- instruction / boundary 规则启发式与 segment 语义判断：散落在文件中段与后段，尤其是 `windowing.py:1938-2233`
- 最终 Stage 1 segment 装配：`windowing.py:2304-2600`

最关键的问题不是“一个文件太长”，而是 `llm_merge.py` 直接跨文件依赖了 `windowing.py` 的私有语义函数：

- `_boundary_support_between`
- `_has_distinct_sequence_markers`
- `_should_split_on_instruction_drift`

这说明当前真实边界并不是：

- `windowing.py` 负责 Stage 1
- `llm_merge.py` 负责 Stage 2

而是：

- 两边共享一套没有正式建模的 segment 语义规则，只是暂时寄存在 `windowing.py` 里

如果不先把这层公共语义提出来，后面无论怎么拆 `windowing.py` 或 `llm_merge.py`，都会继续出现“拆完还要跨文件 import 私有函数”的假拆分。

### 3. `llm_merge.py` 实际上是 Stage 2 总装层，不只是 merge 模块

当前 `src/video2tasks/server/llm_merge.py` 有 1979 行，内部实际包含：

- merge payload 校验、partition 校验、guard 与 coarse consensus：`llm_merge.py:145-1497`
- summary partition 校验与 hierarchy 构造：`llm_merge.py:987-1635`
- subtitle localization 与 export subtitle 兼容层：`llm_merge.py:1637-1745`、`llm_merge.py:1785-1979`
- Stage 2 总装入口：`llm_merge.py:1748-1782`
- provider 初始化与 structured request 重试：`llm_merge.py:648-730`，以及 `llm_merge.py:1324-1333`、`1567-1576`、`1687-1696`、`1851-1860`

这里的边界问题主要有三类：

- 模块名叫 `llm_merge`，但文件里远不止 merge，导致名字已经不能表达职责。
- OpenAI backend 初始化逻辑重复出现，说明 provider 解析没有统一边界。
- Stage 2 的 merge / summary / subtitle 虽然行为上已经开始分开，但代码边界仍然纠缠在一个文件里。

现状的直接后果是：  
你无法只改 summary 而不碰 merge 边界，也无法只改 subtitle request 而不同时理解 Stage 2 其它路径。

### 4. `runtime.py` 和 `protocol.py` 已经是可保留的“正确方向”

这次审查不建议推倒重来，恰恰相反，`src/video2tasks/server/runtime.py` 和 `src/video2tasks/server/protocol.py` 已经代表了应该继续强化的方向：

- `runtime.py` 已经把显式 start / stop / join 生命周期抽成了稳定对象
- `protocol.py` 已经把 job/result transport 收口成了 typed envelope

后续拆分应围绕这两个模块继续推进，而不是重新造一套新的 lifecycle 或 transport 抽象。

## 推荐方案

### 1. `app.py` 的目标边界

`app.py` 最终应只保留三类职责：

- 应用装配：`create_app(config)` 负责依赖组装与 route 注册
- 运行入口：`run_server(config)` 负责 runtime.start / uvicorn.run / runtime.stop
- 兼容 facade：保留少量向后兼容导出，避免测试和外部调用一次性断裂

不应继续留在 `app.py` 的内容如下。

推荐新增模块：

- `src/video2tasks/server/server_state.py`  
  定义 `DispatchState`、`ProducerState` 或等价 dataclass/class，承载 `job_queue`、`inflight`、retry counts、artifact reuse cache、manifest status 等内存态。  
  目标是把“散落在 create_app 局部变量里的运行时状态”变成一个可注入对象。

- `src/video2tasks/server/sample_store.py`  
  负责样本路径解析、`windows.jsonl` / `boundary_refinements.jsonl` / `segment_labels.jsonl` 加载、结果落盘、`.DONE` / `.FAILED` / `failure.json` 管理。  
  这里应吸收 `app.py:335-647` 这一大段持久化逻辑。

- `src/video2tasks/server/job_builder.py`  
  负责 job metadata 组装和 `JobEnvelope` 构建，包括 window、refinement、boundary refinement、segment label 四类 job。  
  这里应吸收 `app.py:649-703` 以及 `app.py:1016-1044`、`1149-1177`、`1277-1306`、`1386-1414` 这些重复 metadata 片段。

- `src/video2tasks/server/routes.py`  
  负责 `/get_job`、`/submit_result`、`/health` 注册。路由层只依赖 `DispatchState`、`SampleStore`、`Protocol`，不直接知道 producer 细节。

- `src/video2tasks/server/producer.py`  
  负责 `producer_loop()`、Step A / Finalize 状态推进、超时回收、sample failure 闭环。  
  它应成为真正的“批处理编排器”，而不是内嵌在 app factory 里的局部函数。

推荐拆分后的依赖方向：

- `app.py` -> `server_state.py`, `sample_store.py`, `job_builder.py`, `routes.py`, `producer.py`, `runtime.py`
- `routes.py` 只依赖 `DispatchState`、`SampleStore`、`protocol.py`
- `producer.py` 依赖 `DispatchState`、`SampleStore`、`JobBuilder`、`windowing facade`、`llm_merge facade`

这条边界的核心价值是：  
先把“Web API”和“批处理编排器”从同一个函数里拆出来，而不是一开始就把所有算法一起移动。

### 2. `windowing.py` 的目标边界

`windowing.py` 最终不应继续作为“Stage 1 相关所有东西”的容器。建议按下面四块拆。

推荐新增模块：

- `src/video2tasks/server/window_media.py`  
  负责 `read_video_info()`、图片编码、contact sheet 构建、`FrameExtractor`、artifact 持久化适配。  
  这里承载媒体 I/O 和 OpenCV/ffmpeg 依赖。

- `src/video2tasks/server/window_plan.py`  
  负责 `Window`、`BoundaryRefinementWindow`、`build_windows()`、`build_refinement_windows()`、`build_boundary_refinement_windows()`、`sample_segment_frame_ids()`、`build_window_prompt_metadata()`。  
  这里承载切窗模型与 frame 采样模型，不夹带 heuristics。

- `src/video2tasks/server/segment_semantics.py`  
  负责边界支持度、sequence marker、instruction drift、instruction specificity 等跨 Stage 1 / Stage 2 共享语义。  
  这里必须把当前被 `llm_merge.py` 依赖的私有函数正式变成公共 API。  
  这是 `windowing.py` 拆分路线里的先决边界。

- `src/video2tasks/server/segmentation.py`  
  负责 `build_segments_via_cuts()`、`apply_boundary_refinement_results()`、`apply_deferred_segment_labels()`、`refine_segment_instructions()` 以及与 segment 装配直接相关的内部 helpers。  
  这里才是 Stage 1 输出构造的核心。

建议保留一个过渡期 `windowing.py facade`：

- 第一阶段不要求所有 import 一次性改完
- `windowing.py` 可以短期做 re-export
- 等测试、调用点和文档都迁移后，再收缩 facade 内容

这条路线的关键不是文件数量，而是强制把“媒体 I/O”和“语义规则”分开。  
否则以后改 ffmpeg/OpenCV 行为也要读一整套 segment 规则，改 merge guard 也要读 contact sheet 逻辑，维护成本不会真正下降。

### 3. `llm_merge.py` 的目标边界

`llm_merge.py` 应从“Stage 2 大杂烩”收缩成“Stage 2 facade”。推荐拆成五块：

- `src/video2tasks/server/stage2_backend.py`  
  负责从 `LLMMergeConfig` 解析 backend，并统一 provider 初始化。  
  这样可以消除 `OpenAIBackend(...)` 在 merge / summary / subtitle 三处重复 new 的问题。

- `src/video2tasks/server/stage2_request.py`  
  负责 structured payload 请求、adapter diagnostics 聚合、max_attempts 语义。  
  这里吸收 `_request_structured_payload()` 及相关 helper。

- `src/video2tasks/server/stage2_merge.py`  
  负责 merge schema、merge validator、guard、candidate 构造、coarse consensus、`run_llm_merge_pass()`。

- `src/video2tasks/server/stage2_summary.py`  
  负责 summary schema、partition validation、hierarchy 构造、`run_llm_summary_pass()`。

- `src/video2tasks/server/stage2_subtitles.py`  
  负责 subtitle payload 校验、subtitle localization、export subtitle 兼容层。

然后由：

- `src/video2tasks/server/llm_merge.py`  
  只保留 `run_llm_postprocess_pass()` 和必要的兼容导出，逐步退化为 facade。

这条路线有两个收益：

- Stage 2 内部可以按 pass 独立测试，不再每次改 summary 都触发 merge 文件级冲突。
- provider 初始化和 structured request 逻辑只保留一份，便于后续真的把 Stage 2 backend 做成统一接缝。

### 4. 推荐迁移顺序

推荐顺序不是按“哪个文件最大”排，而是按“哪个边界是其他拆分的前提”排。

建议的串行顺序：

1. 先抽 `segment_semantics.py`。  
   这是 `windowing.py` 与 `llm_merge.py` 后续并行拆分的前提，因为必须先消除跨文件 import 私有函数。

2. 再抽 `sample_store.py`。  
   这是 `app.py` 后续拆 producer / routes 的前提，因为持久化与路径逻辑必须先从应用装配层拿出来。

3. 再抽 `job_builder.py`。  
   这样可以先消除四类 job metadata 重复，并把 `JobEnvelope` 构建收口。

4. 然后拆 `routes.py`。  
   路由层一旦只依赖 `DispatchState` 和 `SampleStore`，`create_app()` 的职责会明显收缩。

5. 再拆 `producer.py`。  
   此时 producer 迁出时不需要再顺手搬路径、JSONL、metadata 逻辑，风险可控。

6. 最后分别拆 `llm_merge.py` 和 `windowing.py` 的内部实现块。  
   到这一步时，公共语义和 app 边界已经稳定，算法文件拆分才不会反复返工。

## 第一批落地范围

### 建议的“最小可落地第一批”

第一批不要追求“把三个大文件都拆干净”，而是只做以下三件事：

1. 新增 `src/video2tasks/server/segment_semantics.py`。  
   首批只迁出当前被 `llm_merge.py` 直接依赖的公共语义函数，以及它们必须带走的最小依赖。  
   目标：消除跨文件 import 私有函数，而不是一口气整理全部 heuristics。

2. 新增 `src/video2tasks/server/sample_store.py`。  
   首批只迁出 `app.py` 中的：
   - 样本输出路径函数
   - JSONL 追加
   - 窗口/边界/label 结果加载
   - `.DONE` / `.FAILED` / `failure.json` 管理
   - `_persist_result_record()` / `_persist_sample_failure()` 这类落盘逻辑  
   目标：把“应用装配”和“样本持久化”切开。

3. 新增 `src/video2tasks/server/job_builder.py`。  
   首批只迁出：
   - `_build_job_payload()`
   - contact sheet / inline / shared_fs transport 选择逻辑
   - 四类 job 的公共 metadata 字段拼装入口  
   目标：消除 `app.py` 中四份 metadata 重复，给后续 producer 拆分铺路。

### 第一批明确不做的事

- 不修改 `producer_loop()` 的状态推进语义
- 不改变 `/get_job`、`/submit_result` 的 HTTP contract
- 不改变 `windows.jsonl`、`segments.json`、`failure.json` 文件格式
- 不改变 `build_segments_via_cuts()`、`run_llm_postprocess_pass()` 的业务逻辑
- 不在第一批引入新的异步模型、事件总线或 repository framework

### 第一批完成后的预期收益

- `app.py` 体量应明显下降，并开始只剩“装配 + 运行入口 + 过渡 facade”
- `llm_merge.py` 不再依赖 `windowing.py` 私有函数
- 后续拆 `routes.py`、`producer.py`、`stage2_*` 模块时，不必重复搬运路径与 metadata 逻辑

这才是“最小可落地”的关键：  
先把下一步拆分所需的稳定边界建出来，而不是第一次提交就试图把所有逻辑都搬家。

### 哪些任务可以并行，哪些必须串行

必须串行的任务：

1. `segment_semantics.py` 必须先于 `llm_merge.py` / `windowing.py` 大拆分。  
   否则两边会继续共享隐式私有依赖，拆分只会制造新的临时耦合。

2. `sample_store.py` 必须先于 `producer.py` 提取。  
   否则 producer 提取时会同时搬状态机、路径、持久化三层逻辑，提交过大。

3. `job_builder.py` 必须先于 app 路由和 producer 的进一步拆分。  
   否则 job metadata 组装仍然会分散在主循环和 finalize 分支里。

可以并行的任务：

1. 在 `segment_semantics.py` 稳定后，`llm_merge.py` 的 import 迁移与 `windowing.py` facade 调整可以并行。  
   二者共享的是公共 API，不再需要互相改内部实现。

2. 在 `sample_store.py` 和 `job_builder.py` 稳定后，`routes.py` 提取与 `producer.py` 初步提取可以并行。  
   前者主要处理 dispatch API，后者主要处理 sample state machine，交集较小。

3. 在 app 边界稳定后，`stage2_summary.py` / `stage2_subtitles.py` 的拆分可以与 `window_media.py` / `window_plan.py` 的拆分并行。  
   这两条线分别覆盖 Stage 2 文本处理和 Stage 1 媒体/切窗，不共享太多状态。

## 风险与回滚点

### 1. 风险：dict/meta 隐式契约太多，拆分时容易产生“字段没丢但语义变了”的回归

高风险字段集中在：

- `task_id`
- `dispatch_id`
- `subset`
- `sample_id`
- `job_type`
- `window_id` / `boundary_id` / `segment_id`
- `logical_frame_count`
- contact sheet 相关字段

这些字段今天主要靠散落的 dict 约定维持，一旦迁移过程中默认值或字段来源变化，很容易出现：

- resume 可读但逻辑错
- diagnostics 对不上
- 某类 job 在 finalize 时失配

建议：

- 第一批只做“移动代码，不改字段”
- 先补 characterization tests，再做 import rewiring
- 新模块的输入输出保持和旧 helper 一致

### 2. 风险：测试当前大量 monkeypatch 原模块符号，拆分后容易整片失效

现有测试明显直接 patch：

- `app_module.build_segments_via_cuts`
- `app_module.run_llm_postprocess_pass`
- `app_module.run_export_subtitle_localization_pass`
- `app_module.read_video_info`

如果一次性把这些符号彻底移走，短期会造成：

- 不是业务回归，但测试集大面积红
- 开发者误判为拆分行为改变

建议：

- 至少一个批次内保留 `app.py` / `windowing.py` / `llm_merge.py` 的兼容 re-export
- 先迁测试到新模块，再删除旧 facade

### 3. 风险：`windowing.py` 算法拆分最容易引入行为漂移

`build_segments_via_cuts()` 周围已经有较密测试覆盖，这是好事，也意味着它对行为漂移非常敏感。  
因此不建议在“建立公共边界”的同一批次里同时修改：

- cut clustering
- instruction timeline
- cleanup / drift split
- boundary refinement

建议：

- 先分模块，再考虑局部逻辑优化
- 算法拆分提交和算法修正提交必须分开

### 4. 风险：Stage 2 backend 解析若和拆分一起改，会把“模块搬家”和“行为变化”混成同一次回归

`llm_merge.py` 当前虽然不优雅，但行为路径清楚。  
第一批如果同时做：

- 拆文件
- 统一 backend 工厂
- 修改 fallback 语义

就很难判断问题来自哪里。

建议：

- 第一轮 Stage 2 拆分只抽 provider resolver 与 request helper，不改 fallback 规则
- merge / summary / subtitle 的行为修正留到拆分稳定后再做

### 回滚点建议

建议把回滚点设计成“模块落地但旧 facade 仍在”的形式：

1. `segment_semantics.py` 落地后，`windowing.py` 先做兼容导出。  
   如果回归，回滚 import rewiring 即可，不必删除新模块。

2. `sample_store.py` / `job_builder.py` 落地后，`app.py` 先只做代理调用。  
   如果回归，回滚代理替换提交即可，不必恢复整段旧代码。

3. `routes.py` / `producer.py` 落地后，保留 `create_app()` / `run_server()` 签名不变。  
   这样外部调用与测试入口不需要同步回滚。

## 验证计划

这份方案本身不改代码，但第一批落地时建议按下面顺序验证。重点不是“全量跑一遍就算完”，而是按边界对应测试。

### 1. App 装配与运行时契约

目标：确认 `create_app()` / `run_server()` 对外行为没变，尤其是不引入隐藏启动副作用。

建议执行：

```bash
pytest tests/server/test_runtime.py
pytest tests/server/test_run_manifest.py
pytest tests/test_logging.py
```

重点观察：

- `create_app()` 是否仍然只装配 runtime，不提前启动 producer
- `app.state` 暴露的 manifest / queue / runtime 契约是否还在
- `run_server()` 是否仍按 start -> uvicorn -> stop/join 的顺序执行

### 2. Dispatch / retry / finalize 主路径

目标：确认 `sample_store.py`、`job_builder.py` 提取后，没有改坏路由、重试和 finalize 闭环。

建议执行：

```bash
pytest tests/server/test_app_retry.py
```

重点观察：

- `/get_job` / `/submit_result` 的 payload 与状态推进是否一致
- timeout / empty result / stale dispatch 路径是否保持原语义
- finalize 失败和 postprocess 失败相关用例是否仍然通过

### 3. Stage 1 语义与 segment 装配

目标：确认 `segment_semantics.py` 抽出后，Stage 1 分段行为不漂移。

建议执行：

```bash
pytest tests/server/test_windowing.py
```

重点观察：

- `build_segments_via_cuts()` 的已知 guard / cleanup 行为
- `apply_boundary_refinement_results()` 的边界移动与合并语义
- recovered micro-boundary、light cleanup fallback 等行为是否保持不变

### 4. Stage 2 merge / summary / subtitle

目标：确认 Stage 2 拆边界后，请求重试、summary 独立性和 subtitle fallback 仍然一致。

建议执行：

```bash
pytest tests/server/test_llm_summary.py
```

重点观察：

- merge 失败是否仍不阻断 summary
- export subtitle 的 fallback、语言处理、adapter diagnostics 是否保持原语义
- `run_llm_postprocess_pass()` 是否仍是 merge -> summary 的稳定 facade

### 5. 建议新增的回归验证

当前已有测试足够支撑第一批拆分，但建议补两类 characterization test，进一步降低回归风险：

1. `sample_store.py` golden tests  
   直接对 `windows.jsonl`、`boundary_refinements.jsonl`、`failure.json` 做读写往返验证，确保拆分前后格式一致。

2. `job_builder.py` metadata snapshot tests  
   针对四类 job 分别校验生成的 `JobEnvelope.meta` 字段集合，避免某次重构无意丢掉 `logical_frame_count`、contact sheet 字段或 job-specific id。

## 最终建议

综合来看，这轮“架构拆分与模块边界”方向的建议是成立的，而且下一步应优先做的不是“继续忍受大文件”，也不是“直接重写 server”，而是：

1. 先把共享语义边界正式化。  
   也就是把 `windowing.py` 与 `llm_merge.py` 共享的 segment 规则从私有实现提升为公共模块。

2. 先把 `app.py` 中的持久化与 job 构建拆出来。  
   这样 `create_app()` 才有机会回到应用装配入口，而不是继续承担系统总装。

3. 在第一批稳定后，再分拆 `routes.py`、`producer.py`、`stage2_*`、`window_*` 模块。  
   这是能真正降低耦合、同时控制回归面的路线。

一句话总结：  
当前最值得做的不是“把大文件切成更多文件”，而是先把 `app.py` 的编排边界和 `windowing.py` / `llm_merge.py` 的共享语义边界正式建起来。只要这两个边界不成立，任何拆文件都只是搬家，不是解耦。
