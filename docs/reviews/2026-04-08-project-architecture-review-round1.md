# Video2Tasks 项目架构审查 Round 1

日期：2026-04-08  
范围：`src/video2tasks` 全项目架构，重点覆盖 server / worker / windowing / llm_merge / config / vlm 适配层。  
方式：静态读码 + 现有测试结构审视 + 基线命令验证，不改业务代码。

## 1. 结论先行

这个项目的核心算法和规则工程已经很扎实，但系统架构正在进入一个典型的第二阶段问题：

- 不是“功能没有”，而是“功能都在长”。
- 不是“没有抽象”，而是“最初够用的抽象已经被新需求挤穿了”。
- 不是“没有测试”，而是“测试高度集中在规则函数，生命周期编排和模块契约没有跟上复杂度增长”。

当前最值得优先处理的，不是再加一条启发式规则，而是把下面四个架构边界重新立住：

1. 运行时生命周期边界：把 HTTP 服务、调度状态机、持久化和进程退出逻辑拆开。
2. 算法边界：把 `windowing.py` 从“算法黑洞”拆成几个有明确责任的模块。
3. 后端契约边界：修复已经失真的 `VLMBackend` 抽象。
4. 配置与开发入口边界：让 `src` 布局、环境变量、测试入口和文档说法重新一致。

如果不先做这几步，后面继续往 Stage 1/Stage 2 上叠策略，维护成本会比模型调用成本涨得更快。

## 2. 本次证据基线

### 2.1 读码范围

重点审查文件：

- `src/video2tasks/server/app.py`
- `src/video2tasks/server/windowing.py`
- `src/video2tasks/server/llm_merge.py`
- `src/video2tasks/worker/runner.py`
- `src/video2tasks/config.py`
- `src/video2tasks/prompt.py`
- `src/video2tasks/vlm/base.py`
- `src/video2tasks/vlm/factory.py`
- `src/video2tasks/vlm/gemini_api.py`
- `src/video2tasks/vlm/openai_api.py`

辅助参考：

- `README.md`
- `docs/principles/*.md`
- `docs/main-pipeline-review.md`
- `docs/reviews/round1/*.md`
- `tests/**`

### 2.2 复杂度热点

核心文件行数如下：

- `src/video2tasks/server/windowing.py`：2586 行
- `src/video2tasks/server/llm_merge.py`：1648 行
- `src/video2tasks/server/app.py`：1285 行
- `src/video2tasks/prompt.py`：782 行
- `src/video2tasks/config.py`：737 行
- `src/video2tasks/worker/runner.py`：392 行

这说明当前复杂度主要集中在三个点：

- 运行时编排：`server/app.py`
- 规则/算法：`server/windowing.py`
- Stage 2 文本后处理：`server/llm_merge.py`

### 2.3 测试基线

2026-04-08 实际执行结果：

1. 直接在仓库根执行 `pytest -q`
   - 结果：失败
   - 原因：测试收集阶段无法导入 `video2tasks`，报 `ModuleNotFoundError`

2. 执行 `env PYTHONPATH=src pytest -q`
   - 结果：通过
   - 汇总：`235 passed in 67.91s (0:01:07)`

这意味着当前代码不是“已经坏掉”，而是开发入口没有把 `src` 布局接通到默认测试命令。这本身就是架构和工程化的一部分问题。

## 3. 当前架构轮廓

从职责上看，项目已经形成了一个比较清楚的三段式流水线：

- Server
  - 生成窗口任务
  - 管理队列、重试和结果落盘
  - 汇总阶段一结果，串起 refinement / deferred labeling / llm merge / subtitle / export
- Worker
  - 拉任务
  - 解码图片
  - 生成 prompt
  - 调后端模型
  - 回传结构化 JSON
- Backend adapters
  - 把统一任务转换成不同模型端点的请求
  - 从不同响应形态里提取统一的 `thought/transitions/instructions`

这个总体思路是对的，问题主要出在“实现层边界已经模糊”：

- Server 不是只做编排，它还承担了状态机、文件协议、错误恢复和进程生死控制。
- `windowing.py` 不是只做窗口，它同时负责视频 IO、contact sheet、启发式语义规则、切分聚合、标签清洗。
- `VLMBackend` 名义上是统一接口，实际上已经被 Gemini 的原始图片传输需求打破。

## 4. 优点

先记优点，避免只看到问题。

### 4.1 主流程思路是清楚的

尽管实现偏集中，但从概念模型看，项目并不混乱：

- Stage 1 是 recall-first 的边界召回
- Stage 2 是保守的语义合并和层级摘要
- Export 是与主流程松耦合的下游产物生成

这条主线在代码和文档里基本一致。

### 4.2 算法和 prompt 规则投入很深

`windowing.py` 和 `prompt.py` 里积累了大量任务切分经验。  
这类项目最难沉淀的往往不是框架，而是规则资产；这部分已经形成壁垒。

### 4.3 单元测试密度不低

特别是下面几类：

- `tests/server/test_windowing.py`
- `tests/server/test_llm_merge.py`
- `tests/server/test_llm_summary.py`
- `tests/test_prompt.py`
- `tests/vlm/test_*`

说明团队已经在用测试固定规则行为，而不是完全靠人工感觉迭代。

### 4.4 有恢复意识和诊断意识

例如：

- append-only JSONL 结果记录
- `dispatch_id` 防止旧结果覆盖
- `.DONE` / `.FAILED`
- 大量 `diagnostics`

这些都说明项目不是 demo 心态，而是在朝“长跑批处理系统”靠拢。

## 5. 核心问题与优化建议

以下按重要性排序。

### 5.1 高优先级：运行时生命周期被锁死在 FastAPI app 闭包里

关键位置：

- `src/video2tasks/server/app.py:156-1274`
- `src/video2tasks/server/app.py:480-582`
- `src/video2tasks/server/app.py:585-1268`
- `src/video2tasks/server/app.py:655-665`
- `src/video2tasks/server/app.py:1270-1272`

现状：

- `create_app()` 内部直接持有 `job_queue`、`inflight`、retry 计数、样本锁和各种路径函数。
- API endpoint 与调度逻辑共享同一批闭包状态。
- `producer_loop()` 作为 daemon thread 在 `create_app()` 里启动。
- 全部处理完成时，后台线程直接调用 `os._exit(exit_code)`。

这带来四个问题：

1. HTTP 层、调度层、持久化层、进程生命周期层没有清晰边界。
2. `create_app()` 很难作为纯应用工厂做独立测试，因为它会顺带拉起后台调度线程。
3. `os._exit()` 会跳过常规清理路径，不适合作为长期运行服务的退出机制。
4. 当前状态机只能通过读大段闭包代码理解，几乎没有显式 runtime 对象模型。

我的判断：

这已经不是“文件有点大”，而是 runtime model 没有被一等建模。  
随着 refinement、deferred labeling、subtitle/export 继续叠加，`producer_loop()` 会越来越像一段不可替换的流程脚本。

建议：

1. 引入显式运行时对象，例如 `PipelineCoordinator`。
   - 负责样本生命周期推进
   - 负责与 `JobQueue` / `ResultStore` / `SampleRepository` 协作
2. 把 FastAPI endpoint 退回到薄适配层。
   - `GET /get_job` 只向调度器取 job
   - `POST /submit_result` 只把结果交给结果处理器
3. 用 FastAPI lifespan 或独立命令层管理后台线程，而不是在 `create_app()` 里直接启动。
4. 去掉 `os._exit()`，改成：
   - 更新完成状态
   - 由 CLI 层决定是否退出
   - 或通过事件/标志位让主线程优雅停止

一个比较稳妥的拆分方向：

- `server/api.py`
- `server/runtime.py`
- `server/scheduler.py`
- `server/result_store.py`
- `server/sample_state.py`

不要一口气大改。第一步只要先把 `producer_loop()` 搬出 `create_app()`，就已经能显著改善可测性。

### 5.2 高优先级：`windowing.py` 已经不是 windowing 模块，而是算法总装仓

关键位置：

- `src/video2tasks/server/windowing.py:39-98`
- `src/video2tasks/server/windowing.py:642-982`
- `src/video2tasks/server/windowing.py:984-2296`
- `src/video2tasks/server/windowing.py:2297-2586`

现状：

这个文件同时承载了：

- 视频信息读取
- 窗口构造
- refinement window 构造
- boundary refinement window 构造
- contact sheet 生成
- artifact 落盘
- instruction token/phase/action 规则
- segment merge / cleanup / drift split
- 最终 cut 聚合和 `segments` 生成

具体看结构就很明显：

- `FrameExtractor` 本身已经是一个独立模块体量，见 `642-982`
- 各种 instruction 语义启发式从 `984` 之后一路延伸到两千行以上
- `build_segments_via_cuts()` 则在底部再把前面所有能力重新编排一遍

这种形态的代价：

1. 修改抽帧实现时，很容易顺手碰到语义合并逻辑。
2. 想优化切分算法时，必须在同一个超大文件里跨 IO/语义/聚合来回跳。
3. 测试虽多，但模块边界不清，最终会形成“只能继续往这个文件里加函数”的演化路径。

建议：

按责任拆，不按技术层拆。

推荐拆分为：

- `server/window_sampling.py`
  - `Window`
  - `BoundaryRefinementWindow`
  - `build_windows`
  - `build_refinement_windows`
  - `build_boundary_refinement_windows`
- `server/frame_io.py`
  - `read_video_info`
  - `FrameExtractor`
  - contact sheet 与 artifact 逻辑
- `server/cut_voting.py`
  - 重复窗口投票
  - cut 聚类
  - dense micro-boundary 恢复
- `server/instruction_semantics.py`
  - 各种 token/focus/action family 判定
  - drift / merge 启发式
- `server/segment_builder.py`
  - `build_segments_via_cuts`
  - `merge_task_level_segments`
  - `cleanup_auxiliary_segments`
  - `refine_segment_instructions`

拆分原则：

- 先移动代码，不改逻辑。
- 先把“可读边界”拆出来，再决定是否重写内部算法。

当前最值得优先做的是把 `FrameExtractor` 和 instruction 语义规则分出去。  
这两个子域天然独立，拆出去收益最高，风险也最低。

### 5.3 高优先级：`VLMBackend` 抽象已经被实现细节打穿

关键位置：

- `src/video2tasks/vlm/base.py:9-41`
- `src/video2tasks/vlm/base.py:12-27`
- `src/video2tasks/worker/runner.py:306-311`
- `src/video2tasks/vlm/gemini_api.py:37-55`
- `src/video2tasks/vlm/gemini_api.py:341-357`

现状：

`VLMBackend.infer()` 的接口声明是：

- 输入：`List[np.ndarray]`
- 语义：BGR 图像数组列表

但在 worker 里，当 backend 是 Gemini 时，实际传进去的是：

- `{"raw_bytes": ..., "mime_type": ...}` 字典列表

也就是说：

- 抽象层说“所有后端都吃 numpy”
- 调用方却已经知道“Gemini 例外”
- Gemini backend 内部也显式兼容这种例外格式

这类问题的本质不是 typing，而是 abstraction leak：

- worker 不该知道具体后端如何消费图片
- backend interface 也不该靠“某些实现偷偷接受另一种数据形态”维持兼容

这会带来两个长远问题：

1. 后续如果再加一个需要原始字节流或远端对象引用的 backend，worker 会继续长 `if backend == ...`。
2. 所谓 backend pluggability 会退化成“工厂可插拔，但调用协议不可插拔”。

建议：

把图片输入建成共享领域模型，而不是假装所有后端都只吃 numpy。

可选方案：

1. 定义统一的 `ImagePayload` dataclass
   - `raw_bytes: bytes | None`
   - `mime_type: str`
   - `bgr: np.ndarray | None`
   - `path: str | None`
2. `load_job_image_records()` 直接返回 `ImagePayload`
3. `VLMBackend.infer()` 接受 `List[ImagePayload]`
4. 每个 backend 自己决定用原始字节、路径还是 numpy

这样有三个好处：

- worker 不再对 Gemini 特判
- backend interface 终于和真实输入协议一致
- 后续如果改成对象存储 URL、共享内存或 mmap，也有地方可扩展

### 5.4 中高优先级：配置层是手工汇编的单体，扩展成本已经开始偏高

关键位置：

- `src/video2tasks/config.py:439-478`
- `src/video2tasks/config.py:498-700`
- `pyproject.toml:26-43`

现状：

- `Config` 把 schema、默认值、YAML 加载、env override 合并全部放在一个文件。
- 环境变量映射通过 `_collect_env_override_data()` 手工一项项拼。
- 项目依赖里已经有 `pydantic-settings`，但这里没有真正用上。

这类设计前期很快，后期问题主要体现在“每加一项配置都要改很多地方”：

- schema
- validator
- env parser
- README
- example config
- 测试

现在这个文件还能维护，但信号已经很明显：

- 737 行体量
- 大量 one-off env mapping
- 注释和实际优先级容易漂移

另外一个直接暴露出来的问题，是开发入口也没统一：

- 仓库根直接 `pytest -q` 失败
- 必须显式加 `PYTHONPATH=src`

这说明“包布局约定”和“默认开发命令”还没有打通。

建议：

1. 保守方案
   - 保留 `BaseModel`
   - 但把 `server/worker/windowing/llm_merge/export` 分成独立 settings 文件
   - 在顶层 `Config` 只做组合
2. 更进一步的方案
   - 改用 `pydantic-settings` 管理 env source
   - 少写手工 `_collect_env_override_data()`
3. 工程化补齐
   - 在 `pyproject.toml` 增加 pytest 的 `pythonpath = ["src"]` 或统一通过 editable install 运行测试
   - 让 README、`config.example.yaml`、实际默认值来自同一来源

一个非常实际的目标是：

- 新增一个配置项时，不需要在 5 个位置手动同步
- 新同学 clone 下来后，直接 `pytest -q` 就能跑

### 5.5 中优先级：Stage 2 文本后处理被硬绑到 OpenAI 客户端，且重复初始化

关键位置：

- `src/video2tasks/server/llm_merge.py:1210-1224`
- `src/video2tasks/server/llm_merge.py:1452-1462`
- `src/video2tasks/server/llm_merge.py:1563-1573`
- `src/video2tasks/server/llm_merge.py:1618-1648`

现状：

- `llm_merge.backend` 只允许 `"openai"`
- merge / summary / subtitle 三个流程分别构造 `OpenAIBackend`
- 尽管函数参数允许传 `backend`，但默认路径仍然是每段自己初始化

这有两个架构层面的后果：

1. Stage 2 实际上不是“文本模型后处理框架”，而是“OpenAIBackend 驱动的若干功能函数”。
2. 三段文本处理共享的是配置，不是真正共享的客户端抽象。

如果团队已经明确只会用 OpenAI 兼容接口，这不是马上要改的功能 bug。  
但从可演化性看，这会卡住下面几类需求：

- 替换成别的 text-only provider
- 复用连接/诊断状态
- 统一 Stage 2 请求预算和限流

建议：

1. 单独引入 `TextPostprocessClient` 或 `StructuredTextBackend`
2. 让 merge / summary / subtitle 依赖这个抽象，而不是直接 new `OpenAIBackend`
3. 把客户端生命周期收口到一次 sample 级别，而不是每个 pass 自己初始化

这项优先级低于前面三项，但在继续加 Stage 2 能力前最好先收口。

### 5.6 中优先级：测试布局偏向规则函数，生命周期编排和产物契约测试不足

证据来源：

- `tests/server/test_windowing.py`
- `tests/server/test_llm_merge.py`
- `tests/test_prompt.py`
- `tests/server/test_app_retry.py`

现状特征：

- 规则类测试很多，覆盖很好。
- backend adapter 测试也不少。
- 但 `server/app.py` 这种真正控制样本生命周期的核心模块，只看到 retry / submit 级别测试，几乎没有完整 sample 生命周期集成测试。

这和当前代码结构是互相强化的：

- 因为 `producer_loop()` 太大，所以难测
- 因为难测，所以更多新增测试落在纯函数
- 因为纯函数测试足够多，复杂编排逻辑继续留在大函数里

风险不在于现在完全没测，而在于未来重构 runtime 时缺少安全网。

建议：

补三类测试，而不是继续堆更多 token 规则测试：

1. sample 生命周期测试
   - 从窗口生成到 `.DONE` / `.FAILED`
   - 用 fake backend / fake extractor / temp dataset
2. 输出契约测试
   - 最终 `segments.json` 的必有字段、可选字段、fallback 形态
3. 调度恢复测试
   - 基于已有 JSONL、`.DONE`、`.FAILED` 的 resume 行为

如果只允许做一件事，我建议先加“完整 sample 生命周期测试”。  
这是未来拆 `producer_loop()` 的最低成本保险。

## 6. 建议的重构顺序

不建议一次性大重写。建议分三阶段。

### 阶段一：止血，先把边界说清楚

目标：不改变主流程行为，只降低继续演化的风险。

建议项：

1. 让 `pytest -q` 默认可运行
2. 把 `producer_loop()` 从 `create_app()` 里抽成独立 runtime 对象
3. 把 `FrameExtractor` 从 `windowing.py` 中拆出
4. 定义统一 `ImagePayload`

这一阶段不碰算法本身，主要做结构迁移和接口收口。

### 阶段二：算法模块化

目标：把 Stage 1 规则体系拆成可维护的包结构。

建议项：

1. 拆出 `instruction_semantics.py`
2. 拆出 `cut_voting.py`
3. 拆出 `segment_builder.py`
4. 为拆出的模块补边界测试和契约测试

这一阶段完成后，`windowing.py` 应该只剩“窗口相关能力”。

### 阶段三：统一 Stage 2 和产物模型

目标：让 Stage 2 成为真正可扩展的后处理子系统。

建议项：

1. 抽象 text-only backend/client
2. 统一 `segments.json` 的结果 schema
3. 把 diagnostics 从“动态拼装字典”逐步收束为显式结果模型

## 7. 一个更健康的目标架构

如果用一句话描述目标状态，我会这样定：

> Server 负责协调，Worker 负责执行，算法模块负责决策，配置模块负责装配，结果模型负责对外契约。

对应到代码组织，可以收敛到下面这种形态：

- `video2tasks/server/api.py`
- `video2tasks/server/runtime.py`
- `video2tasks/server/jobs.py`
- `video2tasks/server/result_store.py`
- `video2tasks/server/window_sampling.py`
- `video2tasks/server/frame_io.py`
- `video2tasks/server/cut_voting.py`
- `video2tasks/server/instruction_semantics.py`
- `video2tasks/server/segment_builder.py`
- `video2tasks/postprocess/text_backend.py`
- `video2tasks/postprocess/merge.py`
- `video2tasks/postprocess/summary.py`
- `video2tasks/postprocess/subtitles.py`
- `video2tasks/models/job.py`
- `video2tasks/models/results.py`
- `video2tasks/settings/*.py`

不需要追求“设计最优雅”，只需要让新增一个能力时，工程师知道应该落在哪一层，而不是继续把逻辑塞回原来的大文件。

## 8. 最终判断

这个项目当前最强的部分，是切分经验和规则资产。  
当前最弱的部分，不是模型效果，而是“这些能力正在通过几个超大文件耦合在一起”。

短期内，这套结构还能继续跑。  
但如果下一阶段还要继续迭代：

- 更复杂的 refinement
- 更多 backend
- 更稳定的导出契约
- 更强的批处理恢复

那现在就应该开始做结构性减压。

我给出的优先级是：

1. 先重构 runtime 边界
2. 再拆 `windowing.py`
3. 然后修复 backend 输入契约
4. 最后统一配置和 Stage 2 客户端抽象

这条路线最稳，也最符合这个仓库现有积累方式。
