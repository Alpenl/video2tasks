# Video2Tasks 项目架构复核 Round 2（独立审核）

日期：2026-04-08  
审查对象：`src/video2tasks` 主流程架构，以及 `/docs/reviews/2026-04-08-project-architecture-review-round1.md` 的第一轮结论  
方式：静态读码 + 本地命令复核 + 测试入口复核；不修改业务代码

## 1. 独立结论

第一轮抓住了几个真正的复杂度中心，但我认为它还没有把当前项目最核心的架构约束说透。

我对当前系统的独立判断是：

- 这个项目现在并不是一个“通用分布式 server/worker 架构”，而是一个“偏单机、共享磁盘、append-only 结果文件驱动的批处理流水线”。
- 只要团队明确接受这个定位，它仍然可以继续演进；但如果继续沿 README 里的“多机 worker 横向扩展”叙事前进，现有传输协议和产物协议会先出问题。
- 当前最该优先解决的，不是 `config.py` 还是 `llm_merge.py` 哪个更大，而是三个更基础的问题：
  1. 任务传输契约与部署叙事不一致。
  2. 运行时生命周期仍然主要靠 `create_app()` 闭包和后台线程隐式维持。
  3. 恢复/续跑机制没有把“本次运行身份”定义清楚，旧结果和新配置可以被静默混用。

换句话说，Round 1 更像是在看“模块边界”；Round 2 看到的更大问题是“系统边界和运行边界还不够明确”。

## 2. 对 Round 1 的复核

### 2.1 我同意并复核成立的判断

以下结论，我认为第一轮判断基本成立，而且代码证据足够：

- `server/app.py` 的运行时耦合问题成立。`create_app()` 在建应用的同时创建内存队列、状态字典、样本锁、磁盘路径规则，并直接启动 daemon thread；`producer_loop()` 里还负责超时重试、样本推进、持久化与最终退出，见 `src/video2tasks/server/app.py:156-486`、`620-1260`。
- `windowing.py` 已经远超“窗口构造模块”的责任范围。`FrameExtractor`、contact sheet、artifact 持久化、cut 聚合、instruction 语义启发式、最终 segment 构建都塞在一个文件里，拆分方向判断没有问题，见 `src/video2tasks/server/windowing.py:642-980` 以及后续两千多行规则与聚合代码。
- 默认测试入口没有接通 `src` 布局这一点成立。我本地复核结果与第一轮一致：
  - `pytest -q`：失败，测试收集阶段 `ModuleNotFoundError: No module named 'video2tasks'`
  - `env PYTHONPATH=src pytest -q`：通过，`235 passed in 69.68s`

### 2.2 我认为需要降调或更精确表述的地方

#### A. `create_app()` “很难测试”这个说法方向对，但措辞偏重

第一轮的核心担心是对的，但更准确的说法应该是：

- 不是“无法测试”，而是“应用工厂带副作用，导致测试不再是纯构造”。
- 现有测试实际上已经在实例化它，例如 `tests/server/test_app_retry.py` 直接调用 `create_app(config)` 并使用 `TestClient`。
- 真正的问题是：只要创建 app，后台生产线程就会启动，测试必须容忍额外时序和后台状态变化。这会损害测试的确定性，也提高重构成本。

因此，这项问题应表述为“应用构造与运行时启动耦合，降低可测性与可控性”，比“很难作为纯工厂独立测试”更贴近现状。

#### B. `VLMBackend` 抽象泄漏问题存在，但优先级应略低于第一轮给出的强度

第一轮把这项列为高优先级，我同意它是设计缺口，但我会把它降到“中高优先级”。

原因是：

- 问题是真实存在的：`VLMBackend.infer()` 类型声明仍写着 `List[np.ndarray]`，但 worker 在 `config.worker.backend == "gemini"` 时会传 `{"raw_bytes": ..., "mime_type": ...}` 字典列表，见 `src/video2tasks/vlm/base.py:8-24` 与 `src/video2tasks/worker/runner.py:304-311`。
- 但代码里其实已经有一个过渡中的统一载体：`LoadedImage` dataclass，包含 `raw_bytes`、`mime_type`、`bgr`，见 `src/video2tasks/worker/runner.py:45-49`。这说明系统并不是完全没有抽象，只是“共享输入模型已经在 worker 内部形成，但没有被提升为正式 backend 契约”。

所以，我更建议把这项视为“协议建模未完成”，而不是“后端抽象已经整体失真到必须立即大改”。

#### C. Stage 2 绑死 OpenAI 的问题成立，但不应排在最前面

`llm_merge.py` 的 merge / summary / subtitle 都会在未传 `backend` 时自行构造 `OpenAIBackend`，而且配置只接受 `openai`，见 `src/video2tasks/server/llm_merge.py:1215-1224`、`1453-1462`、`1564-1573`。

这确实限制未来扩展，但在我看来它的优先级低于下面几项：

- 任务传输协议不清
- runtime 生命周期不清
- 续跑身份不清

如果团队当前明确只打算用 OpenAI 兼容接口推进 Stage 2，那么这更像是“下一阶段扩展性债务”，不是眼下最容易造成错误运行模式的架构风险。

## 3. 我认为 Round 1 漏掉的关键问题

### 3.1 高优先级：系统默认强依赖“服务端与 worker 共享本地文件系统”，这与 README 的分布式叙事冲突

这是我认为第一轮主文档里最值得补上的问题，而且优先级应排到最前面。

证据链是完整的：

- README 直接把架构描述为可横向扩展的 distributed processing，并写了“Run Server on one 4090, then connect 10 machines running Workers”，见 `README.md:156-162`。
- 但 server 端默认总会创建 `TaskArtifactWriter`，见 `src/video2tasks/server/app.py:173-174`。
- `_build_job_payload()` 调用 `extractor.get_many_b64_with_artifacts(...)` 时，`return_images` 的条件是 `artifact_writer is None`；而这里传入的 `FrameExtractor` 总是带 `artifact_writer=task_artifact_writer`，所以默认走“落盘 artifact，再把路径塞进 job”这条链路，见 `src/video2tasks/server/app.py:457-475`、`750`、`902`、`1022`、`1108`。
- worker 收到任务后，如果存在 `image_paths`，会直接 `Path(path).read_bytes()` 从本地路径读图，见 `src/video2tasks/worker/runner.py:109-132`。

这意味着当前主链路的真实部署假设是：

- server 和 worker 共享同一套文件路径
- 而且 worker 看到的路径必须和 server 写出的路径完全一致

这与 README 强调的“多机 worker 横向扩展”并不一致。  
如果团队确实要支持多机、多容器或远程 worker，这不是文档问题，而是运输层架构问题。

建议：

1. 先明确支持的部署模式，不要继续模糊表达。
   - 模式 A：单机/共享盘模式，`image_paths` 合法
   - 模式 B：网络分离模式，只允许内联 payload 或对象存储 URI
2. 把 job 的图片传输方式显式建模成字段，而不是靠 `image_paths` / `images` 两套隐式约定。
3. 如果短期只支持共享盘模式，应在 README 和 CLI 文档里明确写成硬约束，而不是继续暗示“任意多机 worker”。

### 3.2 高优先级：续跑机制没有“运行身份”与“结果 schema 版本”，旧结果可能被静默复用

当前恢复/续跑依赖的关键身份只有：

- `run.base_dir`
- `subset`
- `run_id`
- `sample_id`
- 以及 `.DONE` / `.FAILED` / `windows.jsonl` / `segments.json` 这些固定文件名

相关代码：

- `RunConfig.run_id` 默认值就是 `"default"`，见 `src/video2tasks/config.py:18-21`
- 输出目录由 `base_dir/subset/run_id/samples/sample_id` 组成，见 `src/video2tasks/server/app.py:52-75`
- server 启动后直接依据 `.DONE` / `.FAILED` 和既有 JSONL 内容决定是否跳过、续跑或聚合，见 `src/video2tasks/server/app.py:237-340`、`688-750`

这里缺了一个关键约束：

- 当前 run 是用什么配置跑的
- 当前结果文件使用什么 schema / prompt / backend / algorithm 版本生成

所以同一个 `run_id` 下，如果你改了：

- `window_repeat_count`
- prompt 策略
- backend
- `boundary_prompt_mode`
- 甚至代码里的聚合逻辑

系统仍可能直接吃旧的 `windows.jsonl` / `.DONE` 继续推进。  
这不是“有点不优雅”，而是会直接破坏实验可复现性和续跑正确性。

建议：

1. 在每个 `run_dir` 写入 `run_manifest.json`。
   - 包含 config hash
   - 代码版本信息（至少 git commit 或 dirty 标记）
   - 结果 schema version
   - worker/backend 摘要
2. 恢复时先做 manifest 校验。
   - 默认 mismatch 就拒绝 resume
   - 需要强制续跑时，显式传 `--force-resume`
3. 把 `.DONE` 从“只表示样本目录里有产物”升级为“某个 manifest 身份下完成”。

### 3.3 中高优先级：任务/结果协议大量依赖松散 dict 和字符串约定，导致多处边界一起漂移

这项问题是我认为第一轮只部分触及、但没有上升成“架构问题”的地方。

当前主流程里有很多关键契约并没有被正式建模：

- server 用 `_build_job_payload()` 拼裸字典，字段可能包括 `task_id`、`meta`、`image_paths`、`images`、`artifact_manifest_path`，见 `src/video2tasks/server/app.py:457-475`
- worker 再从这些松散字段中推断语义，例如 `job_type`、`contact_sheet_rows`、`logical_frame_count`、`frame_ids` 等，见 `src/video2tasks/worker/runner.py:281-328`
- 结果持久化也是按 `job_type` 分支写不同 JSONL，并用 `terminal_error`、`window_id`、`segment_id`、`boundary_id` 等字符串键约定协议，见 `src/video2tasks/server/app.py:360-414`

这类设计在早期迭代非常快，但现在已经开始产生连锁效应：

- Gemini 例外输入会穿透到 worker/backend 交界
- resume 行为只能靠文件名和字段名推断
- sample 生命周期状态只能靠裸整数和分支隐式表达
- 新增一种 job 类型时，需要同步改 server 造任务、worker 解任务、server 存结果、resume 读结果

这项问题和 `VLMBackend` 泄漏本质上是同一类债务：缺少正式的跨边界协议模型。

建议：

1. 先定义最小协议模型，而不是马上大重构。
   - `JobEnvelope`
   - `JobMeta`
   - `ImageRef` / `ImagePayload`
   - `ResultEnvelope`
2. 先让 server 和 worker 之间的 HTTP 契约类型化。
3. 再把 on-disk JSONL 记录也统一为带 `schema_version` 的结果记录模型。

## 4. 我建议调整后的优先级

如果只看“未来半年最容易先踩中的架构风险”，我建议优先级重排如下：

1. 明确任务传输与部署模式：共享盘模式还是网络分离模式
2. 给 run / resume 增加 manifest 与版本校验
3. 把 `create_app()` 中的 runtime 生命周期抽成显式协调对象
4. 为 server/worker 生命周期补真正的端到端契约测试
5. 再拆 `windowing.py`
6. 再做 config 模块化
7. 最后再统一 Stage 2 的 text backend 抽象

这个顺序和第一轮最大不同之处在于：

- 我把“运输契约”和“续跑身份”放到了“配置整理”和“Stage 2 抽象”之前
- 因为前两者一旦出错，会直接让系统在错误的部署方式或错误的实验状态下运行

## 5. 更可执行的建议

### 5.1 不要先全量拆模块，先把系统边界钉牢

建议第一批动作控制在 3 个最小增量内：

1. 新增运行 manifest
   - 每次 run 写 `run_manifest.json`
   - resume 前先校验
2. 新增显式 job/result schema
   - 不必先全面替换内部实现
   - 先把 HTTP 边界和 JSONL 记录边界类型化
3. 新增部署模式开关
   - `transport_mode = shared_fs | inline_payload | object_store`
   - 先把共享盘模式写清，再决定是否支持其他模式

这三步做完，后续重构 `app.py` 和 `windowing.py` 的风险会明显下降。

### 5.2 生命周期重构建议更保守一些

第一轮建议引入 `PipelineCoordinator` 是合理方向，但第一步不必上来就拆很多文件。

我建议的更稳妥路径是：

1. 先把 `producer_loop()` 提升成独立对象方法
2. 引入显式 `SampleStatus` 枚举，替代当前 `0/3/4/...` 的裸整数状态
3. 把“HTTP endpoint”和“runtime state mutation”分离
4. 最后再考虑拆 `scheduler.py` / `result_store.py` / `sample_state.py`

这样做的好处是：

- 先拿到更清晰的状态机
- 不会在缺少测试护栏时一次性搬太多文件

### 5.3 测试补强方向也应调整

第一轮建议补 sample 生命周期测试，我同意，而且我建议再多补一类：

1. 完整 sample 生命周期测试
2. resume/manifest 兼容性测试
3. 非共享路径场景的失败测试

第三类测试很关键，因为它能把“当前只支持共享盘”这件事从隐式事实变成显式约束。

## 6. 最终判断

这个项目的问题不是“架构已经坏掉”，而是“已经从规则工程主导阶段，进入系统边界必须说清的阶段”。

第一轮正确指出了几个大文件和几个高复杂度抽象的问题；第二轮补充后的更完整判断是：

- 当前系统的最大风险不只是代码集中，而是部署模型、任务协议和续跑身份都还比较隐式。
- 如果团队下一步继续往“更大规模、多机 worker、更长周期运行”推进，最先需要补的不是更多规则，而是把这些隐式前提正式化。
- 只有在 transport contract、runtime lifecycle、resume identity 三件事说清之后，`windowing.py` 的拆分和 Stage 2 抽象收口才会真正稳。

## 7. 本轮复核基线

本轮额外确认过的事实：

- `pytest -q` 在仓库根会因 `ModuleNotFoundError: video2tasks` 失败
- `env PYTHONPATH=src pytest -q` 通过，结果为 `235 passed in 69.68s`
- README 的“多机 worker”叙事与当前 `image_paths` 默认主链路不一致
- `create_app()` 实例化即启动后台线程，这一点确实会影响运行边界与测试边界
