# 2026-04-08 Comprehensive Review Synthesis

## 0. 说明

本文件只综合 `docs/reviews` 下既有的 18 份审计文档，不重新做一轮代码审查，也不改任何业务代码。  
判断口径以“第一轮提出问题，第二轮复核收敛”的结果为主。凡是第一轮和第二轮存在分歧的地方，本文都会明确写出：

- 第一轮怎么说
- 第二轮怎么修正
- 最终建议怎么定

这份综合文档的目标不是复述 18 份原文，而是把重复问题合并、把争议点定级、把整改顺序排清楚。

## 1. 审计输入清单

### 1.1 顶层总览文档（2 份）

- `docs/reviews/2026-04-08-project-architecture-review-round1.md`  
  第一轮全局架构总览。重点提出 runtime 边界、大文件、`VLMBackend` 抽象、配置入口和测试入口问题。

- `docs/reviews/2026-04-08-project-architecture-review-round2-subagent.md`  
  第二轮独立复核。重点把“共享文件系统依赖”“run/resume 身份缺失”“任务/结果协议未正式建模”提到了更前面。

### 1.2 Round 1 专题文档（8 份）

- `docs/reviews/round1/01-stage1-pipeline-review.md`  
  主流程、样本状态机、窗口任务、worker 提交、finalize、`segments.json` 产出。

- `docs/reviews/round1/02-windowing-and-artifacts-review.md`  
  切窗、抽帧、联系图、中间产物落盘、上传前准备、`tmp` 路径与共享文件系统依赖。

- `docs/reviews/round1/03-vlm-integration-review.md`  
  Gemini、OpenAI、RemoteAPI 接入层，请求重试、结构化提取、SSE、诊断字段。

- `docs/reviews/round1/04-stage2-postprocess-review.md`  
  merge、summary、字幕本地化、`summary_levels`、最终 JSON 稳定性、Stage 2 慢点。

- `docs/reviews/round1/05-export-review.md`  
  `annotated.mp4`、clips、音频保留、字体、回退路径、导出失败后的行为。

- `docs/reviews/round1/06-config-and-ops-review.md`  
  配置文件、README、环境变量、目录结构、运行目录、默认值漂移、密钥管理。

- `docs/reviews/round1/07-code-quality-and-architecture-review.md`  
  模块边界、编排层耦合、后端抽象、重复逻辑、大文件、日志与异常风格。

- `docs/reviews/round1/08-tests-eval-and-observability-review.md`  
  测试覆盖、评估逻辑、失败落盘、调试产物、日志、健康检查和指标缺口。

### 1.3 Round 2 专题文档（8 份）

- `docs/reviews/round2/01-stage1-pipeline-review-round2.md`  
  复核 Stage 1 主流程问题，确认真正的 correctness 缺陷，并下调部分表述过重的问题。

- `docs/reviews/round2/02-windowing-and-artifacts-review-round2.md`  
  复核共享文件系统依赖、坏图入队、`VIDEO2TASKS_DUMP_INTERMEDIATE` 语义和编码路径。

- `docs/reviews/round2/03-vlm-integration-review-round2.md`  
  复核 Gemini 重试放大、OpenAI stream-first 范围、RemoteAPI 异常处理和诊断能力。

- `docs/reviews/round2/04-stage2-postprocess-review-round2.md`  
  复核 Stage 2 契约，尤其是字幕是否属于正式 Stage 2 产物、是否写回最终 JSON。

- `docs/reviews/round2/05-export-review-round2.md`  
  复核导出契约、音频丢失、manifest 语义、`.DONE` 的含义和测试覆盖。

- `docs/reviews/round2/06-config-and-ops-review-round2.md`  
  复核配置真相、默认值、自动加载 `config.yaml`、无限空结果重试和 Gemini 文档缺口。

- `docs/reviews/round2/07-code-quality-and-architecture-review-round2.md`  
  复核状态机闭环、`create_app` 副作用、后端契约、私有函数跨文件引用和 metadata 重复。

- `docs/reviews/round2/08-tests-eval-and-observability-review-round2.md`  
  复核评估口径错误、失败落盘测试空白、产物开关语义和可观测性定级。

## 2. 全局结论

这 18 份文档收敛出来的结论很清楚：

- 项目最强的部分不是运行时，而是规则资产、切分经验、merge guard 和相当数量的规则类单元测试。
- 项目当前的真实形态不是“已经抽象清楚的通用分布式 server/worker 架构”，而是“共享磁盘、append-only 结果文件驱动的批处理流水线”。
- 第一轮大多数主判断是对的；第二轮的主要价值不是推翻，而是把真正会伤到正确性的问题提到前面，把部署契约问题从“大文件焦虑”里分离出来，再把一些性能/命名/表达过重的问题降级。
- 当前最紧急的不是再加规则，也不是先拆大文件，而是先止血：修补生命周期闭环、冻结 run/resume 身份、明确运输契约、修正最终产物契约、纠正评估口径。
- 如果这些基础问题不先处理，后续继续叠加 Stage 1/Stage 2 策略、增加 backend、增加批量任务规模，只会让维护成本和错误排查成本比模型调用成本涨得更快。

一句话概括：  
系统已经从“规则工程主导阶段”进入“系统边界必须说清的阶段”。

## 3. 跨文档收敛后的高优先级问题

### 3.1 已入库密钥和默认可加载配置源构成真实运维风险

归类：运维问题。  
补充说明：这也是直接安全风险。

收敛判断：

- `config.g3flash.yaml` 中存在真实密钥，而且已受 Git 跟踪，这是所有文档里最直接、最不应拖延的问题。
- 根目录 `config.yaml` 虽然被 `.gitignore` 忽略，但它是 `Config.load()` 的默认自动发现入口之一，实际运行中很容易被命中。
- 这不是“示例写得不规范”，而是“仓库内和默认配置源里同时存在真实凭证与隐式配置真相”。

最终建议：

- 立即轮换 `config.g3flash.yaml` 中已暴露的密钥。
- 受版本控制的配置文件一律不再保存真实密钥。
- 明确 `config.yaml` 只允许作本地私有配置，不允许继续被当作可分享模板。

### 3.2 样本生命周期不闭环，存在静默丢样本和整批卡死路径

归类：正确性问题、架构问题、测试问题。

收敛判断：

- 缺少 `Frame_*.mp4` 会静默跳过样本。
- Step A 异常会静默跳过样本。
- finalize 异常会让当前样本一直停住，并拖住后续样本。

这三条路径都说明同一个问题：  
项目已经有 `.FAILED`、`failure.json` 和统一失败落盘能力，但 producer/finalize 这几条路径没有接上它们。

最终建议：

- 这是代码层第一优先级，必须先收口到统一失败闭环。
- 每条异常路径最终都只能落到两种终态：`done` 或 `failed`。不允许灰色状态。

### 3.3 run/resume 没有“运行身份”，旧结果可以和新配置静默混用

归类：正确性问题、运维问题、架构问题、测试问题。

收敛判断：

- 当前恢复主要靠 `base_dir / subset / run_id / sample_id` 和几个固定文件名。
- 没有 config hash、代码版本、schema version、backend 摘要，也没有 manifest 校验。
- 在同一个 `run_id` 下改 prompt、改 backend、改参数、改算法，都可能直接复用旧的 `windows.jsonl`、`.DONE`、`segments.json`。

这不是“恢复逻辑不够优雅”，而是会直接破坏可复现性和续跑正确性。

最终建议：

- 每个 run 写 `run_manifest.json`。
- resume 默认先校验 manifest，不匹配就拒绝；需要强行续跑时再显式 `--force-resume`。
- `.DONE` 必须和某个 manifest 身份绑定，而不是和目录名松耦合。

### 3.4 任务传输契约与部署叙事不一致，默认链路强依赖共享文件系统

归类：架构问题、运维问题。

收敛判断：

- 当前主链路不是“server 经 HTTP 把图发给 worker”，而是“server 落本地图片，再把本地路径通过 job 交给 worker 读取”。
- README 仍在讲“一个 server 连接多台 worker 并行”，这和代码默认部署假设直接冲突。
- `VIDEO2TASKS_DUMP_INTERMEDIATE` 在 server 主路径上基本不起决定作用，因为 app 初始化时已经显式创建了 `TaskArtifactWriter`。

最终建议：

- 必须先选定真实支持的部署模式。
- 如果短期只支持 `shared_fs`，就在 README、CLI 和运维文档里把它写成硬约束。
- 如果要支持网络分离或多机 worker，就不能继续把 `image_paths` 当主链路。

### 3.5 已知坏产物会进入正式任务流，最后伪装成 worker 超时或重试

归类：正确性问题、运维问题、测试问题。

收敛判断：

- 空字节或解码失败的图片会先被写进 artifact。
- server 构任务时不检查 `decode_ok` 或 `byte_size`，照样把路径塞进 `image_paths`。
- worker 读到这类路径时才崩，server 再通过 inflight timeout 回收。

这使得本来属于 server 抽帧/产物准备阶段的问题，被拖成了 worker 慢失败和队列重排问题。

最终建议：

- 在 server 入队前直接拦截坏图。
- 发现空图或坏图时，应记为抽帧失败或样本失败，而不是继续投进 job queue。

### 3.6 Stage 1 最终输出与函数内部意图不一致

归类：正确性问题、测试问题。

收敛判断：

- `build_segments_via_cuts()` 计算了 `merged_segments`，也计算了与 merge 相关的诊断字段，但最终返回的 `segments` 被固定成 `light_segments`。
- 这会让 `diagnostics` 和实际输出长期不一致，也让部分 merge 参数对最终产物不起作用。

最终建议：

- 这是明确的 correctness 缺陷，不是风格问题。
- 需要让“最终采用哪套 segment 结果”变成清晰且可测试的分支，并让 diagnostics 如实反映。

### 3.7 Stage 2 结果契约没有定稳，字幕和 summary 的职责边界仍然含糊

归类：正确性问题、架构问题、测试问题。

收敛判断：

- 本地化字幕不会写回最终 `segments.json`。
- 字幕本地化被 `export.enabled` 和 `export.subtitles.enabled` 硬绑定，不是独立 Stage 2 产物。
- merge 失败会直接把 summary 一起降级为 identity fallback。

这说明当前 Stage 2 的问题不是“没有 fallback”，而是“哪些是正式产物、哪些只是导出附带行为，还没有统一契约”。

最终建议：

- 先回答一个产品问题：字幕是不是 Stage 2 的正式产物。
- 如果答案是“是”，就必须稳定写回最终结果，并和导出解耦。
- merge 失败不应自动取消 summary 尝试；summary 应按自己的成功与失败独立决策。

### 3.8 边界评估口径存在真实错误，会高估召回

归类：正确性问题、测试问题。

收敛判断：

- 现在一条预测边界可以重复命中多个 GT 边界。
- 这是评估指标本身的口径错误，不是“测试写少了”那么简单。

最终建议：

- 立即修正匹配逻辑，确保一条预测最多命中一个 GT。
- 把最小反例直接固化成测试。

### 3.9 运行时边界和协议模型还没有正式建起来

归类：架构问题、测试问题。

收敛判断：

- `create_app()` 既构造 Web 应用，又启动后台 producer thread，还夹带队列、状态机、持久化与条件退出逻辑。
- worker 知道 Gemini 的特殊图片载荷，`llm_merge` 又绕过统一工厂直接依赖 `OpenAIBackend`。
- job/result 协议仍然主要靠松散 dict、字符串字段和文件名约定在维持。

最终建议：

- 这是高优先级架构债务，但不应压过前面的 correctness 止血。
- 正确顺序是：先补失败闭环和契约测试，再把 runtime 与协议模型正式建起来。

### 3.10 配置真相分散，且有不安全默认行为

归类：运维问题、测试问题。

收敛判断：

- `config.example.yaml` 不能再被当作“完整配置真相”。
- 代码默认值、示例值、README、环境变量映射已经明显漂移。
- `max_empty_retries_per_job=0` 不是示例文件失误，而是代码默认行为，本身就代表无限空结果重试风险。
- `Config.load()` 还受当前工作目录影响。

最终建议：

- 配置优先级必须明确写成：`env > yaml > defaults`。
- 示例文件要么改成完整默认镜像，要么明确标注“调优示例，不代表默认值”。
- 空结果无限重试必须收紧，至少给批量跑建议一个有限默认或强提示。

## 4. 被第二轮下调或修正的问题

### 4.1 `create_app` 单体化问题

第一轮怎么说：  
把 `create_app()` 视为严重问题，认为它把 Web 应用、调度器、状态机、持久化和进程退出都揉在了一起。

第二轮怎么修正：  
问题成立，但“严重”偏高。真正更直接的风险是：

- app 构造阶段就启动 producer thread
- 测试也会直接踩到这个副作用
- `os._exit()` 是条件路径，不是默认路径

最终建议怎么定：  
把它定为 `P1` 架构整改，而不是 `P0` correctness 修复。先修样本失败闭环，再拆 runtime 生命周期。

### 4.2 `VLMBackend` 抽象泄漏问题

第一轮怎么说：  
把 backend 抽象打穿列成高优先级问题，强调 worker 已经知道 Gemini 例外输入。

第二轮怎么修正：  
问题真实存在，但更准确的说法是“共享输入模型已经在 worker 内部萌芽，却还没被提升为正式契约”。优先级应低于运输契约、run 身份和生命周期闭环。

最终建议怎么定：  
保留为 `P1`。不要继续假装所有 backend 都只吃 `np.ndarray`，但也不用先于生命周期止血去大改。

### 4.3 Stage 2 绑死 OpenAI

第一轮怎么说：  
指出 `llm_merge.py` 中 merge / summary / subtitle 默认都直接构造 `OpenAIBackend`，限制了后续扩展。

第二轮怎么修正：  
问题成立，但当前它主要影响文本后处理路径，不是主 worker 图片推理链路。更像下一阶段扩展性债务，不是眼下最先出事故的点。

最终建议怎么定：  
定为 `P2`。在 transport、run identity、runtime lifecycle 说清之前，不需要先做大范围 Stage 2 backend 抽象。

### 4.4 “导出失败后仍 `.DONE`”的定级

第一轮怎么说：  
把它列成高风险，强调样本会被标记为完成，同时可能留下半成品。

第二轮怎么修正：  
问题成立，但更准确的性质是“契约风险”。当前代码明显把主流程结果和导出结果视为两层不同成功标准。

最终建议怎么定：  
定为 `P1` 契约问题。核心不是马上把所有导出失败都升级成样本失败，而是先把 `.DONE` 的含义写清楚，或拆出独立导出级完成标记。

### 4.5 “字幕语言切换不对称”问题

第一轮怎么说：  
把 `zh/en` 行为不对称列入主要问题，认为 `en` 实际只是复用 source instruction。

第二轮怎么修正：  
这条说重了。文档和配置本来就在暗示当前更接近“中文翻译 + 英文直通”的能力，而不是双向对称翻译。

最终建议怎么定：  
定为 `P2` 命名与契约清晰度问题。短期先把能力说明写明白，不需要挤进前排修复序列。

### 4.6 `summary_levels` 与 fallback 可读性

第一轮怎么说：  
把位置相关的 `[coarse, medium, fine]` 数组和 identity fallback 的可读性列成中高问题。

第二轮怎么修正：  
方向是对的，但更像契约可读性，而不是当前行为错误。

最终建议怎么定：  
定为 `P2`。等 Stage 2 产物契约先定稳后，再把配置命名化和 fallback 状态显式化。

### 4.7 OpenAI stream-first 与兼容策略

第一轮怎么说：  
认为“非 `api.openai.com` 主机一律优先走流式”是中等级别问题。

第二轮怎么修正：  
问题存在，但只影响 `infer_text_json()` 这类文本后处理，不影响 worker 图片推理主链路；而且 stream-first 失败后仍会继续尝试更严格路径。

最终建议怎么定：  
定为 `P2`。修正范围应收窄到“文本后处理的额外延迟和更弱约束”，不要再把它写成主流程级风险。

### 4.8 “编码链路重复成本”与“性能瓶颈”表述

第一轮怎么说：  
把若干结构性热点写得比较像已经证实的性能瓶颈，例如联系图编码链路、`job_queue.pop(0)`、长 `drawtext` 过滤链等。

第二轮怎么修正：  
这些点大多成立，但更准确的表述应是“结构性热点”或“高概率成本项”，而不是已经用 profile 证实的瓶颈。尤其 Gemini 路径已经做了原始字节直传优化，不能把所有 backend 混写。

最终建议怎么定：  
统一降到 `P2`，并且加一句前提：先 profile，再优化。

## 5. 按 P0 / P1 / P2 排序的整改清单

### 5.1 P0

- 立即轮换 `config.g3flash.yaml` 中已暴露的真实密钥，并从受版本控制的配置文件中彻底移除真实凭证。
- 为缺少 `Frame_*.mp4`、Step A 异常、finalize 异常三条路径补统一失败闭环，确保样本只能落到 `done` 或 `failed`。
- 给每个 run 增加 `run_manifest.json`，包含 config hash、代码版本、schema version、backend 摘要；resume 默认先校验 manifest。
- 明确 transport mode。短期若只支持共享文件系统，就把 `shared_fs` 写成硬约束；若要支持多机 worker，就停止让 job 主链路依赖 server 本地路径。
- 在 server 入队前拦截坏产物；发现 `decode_ok=false` 或 `byte_size=0` 时，不允许继续生成正式 job。
- 修正 `build_segments_via_cuts()` 的最终输出选择，让 diagnostics 与最终 `segments` 一致。
- 修正 `score_boundary_recall(...)` 的一预测多命中错误，并固化最小反例测试。

### 5.2 P1

- 把 `create_app()` 的构造副作用和 producer 生命周期拆开，补明确的 start / stop / join 句柄。
- 定义正式的 `JobEnvelope`、`ImagePayload`、`ResultEnvelope`，去掉 worker 对 Gemini 的后端特判。
- 修正窗口结果重载时 `logical_frame_count` 的读取位置与真实落盘格式不一致的问题。
- 按已确认的 Stage 2 契约整改：字幕本地化是正式 Stage 2 产物，不能再被 `export.enabled` 绑住；应独立生成，并写回最终结果文件。
- merge 失败后仍允许 summary 独立尝试，不再把 merge 失败自动放大成 summary 降级。
- 按已确认的导出契约整改：`clips` 必须保留音频；manifest 需要区分“计划字幕”和“实际渲染事实”；`.DONE` 明确表示“本次配置要求的全部必需阶段完成”。
- 收紧 `max_empty_retries_per_job`，避免默认无限空结果重试继续拖住批量任务。
- 给 `_persist_sample_failure(...)`、坏图拦截、Stage 2 最终写盘契约、导出语义补直接测试。
- 给 OpenAI 中途断流和 RemoteAPI 请求异常补接入层捕获与统一诊断对象。

### 5.3 P2

- 停止 `llm_merge.py` 对 `windowing.py` 私有函数的跨文件依赖，再按责任拆 `windowing.py`、`llm_merge.py`。
- 把日志从 `print` 收口到统一 logger，并把关键字段固定下来；`logging.level` 也要真正控制应用自身输出。
- 重定 `config.example.yaml` 的角色：它应是最小可运行模板，不再同时承担“完整默认镜像”和“调优示例”两种职责；调优示例应移到独立配置文件。
- 把 `summary_levels` 从位置数组改成更直白的命名结构，并为 fallback 增加显式状态位。
- 把语言契约写清楚：`source instruction` 长期保证为英文；`language=en` 表示保留英文源指令，`zh` 表示正式本地化产物。
- 在做 queue、JSONL 扫描、编码链路、`instruction_timeline`、`completed_dispatch_ids` 等性能优化前，先做 profile，不要继续凭结构直觉排优先级。
- 给 `/health`、artifact 路径清洗、评估文件读取、`BoundaryRecallSummary.to_dict()` 等补契约测试。

## 6. 问题分类

### 6.1 正确性问题

- 样本静默跳过和 finalize 卡死。
- `build_segments_via_cuts()` 最终输出与诊断不一致。
- run/resume 在不同配置和代码版本下静默复用旧结果。
- 坏图进入 job queue，最后伪装成 worker timeout。
- merge 失败直接压掉 summary。
- 边界评估一预测多命中。
- 窗口结果重载时 `logical_frame_count` 二次校验位置不一致。

### 6.2 性能问题

- Gemini 重试链和 worker 本地重试叠加，可能把单任务拖得很长。
- `job_queue.pop(0)`、全量 JSONL 扫描、逐帧 `instruction_timeline`、重复编解码、长 `drawtext` 过滤链等都是结构性热点。
- `completed_dispatch_ids` 等常驻结构会随任务量单调增长。
- 但这些大多还属于“应先 profile 的热点”，不是全部都已经坐实为当前第一优先级瓶颈。

### 6.3 架构问题

- 系统真实依赖共享磁盘，但对外叙事仍像通用多机架构。
- `create_app()` 同时承担 app 构造、调度、状态机、持久化和条件退出。
- job/result/image 协议没有正式建模，仍依赖松散 dict 和字符串约定。
- backend 抽象被具体实现打穿，worker 和 Stage 2 都知道具体后端私有差异。
- `windowing.py`、`llm_merge.py` 过大，并存在跨文件 import 私有函数。

### 6.4 运维问题

- 真实密钥出现在仓库配置中。
- `config.yaml` 是本地忽略文件，但又是默认自动发现入口。
- `config.example.yaml`、README、代码默认值、环境变量映射长期漂移。
- `tmp/` 不是普通缓存，而是 worker 实际输入源之一，误清理会直接打坏任务。
- `.DONE`、导出成功、Stage 2 成功目前不是同一回事，但文档与目录约定没有讲清。

### 6.5 测试问题

- producer/finalize 失败闭环缺少直接测试。
- `_persist_sample_failure(...)` 没有直接锁定测试。
- 评估模块测试过薄，没覆盖反例和异常输入。
- artifact 路径清洗、坏图拦截、`artifact_manifest_path` 契约没有被固定。
- app 级 Stage 2 写盘契约没有直接测试。

## 7. 建议的修复顺序

### 第一步：先止血

- 处理密钥。
- 收紧无限空结果重试。
- 修掉样本静默跳过、finalize 卡死、评估口径错误。

原因：  
这些问题不是“代码不优雅”，而是已经会让结果失真、任务卡死或运维暴露。

### 第二步：冻结运行身份和部署前提

- 增加 `run_manifest.json`。
- 明确 `shared_fs` 还是 `network-separated`。
- 把 README 和配置说明同步到真实行为。

原因：  
不先把系统边界说清楚，后面的所有重构都会继续建立在隐式前提上。

### 第三步：补上输出契约

- 落实 `.DONE` 的新定义：它表示“本次配置要求的全部必需阶段完成”。
- 落实 Stage 2 契约：字幕本地化是正式 Stage 2 产物，应独立于 export 生成并写回最终结果。
- 落实结果文件分工：`segments.json` 只承载切分与 Stage 2 文本结果；run/export/fallback 状态写进独立 manifest，不再混装成一个“最终真相”文件。
- 落实导出契约：`clips` 必须保留音频，export manifest 要记录实际渲染事实。

原因：  
现在很多争议都不是“算错了”，而是“写出来的结果到底代表什么”没有定稳。

### 第四步：补测试护栏

- failure 闭环测试
- Stage 2 app 级契约测试
- eval 反例测试
- artifact 坏路径与路径清洗测试

原因：  
没有这些测试，后面的 runtime 和协议重构风险太高。

### 第五步：再做架构收口

- 拆 `create_app()` 生命周期副作用
- 建正式 job/result/image 协议
- 去掉 worker backend 特判
- 切断跨文件私有函数依赖

原因：  
这是必要工作，但它应该建立在 P0 correctness 问题已经止血、契约已经固定之后。

### 第六步：最后做性能优化

- 对 queue、JSONL、编码链路、长视频 timeline、Gemini 重试链做 profile
- 依据实测再选优化项

原因：  
第二轮已经明确指出，多数“性能瓶颈”目前还只是结构性热点，不宜先凭感觉做大改。

## 8. 已确认的决策（2026-04-08）

以下事项已有人为拍板，不再作为开放问题保留：

- 部署模式按“单机共享盘”定义。当前主链路允许依赖 `image_paths` 与共享文件系统，不再对外暗示已支持多机、多容器、不同挂载点的 worker。
- `.DONE` 的定义固定为：“本次配置要求的全部必需阶段完成”。它不再表示“主流程大致成功”或“目录里已经有部分产物”。
- 字幕本地化是正式 Stage 2 产物，不只是导出前的临时步骤。后续实现应把它与 export 解耦，并稳定写回最终结果层。
- `segments.json` 的职责需要收窄。它应承载切分结果和 Stage 2 文本结果；run 身份、export 状态、fallback 状态等运行信息应写入独立 manifest，而不是继续全压进一个“最终真相”文件。
- `source instruction` 长期保证为英文。基于这个前提，`language=en` 代表保留英文源指令，`language=zh` 代表正式中文本地化输出。
- `clips` 导出必须保留音频。凡是不保留音频的路径，都不应再被当成“正常成功导出”。
- resume 默认不允许跨配置、跨 prompt、跨 backend 继续跑。若确需继续，必须显式使用 `--force-resume`，并把差异和强制续跑事实写入 manifest。
- `config.example.yaml` 的角色重定为“最小可运行模板”。调优示例、实验配置和模型特定配方应拆到独立配置文件中，不再混在 example 里。

这些决策会直接改变后续整改优先级：与其继续讨论定义，不如把对应契约落进代码、manifest 和测试。

## 9. 最终判断

18 份文档合起来，没有得出“项目已经不可维护”这种结论；相反，它们清楚地表明：  
项目的规则工程基础并不弱，真正的问题在于系统边界、运行边界和结果边界还没有被正式说清。

因此，最合理的路线不是“立刻重写”，而是：

- 先修 correctness 和运维暴露点
- 再冻结 run / transport / output 契约
- 再补测试护栏
- 最后再做 runtime 和协议层重构

这也是 18 份文档收敛后最一致的意见。
