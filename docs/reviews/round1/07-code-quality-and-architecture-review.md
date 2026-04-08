# Round 1 代码质量与架构审计 07

更新时间：2026-04-08  
范围：`src/video2tasks/config.py`、`src/video2tasks/cli/*`、`src/video2tasks/server/*`、`src/video2tasks/worker/*`、`src/video2tasks/vlm/*`、`src/video2tasks/prompt.py`。  
补充说明：本次只做审计，不改代码。测试目录只做少量抽查，用来判断是否有兜底，不作为主要审计对象。

## 1. 审计范围

本轮重点看下面几类问题：

- 模块边界是否清楚，是否把 HTTP、状态机、算法、文件落盘、外部 API 协议揉在一起。
- 模块之间是否通过稳定接口协作，还是靠私有函数、字典字段和约定俗成硬连。
- 是否存在明显重复逻辑，后续加一个能力就要改很多处。
- 异常处理是否闭环，失败后是明确失败，还是静默跳过、卡死、或者直接把进程打掉。
- 命名是否能支撑多人维护，长期新增功能时是否容易继续变成“大文件 + 更多 if”。

本次重点模块：

- 编排入口：`src/video2tasks/server/app.py`、`src/video2tasks/worker/runner.py`
- 核心算法：`src/video2tasks/server/windowing.py`、`src/video2tasks/server/llm_merge.py`
- 外部模型接入：`src/video2tasks/vlm/base.py`、`src/video2tasks/vlm/openai_api.py`、`src/video2tasks/vlm/gemini_api.py`、`src/video2tasks/vlm/remote_api.py`
- 配置与入口：`src/video2tasks/config.py`、`src/video2tasks/cli/*.py`

一些体量数据，能直接说明维护压力已经不小：

- `src/video2tasks/server/windowing.py`：2586 行
- `src/video2tasks/server/llm_merge.py`：1648 行
- `src/video2tasks/server/app.py`：1285 行
- `src/video2tasks/prompt.py`：782 行
- `src/video2tasks/config.py`：737 行

## 2. 系统优点

先说做得好的地方，这些是后面重构时应该保留的。

1. 目录分层至少有基本雏形。`cli`、`server`、`worker`、`vlm` 是分开的，说明项目一开始不是完全无边界地写在一起。

2. 配置模型用了 Pydantic，并且字段校验比较实在。比如 `src/video2tasks/config.py:38-61`、`src/video2tasks/config.py:77-83`、`src/video2tasks/config.py:154-160`、`src/video2tasks/config.py:382-417`，这类校验能提前拦住不少坏配置。

3. 有一定单元测试基础。`tests/server/test_windowing.py`、`tests/server/test_llm_merge.py`、`tests/server/test_exporter.py`、`tests/server/test_app_retry.py`、`tests/worker/test_runner.py`、`tests/vlm/test_openai_api.py` 这些文件说明作者至少在给核心模块补回归测试。

4. 中间产物写盘已经抽成了单独模块。`src/video2tasks/server/task_artifacts.py:71-198` 这一块边界相对清楚，是目前少数“职责单一”的模块。

5. Structured output 的基础约束是存在的。`src/video2tasks/vlm/base.py:44-116` 的 `normalize_task_window_result` 虽然被多处重复调用，但它至少给系统建立了统一的结果底线。

## 3. 发现的问题

下面按严重程度排序。

### 3.1 严重：样本状态机的异常路径不闭环，会导致静默丢样本或卡死在单个样本上

最危险的问题在 `src/video2tasks/server/app.py` 里，不在模型算法，而在编排异常路径。

第一处：

- `src/video2tasks/server/app.py:817-821`

这里 Step A 失败后只做了：

- 打印错误
- `cur_idx += 1`

但没有：

- 把 `sample_status[sid]` 标成失败
- 写 `.FAILED`
- 写 `failure.json`
- 统一复用 `_persist_sample_failure`

结果就是这个样本会被直接跳过，但系统里没有明确失败记录。后面 `dataset_idx` 仍然可能推进，`_count_failed_samples` 也统计不到这类异常，因为它只数状态值 `4`，见 `src/video2tasks/server/app.py:143-149`。这类失败会变成“既没成功，也没失败”的灰色状态。

第二处：

- `src/video2tasks/server/app.py:1265-1266`

Finalize 阶段异常时，只打印 `[Err-Finalize]`，没有：

- 标记失败
- 推进 `cur_idx`
- 落失败文件

这意味着如果某个样本在 finalize 上是确定性异常，它会一直停留在 `sample_status == 2` 的阶段，被主循环反复碰到，形成“单样本卡死”。这比普通失败更糟，因为会拖住整批任务。

现有测试也没有覆盖这类异常路径。`tests/server/test_app_retry.py:27-255` 主要覆盖的是重试、过期 dispatch、空结果等提交链路，没有覆盖 producer/finalize 出错后的样本状态闭环。

结论：这里不是“日志不够好”，而是状态机本身不完整。

### 3.2 严重：`create_app` 同时承担 Web 应用、调度器、状态机、持久化和进程退出，边界已经失真

`src/video2tasks/server/app.py` 现在是整个系统最重的耦合点。

直接证据：

- `create_app` 从 `src/video2tasks/server/app.py:156` 开始，到 `src/video2tasks/server/app.py:1274` 才结束。
- 它里面同时做了：
  - 数据集解析和目录创建：`src/video2tasks/server/app.py:160-175`
  - 队列、inflight、重试计数、dispatch 状态管理：`src/video2tasks/server/app.py:165-175`
  - 路径拼装和 JSONL 持久化：`src/video2tasks/server/app.py:186-404`
  - HTTP API：`src/video2tasks/server/app.py:489-582`
  - 样本生产和 finalize 状态机：`src/video2tasks/server/app.py:585-1268`
  - 后台线程启动：`src/video2tasks/server/app.py:1270-1272`

这会带来几个长期问题：

1. `create_app` 不再是“构造 FastAPI app”，而是“顺便把整个批处理系统跑起来”。  
   这让测试、复用、替换运行时都变得困难。

2. App 构造本身带副作用。  
   只要调用 `create_app(config)`，producer 线程就启动了，见 `src/video2tasks/server/app.py:1270-1274`。这会让“导入应用”和“启动业务”无法分开。

3. 生命周期管理很粗暴。  
   `auto_exit_after_all_done` 打开后，后台线程直接 `os._exit(exit_code)`，见 `src/video2tasks/server/app.py:655-665`。这会绕过正常的 FastAPI/Uvicorn 关闭流程、日志 flush、资源释放和测试收尾。

这里已经不是单一职责被破坏，而是几个本该独立的运行层概念被写进一个函数里：

- Web 服务
- 作业编排器
- 持久化仓库
- 后台 worker 协调
- 进程生命周期控制

### 3.3 高：后端抽象接口已经失真，worker 被迫知道具体后端细节

`VLMBackend` 看起来是统一抽象，但现在这个抽象已经被特殊情况打穿。

直接证据：

- 基类把 `infer` 定义为 `List[np.ndarray] -> Dict[str, Any]`，见 `src/video2tasks/vlm/base.py:13-26`
- 但 worker 在 `src/video2tasks/worker/runner.py:306-311` 里，为了 Gemini 会把 `images` 改成 `{"raw_bytes": ..., "mime_type": ...}` 的字典列表
- `GeminiBackend.infer` 仍然保留 `List[np.ndarray]` 签名，见 `src/video2tasks/vlm/gemini_api.py:287-290`，但内部已经偷偷兼容字典负载，见 `src/video2tasks/vlm/gemini_api.py:37-55`、`src/video2tasks/vlm/gemini_api.py:341-357`

这说明现在的真实接口不是“统一图像数组”，而是：

- 有的 backend 吃 `np.ndarray`
- 有的 backend 吃原始图片字节
- worker 需要知道谁吃什么

这会直接破坏两个设计目标：

1. `worker` 本来应该只依赖统一 backend 协议，现在却必须为特定 backend 写分支。
2. 新增一个 backend 时，不能只改 `vlm/*`，还要反向修改 worker 编排逻辑。

同类问题还不止一处：

- `llm_merge` 没走统一 factory，而是直接依赖 `OpenAIBackend`，见 `src/video2tasks/server/llm_merge.py:11-19`
- 配置层也把 merge backend 限死为 `openai`，见 `src/video2tasks/config.py:382-388`

说明项目里现在其实存在两套“后端扩展方式”：

- 第一套：`worker` + `vlm.factory`
- 第二套：`llm_merge` 直接 new 具体 backend

这会让未来扩展越来越乱。

### 3.4 高：结果规范化、JSON 提取和重试策略散在多层，职责分布不清

现在“谁负责把模型返回值变成系统可用结果”并不清楚，至少分散在三层：

1. backend 自己规范化  
   `src/video2tasks/vlm/openai_api.py:631-658`  
   `src/video2tasks/vlm/gemini_api.py:210-215`

2. worker 再规范化一遍  
   `src/video2tasks/worker/runner.py:337-353`

3. server 读取落盘结果时又规范化一遍  
   `src/video2tasks/server/app.py:114-140`、`src/video2tasks/server/app.py:321-325`、`src/video2tasks/server/app.py:537-540`

JSON 提取逻辑也已经开始复制：

- `src/video2tasks/vlm/openai_api.py:29-55`
- `src/video2tasks/vlm/gemini_api.py:70-89`
- `src/video2tasks/vlm/remote_api.py:21-39`

副作用是：

- 一处规则变更，很容易漏改另两处
- 出问题时，不容易判断到底是哪一层吞掉了错误
- backend 和 worker、server 的边界越来越模糊

这类问题短期看只是“有点重复”，长期会演变成“同一个输入在不同路径下处理结果不一样”。

### 3.5 中：核心算法文件过大，且模块边界已经被私有函数穿透

`windowing.py` 和 `llm_merge.py` 的体量都已经超过“一个人短时间能安全 hold 住”的范围。

最明显的是 `windowing.py`：

- 同一个文件里既有视频读写和接触表拼图：`src/video2tasks/server/windowing.py:642-1047`
- 又有大量指令文本启发式：`src/video2tasks/server/windowing.py:1049-1638`
- 又有 segment merge / cleanup / drift split：`src/video2tasks/server/windowing.py:1639-2296`
- 最后还有总装函数 `build_segments_via_cuts`：`src/video2tasks/server/windowing.py:2297-2586`

`llm_merge.py` 也是类似情况：

- schema 校验
- boundary guard 排序
- coarse merge 共识
- hierarchy summary
- subtitle localization
- 最终 postprocess 总装

更糟的是，`llm_merge.py` 直接 import 了 `windowing.py` 的私有函数：

- `src/video2tasks/server/llm_merge.py:14-19`

也就是：

- `_boundary_support_between`
- `_has_distinct_sequence_markers`
- `_should_split_on_instruction_drift`

这些名字前面已经带下划线，按正常约定应该是“文件内部实现细节”。现在另一个大模块直接跨文件依赖这些私有函数，说明模块边界已经不稳。以后只要改一个私有实现，就可能把另一个模块打坏。

### 3.6 中：重复的 job metadata 组装和配置映射已经开始失控

`server/app.py` 里构建 job metadata 的代码重复很多次，而且字段集几乎一样，只是 job_type 不同。

重复片段至少有这些：

- window job：`src/video2tasks/server/app.py:775-792`
- refinement window job：`src/video2tasks/server/app.py:924-941`
- boundary refinement job：`src/video2tasks/server/app.py:1035-1054`
- segment label job：`src/video2tasks/server/app.py:1127-1144`

这些块现在已经有两个明显问题：

1. 一旦元数据字段要变，必须多处同步修改。
2. 不同 job 类型很容易悄悄出现字段不一致，后面排查会非常慢。

配置层也有同样趋势。`src/video2tasks/config.py:498-695` 的 `_collect_env_override_data()` 基本是一整段手写映射表。这个方法短期可用，但随着配置项增长，它会持续放大两个问题：

- 忘记补环境变量映射
- 补了映射但和模型字段、文档、默认值不同步

这已经不是“代码有点长”，而是维护方式本身在鼓励 drift。

### 3.7 中：核心路径上大量 `print`、宽泛异常和短变量名，排障成本偏高

几个典型信号：

- `server/app.py` 和 `worker/runner.py` 大量直接 `print(...)`，见 `src/video2tasks/server/app.py:554-665`、`src/video2tasks/server/app.py:743-1266`、`src/video2tasks/worker/runner.py:220-389`
- 多处 `except Exception` 后直接吞掉或转空结果，见 `src/video2tasks/worker/runner.py:64-65`、`src/video2tasks/worker/runner.py:83-84`、`src/video2tasks/worker/runner.py:97-100`、`src/video2tasks/server/task_artifacts.py:31-34`
- 关键编排代码里充满 `sid`、`st`、`cnt`、`mp4s`、`by_wid` 这类短名，见 `src/video2tasks/server/app.py:669-715`、`src/video2tasks/server/app.py:834-854`

这些问题单独看都不算致命，但叠在一起会有明显后果：

- 线上日志不好检索，也不容易挂监控
- 失败原因没有稳定错误类型
- 新人需要先翻上下文猜变量含义，再敢改代码

现在项目还没大到必须上复杂框架，但已经大到不适合继续靠 `print + dict + 猜字段` 来维持。

## 4. 长期风险

如果不处理上面这些问题，后面最容易发生的是下面几件事：

1. 新增一个 job 类型时，要同时改 `server/app.py`、`worker/runner.py`、`prompt.py`、结果落盘逻辑和 finalize 逻辑，改动面太大，回归风险很高。

2. 新增一个模型后端时，不只是加一个 backend 文件，还会反向污染 worker、merge pass 甚至配置层，导致“抽象层越加越多，真正统一接口越来越少”。

3. 任何一次 finalize 阶段的小改动，都可能把整批任务卡死在某个样本上，或者悄悄漏掉样本，但退出码和失败统计看起来还正常。

4. `windowing.py` 和 `llm_merge.py` 继续长下去后，后续维护者会更倾向于复制一段逻辑再改，而不是安全复用已有逻辑，重复会越来越多。

5. 配置项继续增加后，YAML、环境变量、README、代码默认值四处漂移会更明显，最后排障时间会远大于开发时间。

## 5. 重构建议

这里不谈“大改重写”，只谈能落地的拆法。

### 5.1 先把编排层从 FastAPI app 里拆出来

建议新增一个明确的 orchestrator 层，至少拆成下面几块：

- `JobQueueService`：只负责队列、inflight、dispatch、retry 计数
- `SamplePipelineRunner`：只负责单个 sample 的状态推进
- `ResultRepository`：只负责 JSONL、segments、marker 文件读写
- `FastAPI` 路由层：只负责 HTTP 输入输出

这样 `create_app` 就回到“构造 Web 应用”这件事本身。

### 5.2 把 sample 状态做成显式状态机

不要再靠 `sample_status[sid] == 0/2/3/4` 和大段 `if` 推进。  
建议把状态和失败原因收敛成固定对象，例如：

- `queued`
- `windowing`
- `refinement`
- `boundary_refinement`
- `segment_labeling`
- `finalizing`
- `done`
- `failed`

并且规定：

- 每个阶段异常都必须落统一失败记录
- 每个阶段退出都必须有明确状态转移
- 不允许“跳过但不记失败”的灰色路径

### 5.3 统一 backend 输入输出协议

建议把 backend 契约收紧成两层：

1. `InferenceInput`
   - 统一承载文本 prompt、图像数组、原始字节、mime、逻辑帧数
2. `InferenceResult`
   - 统一承载 raw payload、normalized payload、诊断信息

worker 只和这两个对象打交道，不再知道某个 backend 是走 ndarray 还是 raw bytes。

### 5.4 把 `windowing.py` 至少拆成四块

建议先按责任拆，不要按“技术层”拆：

- `frame_extraction.py`
- `boundary_voting.py`
- `instruction_heuristics.py`
- `segment_assembly.py`

`llm_merge.py` 也建议至少拆成：

- `merge_validation.py`
- `merge_guard.py`
- `summary_builder.py`
- `subtitle_localizer.py`
- `postprocess.py`

同时禁止跨文件 import 私有函数；能跨模块复用的能力，升级成公开 helper。

### 5.5 收敛重复逻辑

优先抽公共函数的不是算法，而是重复最明显、出错最容易扩散的几类逻辑：

- 模型 JSON 提取与 structured payload 解析
- job metadata 组装
- logical frame count 推导
- config env override 映射
- 重试策略和错误分类

这些一旦收口，后面很多新增功能就不需要改四五个文件。

### 5.6 把日志和异常语义补齐

建议至少做到：

- 用标准 `logging` 替代核心路径 `print`
- 定义少量稳定异常类型，例如 `TaskDecodeError`、`BackendProtocolError`、`SampleFinalizeError`
- 日志里固定带 `subset`、`sample_id`、`task_id`、`dispatch_id`

这一步不花哨，但会直接改善线上排障效率。

## 6. 实施优先级

### P0：必须先做

1. 修补 `server/app.py` 的样本异常闭环，确保每条异常路径都明确进入 `failed` 或 `done`，不能再出现静默跳过和 finalize 卡死。

2. 去掉后台线程中的 `os._exit`，把退出动作改成可控的生命周期信号。

3. 把 app 构造和 producer 启动分开，避免 `create_app()` 自带后台副作用。

### P1：下一轮应该做

1. 明确 backend 统一协议，消除 worker 里的 backend 特判。

2. 抽出 `ResultRepository` 和 job metadata builder，先把最重复的编排代码收口。

3. 拆 `windowing.py` 和 `llm_merge.py`，先从“视频读写”和“文本启发式”这两个边界开始拆。

### P2：持续治理

1. 收敛配置 env override 映射方式，减少手写同步点。

2. 统一日志格式和异常类型。

3. 清理短变量名和隐式字典协议，把核心路径改成更明确的数据对象。

## 7. 总结

这个项目现在的主要问题，不是算法写得不够聪明，而是编排层已经开始超过“单文件手工维护”的承载范围。

最先要处理的不是再加一个 prompt 或再换一个模型，而是把下面三件事做实：

1. 样本状态机异常必须闭环。
2. FastAPI app 和后台批处理编排必须拆开。
3. backend 契约必须重新收紧，不能继续让 worker 知道后端内部细节。

这三件事一旦做了，后面的代码质量问题才会开始进入可持续维护状态。
