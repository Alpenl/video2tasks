# Round 1 测试、评估与可观测性审计 08

更新时间：2026-04-08  
范围：`tests/`、`src/video2tasks/eval/`、服务端/worker 日志输出、调试产物落盘、健康检查与失败诊断。  
约束：本次只做审计，不改代码；只新增本文档。

## 审计范围

这次重点看了下面几块：

- 评估代码：`src/video2tasks/eval/official_boundaries.py`
- 评估测试：`tests/eval/test_official_boundaries.py`
- 服务端重试、失败记录、健康检查：`src/video2tasks/server/app.py`、`tests/server/test_app_retry.py`
- 调试产物和中间图片落盘：`src/video2tasks/server/task_artifacts.py`、`src/video2tasks/server/windowing.py`、`tests/server/test_task_artifacts.py`
- worker 输出与相关测试：`src/video2tasks/worker/runner.py`、`tests/worker/test_runner.py`
- 现有“诊断字段”测试：`tests/server/test_exporter.py`、`tests/server/test_llm_merge.py`、`tests/server/test_llm_summary.py`、`tests/vlm/test_openai_api.py`

## 现状

先说整体判断：

- 测试不是空白。`worker` 的重试、解码失败、提交失败重试有覆盖，见 `tests/worker/test_runner.py`。
- 服务端对空结果、超时、终态错误写入 `windows.jsonl` 也有覆盖，见 `tests/server/test_app_retry.py`。
- `llm_merge`、`llm_summary`、OpenAI 适配层的“诊断字段”测试相对完整，说明项目已经开始重视“出问题时要留痕”。
- 但是，评估模块很薄，现在只有 2 个测试，而且都是顺路通过的正向例子，见 `tests/eval/test_official_boundaries.py:4-33`。
- 应用级日志基本还是 `print`，不是统一日志；`logging.level` 只明显接到 `uvicorn.run(...)`，见 `src/video2tasks/config.py:425-436`、`src/video2tasks/server/app.py:1277-1284`。
- 调试图片和 manifest 的落盘路径已经成了默认运行路径的一部分，不只是“临时调试开关”。

## 发现的问题

按严重程度排序如下。

### 高：边界评估会把同一个预测边界重复算成多个命中，召回率会被抬高

位置：

- `src/video2tasks/eval/official_boundaries.py:49-53`
- `src/video2tasks/eval/official_boundaries.py:68-86`
- `tests/eval/test_official_boundaries.py:4-33`

问题说明：

- 现在的做法是“对每个 GT 边界，找最近的预测边界”，但不会把这个预测边界标记为“已经用过”。
- 结果就是一个预测点可以同时命中两个很近的 GT 点。
- 我用最小例子验证过：`gt=[100, 103]`、`pred=[101]`、`tolerance=5`，当前结果是 `hit_count=2`、`recall=1.0`。这明显会高估表现。

影响：

- 评估结果会偏乐观。
- 如果后面用这个指标比较模型、提示词、窗口参数，结论会失真。

为什么测试没有拦住：

- 现有评估测试只覆盖“额外预测点不影响结果”和“阈值内外命中”，没有覆盖“一对多重复命中”的反例，见 `tests/eval/test_official_boundaries.py:4-33`。

建议优先级：`P0`

### 中高：`logging.level` 基本没有真正管到应用自身输出，线上排障会很乱

位置：

- 配置定义：`src/video2tasks/config.py:425-436`
- server 里大量 `print`：`src/video2tasks/server/app.py:554-563`、`src/video2tasks/server/app.py:596-602`、`src/video2tasks/server/app.py:634-645`、`src/video2tasks/server/app.py:817-820`、`src/video2tasks/server/app.py:1263-1284`
- worker 里大量 `print`：`src/video2tasks/worker/runner.py:222-225`、`src/video2tasks/worker/runner.py:239-267`、`src/video2tasks/worker/runner.py:346-389`
- 现有输出测试主要是直接断言 stdout：`tests/worker/test_runner.py:40-78`、`tests/worker/test_runner.py:527-574`

问题说明：

- 配置里有 `logging.level`，但代码里大部分关键事件不是 logger，而是直接 `print`。
- 现在能确定接到日志等级的地方，主要是 `uvicorn.run(... log_level=...)`，见 `src/video2tasks/server/app.py:1280-1284`。
- 这意味着把等级调到 `ERROR`，应用自己的 `[Warn]`、`[Progress]`、`[Done]` 仍然会照常刷出来。

影响：

- 不能按等级收敛日志量。
- stdout 文本格式不稳定，不适合做机器采集。
- server 和 worker 的日志风格分散，排障时很难按任务、样本、dispatch 过滤。

为什么测试没有拦住：

- 现有测试验证的是“某句打印在 stdout 里”，而不是“日志等级是否生效”或“关键字段是否稳定可采集”。
- 没看到 `caplog` 或 server 端日志行为测试。

建议优先级：`P1`

### 中：调试产物默认一直落盘，没有配置开关、配额或清理验证，容易带来磁盘和隐私压力

位置：

- server 启动时总是建 writer：`src/video2tasks/server/app.py:173-174`
- 建 job 时总走带产物的提取路径：`src/video2tasks/server/app.py:449-468`
- 抽帧默认 `persist_artifacts=True`：`src/video2tasks/server/windowing.py:879-907`、`src/video2tasks/server/windowing.py:945-980`
- 实际写盘逻辑：`src/video2tasks/server/task_artifacts.py:112-150`
- 现有测试只覆盖 happy path：`tests/server/test_task_artifacts.py:13-102`

问题说明：

- 现在中间图片、manifest 不是“必要时打开”，而是默认路径的一部分。
- 默认目录还是仓库根附近的 `tmp`，来自环境变量 `VIDEO2TASKS_TMP_DIR`，见 `src/video2tasks/server/app.py:173`。
- 代码里没有看到明确的配置项去关闭落盘，也没有看到保留天数、大小上限、自动清理。

影响：

- 批量跑时会持续堆积图片文件。
- 原始帧或 contact sheet 可能长时间留在磁盘。
- 出现磁盘满、误删正在使用的文件、隐私数据残留时，测试目前拦不住。

为什么测试没有拦住：

- 现在只测了“能写出来”和“source/frame_ids 对不对”。
- 没测关闭路径、环境变量路径、非法 base64、路径清洗、重复运行清理、磁盘压力下的行为。

建议优先级：`P1`

### 中低：评估测试覆盖太窄，文件读取和边界条件几乎没保护

位置：

- 评估代码：`src/video2tasks/eval/official_boundaries.py:32-46`、`src/video2tasks/eval/official_boundaries.py:61-98`
- 现有测试：`tests/eval/test_official_boundaries.py:4-33`

问题说明：

- 现在评估测试只有 2 个。
- 下面这些入口都没看到测试：
  - `predicted_boundary_frames_from_segments_file(...)`
  - `official_boundary_frames_from_file(...)`
  - `BoundaryRecallSummary.to_dict()`
  - `tolerance_frames < 0` 的异常分支
  - 空 GT、空预测、重复预测、乱序输入

影响：

- 评估脚本最容易坏的其实不是“正常样例”，而是空文件、格式变动、边界点重复、排序问题。
- 一旦后面有人把评估接到批量跑流程里，这些洞会直接变成排障时间。

建议优先级：`P2`

### 低：有一部分测试依赖真实睡眠时间和后台线程，套件会慢，也更容易偶发失败

位置：

- `tests/server/test_app_retry.py:223-249`
- 后台 producer 线程：`src/video2tasks/server/app.py:584-667`、`src/video2tasks/server/app.py:1270-1272`

问题说明：

- `test_timeout_exhaustion_records_terminal_error` 直接 `time.sleep(1.2)` 两次。
- 这个测试同时依赖后台线程轮询超时逻辑。
- 在忙机器、CI 抖动、并发审计/测试同时跑时，这类用真时间等状态变化的测试比较容易拖慢或偶发不稳。

影响：

- 套件时长增加。
- 出现“本地过、CI 偶发红”的概率更高。

建议优先级：`P3`

### 低：服务端可观测性还停留在“能活着就返回 ok”，没有真正的运行视图

位置：

- 健康检查：`src/video2tasks/server/app.py:580-582`
- 失败报告写盘：`src/video2tasks/server/app.py:405-428`

问题说明：

- `/health` 只返回 `{"status": "ok"}`。
- 没有现成的队列长度、inflight 数、重试次数、失败样本数、空结果次数。
- 全仓搜索也没看到 `/metrics`、Prometheus、OTel 之类入口。
- 失败会写 `failure.json`，但我没看到对应测试去校验文件结构和关键字段。

影响：

- 服务“活着”不代表“还能处理任务”。
- 出问题时只能靠 stdout 和手动翻目录。

建议优先级：`P2`

## 建议的测试补位

建议按下面顺序补。

### P0 先补

- 给 `score_boundary_recall(...)` 加“一条预测不能重复命中两个 GT”的测试。
- 给评估模块加“空 GT / 空预测 / 重复预测 / 乱序输入”的测试。
- 给 `tolerance_frames < 0` 加异常测试。

### P1 再补

- 给日志行为加测试，不再只看“某句字符串有没有打印”，而是看：
  - `logging.level=ERROR` 时，普通进度信息是否还会冒出来
  - 失败事件是否稳定带 `task_id`、`sample_id`、`dispatch_id`
- 给调试产物加测试：
  - 关闭落盘时不写文件
  - `VIDEO2TASKS_TMP_DIR` 生效
  - 非法 base64 时 `decode_ok`、manifest 是否符合预期
  - 路径清洗是否拦住奇怪输入

### P2 补齐

- 给 `predicted_boundary_frames_from_segments_file(...)` 和 `official_boundary_frames_from_file(...)` 各加 1 到 2 个文件读取测试。
- 给 `BoundaryRecallSummary.to_dict()` 加序列化测试，避免后面接 CLI/报表时字段漂移。
- 给 `_persist_sample_failure(...)` 加测试，确认 `failure.json`、`.FAILED`、旧的 `.DONE`/`segments.json` 清理逻辑都对。
- 给 `/health` 至少加一条接口测试，先把契约固定住。

### P3 可选优化

- 把 `tests/server/test_app_retry.py:223-249` 这类真睡眠测试改成假时间或更短轮询驱动的测试。
- 给长链路测试加统一的超时保护，减少 CI 卡住时的排障成本。

## 建议的日志/指标补位

这一块我建议少一点花样，先做最实用的。

### 日志

- 统一用 logger，不要继续混着 `print`。
- 每条关键日志至少带这些字段：
  - `event`
  - `task_id`
  - `dispatch_id`
  - `subset`
  - `sample_id`
  - `job_type`
  - `attempt`
- 把下面这些事件固定下来，后面测试也好写：
  - job 下发
  - worker 收到 job
  - 本地推理失败
  - 空结果重试
  - 超时重试
  - 终态失败落盘
  - sample finalize 成功

### 指标

- server 侧至少补 6 个数字：
  - 当前 `job_queue` 长度
  - 当前 `inflight` 数
  - 累计空结果重试次数
  - 累计超时重试次数
  - 累计终态失败样本数
  - 累计完成样本数
- worker 侧至少补 4 个数字：
  - 推理请求数
  - 推理失败数
  - 提交重试数
  - 图片解码失败数

### 调试产物

- 给调试图片落盘增加明确的总开关。
- manifest 里补两个字段会更实用：
  - 写盘原因，比如 `job_dispatch_debug` / `retry_debug`
  - 来源时间或 dispatch 关联信息
- 失败报告 `failure.json` 建议固定包含：
  - `reason`
  - `subset`
  - `sample_id`
  - `failed_window_ids` 或 `failed_boundary_ids`
  - 对应的 `terminal_error`

## 优先级建议

按处理顺序，我建议这样排：

| 优先级 | 事项 | 原因 |
| --- | --- | --- |
| P0 | 修正边界评估重复命中问题，并补对应测试 | 这是评估口径错误，会直接误导效果判断 |
| P1 | 统一日志出口，别再让 `logging.level` 成为“看起来有用” | 这是日常排障效率问题，也是后续接指标的前提 |
| P1 | 给调试产物增加开关和覆盖测试 | 这是磁盘、隐私、运行成本问题 |
| P2 | 补齐评估文件读取、序列化、异常路径测试 | 成本不高，收益直接 |
| P2 | 给失败报告和健康检查补契约测试，并补基础运行指标 | 方便批跑和线上排障 |
| P3 | 收拾真睡眠测试，降低套件抖动 | 影响较小，但长期值得做 |

## 一句话结论

这轮最需要先修的不是“再多写几条测试”，而是先把评估口径修正、把日志出口统一、把调试产物从“默认长期落盘”改成“有开关、可控、可测”。这三件事处理完，后面的测试补位和线上排障都会容易很多。
