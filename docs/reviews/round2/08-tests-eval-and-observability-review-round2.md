# Round 2 复核 08：测试、评估与可观测性

更新时间：2026-04-08

## 复核对象

- 主审文档：`docs/reviews/round1/08-tests-eval-and-observability-review.md`
- 复核代码：
  - `src/video2tasks/eval/official_boundaries.py`
  - `tests/eval/test_official_boundaries.py`
  - `src/video2tasks/server/app.py`
  - `src/video2tasks/server/windowing.py`
  - `src/video2tasks/server/task_artifacts.py`
  - `tests/server/test_app_retry.py`
  - `tests/server/test_task_artifacts.py`
  - `src/video2tasks/worker/runner.py`
  - `tests/worker/test_runner.py`
  - `tests/server/test_exporter.py`
  - `tests/server/test_llm_merge.py`
  - `tests/server/test_llm_summary.py`
  - `tests/vlm/test_openai_api.py`
- 复核方式：
  - 逐条核对第一轮文档里的结论和引用位置
  - 回看相关源码与测试
  - 做了一个最小复现实验：`gt=[100,103]`、`pred=[101]`、`tolerance=5`，当前 `score_boundary_recall(...)` 返回 `hit_count=2`、`recall=1.0`

## 确认成立的点

### 1. 边界评估存在“一条预测命中多个 GT”的口径错误

这个判断成立，而且证据充分，优先级维持 `P0`。

- `src/video2tasks/eval/official_boundaries.py:49-53` 的 `_nearest_predicted_boundary(...)` 只找最近预测点，不做“已使用”标记。
- `src/video2tasks/eval/official_boundaries.py:68-86` 在每个 GT 上重复使用同一份 `normalized_pred`。
- 复现实验已经证明一条预测边界可以同时命中两个 GT，召回被抬高。
- `tests/eval/test_official_boundaries.py:4-33` 只有 2 个正向测试，没有覆盖这个反例。

这不是“测试写少了”这么简单，而是评估口径会直接偏乐观。

### 2. `logging.level` 只管到了 Uvicorn，没有管到应用本身的大部分输出

这个判断也成立，但我建议把严重级别从第一轮的 `P1` 下调到 `P2`。

- `src/video2tasks/config.py:423-433` 只有一个 `LoggingConfig.level`。
- 全仓搜索里，`config.logging.level` 的实际使用点只有 `src/video2tasks/server/app.py:1280-1284` 的 `uvicorn.run(... log_level=...)`。
- `src/video2tasks/server/app.py:554-563`、`596-602`、`634-645`、`659-664` 以及 `src/video2tasks/worker/runner.py:222-225`、`239-267`、`346-389` 仍然大量直接 `print(...)`。
- `tests/worker/test_runner.py:40-78`、`484-574` 等测试主要是断言 stdout 文本，不是断言日志等级或结构化字段。

问题是真实存在的，但它更像运维和排障效率问题，不是直接的数据正确性错误。

### 3. 评估模块测试覆盖很薄

这个判断成立，优先级维持 `P2`。

- `tests/eval/test_official_boundaries.py` 只有 2 个测试。
- `predicted_boundary_frames_from_segments_file(...)`、`official_boundary_frames_from_file(...)`、`BoundaryRecallSummary.to_dict()`、`tolerance_frames < 0` 都没有测试。
- 当前测试也没有覆盖空输入、重复预测、乱序输入、重复命中等边界条件。

### 4. `test_timeout_exhaustion_records_terminal_error` 依赖真实睡眠和后台线程

这个判断成立，优先级维持 `P3`。

- `tests/server/test_app_retry.py:223-256` 两次 `time.sleep(1.2)`。
- `src/video2tasks/server/app.py:584-667` 的 producer 线程在后台轮询 inflight 超时。

这个问题更偏“测试稳定性和时长”而不是功能错误，所以不应排得太前。

## 需要修正的点

### 1. “调试产物没有任何开关”这个表述不准确

第一轮这里说得过满了，需要改成更精确的说法。

- `src/video2tasks/server/windowing.py:635-639` 已经有 `VIDEO2TASKS_DUMP_INTERMEDIATE` 环境开关。
- `src/video2tasks/server/windowing.py:655` 说明 `FrameExtractor` 在默认情况下会尊重这个开关。
- 真正的问题在 `src/video2tasks/server/app.py:173-174`：服务端启动时总是创建 `TaskArtifactWriter`。
- 同时 `src/video2tasks/server/app.py:457-467` 总是走 `get_many_b64_with_artifacts(...)`。

更准确的表述应当是：

- 仓库里并非完全没有开关。
- 但服务端主流程绕过了已有的 opt-in 设计，导致落盘在 server 路径上变成了默认行为。

这比“完全没开关”更接近代码事实，也更便于后续修正。

### 2. “服务端可观测性缺失”这个点成立，但第一轮把“没有 Prometheus/OTel”写得像缺陷，级别偏高

我建议把这个点收敛成两个更具体的问题，而不是把“没上指标系统”本身当成 bug。

- `src/video2tasks/server/app.py:580-582` 的 `/health` 的确只返回 `{"status": "ok"}`。
- 复核时没有搜到任何 `/health` 测试。
- 也没有搜到 `/metrics`、Prometheus 或 OTel 接口。

但“没有 Prometheus/OTel”更像未建设的能力，不宜直接按缺陷上升处理。更实在的说法是：

- 现有健康检查契约过薄，而且没有测试固定下来。
- 线上排障主要依赖 stdout 和样本目录。

如果要排优先级，我会把它放在 `P3`，低于评估口径错误和样本失败落盘契约。

## 新增发现

### 1. `_persist_sample_failure(...)` 的关键副作用没有任何直接测试

这是第一轮漏掉的更具体问题，我认为优先级应为 `P1`。

- `src/video2tasks/server/app.py:405-428` 这个函数会同时做四件事：
  - 删除 `segments.json`
  - 删除 `.DONE`
  - 创建 `.FAILED`
  - 写入 `failure.json`
- 复核时没有搜到针对 `failure.json`、`.FAILED`、旧 `.DONE` 清理行为的直接测试。

第一轮提到“失败报告结构没测”，但没有把这个函数的副作用完整点出来。这里不只是“少一个 JSON 断言”，而是样本终态切换逻辑本身没有独立保护。

### 2. 产物路径清洗是关键文件系统边界，但没有测试锁住

这个点第一轮只在建议里轻描淡写提到，我认为值得单独列出来，优先级 `P2`。

- `src/video2tasks/server/task_artifacts.py:17-22` 的 `_sanitize_path_token(...)` 决定了目录名如何从 metadata 落到磁盘。
- `src/video2tasks/server/task_artifacts.py:77-81` 的 `_task_dir(...)` 直接依赖这个清洗结果。
- `tests/server/test_task_artifacts.py:13-102` 只覆盖正常输入，没有覆盖带路径分隔符、空值、奇怪字符的 metadata。

这类逻辑一旦退化，后果不是“日志不好看”，而是路径逃逸、目录污染或难清理。

### 3. 服务端主流程已经把 `artifact_manifest_path` 放进 job，但没有测试把这个契约固定住

这个点优先级不高，记为 `P3`，但它是第一轮没提到的。

- `src/video2tasks/server/app.py:473-475` 在有产物批次时，会把 `artifact_manifest_path` 一起发给 worker。
- 现有 worker 相关测试主要验证 `dispatch_id`、图片加载和 stdout，没有验证这个字段是否存在、路径是否可读、契约是否稳定。

这不一定马上出错，但如果后面有人开始依赖 manifest 做排障或采集，这个字段没有测试保护。

## 重新排序后的建议

### P0

1. 修正 `score_boundary_recall(...)` 的匹配口径，确保一条预测边界不能重复命中多个 GT。
2. 补一条最小反例测试，把这次复现样例直接固化进 `tests/eval/test_official_boundaries.py`。

### P1

1. 给 `_persist_sample_failure(...)` 补直接测试，锁住 `failure.json`、`.FAILED`、旧 `.DONE`、旧 `segments.json` 的行为。
2. 明确服务端是否应该遵守 `VIDEO2TASKS_DUMP_INTERMEDIATE`。如果应该，就不要再在 `create_app(...)` 里无条件创建 `TaskArtifactWriter`；如果不应该，就至少把当前行为写成显式配置而不是隐式绕过。

### P2

1. 扩充评估模块测试，至少补齐文件读取、`to_dict()`、负容忍度、空输入、重复预测、乱序输入。
2. 把“日志等级是否生效”从纯 stdout 断言升级为契约测试。哪怕暂时不全面改成结构化日志，也要先锁住关键事件能否被等级控制。
3. 给 `TaskArtifactWriter` 补路径清洗和异常输入测试。

### P3

1. 给 `/health` 增加一条最基础的接口测试，先把当前契约固定住。
2. 再决定是否需要更丰富的运行状态字段，不要把“先上 Prometheus/OTel”当成第一步。
3. 把 `tests/server/test_app_retry.py:223-256` 这类真时间测试改成可控时钟或更短驱动，降低慢测和偶发失败风险。

## 结论

第一轮最重要的判断是对的：边界评估口径确实有真实错误，必须先修。需要修正的是两处表述：

- 产物落盘不是“完全没开关”，而是“已有开关被服务端主流程绕过”。
- 可观测性问题存在，但“没有 Prometheus/OTel”不应和真实缺陷混成一个级别。

如果按修复顺序排，我会先处理评估口径，再处理样本失败落盘契约和产物开关一致性，最后才是健康检查和更完整的指标建设。
