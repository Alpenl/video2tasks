# Round 2 代码质量与架构复核 07

更新时间：2026-04-08  
主审文档：`docs/reviews/round1/07-code-quality-and-architecture-review.md`  
复核方式：只复核第一轮文档的证据链、问题级别和遗漏项；不改代码，只新增本复核文档。

## 1. 复核对象

本轮实际回看了下面这些文件和路径：

1. 主审文档：`docs/reviews/round1/07-code-quality-and-architecture-review.md`
2. 编排与状态机：`src/video2tasks/server/app.py`
3. Worker 与后端接缝：`src/video2tasks/worker/runner.py`、`src/video2tasks/vlm/base.py`、`src/video2tasks/vlm/openai_api.py`、`src/video2tasks/vlm/gemini_api.py`、`src/video2tasks/vlm/remote_api.py`
4. 后处理与算法边界：`src/video2tasks/server/llm_merge.py`、`src/video2tasks/server/windowing.py`
5. 配置：`src/video2tasks/config.py`
6. 相关测试：`tests/server/test_app_retry.py`、`tests/server/test_llm_summary.py`

本轮目标不是重做第一轮，而是审第一轮文档本身：看它抓到的问题是否真有代码证据、级别是否合适、有没有说大了、有没有漏掉更直接的问题。

## 2. 确认成立的点

### 1. 严重：样本状态机的异常路径不闭环，这一点成立，而且仍然是首要问题

第一轮对这个问题的判断是准确的，级别也不应下调。

证据回看：

- `src/video2tasks/server/app.py:405-428` 已经有 `_persist_sample_failure()`，会写 `.FAILED` 和 `failure.json`。
- `src/video2tasks/server/app.py:430-447` 还有 `_mark_task_terminal_failure()`，说明项目里并不缺失败落盘手段。
- 但 Step A 的兜底异常分支 `src/video2tasks/server/app.py:817-821` 只打印栈和 `cur_idx += 1`，没有改 `sample_status[sid]`，也没有调用失败持久化。
- Finalize 的异常分支 `src/video2tasks/server/app.py:1265-1266` 只打印 `[Err-Finalize]`，既不前进 `cur_idx`，也不标记失败，主循环会反复撞上同一个样本。
- 对照看，超时和空结果已经会走统一终态处理，见 `src/video2tasks/server/app.py:542-566` 与 `src/video2tasks/server/app.py:641-650`。这说明“失败闭环”不是没设计，而是 producer/finalize 这两条分支漏接了。

测试情况也支持第一轮的判断：

- `tests/server/test_app_retry.py:18-24`、`tests/server/test_app_retry.py:27-255` 主要覆盖了 `submit_result`、超时重排队和空结果终态。
- 没有看到针对 `src/video2tasks/server/app.py:817-821` 或 `src/video2tasks/server/app.py:1265-1266` 的回归测试。

结论：

- 这不是“日志不够细”，而是状态机漏了一条失败出口。
- 第一轮把它放在最前面是对的。

### 2. 高：后端抽象已经被具体实现打穿，这一点成立

这一条第一轮证据充分，级别维持高比较合适。

证据回看：

- `src/video2tasks/vlm/base.py:12-26` 把统一接口定义成 `infer(images: List[np.ndarray], prompt: str)`。
- 但 `src/video2tasks/worker/runner.py:306-311` 在 `backend == "gemini"` 时不再传 `np.ndarray`，而是传 `{"raw_bytes", "mime_type"}` 字典。
- `src/video2tasks/vlm/gemini_api.py:37-47`、`src/video2tasks/vlm/gemini_api.py:341-356` 也确实在实现内部偷偷兼容了这种字典载荷，而它的公开签名仍然写着 `List[np.ndarray]`，见 `src/video2tasks/vlm/gemini_api.py:287-290`。
- `llm_merge` 这边没有走统一工厂，而是直接依赖 `OpenAIBackend`，见 `src/video2tasks/server/llm_merge.py:13-19`、`src/video2tasks/server/llm_merge.py:1210-1224`、`src/video2tasks/server/llm_merge.py:1452-1462`。
- 配置层也把 `llm_merge.backend` 限成了 `"openai"`，见 `src/video2tasks/config.py:382-388`。

结论：

- 第一轮说“实际已经有两套后端扩展方式”，这个判断成立。
- 这里不是抽象不够优雅，而是 worker 和后处理都已经知道了具体后端的私有差异。

### 3. 中：核心算法文件过大，且跨文件引用私有函数，这一点成立

这一条第一轮没有夸大。

证据回看：

- `windowing.py` 2586 行，`llm_merge.py` 1648 行，`app.py` 1285 行，`prompt.py` 782 行，`config.py` 737 行。
- `src/video2tasks/server/llm_merge.py:14-18` 直接导入了 `windowing.py` 的私有函数：
  - `_boundary_support_between`
  - `_has_distinct_sequence_markers`
  - `_should_split_on_instruction_drift`

结论：

- 大文件本身不一定是 bug，但和“跨文件 import 私有实现”叠在一起，就说明模块边界已经不稳。
- 第一轮把它作为中优先级的结构问题是合理的。

### 4. 中：job metadata 组装重复，这一点成立

第一轮这里抓到了一个真实维护问题。

证据回看：

- window job 元数据：`src/video2tasks/server/app.py:775-792`
- refinement window 元数据：`src/video2tasks/server/app.py:924-942`
- boundary refinement 元数据：`src/video2tasks/server/app.py:1035-1054`
- segment label 元数据：`src/video2tasks/server/app.py:1127-1145`

这些片段明显共享同一批字段：

- `subset`
- `sample_id`
- `logical_frame_count`
- `use_contact_sheets`
- `contact_sheet_rows`
- `contact_sheet_cols`

结论：

- 第一轮关于“字段一变要改多处”的判断成立。
- 这条值得保留，但它应排在状态机闭环和后端接口失真之后。

## 3. 需要修正的点

### 1. 第一轮第 3.2 条方向是对的，但“严重”偏高，且把两类问题揉在了一起

第一轮说 `create_app` 同时承担 Web 应用、调度器、状态机、持久化和进程退出，这个方向没错；但把它整体放到“严重”一档，会把优先级拉歪。

更准确的拆法应该是：

- 高：`create_app` 构造阶段就启动 producer thread，App 构造和业务运行耦合，见 `src/video2tasks/server/app.py:480-487`、`src/video2tasks/server/app.py:1270-1272`。
- 高：测试也直接走这个入口，见 `tests/server/test_app_retry.py:18-24`，说明副作用已经进入测试面。
- 中到高：`os._exit()` 的确粗暴，但它是条件路径，不是默认行为。`auto_exit_after_all_done` 默认是 `False`，见 `src/video2tasks/config.py:86-97`；真正的退出捷径在 `src/video2tasks/server/app.py:655-665`。

结论：

- 第一轮的核心观察成立。
- 但建议把这一条从“严重”下调到“高”，并拆成“构造副作用”和“条件性进程退出”两部分写，避免和第 3.1 条那种已经能导致样本丢失或卡死的问题混在同一层。

### 2. 第一轮第 3.4 条把几件不同的事绑成了一个“高优先级”问题，证据不够紧

第一轮这一条至少包含了三类不同问题：

1. 规范化在多层重复
2. JSON 提取逻辑有重复
3. 重试策略散在多层

前两类是成立的，第三类证据不足。

证据回看：

- backend 自己做规范化：`src/video2tasks/vlm/openai_api.py:631-658`、`src/video2tasks/vlm/gemini_api.py:210-215`
- worker 再做一次：`src/video2tasks/worker/runner.py:349-353`
- server 读取落盘结果再做一次：`src/video2tasks/server/app.py:114-140`
- JSON 提取逻辑确实重复：`src/video2tasks/vlm/openai_api.py:29-54`、`src/video2tasks/vlm/gemini_api.py:70-88`、`src/video2tasks/vlm/remote_api.py:21-39`

但“重试策略散在多层”不宜直接定性成职责失真，因为这几层其实处理的是不同失败域：

- Gemini 接口层的 payload 级重试：`src/video2tasks/vlm/gemini_api.py:293-339`
- worker 本地重试：`src/video2tasks/worker/runner.py:342-359`
- server 队列级重排队和终态处理：`src/video2tasks/server/app.py:542-566`、`src/video2tasks/server/app.py:641-650`

结论：

- 建议把这一条拆开。
- “规范化和 JSON 提取重复”可以保留，级别建议下调到中。
- “多层重试策略”更像需要补设计说明，而不是已经坐实的高优先级结构缺陷。

### 3. 第一轮第 3.6 条前半段成立，后半段证据偏弱

前半段，也就是 job metadata 组装重复，证据足够，前文已确认。

但后半段把 `_collect_env_override_data()` 的长度直接推成“已经失控”，这一步证据不够紧。当前能直接看到的是：

- `src/video2tasks/config.py:498-695` 这段映射确实很长。

问题在于：

- 第一轮没有给出一个已经发生的错配样本。
- 这次复核也没有在抽看的字段里看到“这里已经写错映射”这种直接证据。

结论：

- 建议把这一条拆开写。
- “metadata 重复”保留中优先级。
- “环境变量映射过长”可以作为低到中优先级的维护提醒，但不应和状态机、后端接口问题排在一组。

### 4. 第一轮第 3.7 条应当下调到低

`print`、宽泛异常、短变量名这些问题都是真实存在的，但更像代码卫生问题，不适合和前几条放在同一梯队。

原因很直接：

- 它们会增加排障成本。
- 但第一轮同一份文档里已经有更直接的行为缺陷，例如隐藏失败、卡死样本、接口抽象被打穿。

结论：

- 这一条保留可以。
- 但建议明确下调到低优先级，作为后续治理项，而不是本轮最前排建议。

## 4. 新增发现

### 1. 高：缺少 `Frame_*.mp4` 的样本会被静默跳过，第一轮漏掉了

这是和第 3.1 条同类的失败闭环问题，而且证据更直接。

证据：

- `src/video2tasks/server/app.py:705-710` 找不到 `Frame_*.mp4` 时，只做了 `st["cur_idx"] += 1` 和 `continue`。
- 这里没有：
  - `sample_status[sid] = 4`
  - `.FAILED`
  - `failure.json`
  - 任何错误原因记录

影响：

- 这类样本会直接从队列里消失。
- 最终统计也不会把它当失败样本计入。

复核结论：

- 第一轮已经指出了异常分支会形成“灰色状态”，但漏掉了这个更简单、也更常见的静默跳过路径。
- 这一条应该补进前排建议。

### 2. 中：窗口结果重载时的二次校验读错了 `logical_frame_count` 位置，现有测试把这个缺口遮住了

这是第一轮完全没提到的一个具体一致性问题。

证据：

- `src/video2tasks/server/app.py:372-374` 落盘时把 `logical_frame_count` 存在顶层字段。
- 窗口结果实际写盘记录只包含顶层字段，见 `src/video2tasks/server/app.py:398-403`。
- 但 `_normalize_loaded_window_vlm_json()` 却从 `record.get("meta", {})` 里取 `logical_frame_count`，见 `src/video2tasks/server/app.py:114-120`。
- 现有测试 `tests/server/test_app_retry.py:259-265` 手工构造的是 `{"meta": {"logical_frame_count": 4}}`，这和真实落盘格式不一致。

影响：

- 对真实 `windows.jsonl` 记录来说，重载阶段的 `max_transition_index` 约束可能失效。
- 好在 `submit_result` 入库前已经做过一次上界校验，见 `src/video2tasks/server/app.py:531-540`，所以它目前更像潜在正确性缺口，而不是立刻爆炸的主路径故障。

复核结论：

- 这条建议放中优先级。
- 它也说明第一轮把“多层规范化”写得太抽象了，漏掉了这种更具体、可落到行号的问题。

### 3. 中：producer thread 没有任何 stop/join 句柄，测试隔离风险比第一轮写得更实

第一轮提到了 `create_app` 有副作用，但没有把“没有停止句柄”这个更具体的问题单独指出来。

证据：

- `src/video2tasks/server/app.py:1270-1272` 直接启动 daemon thread。
- 没有看到它被挂到 `app.state`，也没有显式的 stop/join 接口。
- `tests/server/test_app_retry.py:18-24` 的 `_make_app()` 直接调用 `create_app()`，也就是单测本身已经在后台起了真实循环。

影响：

- 这会让和 producer/finalize 相关的测试更难写，也更难稳定复现边界情况。
- 未来只要 producer loop 多一点副作用，测试就更容易互相污染。

复核结论：

- 这条不一定要单列成最高优先级问题，但应该作为第 3.2 条的补强证据写出来。

## 5. 重新排序后的建议

1. 先把所有“样本会悄悄消失或卡住”的路径收口到统一失败闭环。  
至少包括 Step A 异常、Finalize 异常、找不到 `Frame_*.mp4` 三类路径，都应该走统一的 sample failure 记录，而不是继续留灰色状态。

2. 把 App 构造和运行时生命周期拆开。  
`create_app()` 应尽量只负责组装 FastAPI；producer thread 的启动、停止、join 和退出码处理应单独挂到明确的运行入口或 lifespan。

3. 统一 backend 的输入契约和扩展方式。  
worker 不应知道 Gemini 的特殊载荷；`llm_merge` 也不应绕过统一 backend 工厂直接 new `OpenAIBackend`。

4. 先切断跨文件私有函数依赖，再考虑拆大文件。  
`llm_merge.py` 对 `windowing.py` 私有函数的直接 import 是边界破坏的明确信号，应优先处理。

5. 把 job metadata 构造收成共享 builder。  
这件事价值是降低重复和字段漂移，但优先级应低于状态机修复和接口收口。

6. 把“规范化重复”和“重试层次”分开治理。  
前者适合收敛到公共 helper；后者更适合先写清每层负责什么，不要继续混写成一个大问题。

7. `print`、短变量名、宽泛异常作为第二梯队整理项。  
这些问题应改，但不应挤占前面那些已经影响正确性的修复顺序。

## 6. 复核结论

第一轮文档的大方向是对的，尤其是样本状态机不闭环、后端抽象被具体实现打穿、以及大文件加跨文件私有依赖这三点，证据都成立。  
需要修正的主要有三处：一是把 `create_app` 那条从“严重”下调到“高”并拆开写，二是把“规范化/JSON 提取/重试策略”拆开，不要绑成一个高优先级问题，三是把 `print` 和命名这类卫生问题下调。  
另外，第一轮漏掉了两个很具体的点：缺少 `Frame_*.mp4` 会静默跳过样本，以及窗口结果重载时二次校验读取了和真实落盘格式不一致的字段。
