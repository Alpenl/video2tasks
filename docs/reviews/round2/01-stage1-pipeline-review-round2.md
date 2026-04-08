# Stage 1 Pipeline Review Round 2

## 复核对象

- 主审文档：`/home/alpen/DEV/video2tasks/docs/reviews/round1/01-stage1-pipeline-review.md`
- 复核方式：不重复做整轮代码审查，只核对第一轮关键结论是否有足够代码证据，并补充第一轮漏掉但同样重要的问题。
- 本轮回看代码的重点文件：
  - `src/video2tasks/server/app.py`
  - `src/video2tasks/server/windowing.py`
  - `src/video2tasks/worker/runner.py`
  - `src/video2tasks/server/llm_merge.py`

## 确认成立的点

### 1. `build_segments_via_cuts()` 没有把 `merged_segments` 用到最终输出里

这个结论成立，而且证据是直接的。

- `light_segments` 和 `merged_segments` 都被计算出来了，见 `src/video2tasks/server/windowing.py:2523-2528`。
- `missing_strong_boundaries_if_merged`、`adaptive_light_fallback`、`selection_policy` 也都基于“如果采用 merged 结果会怎样”继续计算，见 `src/video2tasks/server/windowing.py:2529-2567`。
- 但真正返回前，`final_output` 被固定设成 `light_segments`，后面也没有改成 `merged_segments` 的分支，见 `src/video2tasks/server/windowing.py:2555-2577`。

第一轮把它列为高优先级是合理的，因为这不是调参问题，而是最终产物和函数内部意图已经分叉。

### 2. Step A 遇到缺视频或异常时，样本会被静默跳过

这个结论也成立。

- 找不到 `Frame_*.mp4` 时，只做 `cur_idx += 1` 后继续，没有 `.FAILED`、没有失败报告，见 `src/video2tasks/server/app.py:705-710`。
- Step A 外层 `except` 里只打印异常并 `cur_idx += 1`，同样没有失败落盘，见 `src/video2tasks/server/app.py:817-821`。
- 退出码只统计 `sample_status == 4` 的样本，见 `src/video2tasks/server/app.py:143-153`。这说明被静默跳过的样本既不会计入成功，也不会计入失败。

第一轮把这条放在高优先级是对的。它影响的是结果完整性，不只是日志可读性。

### 3. finalize 异常会让当前样本反复重试，并卡住后续样本

这个结论成立。

- finalize 主体包在一个大 `try` 里，见 `src/video2tasks/server/app.py:824-1266`。
- `except` 里只有 `print(f"[Err-Finalize] ...")`，没有写 `.FAILED`，没有推进 `cur_idx`，也没有重试上限，见 `src/video2tasks/server/app.py:1265-1266`。
- producer loop 是按当前 `cur_idx` 串行推进样本的，所以当前样本不前进，后面的样本也排不到，见 `src/video2tasks/server/app.py:684-690` 和 `src/video2tasks/server/app.py:824-1266`。

第一轮把这条列为高优先级也是合理的。这是明确的停机点。

### 4. worker 的非推理异常不会显式上报，只能靠 server 超时回收

这个结论成立，但优先级我会往下调半档。

- 图片加载、prompt 构造、提交失败都在 worker 主循环外层 `except Exception` 范围内，见 `src/video2tasks/worker/runner.py:306-382` 和 `src/video2tasks/worker/runner.py:387-389`。
- 外层 `except` 只打印 `Loop crashed` 并 sleep，不会向 server 回传终态，见 `src/video2tasks/worker/runner.py:387-389`。
- submit 失败在本地最多重试 3 次，耗尽后直接抛异常，见 `src/video2tasks/worker/runner.py:201-230`。
- server 端对这类未提交任务的回收路径确实是 inflight timeout，见 `src/video2tasks/server/app.py:619-650`。

所以“慢失败拖吞吐”是成立的。但它更像吞吐和时延问题，不像前三条那样直接破坏结果正确性或导致整批卡死。

### 5. 可选二次流程失败时，样本仍可能被写成 `.DONE`

这个结论成立，但应当更准确地表述为“完成语义变弱”，而不是直接等同于主流程错误。

- boundary refinement 的失败只进入 diagnostics，不阻止继续写 `segments.json` 和 `.DONE`，见 `src/video2tasks/server/app.py:1001-1096` 和 `src/video2tasks/server/app.py:1246-1258`。
- deferred segment labeling 的失败也是同样处理，见 `src/video2tasks/server/app.py:1098-1183` 和 `src/video2tasks/server/app.py:1246-1258`。

第一轮提醒这件事是有价值的，但它更像“状态语义不清晰”，而不是无条件的功能错误。

## 需要修正的点

### 1. “`merge_task_level_segments()` 基本是死代码”这个说法有点过头

更准确的说法应当是：

- `merge_task_level_segments()` 不影响最终返回的 `segments`。
- 但它仍然影响 diagnostics 和 `selection_policy` 的计算。

证据见 `src/video2tasks/server/windowing.py:2524-2567`。所以“对最终分段结果无效”是准确的，“基本死代码”则说重了。

### 2. “`adaptive_merge_*` 参数不影响最终产物”需要限定范围

这个判断只有在“最终输出的 `segments`”这个层面成立。

- 这些参数确实不会改变当前返回的 `segments`，因为 `final_output` 固定是 `light_segments`。
- 但它们会影响 `adaptive_light_fallback` 和 `selection_policy`，也就是 diagnostics。

证据同样在 `src/video2tasks/server/windowing.py:2543-2567`。建议第一轮把表述改成“不会影响当前返回的分段结果”，不要写成完全无影响。

### 3. 第四条问题的优先级可以从“中高”降到“中”

原因很直接：

- 它会放大超时和重试，拖慢吞吐。
- 但 server 最终仍有 timeout 回收路径，结果不是必丢。

相比之下，静默跳过样本、finalize 卡死整批、最终分段逻辑失真，都是更硬的主流程问题。把 worker 这条排在它们后面更合适。

### 4. “性能瓶颈”一节的证据主要是静态结构，不是实测结果

第一轮指出的几个点基本都能从代码里看出来：

- `job_queue.pop(0)` 是 O(n)，见 `src/video2tasks/server/app.py:494`。
- `instruction_timeline = [[] for _ in range(nframes)]` 是按总帧数分配，见 `src/video2tasks/server/windowing.py:2320`。
- JSONL 恢复依赖全量扫描，见 `src/video2tasks/server/app.py:212-300`。

但这些更适合写成“高概率热点”或“结构性成本”，不宜直接写成已经证实的瓶颈。第一轮缺少任何 profiling 或实测数据来支持“瓶颈”这个词。

### 5. 第五条问题最好明确前提条件

这条只在以下配置打开时才成立：

- `enable_boundary_refinement`
- `segment_labeling_mode == "deferred"`

如果功能没开，这条风险不存在。建议第一轮在标题或第一句里明确这个前提，避免读者误以为 `.DONE` 在所有配置下都代表“可能缺少核心输出”。

## 新增发现

### 1. `.DONE` 语义被削弱的不只是 boundary refinement 和 segment labeling

第一轮漏掉了 finalize 后半段的另外两类降级完成路径。

- `run_llm_postprocess_pass()` 无论 merge 或 summary 过程里发生什么，都会返回一份 `cleaned_segments` 和 diagnostics，调用点在 `src/video2tasks/server/app.py:1185-1195`，函数出口在 `src/video2tasks/server/llm_merge.py:1618-1648`。
- 导出字幕本身就有多条 fallback 路径，失败时继续使用源 instruction 作为字幕，见 `src/video2tasks/server/llm_merge.py:1528-1615`。
- 真正的导出异常也被 `try/except` 吞成 diagnostics，不会阻止 `.DONE`，见 `src/video2tasks/server/app.py:1225-1256`。

所以更准确的结论应该是：

`.DONE` 当前代表“finalize 写出了一份可接受的结果文件”，不代表“所有启用的后处理和导出步骤都成功完成”。

这和第一轮第五条是同一个方向，但范围更大，外部系统如果拿 `.DONE` 当全链路成功标记，会被误导。

### 2. `completed_dispatch_ids` 会随任务总数单调增长，第一轮漏掉了这个常驻内存点

- 这个字典在 server 启动时初始化，见 `src/video2tasks/server/app.py:172`。
- 每个任务一旦收到终态结果，就会写入一个条目，见 `src/video2tasks/server/app.py:438` 和 `src/video2tasks/server/app.py:573`。
- 当前文件里没有任何清理逻辑，查验点只在 `src/video2tasks/server/app.py:512`。

这不是马上炸掉主流程的 bug，但对长跑任务来说，内存占用会和历史 task 数量线性增长。它至少值得放进性能/稳定性清单，而不是完全漏掉。

## 重新排序后的建议

### P0

1. 补齐样本状态语义，禁止 Step A 静默跳过。缺视频和 Step A 异常都必须落到 `.FAILED` 或明确可重试状态。
2. 给 finalize 增加有限重试或失败落盘，避免单样本把整批卡死。
3. 修正 `build_segments_via_cuts()` 的最终输出选择，让 diagnostics 和真实 `segments` 一致。

### P1

1. 明确 `.DONE` 的定义。至少要决定它表示“主流程成功”还是“所有启用步骤成功”，并让 boundary refinement、segment labeling、LLM postprocess、字幕和导出都遵守同一套语义。
2. 让 worker 对 submit 前的本地异常显式上报，避免把本地快失败拉长成 timeout 失败。

### P2

1. 把第一轮“性能瓶颈”改写成“结构性热点”，再决定是否做 profiling。
2. 替换 `job_queue.pop(0)` 和线性查重结构，降低 server 调度开销。
3. 给 `completed_dispatch_ids` 增加回收策略，避免长跑任务的常驻内存线性增长。
4. 如果长视频是主要场景，再考虑 `instruction_timeline` 和 JSONL 扫描的优化。

## 结论

第一轮最重要的三条判断是站得住的：

- 最终分段逻辑和函数内部意图不一致。
- Step A 有静默丢样本路径。
- finalize 有稳定卡死整批的路径。

需要收敛的是表述力度和优先级：

- `merge_task_level_segments()` 不是完全死代码，只是没有影响最终输出。
- worker 非推理异常更像 P1 吞吐问题，不应压过主流程 correctness 问题。
- `.DONE` 语义问题比第一轮写得更广，但本质上是“成功定义不清”，不是单点逻辑 bug。
