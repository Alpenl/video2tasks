# Stage 1 Pipeline Review

## 审计范围

本轮只看 Stage 1 主流程，也就是从 server 生成窗口任务，到 worker 拉取任务、执行推理、结果落盘，再到 server 汇总成 `segments.json` 的整条链路。

重点文件：

- `src/video2tasks/server/app.py`
- `src/video2tasks/server/windowing.py`
- `src/video2tasks/server/task_artifacts.py`
- `src/video2tasks/worker/runner.py`

辅助看了但没有深挖：

- `src/video2tasks/server/exporter.py`
- `src/video2tasks/server/llm_merge.py`
- `src/video2tasks/config.py`

不在本轮重点范围里的内容：

- 模型能力本身是否足够
- Stage 2 合并/摘要策略本身的质量
- 导出视频细节

## 主流程解释

### 1. Server 扫样本并生成窗口任务

`create_app()` 里启动了一个后台 `producer_loop()`，按数据集顺序推进样本状态机。主入口在 `src/video2tasks/server/app.py:584-1272`。

对每个样本，server 会：

1. 找到视频文件 `Frame_*.mp4`
2. 调 `read_video_info()` 读取 FPS 和总帧数
3. 调 `build_windows()` 生成重叠窗口
4. 读取已有的 `windows.jsonl`，用于断点续跑
5. 找出还没完成的窗口 repeat
6. 用 `FrameExtractor` 抽帧或生成 contact sheet
7. 把任务塞进内存队列 `job_queue`

### 2. Worker 拉任务并推理

worker 主循环在 `src/video2tasks/worker/runner.py:233-392`。

它会：

1. `GET /get_job`
2. 从 `image_paths` 或内联 `images` 加载图片
3. 按 `job_type` 选 prompt
4. 调后端 `backend.infer()`
5. 用 `normalize_task_window_result()` 规整输出
6. `POST /submit_result`

worker 本身不做持久化，也不决定全局切分结果。它只是把单个任务的结果回传给 server。

### 3. Server 收结果并落盘

server 的 `submit_result()` 在 `src/video2tasks/server/app.py:506-578`。

它会：

1. 校验 `dispatch_id`，避免旧结果覆盖新结果
2. 对空结果做重试或终止
3. 把成功结果按任务类型追加写入：
   - `windows.jsonl`
   - `boundary_refinements.jsonl`
   - `segment_labels.jsonl`

窗口任务的核心持久化函数是 `_persist_result_record()`，在 `src/video2tasks/server/app.py:356-403`。

### 4. Server 汇总成 segments

当一个样本的窗口都完成后，server 在 `producer_loop()` 的 finalize 分支里调用 `build_segments_via_cuts()`，位置在 `src/video2tasks/server/app.py:972-983`。

`build_segments_via_cuts()` 的主要步骤在 `src/video2tasks/server/windowing.py:2297-2586`：

1. 把每个窗口里的局部 cut 投票到全局帧号
2. 聚类成全局候选边界
3. 根据窗口内 instruction 构造逐帧 instruction timeline
4. 按边界切出原始 segments
5. 做长段拆分、轻清理、任务级合并
6. 返回 `segments` 和诊断信息

最终结果由 server 写到 `segments.json`，再打 `.DONE` 标记，位置在 `src/video2tasks/server/app.py:1246-1258`。

## 关键实现细节

### 任务状态是纯内存的，样本结果才落盘

- 队列 `job_queue`
- 飞行中任务 `inflight`
- 各种 retry 计数

这些都只存在内存里，server 重启后依赖磁盘上的 `windows.jsonl`、`segments.json`、`.DONE`、`.FAILED` 恢复进度。

### 任务去重靠 `task_id + dispatch_id`

- `task_id` 用来识别逻辑任务
- `dispatch_id` 用来识别某次派发
- server 只接受当前 inflight 的 dispatch 结果

这能避免“超时后旧 worker 结果”把新结果冲掉。

### 空结果和超时是两套重试机制

- worker 本地先做最多 4 次推理重试
- 仍然空时，把空结果交给 server
- server 再按 `max_empty_retries_per_job` 决定是否重新入队
- 如果 worker 根本没提交，server 只能等 `inflight_timeout_sec` 到期后重排

### 结果文件是 append-only JSONL

`windows.jsonl` 不是覆盖写，而是追加写。恢复时重新扫描文件，再按 `window_id + repeat_index` 归并。

这个设计适合断点续跑，但也意味着文件会持续变大，恢复成本会越来越高。

## 发现的问题

按严重程度排序。

### 1. 高: `build_segments_via_cuts()` 实际上从不采用合并后的 segments

位置：

- `src/video2tasks/server/windowing.py:2523-2558`
- `src/video2tasks/server/windowing.py:2569-2577`

现象：

- 函数先算出了 `light_segments`
- 又算出了 `merged_segments`
- 还算了 `use_light_fallback`
- 但 `final_output` 被固定赋值为 `light_segments`
- 后面只在 `light_segments` 上做 `refine_segment_instructions()`
- 返回时 `segments` 永远是 `light_segments`

直接影响：

- `merge_task_level_segments()` 的结果基本是死代码
- `adaptive_merge_guard`、`adaptive_merge_*` 这些参数不影响最终产物
- 诊断里会写 `merged_segment_count` 和 `selection_policy`，但真正输出仍是未合并版本
- 这会把 Stage 1 结果固定在“偏保守、偏碎”的状态，增加后续成本

为什么严重：

这不是调参偏差，而是主流程行为和代码意图已经分叉了。只要这段逻辑存在，汇总成 segments 的最终结果就不符合函数名和诊断信息表达的含义。

### 2. 高: 生成窗口阶段遇到异常时，样本会被直接跳过，既不失败也不完成

位置：

- `src/video2tasks/server/app.py:705-710`
- `src/video2tasks/server/app.py:817-821`

现象：

- 视频文件不存在时，只做 `cur_idx += 1`
- Step A 任意异常时，也只做 `cur_idx += 1`
- 这两条路径都不会写 `.FAILED`
- 也不会写失败报告
- 也不会保留一个“待重试”的状态

直接影响：

- 样本可能悄悄消失在处理流程里
- 后续不会再被处理
- `auto_exit_after_all_done` 仍可能以 0 退出
- 最终目录里既没有 `.DONE` 也没有 `.FAILED`

为什么严重：

这是数据完整性问题。主流程表面上“跑完了”，但实际上有样本被静默丢掉，后面很难靠日志之外的证据补查。

### 3. 高: finalize 分支异常只打印日志，不会失败、不会前进，会把整批数据卡死

位置：

- `src/video2tasks/server/app.py:824-1266`

现象：

- finalize 阶段的大块逻辑包在一个 `try`
- `except` 里只打印 `[Err-Finalize]`
- 没有写 `.FAILED`
- 没有推进 `cur_idx`
- 没有重试预算

直接影响：

- 如果异常是稳定可复现的，比如某个样本结构异常、后处理返回脏数据、写文件失败
- 当前样本会一直停在 `sample_status == 2`
- producer loop 会每 0.1 秒重试一次
- 由于流程按样本顺序串行推进，整个 dataset 都会被这个样本堵住

为什么严重：

这是明显的停机点。一旦触发，队列可能空着，但主流程永远过不去。

### 4. 中高: worker 的非推理异常不会主动上报，server 只能靠超时回收

位置：

- `src/video2tasks/worker/runner.py:306-333`
- `src/video2tasks/worker/runner.py:373-389`
- `src/video2tasks/server/app.py:619-650`

现象：

- 图片加载失败、prompt 构造异常、submit 之前的其他异常
- 都会落到 worker 外层 `except Exception`
- worker 只打印 `Loop crashed`
- 不会回传一个显式失败结果

直接影响：

- server 认为这个任务还在 inflight
- 只能等 `inflight_timeout_sec` 到期
- 默认 300 秒，再叠加 `max_retries_per_job`
- 一个本地就能立刻判死的任务，会变成几分钟级的慢失败

为什么重要：

这会把系统从“快速失败”变成“长时间占坑”。在队列不大或 worker 数量固定时，这种失败方式会显著拖慢吞吐。

### 5. 中: 启用可选二次流程时，部分任务失败会被当作“已完成样本”

位置：

- `src/video2tasks/server/app.py:1001-1096`
- `src/video2tasks/server/app.py:1098-1183`
- `src/video2tasks/server/app.py:1246-1258`

现象：

- `boundary_refinement` 和 `segment_label` 都会收集 `*_failures`
- 但 finalize 不会因为这些失败而标记样本失败
- 失败的 boundary 或 segment 只写进 diagnostics
- 之后仍然写 `segments.json` 并打 `.DONE`

直接影响：

- 如果用户开启了这些功能，最终产物可能是“部分降级成功”
- 但外部只看到 `.DONE`
- 无法从完成标记判断这个样本是不是完整跑通了全部启用步骤

为什么要注意：

这可能是有意设计，但它让“完成”的语义变弱了。若后续要统计全链路成功率，单靠 `.DONE` 不可靠。

## 失败模式

这条链路现在比较容易进入下面几类失败模式：

### 1. 静默丢样本

- 视频不存在
- Step A 抛异常
- server 直接跳过样本
- 最终没有 `.DONE` 也没有 `.FAILED`

### 2. 单样本卡死整批

- finalize 抛稳定异常
- 当前样本不前进
- 后续样本都排不到

### 3. 慢失败拖死吞吐

- worker 在 submit 前崩掉
- server 只能等 inflight timeout
- 一个坏任务长时间占着 inflight 和重试预算

### 4. 结果名义完成、实际降级

- 可选二次流程部分失败
- 样本仍然 `.DONE`
- 只有看 diagnostics 才知道并不完整

### 5. 过切结果长期存在

- 任务级合并结果没有被采用
- segments 容易偏碎
- 后续步骤承担额外清理成本

## 性能瓶颈

### 1. `job_queue.pop(0)` 和重复线性扫描是 O(n)

位置：

- `src/video2tasks/server/app.py:494`
- `src/video2tasks/server/app.py:769`
- `src/video2tasks/server/app.py:918`
- `src/video2tasks/server/app.py:1029`
- `src/video2tasks/server/app.py:1115`

说明：

- `list.pop(0)` 会整体搬移后面的元素
- 入队前还反复 `any(...)` 扫描整个队列
- 任务多时，server 端调度成本会持续上升

### 2. `instruction_timeline` 按总帧数建整表，长视频内存占用会很高

位置：

- `src/video2tasks/server/windowing.py:2320`

说明：

- 每一帧都分配一个列表
- 长视频会产生很大的 Python 对象开销
- 这类结构对 CPU cache 和 GC 都不友好

### 3. 抽帧是随机 seek 模式，CV2 路径成本高

位置：

- `src/video2tasks/server/windowing.py:672-677`
- `src/video2tasks/server/windowing.py:887-907`
- `src/video2tasks/server/windowing.py:918-930`

说明：

- `_read_frame_bgr()` 每次都 `cap.set(...POS_FRAMES...)`
- 窗口和 refinement 阶段会反复打开视频并随机取帧
- contact sheet ffmpeg 失败后，还会退回到逐帧 CV2 方案

### 4. 结果恢复依赖全量扫描 JSONL

位置：

- `src/video2tasks/server/app.py:237-300`
- `src/video2tasks/server/app.py:212-235`

说明：

- 每次进入样本处理和 finalize 都会重新扫结果文件
- 文件越大，恢复越慢
- repeat 和多轮 refinement 会进一步放大这个问题

## 可泛化的优化建议

### 1. 先把“状态语义”补齐

- 每个样本最终必须落到 `.DONE` 或 `.FAILED`
- 禁止无标记跳过
- finalize 失败要有明确的失败或有限重试语义

### 2. 把“快失败”和“慢失败”分开

- worker 能本地判死的错误，应该显式提交失败
- 只把真正的网络丢包、worker 消失交给 inflight timeout

### 3. 让 diagnostics 和真实输出保持一致

- 既然代码已经算了 `merged_segments`
- 最终到底选 `light_segments` 还是 `merged_segments`
- 应该由清晰的分支决定，并在 diagnostics 里如实反映

### 4. 把队列和索引结构换成更适合调度的数据结构

- `deque` 适合 FIFO 出队
- 单独维护 `queued_task_ids` / `inflight_task_ids`
- 避免每次线性扫整个队列做重复检查

### 5. 降低长视频的常数成本

- 用稀疏结构替代逐帧 `instruction_timeline`
- 对抽帧和 contact sheet 做缓存或批量读取
- 对 JSONL 恢复建立轻量索引或阶段性快照

### 6. 给主流程补最小闭环测试

- 至少要有以下场景：
- 窗口任务正常完成并产出 `segments.json`
- Step A 抛异常时样本被明确标失败
- finalize 抛异常时不会无限卡死
- worker 在 submit 前失败时 server 能快速回收
- 合并路径确实会影响最终输出

## 优先级排序

### P0

1. 修正 `build_segments_via_cuts()` 的最终输出选择逻辑，保证 diagnostics 和真实 segments 一致。
2. 修正样本状态机，禁止 Step A 静默跳过样本。
3. 给 finalize 增加明确的失败或有限重试逻辑，避免单样本卡死整批。

### P1

1. 让 worker 对 submit 前的本地异常显式上报，减少 timeout 型慢失败。
2. 明确 `.DONE` 的语义；如果启用了可选二次流程，决定失败时是降级完成还是整样本失败。

### P2

1. 替换队列和查重的数据结构，降低 server 调度开销。
2. 优化长视频的内存和随机抽帧成本。
3. 给 JSONL 恢复建立更便宜的读取路径。
