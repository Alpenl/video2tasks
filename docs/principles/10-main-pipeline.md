# 主流程怎么跑（Stage 1 -> Stage 2 -> Export）

这份文档按“谁负责什么”讲清楚主流程。
如果你想知道“到底是谁生成了 windows.jsonl”“为什么 worker 跑完了但 segments.json 还没变”，从这里开始看。

## 三个角色（概念上）

- Server：负责产出任务、收结果、落盘 windows/segments、触发第二阶段和导出。
- Worker：负责拿到任务后准备图片、调用模型、把 JSON 回传。
- 模型端点：外部服务（Gemini/OpenAI 兼容等）。绝大多数时间消耗在这里。

一句话：worker 负责并行干活，server 负责流程和持久化。

## Stage 1：窗口任务（最重）

### 1) 生成窗口（Server）

server 会把视频变成一组窗口：

- `window_sec`：窗口覆盖多长时间。
- `step_sec`：每次向前走多少时间。

例子：`window_sec=12`、`step_sec=6`。

- 意味着每 6 秒会产生一个新窗口。
- 相邻窗口会重叠 6 秒。

重叠不是浪费，是为了让“动作刚开始出现”的那一刻更容易被某个窗口捕捉到。

### 2) 抽帧并拼联系图（Server）

每个窗口会抽 `frames_per_window` 张逻辑帧。

如果启用联系图：

- 每张联系图是一个网格：`contact_sheet_rows` x `contact_sheet_cols`。
- 一张联系图能放 `rows * cols` 张帧。
- 需要的联系图张数约等于：

```text
sheet_count ~= ceil(frames_per_window / (rows * cols))
```

例子：`frames_per_window=128`，`rows=4`，`cols=4`。

- `rows*cols=16`
- `128/16=8`
- 每个窗口需要 8 张联系图。

这一步决定了单窗口上传给模型的数据量，是 Stage 1 主要成本之一。

### 3) 推理（Worker）

worker 的工作是“把一个窗口变成一个 JSON 结果”：

1. 从 server 拉一个 job。
2. 读取/解码 job 里的图片（联系图或单帧）。
3. 生成提示词（prompt）。
4. 调用模型端点。
5. 把返回结果规范化成我们需要的 JSON 格式。

对边界检测来说，核心输出通常是：这个窗口里哪些位置是“新动作开始”的候选点。
这些位置是窗口内的相对索引，最终会被换算成全局时间戳。

### 4) 落盘窗口结果（Server）

server 收到 worker 的 JSON 后，会把它追加写入：

- `samples/<sample_id>/windows.jsonl`

这是排查 Stage 1 的第一手证据。
你可以在里面看到：

- 一共有多少个窗口。
- 每个窗口输出了哪些 cut。
- 是否有窗口输出为空（会触发重试/降级）。

### 5) 汇总成全局切分（Server）

当窗口结果足够多，server 会把窗口内的 cut 汇总成全局 segments：

- 把窗口内索引换算成视频时间戳。
- 对重叠窗口的证据去重/合并。
- 做一些保护，避免极端过切导致段数爆炸（具体取决于配置和实现）。

输出文件是：

- `samples/<sample_id>/segments.json`

对后续所有步骤来说，segments.json 是“最终真相”。

## Stage 2：后处理（合并/字幕语言等）

Stage 2 是可选的，目的是把 Stage 1 的结果变得更可用。

为什么需要：

- Stage 1 的取向通常是“宁可多切，也别漏边界”。
- 这会带来很多很短、意义不大的碎段。

Stage 2 常做的事：

- 合并碎段：把明显的过切合回更粗粒度的段。
- 整理描述：把每段动作写成更可读的指令。
- 字幕本地化：导出字幕可选中文/英文（通常是把英文指令翻译成中文字幕）。

Stage 2 的常见失败点：

- 模型返回空内容（代理层 bug、解析 bug、输出格式不满足 JSON 要求）。
- 合并提示词太激进，把真实边界也合掉了。
- 校验条件太松，接受了不完整或乱序的合并结果。

设计上应该允许 Stage 2 失败后回退：至少还能用 Stage 1 的 segments.json 导出。

## Export：导出带字幕视频

导出会读取：

- 原视频
- `segments.json`
- 每段字幕文本（如果启用字幕）

然后生成：

- `exports/<sample_id>/annotated.mp4`

导出慢通常是本地 ffmpeg 的计算/IO，不是模型端点。

常见导出问题：

- 系统没装 ffmpeg。
- 中文字幕字体选择失败（找不到可用的 CJK 字体）。
- ffmpeg drawtext 参数转义错误（引号、冒号等）。

## 出问题时先看哪里

从“最终结果”往回查最快：

- 先看 `samples/<sample_id>/segments.json`：段数、时间戳是否合理。
- 若 segments 不合理，再看 `samples/<sample_id>/windows.jsonl`：哪些窗口贡献了错误 cut。
- 若字幕不对，看 `exports/<sample_id>/seg_XX.caption.txt` 和 Stage 2 的输出。

下一份文档会专门讲：哪些配置最影响速度/质量，以及怎么做最小代价的调参试验。

