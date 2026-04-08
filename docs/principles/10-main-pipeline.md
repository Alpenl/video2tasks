# 主流程怎么跑（Stage 1 -> Stage 2 -> Export）

这份文档按“谁负责什么”讲清楚主流程。
如果你想知道“到底是谁生成了 windows.jsonl”“为什么 worker 跑完了但 segments.json 还没变”，从这里开始看。
本文统一用 `<run_dir>` 表示 `<run.base_dir>/<subset>/<run_id>`；默认示例是 `./runs/<subset>/<run_id>`。

## 三个角色（概念上）

- Server：负责产出任务、收结果、落盘 windows/segments、触发第二阶段和导出。
- Worker：负责拿到任务后准备图片、调用模型、把 JSON 回传。
- 模型端点：外部服务（Gemini/OpenAI 兼容等）。绝大多数时间消耗在这里。

一句话：worker 负责并行干活，server 负责流程和持久化。

## 运行契约（冻结语义）

- 部署模式是 `single-machine shared-fs`，Server/Worker 必须共享同一路径视图。
- `.DONE` 表示当前配置要求的全部必需阶段完成（由 `<run_dir>/run_manifest.json.required_stages` 定义）。
- `segments.json` 只放 segmentation + Stage 2 文本产物，不承载 run/export/fallback 的最终真相。
- source instruction 永远英文；subtitle localization 是 Stage 2 正式 artifact。
- resume 默认拒绝跨 config/prompt/backend/required-stages 续跑，除非显式 force。
- clips 导出必须保留音频；音频不保留属于导出契约失败。

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

- `<run_dir>/samples/<sample_id>/windows.jsonl`

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

- `<run_dir>/samples/<sample_id>/segments.json`

`segments.json` 只承载分段结果与 Stage 2 文本层结果。
run/resume/export/fallback 等运行态事实在 manifest 与 diagnostics，不和分段结果混成一个“最终真相”。

## Stage 2：后处理（合并/字幕语言等）

Stage 2 是可选的，目的是把 Stage 1 的结果变得更可用。

为什么需要：

- Stage 1 的取向通常是“宁可多切，也别漏边界”。
- 这会带来很多很短、意义不大的碎段。

Stage 2 常做的事：

- 合并碎段：把明显的过切合回更粗粒度的段。
- 整理描述：把每段动作写成更可读的指令。
- 字幕本地化：导出字幕可选中文/英文（通常是把英文指令翻译成中文字幕）。
  这是 Stage 2 的正式文本产物；source instruction language 固定为 `en`。

Stage 2 的常见失败点：

- 模型返回空内容（代理层 bug、解析 bug、输出格式不满足 JSON 要求）。
- 合并提示词太激进，把真实边界也合掉了。
- 校验条件太松，接受了不完整或乱序的合并结果。

关于失败处理（required-stages）：如果 Stage 2 在当前配置下属于必需阶段，Stage 2 失败就不能标记 `.DONE`。

## Export：导出带字幕视频

导出会读取：

- 原视频
- `<run_dir>/samples/<sample_id>/segments.json`
- 每段字幕文本（来自 Stage 2 字幕本地化产物）

然后（若启用导出且导出成功）生成的产物取决于 `export.mode`：

- `export.mode=annotated|both`：`<run_dir>/exports/<sample_id>/annotated.mp4`（默认目录名 `exports`，文件名 `annotated.mp4`）
- `export.mode=clips|both`：`<run_dir>/clips/<sample_id>/seg_XX_*.mp4` 以及 `<run_dir>/clips/<sample_id>/manifest.json`（默认目录名 `clips`）

对 clips，`manifest.json` 里每条记录都应满足 `audio_preserved=true`。

导出慢通常是本地 ffmpeg 的计算/IO，不是模型端点。

常见导出问题：

- 系统没装 ffmpeg。
- 中文字幕字体选择失败（找不到可用的 CJK 字体）。
- ffmpeg drawtext 参数转义错误（引号、冒号等）。

## 出问题时先看哪里

从“最终结果”往回查最快：

- 先看 `<run_dir>/samples/<sample_id>/segments.json`：段数、时间戳是否合理。
- 若 segments 不合理，再看 `<run_dir>/samples/<sample_id>/windows.jsonl`：哪些窗口贡献了错误 cut。
- 若字幕不对，先看 `<run_dir>/samples/<sample_id>/segments.json` 里的 Stage 2 字幕结果，再看对应 `diagnostics`。
- 若导出不对，按 `export.mode` 检查：`<run_dir>/exports/<sample_id>/annotated.mp4`（annotated|both）与 `<run_dir>/clips/<sample_id>/`（clips|both，包括 `manifest.json`），并重点看 clips 的 `audio_preserved` 字段。

下一份文档会专门讲：哪些配置最影响速度/质量，以及怎么做最小代价的调参试验。
