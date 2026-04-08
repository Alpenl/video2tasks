# 这个项目解决什么问题（10 分钟概览）

这个仓库把一个长视频变成两类东西：

1. 一组带时间戳的“任务片段”（segments），每段对应一个相对完整的动作。
2. （可选）一个导出视频（例如 `annotated.mp4`），用于把每段动作的字幕渲染到视频里。是否产出以及字幕是否实际烧录，取决于导出相关开关与导出是否成功。

核心思路很朴素：

- 不尝试一次看懂整段视频。
- 把视频切成很多重叠的时间窗口（window）。
- 每个窗口只问模型一个窄问题：哪里出现了“新动作的开始”。
- 把所有窗口的答案拼起来，得到全局的切分结果。
- 可选做第二阶段：合并过切、整理字幕语言，然后导出。

这么设计主要是为了三件事：

- 可以并行：窗口之间互不依赖，同一台机器上可以跑多个 worker 进程。
- 可以恢复：少量窗口失败不会把整段视频的结果毁掉。
- 可以调参：速度和质量主要由几个配置控制，而不是改代码。

## 你会得到哪些文件

默认输出根目录是 `./runs`（可在配置里改）。本文统一用 `<run_dir>` 表示 `<run.base_dir>/<subset>/<run_id>`；默认示例是 `./runs/<subset>/<run_id>`。
部署模式固定为 `single-machine shared-fs`：Server 和 Worker 必须看到同一套本地路径。

对单个样本（sample）通常会看到：

- `<run_dir>/samples/<sample_id>/windows.jsonl`
  第一阶段每个窗口的原始结果（每行一个窗口）。
- `<run_dir>/samples/<sample_id>/segments.json`
  结果层产物：只放全局切分结果 + Stage 2 文本产物（merge/summary/subtitle localization）。
  source instruction 永远是英文；字幕本地化只改变字幕文本。
- `<run_dir>/run_manifest.json`
  run 级身份和契约文件：记录 config/prompt/backend identity、`required_stages`、resume 校验信息。
  resume 默认拒绝跨 identity 续跑；只有显式 force（`run.force_resume=true` 或 `RUN_FORCE_RESUME=true`）才放行。
- `<run_dir>/samples/<sample_id>/.DONE`
  样本完成标记：表示该样本完成了当前配置要求的全部必需阶段（即 `run_manifest.json.required_stages`）。
- `<run_dir>/samples/<sample_id>/.FAILED`（以及 `<run_dir>/samples/<sample_id>/failure.json`）
  样本失败标记与失败详情。
- `<run_dir>/exports/<sample_id>/annotated.mp4`（若 `export.mode=annotated|both` 且导出成功）
  annotated 导出产物。
- `<run_dir>/clips/<sample_id>/...`（若 `export.mode=clips|both` 且导出成功）
  clips 导出产物；同时会写 `<run_dir>/clips/<sample_id>/manifest.json`。
  clips 导出必须保留音频（`audio_preserved=true`）。
- `diagnostics` / 各类 manifest
  运行态事实（run/export/fallback state）放在 manifest 与 diagnostics，不和最终切分真相混在一起。

调试时你可能还会看到 `tmp/` 下的中间产物（例如联系图、日志等），是否写入取决于配置和代码路径。

## 三个阶段的直觉

### 阶段 1：窗口级边界检测（最耗时）

输入：原视频。

做的事：

1. 生成窗口：`window_sec` 的窗口长度，每次前进 `step_sec`。
2. 在窗口里抽 `frames_per_window` 张逻辑帧。
3. 把帧拼成联系图（一个网格：`contact_sheet_rows` x `contact_sheet_cols`）。
4. 把联系图 + 提示词发给视觉模型。
5. 模型返回：这个窗口里哪些位置是“新动作开始”。

输出：`<run_dir>/samples/<sample_id>/windows.jsonl`。

为什么慢：每个窗口都是一次多模态请求，而且窗口通常有重叠（这是为了不漏边界）。

### 阶段 2：后处理（合并/总结/字幕本地化，通常更便宜）

输入：阶段 1 的 segments / windows。

做的事（取决于开关）：

- 把明显的过切合并掉（从很多小段变成更可读的段）。
- 生成或整理每段的动作描述。
- 字幕本地化：把字幕在中英文之间转换。它是 Stage 2 的正式 artifact，而不是临时导出副产物。
  source instruction language 固定为 `en`，不会因为字幕语言切换而改写。

输出：更新后的 `<run_dir>/samples/<sample_id>/segments.json`。
这里的 Stage 2 文本结果就是结果层的一部分，不需要再去导出目录里找“最终真相”。

这个阶段通常比阶段 1 便宜，因为主要是文本处理，不需要上传图片。

### 导出：渲染带字幕的视频

输入：原视频 + `<run_dir>/samples/<sample_id>/segments.json` + 每段字幕（若有）。

做的事：

- 按 segments 切片。
- （若启用字幕且渲染成功）用 ffmpeg 把字幕烧录进去。
- （若导出成功）生成 `<run_dir>/exports/<sample_id>/annotated.mp4`（`export.mode=annotated|both`）或 `<run_dir>/clips/<sample_id>/...`（`export.mode=clips|both`）。
- clips 导出必须保音频；音频丢失属于导出契约失败。

导出一般比阶段 1 快很多；如果导出慢，多数是本机 CPU/磁盘或 ffmpeg 参数问题。

## 为什么要“窗口 + 联系图”

长视频很难一次请求就稳定地产生可靠切分。

窗口的好处：

- 把问题缩小到 10 到 20 秒级别，模型更容易回答。
- 可并行，吞吐主要靠同机 worker 并发堆起来。
- 允许用重叠换边界命中（`step_sec` 越小越密）。

联系图的好处：

- 把很多帧压成少量图片，避免一次请求上传上百张独立图片。

代价：联系图越清晰、越多，上传字节越大，模型推理时间也越长。

## “慢”一般慢在哪里

经验上：只要阶段 1 没跑完，先默认慢在阶段 1。

常见原因：

- 窗口太多：`step_sec` 太小，或视频太长。
- 单窗口太重：联系图太多/分辨率太高。
- `window_repeat_count > 1` 直接把阶段 1 的工作量翻倍。
- 远端模型端点排队/限流导致单窗口延迟波动很大。
- 重试触发（空 JSON、超时、HTTP 429/5xx）。

下一份文档会把主流程按“Server 做什么、Worker 做什么、哪些文件什么时候出现”串起来。
