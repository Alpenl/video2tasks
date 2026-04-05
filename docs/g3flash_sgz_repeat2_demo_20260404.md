# Gemini 3 Flash 这一轮有效配置记录

本文档记录当前项目里一次结果相对稳定、边界表现还可以的实跑配置，便于后续复现、对比和继续优化。

## 1. 这轮结果对应的 run

- 样本：`yc_sGzBQrg1adY`
- 数据源：[data/youcook2_stitched/yc_sGzBQrg1adY](/home/alpen/DEV/video2tasks/data/youcook2_stitched/yc_sGzBQrg1adY)
- 本轮 run id：`g3flash_sgz_repeat2_demo_20260404`
- 输出目录：[runs/demo_single_sGzBQrg1adY/g3flash_sgz_repeat2_demo_20260404/samples/yc_sGzBQrg1adY](/home/alpen/DEV/video2tasks/runs/demo_single_sGzBQrg1adY/g3flash_sgz_repeat2_demo_20260404/samples/yc_sGzBQrg1adY)
- 最终切分结果：[segments.json](/home/alpen/DEV/video2tasks/runs/demo_single_sGzBQrg1adY/g3flash_sgz_repeat2_demo_20260404/samples/yc_sGzBQrg1adY/segments.json)
- 官方边界来源：[source_meta.json](/home/alpen/DEV/video2tasks/data/youcook2_stitched/yc_sGzBQrg1adY/source_meta.json)

说明：

- 这个 stitched benchmark 没有单独落一份 `official_segments.json`。
- 这里的官方边界取自 `source_meta.json` 里每段 clip 的 `stitched_end_frame`。

## 2. 这轮“实际生效”的配置

### 2.1 配置文件基线

基础配置文件是 [config.g3flash.yaml](/home/alpen/DEV/video2tasks/config.g3flash.yaml)。

关键项如下：

- 后端：`gemini`
- 模型：`gemini-3-flash-preview`
- 接口模式：`openai_compatible`
- Base URL：`https://api.duckcoding.ai`
- `timeout_sec: 60`
- `max_output_tokens: 2048`
- `jpeg_quality: 85`
- `window_sec: 12.0`
- `step_sec: 6.0`
- `frames_per_window: 128`
- `window_repeat_count: 2`
- `use_contact_sheets: true`
- `contact_sheet_rows: 4`
- `contact_sheet_cols: 4`
- `target_width: 720`
- `target_height: 480`
- `adaptive_merge_guard: true`
- `adaptive_merge_min_segments: 8`
- `adaptive_merge_collapse_ratio: 0.6`
- `boundary_support_threshold: 0.9`
- `refine_final_instructions: true`
- `enable_refinement_pass: false`
- `enable_boundary_refinement: false`
- `worker.count: 4`
- `server.max_retries_per_job: 50`
- `server.max_empty_retries_per_job: 0`

### 2.2 这次实跑时的运行时覆盖

这轮最终跑通并产出结果时，不是完全按配置文件默认值运行，而是带了下面这些运行时参数：

```bash
DATASETS="tmp:demo_single_sGzBQrg1adY"
RUN_ID="g3flash_sgz_repeat2_demo_20260404"
PORT=8107
SERVER_URL="http://127.0.0.1:8107"
WORKER_COUNT=2
PYTHONPATH=src python3 -m video2tasks.cli.cluster --config config.g3flash.yaml
```

因此需要明确区分：

- 配置文件默认 `worker.count` 是 `4`
- 这轮最终出结果时，实际使用的是 `WORKER_COUNT=2`

也就是说，如果要“完全复刻这轮”，应以 `config.g3flash.yaml` 为基线，并额外覆盖：

- `DATASETS="tmp:demo_single_sGzBQrg1adY"`
- `RUN_ID="g3flash_sgz_repeat2_demo_20260404"`
- `PORT=8107`
- `SERVER_URL="http://127.0.0.1:8107"`
- `WORKER_COUNT=2`

## 3. 这轮配置的核心含义

### 3.1 窗口策略

- 单窗口长度 `12s`
- 步长 `6s`
- 每个窗口抽 `128` 个 logical frames
- 上传前拼成 contact sheets
- 每个窗口上传 `8` 张大图
- 每张大图是 `4 x 4 = 16` 帧拼图

换句话说，这轮的视觉输入策略可以概括为：

- `12s / 6s`
- `128 logical frames`
- `8` 张上传图
- 每张图 `4x4`

### 3.2 稳定性策略

- `window_repeat_count: 2`
  - 同一窗口重复跑两次，再做 boundary voting
- `boundary_support_threshold: 0.9`
  - 边界支持度足够高时，后续 merge 更难把它吞掉
- `adaptive_merge_guard: true`
  - 防止后处理把切分过度压平
- `server.max_retries_per_job: 50`
  - 普通失败有较高重试预算
- `server.max_empty_retries_per_job: 0`
  - 在当前实现里会被当成 `inf`
  - 也就是空 JSON / 空结构化结果会持续回队，不因为空返次数而停止

### 3.3 这轮没有启用的东西

这轮结果不是靠 refinement pass 拿到的，下面这些都关闭了：

- `enable_refinement_pass: false`
- `enable_boundary_refinement: false`
- `refinement_frames_per_window: 0`
- `boundary_refinement_frames_per_window: 0`

因此，这轮结果的主要来源是：

- 更密的滑窗覆盖
- contact sheet 大图输入
- 同窗口重复两次后的边界投票
- 较强的 retry / requeue 机制

## 4. 这轮结果

最终结果：

- 视频总帧数：`4800`
- 预测段数：`69`
- 预测边界数：`68`
- 官方边界数：`15`

按项目当前 official boundary recall 口径：

- `±5` 帧：`12 / 15 = 0.800`
- `±10` 帧：`13 / 15 = 0.867`
- `±15` 帧：`13 / 15 = 0.867`
- `±30` 帧：`14 / 15 = 0.933`

这轮 miss 的关键点：

- `2350`，最近预测边界 `2373`，偏 `+23`
- `2900`，最近预测边界 `2890`，偏 `-10`
- `3750`，最近预测边界 `3833`，偏 `+83`

其中：

- `2900` 在 `±10` 下已经算命中
- `3750 -> 3833` 是这一轮最明显的失准点

## 5. 对后续复现最重要的结论

如果后面要复现这轮“切得比较碎、边界表现还可以”的状态，优先保留下面这组组合，而不是只盯单个参数：

- `gemini-3-flash-preview`
- `openai_compatible` 模式
- `https://api.duckcoding.ai`
- `12s / 6s`
- `128 logical frames`
- `8` 张 `4x4` contact sheets
- `window_repeat_count = 2`
- `boundary_support_threshold = 0.9`
- `adaptive_merge_guard = true`
- live 运行时用 `WORKER_COUNT=2`
- 空返持续回队，不要因为 empty JSON 提前终止

## 6. 备注

- 这是一份“效果记录文档”，不是最终最优配置定论。
- 当前结果是单样本 `yc_sGzBQrg1adY` 的实跑记录，适合拿来做后续对比基线。
- 如果后续继续调 prompt、merge 逻辑或 representative point 选择，建议都先和这份配置做 A/B 对比。
