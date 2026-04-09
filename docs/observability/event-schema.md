# Structured Event Schema (P0-2 Freeze)

本文档冻结 `video2tasks` 首批 operator-facing 结构化事件契约。

- 适用范围：`src/video2tasks/server/app.py` + `src/video2tasks/server/job_builder.py`
- 输出格式：每条事件为单行 JSON，且包含 `event` 字段
- 稳定性：本文件中的事件名与“必有字段”视为契约，后续改动需要同步更新代码与测试
- 当前实现：`src/video2tasks/logging_utils.py` 中 `FROZEN_EVENT_SCHEMAS` 与本文件一一对应

## Common Field Notes

- `subset`: 数据子集名（如 `demo_smoke`）
- `sample_id`: 样本 ID（如 `sample_001`）
- `job_type`: 任务类型（如 `window_boundary` / `boundary_refinement` / `segment_label`）
- `task_id`: 队列任务唯一 ID
- `dispatch_id`: 下发尝试 ID（如 `d1`, `d2`）
- 耗时字段（毫秒）：`artifact_extract_ms`, `infer_ms`, `submit_ms`, `finalize_ms`

## Frozen Events

### 1) `artifact_extract_done`

- 必有字段：
  `task_id`, `subset`, `sample_id`, `job_type`, `image_count`, `artifact_extract_ms`, `transport_mode`, `artifact_reuse`
- 可选字段：
  无（当前版本）

### 2) `artifact_reuse_hit`

- 必有字段：
  `task_id`, `subset`, `sample_id`, `job_type`, `artifact_reuse`, `artifact_reuse_group`, `artifact_producer_task_id`, `artifact_consumer_task_id`
- 可选字段：
  无（当前版本）

### 3) `job_dispatched`

- 必有字段：
  `task_id`, `dispatch_id`, `subset`, `sample_id`, `job_type`, `source_count`, `transport_mode`, `artifact_reuse`
- 可选字段：
  无（当前版本）

### 4) `infer_attempt`

- 必有字段：
  `task_id`, `dispatch_id`, `subset`, `sample_id`, `job_type`, `infer_ms`
- 可选字段：
  无（当前版本）

### 5) `job_done`

- 必有字段：
  `task_id`, `dispatch_id`, `subset`, `sample_id`, `job_type`, `infer_ms`, `submit_ms`
- 可选字段：
  无（当前版本）

### 6) `result_empty_retry`

- 必有字段：
  `task_id`, `dispatch_id`, `subset`, `sample_id`, `job_type`, `attempt`, `retry_limit`, `infer_ms`, `submit_ms`
- 可选字段：
  无（当前版本）

### 7) `result_timeout_retry`

- 必有字段：
  `task_id`, `dispatch_id`, `subset`, `sample_id`, `job_type`, `attempt`, `retry_limit`
- 可选字段：
  无（当前版本）

### 8) `fallback_applied`

- 必有字段：
  `subset`, `sample_id`
- 可选字段：
  `selection_policy`, `fallback_reason`
- 兼容扩展字段（当前实现可出现）：
  `*_fallback_used`, `*_fallback_reason`, `*used_subtitle_fallback`

### 9) `sample_failed`

- 必有字段：
  `subset`, `sample_id`, `reason`, `details`
- 可选字段：
  无（当前版本）

### 10) `finalize_done`

- 必有字段：
  `subset`, `sample_id`, `finalize_ms`, `segment_count`
- 可选字段：
  无（当前版本）

## Enforcement

- `log_event(...)` 对以上冻结事件执行必有字段校验：缺字段或字段值为 `None` 会抛出 `ValueError`。
- 对标识字段 `subset`、`sample_id`、`job_type`、`task_id`、`dispatch_id`，空白字符串（如 `""`、`"   "`）同样会触发 `ValueError`。
- 非冻结事件不做该校验（保持低开销与兼容性）。
