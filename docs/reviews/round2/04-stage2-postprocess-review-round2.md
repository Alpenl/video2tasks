# 第二轮复核 04：第二阶段后处理

## 复核对象

- 主审文档：`docs/reviews/round1/04-stage2-postprocess-review.md`
- 回看代码：
  - `src/video2tasks/server/app.py`
  - `src/video2tasks/server/llm_merge.py`
  - `src/video2tasks/server/exporter.py`
  - `src/video2tasks/config.py`
  - `src/video2tasks/prompt.py`
- 回看测试：
  - `tests/server/test_llm_summary.py`
  - `tests/server/test_llm_merge.py`
  - `tests/server/test_exporter.py`
  - `tests/server/test_app_retry.py`
- 对照文档：
  - `docs/principles/00-why-and-overview.md`
  - `docs/principles/10-main-pipeline.md`

本轮目标不是重复写一遍第一轮，而是核对第一轮文档本身有没有证据不足、级别不准、遗漏关键点。

## 确认成立的点

### 1. 高：本地化字幕没有写回最终 `segments.json`

这个判断成立，而且是当前最实的问题之一。

证据：

- `run_llm_postprocess_pass(...)` 的结果会写回 `final_res["segments"]`，见 `src/video2tasks/server/app.py:1185-1193`。
- 字幕本地化是在 `export_segments` 这份拷贝上做的，不是直接改 `final_res["segments"]`，见 `src/video2tasks/server/app.py:1197-1209`。
- 最终写盘的是 `final_res`，不是带 `export_subtitle` 的 `export_segments`，见 `src/video2tasks/server/app.py:1243-1247`。
- `export_subtitle` 字段确实只在字幕本地化函数里挂到 segment 上，见 `src/video2tasks/server/llm_merge.py:699-706`、`src/video2tasks/server/llm_merge.py:1549-1615`。
- 导出器消费的也是这个临时字段，见 `src/video2tasks/server/exporter.py:93-97`、`src/video2tasks/server/exporter.py:127-150`、`src/video2tasks/server/exporter.py:333-359`。

影响判断：

- 如果下游把 `samples/<sample_id>/segments.json` 当成唯一稳定输入，它拿不到已经本地化好的字幕文本。
- 这不是导出层的小细节，而是“第二阶段产物是否进入最终结果”的契约问题。

需要补充的一点：

- 第一轮这里说“文档和代码不一致”是对的，但证据应写得更准确。`docs/principles/00-why-and-overview.md:69` 说 Stage 2 输出“更新后的 `segments.json` 和字幕文本”，同时同一文档前面又把字幕文本列成 `exports/<sample_id>/seg_XX.caption.txt` 这类导出产物，见 `docs/principles/00-why-and-overview.md:30-35`。也就是说，这里不是单纯“文档说 A、代码做 B”，而是“文档内部就有两套说法，代码选择了偏导出产物的那一套”。

### 2. 中高：merge 请求失败时，summary 会被一起跳过

这个问题也成立，第一轮证据是够的。

证据：

- `_should_skip_live_summary_after_merge(...)` 会在 `empty_response`、`request_failed:*`、`backend_init_failed:*` 和 adapter 失败前缀时直接返回 `True`，见 `src/video2tasks/server/llm_merge.py:682-696`。
- `run_llm_postprocess_pass(...)` 命中这个分支后，不会再调 `run_llm_summary_pass(...)`，而是直接构造 identity fallback，见 `src/video2tasks/server/llm_merge.py:1618-1648`。
- 测试明确锁定了这个行为，见 `tests/server/test_llm_summary.py:299-324`。

影响判断：

- 这会把“merge 通道偶发失败”放大成“summary 结果整体降级”。
- 不过它仍然有 fallback，不会导致崩溃或空输出，所以级别更接近“中高”，不是最高优先级故障。

### 3. 低到中：`summary_levels` 的表达不直白，fallback 形状也不够显眼

这一条大方向成立，但更像契约可读性问题，不该和前两条放在同一强度。

证据：

- 配置确实是位置相关整数数组 `[coarse, medium, fine]`，见 `src/video2tasks/config.py:294-296`、`src/video2tasks/config.py:406-417`、`src/video2tasks/config.py:614-615`。
- 输出层会额外补 `enabled_level_names`，见 `src/video2tasks/server/llm_merge.py:882-930`。
- fallback 时每个启用层级都会退成逐段 identity 分区，见 `src/video2tasks/server/llm_merge.py:933-976`。

判断：

- 这里的问题不是“代码错了”，而是“人和下游要多记一层映射关系，fallback 也不够一眼可见”。
- 这应归到中低优先级的契约清晰度改进，不宜和写盘缺失、错误耦合并列成主要问题。

### 4. 正向确认：merge 保护逻辑确实存在，而且不是摆设

第一轮这一段判断是稳的，应该保留。

证据：

- coarse 路径会根据配置做 prompt boundary hints、重复采样和候选筛选，见 `src/video2tasks/server/llm_merge.py:1214-1295`。
- 多候选时还会做 boundary 共识或 candidate 共识选择，见 `src/video2tasks/server/llm_merge.py:1314-1347`。
- 相关测试并不少，第一轮对这部分“保护逻辑不薄”的结论基本准确，见 `tests/server/test_llm_merge.py`。

## 需要修正的点

### 1. “字幕语言切换不是对称能力”被说重了

第一轮把这条放进主要问题，不够准确。

原因：

- 配置描述已经明说“Only export subtitles change language; source instructions remain unchanged.”，见 `src/video2tasks/config.py:27-30`。
- 文档也把常见路径写成“通常是把英文指令翻译成中文字幕”，见 `docs/principles/10-main-pipeline.md:101-105`。
- `en` 走 source instruction 直通，在当前“源 instruction 默认就是英文”的假设下，是有意的快捷路径，不一定算 defect，见 `src/video2tasks/server/llm_merge.py:1549-1551`、`tests/server/test_llm_summary.py:442-458`。
- `prompt_segment_subtitles(...)` 虽然支持 `zh/en` 两种提示词模板，见 `src/video2tasks/prompt.py:691-744`，但运行路径对 `en` 做了短路，这更像实现取舍和文档表述问题。

更合适的定级：

- 低优先级。
- 应写成“能力命名和实现路径不完全对称，未来如果 source instruction 不再稳定是英文，会变成真实问题”，而不是当前主要问题。

### 2. “第二阶段主要慢点很明确”不该列进主要问题

第一轮的性能分析方向没错，但证据强度不够支撑它进入主要问题区。

原因：

- 代码只能证明存在串行调用、多次采样和重复初始化，见 `src/video2tasks/server/app.py:1185-1233`、`src/video2tasks/server/llm_merge.py:1214-1224`、`src/video2tasks/server/llm_merge.py:1453-1462`、`src/video2tasks/server/llm_merge.py:1564-1573`。
- 这不能直接证明当前仓库里“已经慢到需要先处理”，因为没有 profile、耗时统计、样本规模数据。
- 仓内原则文档本身也把 Stage 2 慢点描述成一般性风险，不是当前已确认的缺陷，见 `docs/principles/20-speed-quality-knobs.md:93-103`。

更合适的定级：

- 观察项或后续优化项。
- 可以保留建议，但不宜排在契约和结果质量问题前面。

### 3. “最终 JSON 不稳定”表述过宽，应该收窄成更具体的契约问题

第一轮这里有道理，但说法太泛。

原因：

- `task_hierarchy` 可选、`diagnostics` 分支化，并不自动等于“契约不稳定”；这也可能只是设计上的可选字段。
- 当前更实的契约问题其实是两件事：
  - 本地化字幕没有进入最终 `segments.json`
  - fallback 状态主要藏在 diagnostics，中间结果和正常结果对下游不够易辨认

更合适的写法：

- 不要把所有分支字段都打成“最终 JSON 不稳定”。
- 应聚焦在“哪些字段必须稳定存在、哪些结果需要显式状态位”。

### 4. 第一轮遗漏了一个会缓和影响判断的事实：成功导出时，字幕文本并不是完全不落盘

这不推翻第 1 条问题，但会影响影响面的描述。

证据：

- clips 导出会把 `subtitle` 写进 `manifest.json`，见 `src/video2tasks/server/exporter.py:143-150`，测试也覆盖了这一点，见 `tests/server/test_exporter.py:100-103`。
- annotated 导出会把每段字幕写进 `seg_XX.caption.txt`，见 `src/video2tasks/server/exporter.py:333-339`。

结论：

- 真正的问题不是“字幕文本完全没有落盘”。
- 真正的问题是“字幕文本没有进入 `segments.json` 这个被文档描述为最终真相的文件，而且其生成还被导出链路绑定”。

## 新增发现

### 1. 高：字幕本地化被 `export` 开关硬绑定，不是独立的 Stage 2 产物

这是第一轮没有点透，但比“没写回最终 JSON”更根的问题。

证据：

- 只有 `config.export.enabled` 和 `config.export.subtitles.enabled` 都为真时，app 才会调用 `run_export_subtitle_localization_pass(...)`，见 `src/video2tasks/server/app.py:1197-1209`。
- 否则直接写 `export_disabled` 或 `subtitles_disabled` 诊断，不会生成任何本地化字幕结果，见 `src/video2tasks/server/app.py:1210-1223`。
- 但原则文档把 Stage 2 写成“可选做第二阶段：合并过切、整理字幕语言，然后导出”，见 `docs/principles/00-why-and-overview.md:14`；后面又把 Stage 2 输入输出写成独立于导出的过程，见 `docs/principles/00-why-and-overview.md:59-75`。

影响：

- 现在的实现不是“Stage 2 负责字幕本地化，导出负责消费字幕”，而是“只有要导出时，才顺手做字幕本地化”。
- 这会让 Stage 2 的职责边界和文档描述继续打架。
- 这也解释了为什么字幕既没有稳定进入 `segments.json`，也没有作为独立 Stage 2 产物被保留下来。

### 2. 中：缺少 app 级测试来锁定最终写盘契约，尤其是字幕和导出耦合路径

第一轮说“app 层最终写出的 `segments.json` 缺少关键覆盖”是对的，但还可以更具体。

证据：

- 当前测试主要覆盖的是：
  - `run_llm_postprocess_pass(...)` 的 merge/summary 组合，见 `tests/server/test_llm_summary.py:245-324`
  - `run_export_subtitle_localization_pass(...)` 的字幕行为，见 `tests/server/test_llm_summary.py:353-478`
  - `export_sample_outputs(...)` 和 exporter 行为，见 `tests/server/test_exporter.py:76-120`
- `tests/server/test_app_retry.py` 主要覆盖重试和 JSONL 落盘，不覆盖最终 `segments.json` 的第二阶段结果形状。

影响：

- 现在最关键的契约问题恰好跨越 `app.py`、`llm_merge.py`、`exporter.py` 三层，但测试是分开的。
- 这也是第一轮能指出“写回缺失”，却没有现成 app 级测试能直接钉住它的原因。

## 重新排序后的建议

### 1. 先定清楚字幕契约，再谈具体实现

先回答一个问题：本地化字幕到底是不是 Stage 2 的正式产物。

如果答案是“是”：

- 就应该把字幕稳定写进 `segments.json`，或者至少写进一个同级、稳定、可按 `seg_id` 取回的结构。
- 同时把字幕本地化从 `export.enabled` 的条件里拆出来。

如果答案是“不是，只服务导出”：

- 那就应该改文档，明确 `segments.json` 不负责保存本地化字幕。
- 同时把“最终真相”这个说法收窄，不要暗示仅靠 `segments.json` 就能完整重建 Stage 2 结果。

### 2. 然后把 merge 失败和 summary 尝试解耦

这条是第二优先级。

- merge 失败不应自动取消 summary 尝试。
- 只有 summary 自己失败时，再退回 identity fallback，会更符合职责边界。

### 3. 补 app 级契约测试

在决定好字幕契约后，优先补以下测试：

- 最终 `segments.json` 是否包含预期的 Stage 2 产物
- `export.enabled=false` 时字幕本地化是否还会发生，或者明确不会发生
- `export.mode=annotated/clips/both` 下，最终写盘和导出产物是否一致
- fallback 时，下游需要依赖的状态位是否稳定

### 4. 再处理 `summary_levels` 和 fallback 可读性

这部分值得做，但它不该排在契约问题前面。

- 可以考虑名字化配置或更显眼的 fallback 状态位。
- 目标是降低理解成本，而不是修复一个已经会出错的行为。

### 5. 性能优化放最后

在没有 profile 之前，不建议把“提前停止 coarse 采样”“复用 backend”“延后导出”排到前面。

- 它们是合理优化方向。
- 但当前更值得先修的是结果契约和质量回退逻辑。

## 本轮结论

第一轮最有价值的两条结论仍然成立：

- 本地化字幕没有进入最终 `segments.json`
- merge 失败会把 summary 一起降级

但第一轮也有三处需要修正：

- 把“语言切换不对称”说成主要问题，说重了
- 把性能观察放进主要问题，证据还不够
- 对“文档不一致”的描述不够精确，实际上是文档内部先不一致，代码再偏向其中一种解释

另外，第一轮漏掉了一个更关键的点：

- 字幕本地化现在被导出开关硬绑定，尚未形成独立、稳定的 Stage 2 产物契约
