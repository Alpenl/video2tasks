# 第二轮复核 05：导出链路 Review

更新时间：2026-04-08

## 复核对象

- 第一轮文档：`docs/reviews/round1/05-export-review.md`
- 复核代码：
  - `src/video2tasks/server/exporter.py`
  - `src/video2tasks/server/app.py`
  - `src/video2tasks/server/llm_merge.py`
  - `src/video2tasks/config.py`
- 复核测试：
  - `tests/server/test_exporter.py`
  - `tests/server/test_llm_summary.py`

说明：

- 本轮只审文档与代码，不改实现。
- 证据主要来自源码静态复核；没有补做一次真实 `ffmpeg`/多音轨样本的运行验证。
- 因此文中会区分“代码已直接证明”和“根据代码可以稳定推断”。

## 确认成立的点

### 1. `clips` 回退路径会丢音频，这个判断成立

第一轮把这点列为高优先级，方向是对的，证据也足够。

- `ClipExporter._write_clip()` 只有在“字幕文本非空且系统存在 `ffmpeg`”时才尝试 `ffmpeg` 路径，见 `src/video2tasks/server/exporter.py:175-188`。
- 其他情况都会走 OpenCV 写帧路径，见 `src/video2tasks/server/exporter.py:190-219`。
- OpenCV 分支只从 `VideoCapture` 读视频帧，再用 `VideoWriter` 写 `mp4v` 视频，没有任何音频复制逻辑。
- `subtitles.enabled=false` 时，`subtitle_text` 会被直接置空，见 `src/video2tasks/server/exporter.py:130`。这意味着在 `clips` 模式下，只要关闭字幕，就不会走保音频的 `ffmpeg` 分支。

需要补一句限定语：这里的“丢音频”是相对“输入文件原本有音频轨”而言。代码已经足以证明回退产物不含音频轨，不需要再等运行结果才能下结论。

### 2. 导出失败不阻断主流程，样本仍会写 `.DONE`，这个判断成立

第一轮这一点也成立。

- `export_sample_outputs()` 会吞掉 annotated/clips 子导出的异常，把结果折叠成 `export_reason=partial_failure` 或 `failed`，见 `src/video2tasks/server/exporter.py:433-482`。
- `app.py` 在拿到导出诊断后，仍会把 `segments.json` 写盘、清掉旧 `.FAILED`/`failure.json`、再写 `.DONE`，见 `src/video2tasks/server/app.py:1225-1256`。
- 调度层确实会因为 `.DONE` 存在而跳过样本，见 `src/video2tasks/server/app.py:693-697`。

第一轮把这点提出来是必要的，因为它直接影响“`.DONE` 到底表示什么”。

### 3. 中文字体兜底不可靠，这个判断成立

- 默认导出字幕语言是 `zh`，见 `src/video2tasks/config.py:26-30`。
- 默认字体探测把 `DejaVuSans.ttf` 也放进了 CJK 字体兜底列表，见 `src/video2tasks/server/exporter.py:21-32`。
- 代码没有检查所选字体是否覆盖实际中文字符。

第一轮说“导出成功但字幕不可读”的风险存在，这个判断是合理的。这里更准确的说法应当是“环境兼容性风险”，不是“代码必然错误”。

### 4. OpenCV 读帧失败会被当成成功结束，这个判断成立

- OpenCV 写 clip 的循环里，`capture.read()` 失败只会 `break`，不会报错，见 `src/video2tasks/server/exporter.py:210-219`。
- 外层 `export()` 仍然会把 `clip_path` 和原始 `start_frame/end_frame` 记入返回值和 `manifest.json`，见 `src/video2tasks/server/exporter.py:121-157`。

因此第一轮关于“可能产生短片/空片，但 manifest 仍写原区间”的判断成立。

### 5. 测试覆盖确实偏薄，这个判断成立

`tests/server/test_exporter.py` 只覆盖了：

- annotated 在关闭字幕时复制原视频；
- annotated 在开启字幕时调用 `ffmpeg`；
- clips 基础成功路径；
- export disabled 诊断。

第一轮列出的几类关键风险，目前都没有测试直接兜住，尤其是音频保留、回退语义、部分失败和读帧中断。

## 需要修正的点

### 1. “导出失败后样本仍会记为完成”不宜继续定为高，建议降到中

原因不是问题不存在，而是它更像“契约定义不清 + 产物管理偏弱”，不完全是逻辑错误。

- 从 `app.py` 的结构看，主流程成功标准是 `segments.json` 成功产出；导出是附加产物。
- `export_reason`、`export_errors`、`export_annotated_error`、`export_clips_error` 都会进最终诊断，见 `src/video2tasks/server/exporter.py:451-479` 与 `src/video2tasks/server/app.py:1243-1244`。

所以更准确的定级应是：

- 如果下游把 `.DONE` 理解为“主结果完成”，这是设计选择。
- 如果下游把 `.DONE` 理解为“所有导出也完整成功”，这里才会变成较严重的契约问题。

建议在二轮文档里把它写成“中优先级的契约风险”，不要直接写成“高优先级逻辑错误”。

### 2. “中文字幕兼容性不稳”建议从“中高”改成“中”

第一轮的风险判断没错，但级别略高。

- 这是环境依赖问题，不是每台机器都会触发。
- 配置已经允许显式指定 `subtitles.font_file`，见 `src/video2tasks/config.py:35`。
- 真正的问题在于默认配置和默认语言组合不够稳，不是用户完全无解。

因此更准确的表达应是：

- 默认行为对中文环境不够稳；
- 对缺字体的机器缺少更明确的预检、警告或失败策略。

### 3. `annotated.mp4` 的字幕实现扩展性一般，不适合继续放在主要问题里

第一轮第 2.5 点更像工程建议，不像已经证明的缺陷。

- 代码确实会为每个 segment 生成一个 `seg_XX.caption.txt`，也不会清理，见 `src/video2tasks/server/exporter.py:325-401`。
- 但“长视频一定明显变慢”这一层，目前没有基准或报错证据支持。

因此建议把这点移到“维护性建议”或“后续优化”，而不是和音频丢失、失败语义混排在同一层级。

### 4. 第一轮对证据类型区分不够清楚

建议把文档里的结论拆成两类：

- 代码已经直接证明的事实：例如 OpenCV 分支不保音频、读帧失败只 `break`、`.DONE` 会照常写。
- 基于代码的合理推断：例如中文字体可能显示成方块、长过滤链可能带来性能问题。

这样后续读者更容易判断哪些点应立刻修，哪些点应先补回归测试或复现脚本。

## 新增发现

### 1. `manifest.json` 会记录字幕文本，但这不等于视频里真的烧进了字幕

这是第一轮漏掉的点，而且会直接误导下游。

- `manifest_records` 里的 `subtitle` 字段永远来自 `_segment_subtitle(segment)`，见 `src/video2tasks/server/exporter.py:143-150`。
- 这个字段与实际输出解耦：
  - `subtitles.enabled=false` 时，视频里没有字幕，但 manifest 仍会写字幕文案；
  - `ffmpeg` 失败并回退到 OpenCV 时，视频里也没有字幕，但 manifest 仍会写字幕文案。

这意味着 `manifest.json` 当前更像“计划写入的字幕文本”，不是“已渲染字幕事实表”。这个问题的误导性不低，建议至少列为中。

### 2. clip 导出的失败粒度是“整批失败”，不是“单段失败可见”

第一轮提到“中途失败会留下半成品”，但没有把失败粒度说透。

- `ClipExporter.export()` 只有在 for 循环全部走完后才写 `manifest.json`，见 `src/video2tasks/server/exporter.py:121-157`。
- 一旦某个 segment 在 `_write_clip()` 里抛异常，整个 `ClipExporter.export()` 就会中断，前面已写出的 clip 不会被回滚，后面的 clip 也不会继续。
- `export_sample_outputs()` 只能在更高层得到一个 `export_clips_error`，拿不到“第几个片段失败、前面哪些已经写出”的结构化结果。

所以这里不只是“会留半成品”，还包括“失败信息粒度太粗”，排障成本偏高。

### 3. 现有测试没有覆盖第一轮文档最强调的三类核心风险

这个点第一轮有提“测试缺口”，但二轮建议说得更直白一些：

- 没有测试证明 `clips + subtitles.disabled` 时是否按预期保留或丢失音频；
- 没有测试证明 `ffmpeg` 失败后 `used_subtitle_fallback` 的诊断是否足够表达真实产物变化；
- 没有测试证明 `manifest.json` 与实际渲染状态是否一致。

因此第一轮文档里一些措辞最好加上“代码复核表明”而不是写成“测试已证实”。

## 重新排序后的建议

### 1. 高：先把 clip 导出的媒体契约说清楚并补诊断

优先处理：

- 是否必须保留音频；
- 是否允许在 `ffmpeg` 失败后回退到无音频 OpenCV 版本；
- 诊断里是否要明确写出“实际是否带音频”“实际是否烧字”“是否发生编码路径回退”。

这是当前最像真实交付风险的一点。

### 2. 中：把 `.DONE` 的含义写清楚，或拆出导出级完成标记

二选一至少做一个：

- 明确文档约定：`.DONE` 仅代表主结果完成，不代表导出完整成功；
- 或增加独立导出完成/失败标记，避免调度层和下游只看 `.DONE` 时误判。

### 3. 中：修正 `manifest.json` 的语义

建议让 manifest 能区分：

- 计划字幕文本；
- 实际是否烧字；
- 实际输出文件是否保留音频；
- 实际写出的帧区间或帧数。

否则 manifest 现在更像“请求参数回显”，不够像可靠产物清单。

### 4. 中：给 OpenCV 回退路径补完整性保护

至少需要补两类防护：

- 读帧首帧失败或实际写出帧数为 0 时，不要当成功 clip；
- 使用临时文件后再改名，减少半成品残留。

### 5. 中：给中文字体增加预检或更明确的失败/警告

建议目标是避免“导出成功但字幕不可读”的假成功。

### 6. 低：把 `caption.txt` 清理和 `drawtext` 扩展性问题移到后续优化

这类问题值得做，但从当前证据看，不应排在前四项前面。

## 二轮结论

第一轮文档抓住了两个最重要的风险：`clips` 回退丢音频，以及导出结果和 `.DONE` 语义可能脱节。这两点应当保留。

需要调整的是分级和表述：

- “导出失败后仍 `.DONE`”应从高降到中，更准确地写成契约风险；
- “中文字幕兼容性不稳”应从中高降到中；
- “大量 `caption.txt` / `drawtext` 变长”应移出主要问题区，改成后续优化建议。

另外，一轮漏掉了一个实际会误导下游的点：`manifest.json` 里的 `subtitle` 不是实际渲染事实，关闭字幕或回退后它依然会写出字幕文本。这个问题建议加入最终问题列表。
