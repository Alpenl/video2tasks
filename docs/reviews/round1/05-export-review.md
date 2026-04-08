# 第一轮审计 05：导出链路 Review

更新时间：2026-04-07  
范围：只看导出链路，重点是字幕文本、`ffmpeg` 调用、`annotated.mp4`、分段 `clip`、字体与编码、导出失败后的行为。  
核心文件：`src/video2tasks/server/exporter.py`，以及它在 `src/video2tasks/server/app.py`、`src/video2tasks/server/llm_merge.py` 的调用。

## 1. 先给结论

这条导出链路已经能产出两类结果：

1. 整段带字幕视频：`<run_dir>/exports/<sample_id>/annotated.mp4`
2. 分段 clip 和清单：`<run_dir>/clips/<sample_id>/manifest.json` 与若干 `seg_*.mp4`

但当前实现有几个比较实在的问题：

1. `clip` 导出在“无字幕”或“ffmpeg 失败回退”时，会静默丢失音频，而且诊断信息没有把这件事说清楚。
2. 导出失败后，样本仍会被标成 `.DONE`，同时可能留下半成品导出文件；如果下游只看目录，不一定知道这些文件不完整。
3. 中文字幕依赖本机字体，默认兜底里还包含不适合中文的字体，存在“导出成功但中文显示成方块/空白”的风险。
4. `clip` 导出对读帧失败过于宽松，可能生成短片甚至空片，但 `manifest.json` 仍按原区间写成功记录。
5. 长视频或段数很多时，`annotated.mp4` 的字幕方案会变慢，也会在导出目录里留下大量 `.caption.txt` 文件。

## 2. 主要问题

### 2.1 高：`clip` 导出在回退路径里会静默丢失音频

现象：

- `ClipExporter.export()` 里，只有在“字幕文本非空且系统里有 `ffmpeg`”时，才走 `ffmpeg` 分支。
- 只要字幕关闭，或者 `ffmpeg` 失败，就会走 OpenCV 回退分支。
- OpenCV 回退只写视频帧，不处理音频。

代码证据：

- `src/video2tasks/server/exporter.py:130`
- `src/video2tasks/server/exporter.py:165`
- `src/video2tasks/server/exporter.py:176`
- `src/video2tasks/server/exporter.py:190`
- `src/video2tasks/server/exporter.py:204`
- `src/video2tasks/server/exporter.py:217`
- `src/video2tasks/server/exporter.py:269`
- `src/video2tasks/server/exporter.py:281`

影响：

- 当用户配置 `export.mode=clips` 且 `subtitles.enabled=false` 时，所有 clip 默认都走 OpenCV，结果是音频直接丢失。
- 当字幕开启但 `ffmpeg` 因字体、编码、路径或命令问题失败时，也会静默回退到无音频版本。
- 诊断字段只有 `used_subtitle_fallback`，字面意思像是“字幕回退”，但实际发生的是“字幕没了，音频也没了，编码器也变了”。

为什么这是问题：

- 这会让导出输入输出契约变得不稳定。调用方以为自己只是关掉字幕，实际拿到的是另一种媒体文件。
- 这属于潜在数据丢失，而且现在的诊断不够直白。

### 2.2 高：导出失败后样本仍会记为完成，而且可能留下半成品

现象：

- `export_sample_outputs()` 会分别尝试 `annotated` 和 `clips`，失败只记到 `diagnostics`。
- `app.py` 在拿到导出诊断后，仍然会写 `segments.json`、删除旧的失败标记、再写 `.DONE`。
- 导出过程中已经生成的文件不会回滚或清理。

代码证据：

- `src/video2tasks/server/exporter.py:113`
- `src/video2tasks/server/exporter.py:154`
- `src/video2tasks/server/exporter.py:301`
- `src/video2tasks/server/exporter.py:405`
- `src/video2tasks/server/exporter.py:451`
- `src/video2tasks/server/exporter.py:470`
- `src/video2tasks/server/exporter.py:474`
- `src/video2tasks/server/app.py:1225`
- `src/video2tasks/server/app.py:1243`
- `src/video2tasks/server/app.py:1246`
- `src/video2tasks/server/app.py:1251`
- `src/video2tasks/server/app.py:1256`

对比：

- 真正的样本失败路径会删除 `segments.json` 和 `.DONE`，并写 `.FAILED` 与 `failure.json`。
- 这套逻辑在 `src/video2tasks/server/app.py:405` 到 `src/video2tasks/server/app.py:428`。
- 但导出阶段不会走这条失败路径。

影响：

- `both` 模式下，只要其中一类导出成功，另一类失败，目录里就会留下半套产物。
- `clips` 模式下，如果中途某个片段失败，前面已经写出的 clip 会留在目录里，但可能没有完整的 `manifest.json`。
- 调度层后续会因为 `.DONE` 存在而跳过这个样本，见 `src/video2tasks/server/app.py:693`。

这不一定是逻辑错误：

- 从代码看，当前设计明显倾向“导出是附加能力，不阻断主流程”。
- 但这条契约需要写清楚，否则用户很容易把 `.DONE` 理解成“导出也完整成功”。

### 2.3 中高：中文字幕兼容性不稳，可能“导出成功但字不可读”

现象：

- 默认字幕语言是 `zh`。
- 导出时会优先找几种 CJK 字体；如果都没有，就会兜底到 `DejaVuSans.ttf`。
- 代码没有检查这个字体是否真的能覆盖中文字符。

代码证据：

- `src/video2tasks/config.py:23`
- `src/video2tasks/config.py:24`
- `src/video2tasks/server/exporter.py:21`
- `src/video2tasks/server/exporter.py:27`
- `src/video2tasks/server/exporter.py:240`
- `src/video2tasks/server/exporter.py:322`
- `src/video2tasks/server/llm_merge.py:1549`
- `src/video2tasks/server/llm_merge.py:1554`

影响：

- 在没有 Noto/WQY 这类中文字体的机器上，`ffmpeg drawtext` 可能仍然执行成功，但字幕显示成方块、问号或空白。
- 这类问题不会触发异常，`AnnotatedVideoExporter` 和 `ClipExporter` 都可能把它记成“成功导出字幕”。

补充判断：

- 英文路径相对安全。目标语言是 `en` 时，系统直接复用 `instruction`，不会再走字幕本地化请求。
- 中文路径依赖两件事：一是本地化能拿到 `export_subtitle`，二是本机字体真能画出这些字。

### 2.4 中：`clip` 导出对读帧失败过宽松，可能把短片/空片当成功

现象：

- OpenCV 回退里，写帧循环一旦 `capture.read()` 失败，只是 `break`，不会抛错。
- 上层仍会把这个片段加入 `clip_paths`，并把原始 `start_frame/end_frame` 写进 `manifest.json`。

代码证据：

- `src/video2tasks/server/exporter.py:121`
- `src/video2tasks/server/exporter.py:142`
- `src/video2tasks/server/exporter.py:154`
- `src/video2tasks/server/exporter.py:210`
- `src/video2tasks/server/exporter.py:213`

影响：

- 如果源视频尾部损坏、帧数比元数据少、或者 seek 到了读不到的位置，最终可能得到时长不足的 clip。
- 更差的情况是首帧就读不到，还是可能留下一个空的或几乎空的文件。
- `manifest.json` 记录的是“想导出的区间”，不是“实际写出的区间”，这会误导下游。

### 2.5 中：`annotated.mp4` 的字幕实现扩展性一般，目录也会留下很多中间文本

现象：

- 整段带字幕视频的实现，是为每个 segment 各写一个 `seg_XX.caption.txt`，再拼成一长串 `drawtext` 过滤器。
- 这些 `.caption.txt` 文件不会清理。

代码证据：

- `src/video2tasks/server/exporter.py:325`
- `src/video2tasks/server/exporter.py:337`
- `src/video2tasks/server/exporter.py:359`
- `src/video2tasks/server/exporter.py:383`
- `src/video2tasks/server/exporter.py:401`

影响：

- 段数多时，`-vf` 参数会很长，命令构造和 `ffmpeg` 解析都会变重。
- 导出目录里会混入很多中间文件，路径组织不够干净。
- 同样的问题也存在于 clip 的 `*.caption.txt`。

## 3. 现在这条导出链路的输入输出契约

### 3.1 输入

调用入口在 `src/video2tasks/server/app.py:1225`。

传给 `export_sample_outputs()` 的关键输入是：

- `run_dir`
- `sample_id`
- `video_path`
- `fps`
- `segments`
- `export_config`

其中每个 segment 至少默认依赖这些字段：

- `start_frame`
- `end_frame`
- `instruction`

可选字段：

- `seg_id`
- `export_subtitle`

现在的现实情况是：

- 导出前没有对 `start_frame/end_frame` 做更严格的校验和纠正。
- 也没有把“必须保留音频”“必须成功烧字”“必须全部导出成功”这种更强的要求写进函数契约。

### 3.2 输出

`annotated` 模式：

- 输出路径：`<run_dir>/<annotated_dirname>/<sample_id>/<annotated_name>`
- 默认就是 `exports/<sample_id>/annotated.mp4`

`clips` 模式：

- 输出路径：`<run_dir>/<clips_dirname>/<sample_id>/`
- 默认包含 `manifest.json` 和一组 `seg_*.mp4`

诊断输出：

- `export_reason` 只区分 `disabled`、`empty_segments`、`applied`、`partial_failure`、`failed`
- 还会附带 `export_annotated_*` 和 `export_clip_*` 一些字段

契约缺口：

- `export_reason=applied` 不代表字幕一定烧进去了。
- `export_clips_used_subtitle_fallback=true` 也不代表“只是少了字幕”，因为回退路径还可能丢音频。
- 现在没有“实际导出帧区间”“实际是否保留音频”“每个产物是否完整”的明确字段。

## 4. 中文 / 英文字幕兼容情况

### 4.1 英文

- 英文目标语言走的是最简单路径。
- `run_export_subtitle_localization_pass()` 在 `target_language=en` 时直接复用原 `instruction`。
- 这一段相对稳，主要风险反而是长英文句子的换行效果一般。

### 4.2 中文

- 中文目标语言会尝试生成 `export_subtitle`，再由 `ffmpeg drawtext` 烧字。
- 真正的兼容性关键不在文本编码本身，`caption.txt` 已经按 `utf-8` 写了；关键在字体。
- 也就是说，编码处理还可以，字体兜底不够可靠。

### 4.3 混合中英文本

- `_wrap_caption()` 的换行规则比较简单：
  - 有空格就按英文单词切
  - 没空格就按固定字符数切
- 这对纯英文、纯中文都能工作，但对“中英混排、带长 token、带文件名、带路径”的字幕不够友好，显示宽度容易不均匀。

## 5. 导出失败后的行为

当前行为可以概括成：

1. 导出失败会记录到 `diagnostics`
2. 主结果 `segments.json` 仍会落盘
3. 样本仍会被标记 `.DONE`
4. 旧的 `.FAILED` / `failure.json` 还会被清掉

这带来的实际效果是：

- 主流程结果和导出结果被视为两层不同成功标准。
- 如果团队认同“导出只是附加产物”，这套行为可以接受。
- 但如果下游把导出文件当成正式交付物，这套行为就偏危险，因为它不会触发重试，也不会保留失败状态。

## 6. 路径组织与潜在数据丢失点

### 路径组织

优点：

- 每个样本都有独立目录，整体上不乱。
- `annotated` 和 `clips` 分目录，主意图是清楚的。

不足：

- 字幕中间文件直接混在产物目录里。
- clip 文件名对中文不友好。`_slugify()` 只保留 `[a-z0-9]`，中文 instruction 最终大多变成 `segment`，虽然有序号不至于撞名，但可读性差。

### 潜在数据丢失点

1. OpenCV 回退会直接丢音频。
2. OpenCV 读帧失败只会提前结束，不会报错。
3. `ffmpeg` 或 OpenCV 直接写最终路径，没有“先写临时文件、成功后再改名”的保护。
4. 导出中途失败时，已经写出的半成品不会清理。

## 7. 性能与可维护性建议

下面这些更像下一步建议，不是要求立即改：

1. 对整段字幕视频，优先考虑改成单个字幕文件方案，比如 `SRT/ASS`，不要给每个 segment 叠一个 `drawtext`。
2. 对 clip 导出，把“是否保留音频”变成明确契约和显式诊断字段，不要藏在回退路径里。
3. 对每个导出文件使用“临时文件 + 原子改名”，避免留下半成品。
4. 对失败策略补一个更清晰的开关：
   - “导出失败不影响主流程”
   - 或“导出失败视为样本失败”
5. 对中文字幕增加字体检查，至少在 `zh` 模式下不要默认兜底到不可靠字体。
6. 把 `caption.txt` 视为临时文件，导出成功后清理，或统一放进单独临时目录。
7. 给导出结果补更细的诊断：
   - 是否保留音频
   - 是否实际烧字
   - 每个 clip 实际写出多少帧
   - 是否发生回退，以及回退原因

## 8. 测试缺口

当前测试主要覆盖了“禁用导出”和“基础成功路径”，见 `tests/server/test_exporter.py`。

还缺这些更关键的用例：

1. `clips + subtitles.disabled` 时音频是否保留。
2. `ffmpeg` 失败后 clip 回退时，诊断是否能明确反映音频变化。
3. 中文字体缺失时，是否能给出明确失败或警告。
4. `both` 模式下一半成功一半失败时，目录里会留下什么。
5. 读帧中断、空片段、越界帧区间时，`manifest.json` 是否仍然可信。

## 9. 审计判断

如果只看“能不能跑出文件”，这条链路已经可用。  
如果看“导出契约是否稳、字幕中英文是否可靠、失败后是否容易误判成功”，当前实现还不够稳。

我会优先把下面三件事视为最需要尽快处理的点：

1. `clip` 回退路径的静默丢音频
2. 导出失败后仍 `.DONE` 且可能残留半成品
3. 中文字体兜底不可靠但会被记为成功
