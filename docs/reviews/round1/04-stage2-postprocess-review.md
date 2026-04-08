# 第一轮审计 04：第二阶段后处理

## 范围

只看第二阶段后处理相关实现，不改代码。

本次主要审计了这些位置：

- `src/video2tasks/server/app.py`
- `src/video2tasks/server/llm_merge.py`
- `src/video2tasks/server/exporter.py`
- `src/video2tasks/config.py`
- `src/video2tasks/prompt.py`
- `tests/server/test_llm_merge.py`
- `tests/server/test_llm_summary.py`
- `tests/server/test_exporter.py`

重点关注：

- 过切合并策略
- `summary_levels` 的数据结构
- 字幕语言切换
- 最终输出 JSON 是否稳定
- 第二阶段慢点
- 校验与 fallback

## 先说结论

这一段的主线是清楚的：先 merge，再 summary，再字幕本地化，再导出。保护逻辑也不算薄，特别是 merge 这块，不只是靠提示词，代码里还有分区校验、边界保护、过度塌缩保护、粗粒度共识选择，方向是对的。

真正更值得担心的，不是“它会不会乱合”，而是“第二阶段的结果到底有没有稳定写回最终产物”“失败时为什么连 summary 一起降级”“语言切换是不是名实相符”。这几项会直接影响下游可用性。

## 主要问题

### 1. 已本地化的字幕没有写回最终 `segments.json`

这是本轮最明确的问题。

证据：

- 文档把 Stage 2 的输出写成“更新后的 `segments.json` 和字幕文本”，见 `docs/principles/00-why-and-overview.md:63-69`。
- 文档还把 `samples/<sample_id>/segments.json` 定义成“最终真相”，见 `docs/principles/10-main-pipeline.md:86-90`。
- 但实际代码里，`run_export_subtitle_localization_pass(...)` 只作用在单独的 `export_segments` 上，见 `src/video2tasks/server/app.py:1197-1209`。
- 最后写盘的是 `final_res`，不是带 `export_subtitle` 的 `export_segments`，见 `src/video2tasks/server/app.py:1243-1247`。
- `export_subtitle` 字段只在本地化函数里临时挂到 segment 上，见 `src/video2tasks/server/llm_merge.py:699-704`，随后只被导出器消费，见 `src/video2tasks/server/exporter.py:130-155`、`src/video2tasks/server/exporter.py:333-359`。

影响：

- 最终 `segments.json` 里看不到已经生成好的字幕文本。
- 如果后续想重做导出、换导出模式、核对字幕语言，不能只靠“最终真相”文件完成。
- 当前实现更像是“字幕只服务导出”，不是“字幕是第二阶段产物的一部分”。
- 这和仓内文档的说法不一致，会让下游误判。

建议：

- 最简单的做法是把 `export_subtitle` 一并写回 `final_res["segments"]`。
- 如果不想污染主段结构，至少也应该在最终 JSON 里落一个明确的 `subtitles` 块，按 `seg_id` 保存。
- 不管选哪种，建议把“最终 JSON 是否包含字幕文本”定成稳定约定，不要只留在 manifest 或临时变量里。

### 2. merge 请求失败时，summary 会被一起跳过，导致层级结果无谓降级

这不是 crash，但会让输出质量被 merge 的瞬时失败拖下去。

证据：

- `run_llm_postprocess_pass(...)` 先跑 merge，再决定是否跑 summary，见 `src/video2tasks/server/llm_merge.py:1618-1648`。
- `_should_skip_live_summary_after_merge(...)` 会在 merge 空响应、请求失败、后端初始化失败时直接返回 `True`，见 `src/video2tasks/server/llm_merge.py:681-697`。
- 一旦命中这个分支，就不会再请求 summary，而是直接走 identity fallback，见 `src/video2tasks/server/llm_merge.py:1631-1637`。
- 现有测试也明确验证了这一行为，见 `tests/server/test_llm_summary.py:299-324`。

影响：

- 同一份已清洗的 segments，只因为 merge 接口一时超时，`task_hierarchy` 就会从真实 summary 退化成逐段 identity 层级。
- 这会让第二阶段 JSON 的语义波动很大，波动来源不是视频内容变化，而是 merge 通道的偶发性失败。
- 从职责上看，merge 失败并不等于 summary 一定也失败；当前把两者绑得过紧。

建议：

- merge 失败时，仍然可以拿 `cleaned_segments` 继续尝试 summary。
- 只有 summary 自己失败时，再退到 identity fallback。
- 如果担心额外时延，可以加开关，而不是把“跳过 summary”写死在 merge 失败策略里。

### 3. 字幕语言切换不是对称能力，当前更接近“中文翻译 + 英文复用”

这个点不一定马上出错，但名字和实际能力不完全一致。

证据：

- 配置层只接受 `zh/en`，见 `src/video2tasks/config.py:24-61`。
- 提示词层也只支持 `zh/en`，见 `src/video2tasks/prompt.py:691-744`。
- 真正执行时，`zh` 会走 LLM，本地化生成字幕；`en` 则完全不请求模型，直接复用原始 instruction，见 `src/video2tasks/server/llm_merge.py:1549-1551`、`src/video2tasks/server/llm_merge.py:1580-1615`。
- 测试同样把 `en` 定义为“复用 source instruction”，见 `tests/server/test_llm_summary.py:442-457`。

影响：

- 如果未来 source instruction 不再稳定是英文，`language=en` 的语义就会失真。
- 从配置名字看，用户会以为 `zh` 和 `en` 都是“目标语言切换”；实际上只有 `zh` 是翻译，`en` 是直通。
- unsupported language 也只是静默 fallback 到 source instruction，并不会给出真正的语言转换能力，见 `src/video2tasks/server/llm_merge.py:1553-1556`。

建议：

- 要么把能力描述写清楚：当前只支持“source instruction”与“中文导出字幕”。
- 要么把配置改成更直白的值，比如 `source` / `zh`，避免把 `en` 包装成对称翻译。
- 如果后面真要支持双向切换，建议把“源语言”和“目标语言”拆开表达。

### 4. `summary_levels` 对人和下游都不够直白，回退后更容易看不懂

这是结构清晰度问题，不是单点 bug，但已经影响可读性和稳定约定。

证据：

- 配置里 `summary_levels` 是一个位置相关的整数数组 `[coarse, medium, fine]`，见 `src/video2tasks/config.py:294-297`。
- 校验也只认长度为 3 的 `0/1` 数组，见 `src/video2tasks/config.py:406-417`。
- 环境变量入口 `LLM_MERGE_SUMMARY_LEVELS` 也是整数列表，不带名字，见 `src/video2tasks/config.py:614-615`。
- 输出层又同时给了 `enabled_levels` 和 `enabled_level_names`，见 `src/video2tasks/server/llm_merge.py:924-929`。
- summary fallback 会把每个 enabled level 都退成按 segment 一一对应的 identity 分区，见 `src/video2tasks/server/llm_merge.py:865-879`、`src/video2tasks/server/llm_merge.py:965-976`。

影响：

- 人看配置时必须记住固定顺序，成本偏高。
- 下游如果只看 `enabled_levels`，也必须知道“第 0 位是 coarse，第 1 位是 medium，第 2 位是 fine”。
- fallback 后不同层级可能都变成一模一样的逐段结果，JSON 形式上合法，但语义上已经很弱，不容易一眼识别。

建议：

- 配置更适合改成命名结构，例如 `{coarse: true, medium: false, fine: true}`。
- 最终 JSON 可以保留名字化结构，减少位置耦合。
- fallback 时建议再加一个更显眼的层级标记，比如 `task_hierarchy.mode = "identity_fallback"`，不要只藏在 diagnostics 里。

### 5. 第二阶段的主要慢点很明确，尤其在 coarse merge 和串行远程调用

这块不算设计错误，但会在规模上放大。

证据：

- coarse merge 支持 `repeat_count` 多次采样，见 `src/video2tasks/config.py:298-307`、`src/video2tasks/server/llm_merge.py:1257-1291`。
- 每次采样又可能走到 `max_attempts` 重试，见 `src/video2tasks/server/llm_merge.py:587-665`、`src/video2tasks/config.py:289-293`。
- merge、summary、subtitle 三段是严格串行的，入口见 `src/video2tasks/server/app.py:1185-1233`。
- 这三段默认各自新建一次 `OpenAIBackend`，见 `src/video2tasks/server/llm_merge.py:1214-1224`、`src/video2tasks/server/llm_merge.py:1452-1462`、`src/video2tasks/server/llm_merge.py:1563-1573`。
- 导出如果是 clips 模式，每段 clip 都会单独切一次并可能单独 ffmpeg 重编码，见 `src/video2tasks/server/exporter.py:121-139`、`src/video2tasks/server/exporter.py:221-285`。

影响：

- 粗粒度 merge 的远程请求数大致是 `repeat_count * max_attempts` 的级别，随后还要再接 summary 和 subtitle。
- 当视频段数多、coarse 共识开启、接口偶发空响应时，第二阶段耗时会明显拉长。
- 导出 clips 时，段数越多，时间越线性变差。

建议：

- coarse merge 可以在达到足够共识后提前停止，不一定要把 `repeat_count` 全跑完。
- 三个文本阶段可以共用同一个 backend/client，减少重复初始化。
- 如果最终不导出字幕，就没必要做 subtitle localization。
- 如果只需要 `segments.json`，导出应完全延后，避免把视频重编码成本混到主路径里。

## 过切合并策略：好的部分

这部分值得单独记一下，避免只看到问题。

- merge 不是纯提示词驱动，先做 partition 校验，再做 guard 清洗，最后还有输出比例保护，见 `src/video2tasks/server/llm_merge.py:120-164`、`src/video2tasks/server/llm_merge.py:473-503`、`src/video2tasks/server/llm_merge.py:1003-1050`。
- coarse 模式没有盲信模型，而是结合 boundary hints、anchor 保护和多次采样共识，见 `src/video2tasks/prompt.py:513-560`、`src/video2tasks/server/llm_merge.py:324-471`、`src/video2tasks/server/llm_merge.py:1053-1156`、`src/video2tasks/server/llm_merge.py:1314-1347`。
- 这块测试覆盖是够的，特别是 coarse guard、anchor、共识选择，主要都在 `tests/server/test_llm_merge.py`。

我的判断：

- 当前 merge 最大的风险已经不是“完全没保护”，而是“保护和回退很多，但这些结果怎么对外表达还不够稳定”。

## 输出 JSON 稳定性判断

当前最终 JSON 不是一个很稳定的契约，更像一个“按路径动态拼出来的结果对象”。

主要原因：

- `task_hierarchy` 是可选字段，只有非空时才写入，见 `src/video2tasks/server/app.py:1191-1193`。
- `diagnostics` 里大量字段是按分支出现的，例如 merge 成功、merge 共识、summary fallback、subtitle fallback、export partial failure 都会带来不同 key 集合，见 `src/video2tasks/server/llm_merge.py:1169-1387`、`src/video2tasks/server/llm_merge.py:1402-1519`、`src/video2tasks/server/llm_merge.py:1534-1615`、`src/video2tasks/server/exporter.py:414-483`。
- 已生成的字幕文本没有写回最终 `segments.json`，见上面的第 1 条。

如果这个文件后面要给评测脚本、前端、导出器之外的其他消费者长期依赖，建议尽快把“必有字段 / 可选字段 / fallback 时的固定形状”定清楚。

## 校验与 fallback 判断

优点：

- merge、summary、subtitle 都有结构校验，不是直接吃模型输出。
- summary 还额外校验了跨层嵌套关系，见 `src/video2tasks/server/llm_merge.py:838-862`。
- subtitle 校验要求按 `seg_id` 全覆盖，见 `src/video2tasks/server/llm_merge.py:716-738`。

问题不在“有没有 fallback”，而在“fallback 暴露给下游是否足够明显”：

- summary identity fallback 在数据形状上和正常 hierarchy 很像，只是 diagnostics 里写了 reason，见 `src/video2tasks/server/llm_merge.py:933-976`。
- subtitle fallback 也是直接回 source instruction，下游如果只看 `export_subtitle` 文本本身，不看 diagnostics，很难知道这是不是模型翻译产物，见 `src/video2tasks/server/llm_merge.py:1544-1615`。

建议：

- fallback 不要只靠 diagnostics 解释。
- 最终 JSON 最好给出面向下游的显式状态位，例如 `task_hierarchy_status`、`subtitle_status`。

## 测试覆盖和空白

已经覆盖到的：

- merge guard、coarse anchor、coarse 共识：`tests/server/test_llm_merge.py`
- summary 成功、fallback、merge 失败后 summary 跳过：`tests/server/test_llm_summary.py`
- subtitle 中文、本地回退、英文复用：`tests/server/test_llm_summary.py`
- exporter 基础路径、manifest、ffmpeg 路径：`tests/server/test_exporter.py`

我没有找到的关键覆盖：

- app 层最终写出的 `segments.json` 是否真的包含第二阶段应有产物，尤其是字幕文本。
- app 层在不同开关组合下，最终 JSON 形状是否稳定。
- `summary_levels` 不同组合下，最终 JSON 是否容易被下游正确消费。
- `export.mode=both` 的 app 级联动验证。

## 建议优先级

建议优先顺序：

1. 先解决字幕结果没有写回最终 `segments.json` 的问题。
2. 再把 merge 失败与 summary 失败解耦。
3. 然后把 `summary_levels` 和最终 hierarchy 的对外表达改得更直白。
4. 最后再做 coarse merge 提前收敛、backend 复用、导出延后这些性能优化。

## 本轮判断

如果只看“能不能跑”，这段代码已经能跑。

如果看“后续是不是容易接、容易查、容易稳定复现”，第二阶段还有三处要尽快补齐：

- 最终 JSON 和文档定义不一致
- merge 失败会连带压掉 summary
- 语言切换和层级配置的对外语义不够直白
