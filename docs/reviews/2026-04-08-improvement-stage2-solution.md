# 2026-04-08 Stage 2 Backend 抽象与配置契约改造方案

## 结论

这两条建议都值得做，但优先级和做法都需要收敛：

- `Stage 2 backend 抽象还没真正做完` 这个判断成立。
- 但不建议把它理解成“现在就把 Stage 2 全量改造成通用 VLM backend”。
- 更合理的目标是：把 Stage 2 从“直绑 `OpenAIBackend` 的实现细节”提升为“正式的文本结构化后处理 backend 契约”，同时允许第一阶段仍然只有 OpenAI 一个实现。
- `summary_levels / language / subtitle / backend` 的语义确实还不够直白，也值得改。
- 但不建议一次性重命名整套配置或重写 Stage 2 输出结构。当前代码里已经出现了比较清楚的方向：`run_llm_stage2_pass()` 提供独立的 Stage 2 包络，适合被扶正为正式契约。

最终判断：

- 短期执行层面，Stage 2 可以继续只有 OpenAI 这一个可用 provider。
- 中期架构层面，不应该继续把 `OpenAIBackend` 本身当成 Stage 2 契约。
- 应该提炼一个窄接口，只覆盖 Stage 2 真正需要的“文本 prompt -> JSON schema 结果 + 诊断”能力。
- 配置层应做“兼容归一化”，不是“大爆炸改名”。

## 当前实现问题

### 1. Stage 2 已经有稳定产物方向，但主调用链还停留在旧接口

`src/video2tasks/server/llm_merge.py` 里其实已经有两个层次：

- 旧接口：`run_llm_postprocess_pass()`，只返回 merge 后 segments、可选 hierarchy 和合并后的 diagnostics。
- 新接口：`run_llm_stage2_pass()`，把 merge / summary / subtitles 分成三个独立 artifact，且都带独立 diagnostics。

这说明 Stage 2 的“结果契约”并不是完全没有，只是没有成为主集成路径。

当前 `src/video2tasks/server/app.py` 仍然使用：

- `run_llm_postprocess_pass()`
- `run_export_subtitle_localization_pass()`

而不是直接消费 `run_llm_stage2_pass()`。  
结果就是模块内的稳定包络已经存在，但应用层仍在拼装旧形状，导致契约分叉。

### 2. Stage 2 backend 抽象目前是 duck typing，不是正式接口

`src/video2tasks/server/llm_merge.py` 当前对 backend 的真实要求并不是 `VLMBackend`，而是：

- 有 `infer_text_json(...)`
- 最好还带 `last_text_json_diagnostics`

但 `src/video2tasks/vlm/base.py` 的 `VLMBackend` 并没有定义这套能力。  
也就是说，Stage 2 现在实际上依赖的是“OpenAIBackend 私有扩展能力”，不是仓内正式抽象。

这带来三个问题：

- 类型边界不成立：传入 `backend: Any`，调用点只能靠约定。
- 工厂边界不成立：worker 走 `vlm.factory.create_backend(...)`，Stage 2 却在 `llm_merge.py` 里直接 `OpenAIBackend(...)`。
- 能力边界不成立：图像 VLM 推理和文本结构化后处理被强行放在一个“backend”词下，但实际约束不同。

### 3. 当前不适合把 Stage 2 直接并入 `VLMBackend`

表面上看，最省事的方案像是给 `VLMBackend` 加一个 `infer_text_json()`。  
但这不是好主意。

原因很直接：

- `VLMBackend` 当前表达的是“图片输入 + prompt -> 结构化窗口结果”。
- Stage 2 需要的是“纯文本 prompt + JSON schema -> 结构化文本结果”。
- `RemoteAPIBackend`、`Qwen3VLBackend`、`DummyBackend` 是否天然支持 Stage 2 这套严格 schema 契约，当前并不成立。
- 如果强行把 `infer_text_json()` 提升到所有 VLM backend 的统一接口，只会制造一批名义支持、实际空实现或语义不一致的 backend。

所以 Stage 2 的 backend 抽象应该是“独立的文本后处理接口”，而不是继续挤进现有 `VLMBackend`。

### 4. `llm_merge.backend` 这个名字已经失真

`src/video2tasks/config.py` 里的 `LLMMergeConfig` 现在同时控制：

- merge
- summary
- subtitle localization

但字段名还是：

- `llm_merge.backend`
- `llm_merge.summary_levels`

这两个名字都已经偏旧：

- `backend` 听起来像“Stage 1 worker 用的 VLM backend”，实际上这里只是 Stage 2 文本后处理 provider。
- `summary_levels` 虽然技术上可用，但默认形状还是位置相关数组 `[coarse, medium, fine]`，可读性和可维护性都一般。

### 5. `language / subtitle` 语义在模块内比在配置入口更清楚，但对外仍有混叠

模块内部，`run_llm_stage2_pass()` 的 `subtitles` 包络已经比旧接口清楚得多：

- `requested_language`
- `target_language`
- `language`
- `output_language`
- `source_instruction_language`
- `items`
- `diagnostics`

这里的语义已经接近稳定。

但外部仍然存在两层混叠：

- Stage 2 的字幕本地化能力和导出配置 `export.subtitles.language` 仍有强耦合。
- `segments[].export_subtitle` 仍在承担“导出消费字段”和“Stage 2 结果写回字段”两种角色。

这会让下游继续误解：

- canonical Stage 2 字幕真相到底是 `subtitles.items`，还是 `segments[].export_subtitle`？
- `language` 到底指 instruction 语言，还是 subtitle 输出语言？

## 推荐目标接口/契约

### 1. Backend 方向：不要继续直绑 `OpenAIBackend`，也不要把 Stage 2 硬塞进 `VLMBackend`

推荐目标是新增一个窄接口，例如：

```python
class Stage2TextBackend(Protocol):
    name: str
    last_text_json_diagnostics: dict[str, Any] | None

    def infer_text_json(
        self,
        prompt: str,
        *,
        schema_name: str,
        schema: dict[str, Any],
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        raise_on_http_error: bool = False,
    ) -> dict[str, Any]: ...
```

这个接口只表达 Stage 2 真正依赖的能力：

- 文本 prompt
- 强 schema JSON 输出
- 诊断信息

它不承担图片推理职责，也不要求所有 `VLMBackend` 都支持。

推荐配套增加一个小工厂，例如：

```python
create_stage2_backend(config: LLMMergeConfig) -> Stage2TextBackend
```

第一阶段实现可以只有：

- `OpenAIStage2Backend`，或者
- 直接让 `OpenAIBackend` 实现 `Stage2TextBackend`

关键点不是“类名怎么取”，而是：

- `llm_merge.py` 不再自己 `new OpenAIBackend`
- Stage 2 的依赖从具体类切换为正式接口
- “OpenAI 是当前唯一 provider”与“Stage 2 契约等于 OpenAI”这两件事被拆开

结论上，应该选择：

- 不继续直绑 `OpenAIBackend`
- 提炼正式文本后处理 backend 接口
- 但先只接 OpenAI 一个实现

### 2. Stage 2 结果契约：以 `run_llm_stage2_pass()` 为正式方向

当前最值得扶正的不是旧的 `run_llm_postprocess_pass()`，而是 `run_llm_stage2_pass()` 返回的包络：

- `merge`
- `summary`
- `subtitles`

这是比“把字段散落在 `segments` 和 `diagnostics` 上”更稳的结果结构。

推荐把它定义为正式 Stage 2 输出契约：

```json
{
  "stage": "stage2",
  "version": 2,
  "merge": {
    "applied": true,
    "segments": [...],
    "diagnostics": {...}
  },
  "summary": {
    "applied": true,
    "hierarchy": {...},
    "diagnostics": {...}
  },
  "subtitles": {
    "requested_language": "zh",
    "output_language": "zh",
    "source_instruction_language": "en",
    "applied": true,
    "items": [...],
    "diagnostics": {...}
  }
}
```

这套结构的优点是：

- merge / summary / subtitle 成败独立
- 每段 artifact 都有明确 diagnostics
- 下游不必再从 `segments` 字段形状猜测 Stage 2 有没有做、是不是 fallback

### 3. 配置契约：先做“语义归一化”，再做“字段迁移”

不建议直接把 `llm_merge` 整块改名成 `stage2`。  
那会牵涉 YAML、环境变量、README、测试、用户脚本和已有运行配置，风险偏高。

更现实的目标是：

- 保留 `llm_merge` 作为兼容入口
- 在其内部引入更直白的归一化语义

推荐的目标读法：

#### 3.1 `backend`

目标语义：

- 它不是 Stage 1 VLM backend
- 它是 Stage 2 text postprocess provider

建议新增兼容别名：

- `llm_merge.provider`
- 或 `llm_merge.text_backend`

兼容策略：

- 保留 `llm_merge.backend`
- 内部归一化为 `provider`
- 文档和 diagnostics 优先使用 `provider` 这个词

#### 3.2 `summary_levels`

目标语义：

- 它不是“魔法位置数组”
- 它是“启用哪些 summary level”

建议目标配置表达：

```yaml
llm_merge:
  summary:
    enabled_levels: [coarse, medium, fine]
```

兼容策略：

- 继续接受旧格式 `[1, 1, 1]`
- 继续接受现有命名映射 `{coarse: 1, medium: 0, fine: 1}`
- 内部统一归一化成显式 level name 列表，或至少显式 named mapping

也就是说，`summary_levels` 值得重构，但方式应是“新增更直白表示 + 保留旧输入”。

#### 3.3 `language / subtitle`

推荐把语义拆成三层：

- `requested_language`：调用方请求的语言
- `output_language`：最终 `items[].subtitle` 的实际语言
- `source_instruction_language`：instruction 的原始语言

当前 `run_llm_stage2_pass()` 已经基本这样做了，所以这里不需要推倒重来。  
真正要做的是：

- 让这套语义成为对外正式契约
- 不再让 `export.subtitles.language` 充当 Stage 2 canonical 配置来源

建议目标配置表达：

```yaml
llm_merge:
  subtitles:
    target_language: zh
```

兼容策略：

- 如果 `llm_merge.subtitles.target_language` 未设置，则回退读 `export.subtitles.language`
- 保持现有导出配置不失效
- 逐步把“字幕语言真相”从 export 配置迁到 Stage 2 配置

#### 3.4 `subtitle`

建议明确两个概念：

- `subtitles.items`：Stage 2 的 canonical 字幕结果
- `segments[].export_subtitle`：导出层消费的冗余映射字段

如果继续把 `export_subtitle` 当作 Stage 2 真相，下游会一直绕不开旧耦合。  
因此推荐契约上明确：

- `subtitles.items` 才是正式结果
- `export_subtitle` 只是为 exporter 保留的兼容镜像

## 兼容迁移方案

### 1. 第一阶段：抽出接口，不改主业务形状

目标：

- 不改 Stage 2 行为
- 不改最终写盘大形状
- 只把 backend 依赖从具体类切到正式接口

做法：

1. 新增 `Stage2TextBackend` 协议或抽象基类。
2. 新增 `create_stage2_backend(...)`。
3. 让 `llm_merge.py` 统一通过 resolver/factory 获取 backend。
4. `run_llm_merge_pass()`、`run_llm_summary_pass()`、`run_llm_subtitle_localization_pass()` 不再直接 `OpenAIBackend(...)`。

收益：

- 立即消除“Stage 2 契约 = OpenAI 实现”的硬编码。
- 不要求 `RemoteAPIBackend`、`GeminiBackend` 同步支持。
- 不影响当前 OpenAI-only 的实际可用性。

### 2. 第二阶段：扶正 `run_llm_stage2_pass()`，旧接口保留为 wrapper

目标：

- 模块侧只维护一个正式 Stage 2 结果契约。

做法：

1. 把 `run_llm_stage2_pass()` 标成 canonical API。
2. 让 `run_llm_postprocess_pass()` 变成兼容 wrapper，从 Stage 2 包络里裁剪旧返回值。
3. 让导出字幕路径也基于同一个 Stage 2 结果源，而不是单独再组织一套半平铺结果。

收益：

- 避免新旧 Stage 2 形状长期并存。
- merge / summary / subtitles 的状态位和 diagnostics 有统一来源。

### 3. 第三阶段：配置做别名迁移，不强推旧字段失效

目标：

- 提高语义可读性
- 不破坏已有 YAML / env 配置

做法：

1. 为 `llm_merge.backend` 增加 `provider` 别名。
2. 为 `llm_merge.summary_levels` 增加显式的 `summary.enabled_levels` 读法。
3. 为 `export.subtitles.language` 的 Stage 2 用途增加 `llm_merge.subtitles.target_language` 正式入口。
4. 继续兼容旧环境变量，同时新增更直白命名的变量。

建议的兼容顺序：

1. 先新增新字段和归一化访问器
2. README 与示例配置改用新名字
3. diagnostics 输出优先写新语义字段
4. 旧字段至少保留一个稳定版本周期

### 4. 第四阶段：应用层再切换到新包络

目标：

- 避免一次性重写 finalize 流程

做法：

1. `app.py` 先继续产出当前 `segments.json` 主形状。
2. 在内部改为先拿 `run_llm_stage2_pass()` 结果。
3. 再把其中：
   - `merge.segments` 映射回 `final_res["segments"]`
   - `summary.hierarchy` 映射回 `final_res["task_hierarchy"]`
   - `subtitles.items` 映射到导出层需要的 `export_subtitle`
4. 等下游准备好后，再决定是否把完整 `stage2` 包络稳定写盘。

这是最小兼容改造路径，因为它避免了：

- 一次性重写 Stage 2
- 一次性改坏 app finalize
- 一次性让所有下游消费者改读新 JSON

## 第一批落地范围

高收益、低风险的第一批改造建议如下：

### 1. 抽 `Stage2TextBackend` 正式接口，并集中 backend 创建

原因：

- 这是当前 Stage 2 backend 抽象问题的最短闭环。
- 只改依赖边界，不改算法逻辑。

### 2. 把 `run_llm_stage2_pass()` 定为模块正式契约，旧函数留 wrapper

原因：

- 代码里已经有更好的包络，不需要重新设计。
- 这是“契约已存在但未扶正”的典型高收益项。

### 3. 明确 `subtitles.items` 是 canonical，`export_subtitle` 只是兼容镜像

原因：

- 这一步能立刻减少 Stage 2 与导出层的语义混叠。
- 不需要立刻改 exporter 行为。

### 4. 为配置增加更直白的兼容别名

优先建议：

- `llm_merge.provider` 兼容 `llm_merge.backend`
- `llm_merge.subtitles.target_language` 兼容 `export.subtitles.language`
- `llm_merge.summary.enabled_levels` 兼容 `summary_levels`

原因：

- 这是降低理解成本的最低风险做法。
- 比直接推翻现有 YAML 结构安全得多。

### 5. 补接口级测试，而不是先做大规模重构

优先测试方向：

- Stage 2 backend resolver 只接受实现了文本 JSON 契约的 provider
- `run_llm_stage2_pass()` 与 legacy wrapper 返回结果一致
- `provider / summary enabled levels / subtitle target language` 的兼容归一化行为
- `subtitles.items` 与 `segments[].export_subtitle` 的镜像关系

## 验证计划

建议按“契约验证”而不是“功能堆砌”来验收。

### 1. Backend 契约验证

- OpenAI 仍然能跑通 merge / summary / subtitle 三条路径。
- `llm_merge.py` 内不再直接出现 `OpenAIBackend(...)`。
- Stage 2 工厂返回的是正式接口对象，不是 `Any` 约定。

### 2. 兼容性验证

- 旧配置 `llm_merge.backend=openai` 仍然可用。
- 旧配置 `summary_levels=[1,0,1]` 仍然可用。
- 旧配置 `export.subtitles.language=zh` 在未设置 Stage 2 新字段时仍能驱动字幕语言。
- 旧的 `run_llm_postprocess_pass()` 调用点行为不变。

### 3. 结果契约验证

- `run_llm_stage2_pass()` 始终返回 merge / summary / subtitles 三段包络。
- subtitle fallback 时，`requested_language` 与 `output_language` 不混淆。
- 下游无需读取 diagnostics 也能判断 canonical 字幕结果在哪里。

### 4. 应用层回归验证

- 当前 `segments.json` 形状保持兼容。
- 导出路径仍能读取 `export_subtitle`。
- 不导出字幕时，Stage 2 结果契约仍然成立，不因 export 开关而消失。

## 结语

这次改造最关键的取舍不是“要不要支持更多 provider”，而是“要不要先把 Stage 2 的正式边界立住”。

我的建议是：

- 立边界，但先不扩 provider 数量。
- 扶正现有 `run_llm_stage2_pass()`，而不是再设计第三套结果形状。
- 用兼容别名和归一化读法修复 `summary_levels / language / subtitle / backend` 的语义问题。

这样可以用最小改动，把当前 Stage 2 从“OpenAI 特例实现”推进到“只有一个实现、但边界正式”的状态。这是当前收益最高、风险最低的路径。
