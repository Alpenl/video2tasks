# Round 1 VLM Integration Review

## 范围

本轮只看模型接入层，重点是 `src/video2tasks/vlm/` 里的 Gemini、OpenAI 兼容端点、请求构造、重试、超时、流式解析、JSON 提取、失败回退。为了判断实际影响，也补看了少量上层调用点和测试：

- `src/video2tasks/vlm/gemini_api.py`
- `src/video2tasks/vlm/openai_api.py`
- `src/video2tasks/vlm/remote_api.py`
- `src/video2tasks/vlm/base.py`
- `src/video2tasks/vlm/factory.py`
- `src/video2tasks/worker/runner.py`
- `src/video2tasks/server/llm_merge.py`
- `tests/vlm/test_gemini.py`
- `tests/vlm/test_openai.py`
- `tests/vlm/test_openai_api.py`

## 主要发现

### 1. [高] Gemini 的多层重试会把接口抖动放大成很长等待，空结果时尤其明显

证据：

- `src/video2tasks/vlm/gemini_api.py:117` 到 `src/video2tasks/vlm/gemini_api.py:164` 的 `_post_json()` 在请求异常或 408/409/425/429/5xx 时会做 4 次请求，间隔 2/4/6 秒。
- `src/video2tasks/vlm/gemini_api.py:293` 到 `src/video2tasks/vlm/gemini_api.py:339` 的 `_request_with_payload_retries()` 在 HTTP 200 但结构化结果为空时，又会做 5 轮 payload 级重试，之后再做 2 轮 curl 回退。
- `src/video2tasks/worker/runner.py:342` 到 `src/video2tasks/worker/runner.py:364` 的 worker 外层还会再做 4 次本地重试。

影响：

- 接口轻微波动会被放大成分钟级等待。以默认 `timeout_sec=60` 估算，Gemini 一次 `infer()` 在“前几次超时，最后返回 200 但内容仍为空”的路径上，理论上可拖到约 23 分钟；再叠加 worker 的 4 次本地重试，单任务最坏可接近 94 分钟。
- 这不只是慢。任务长时间占住 worker，会让队列积压、超时变多、结果变旧，最后反过来压低整体质量。
- 现在的重试条件太粗，只要“结构化结果为空”就继续重试，没区分是网络抖动、模型空话、解析失败、还是响应形状变了。后几种情况通常不是多试几次就会好。

建议：

- 给整次调用加总时长预算，不要让内层和外层无限叠。
- 只对明确可恢复的情况重试，比如连接错误、429、5xx。
- 对“200 但空内容”“解析失败”“字段形状不对”分开记原因，再决定是否继续。

### 2. [中] OpenAI 文本结构化路径把“非 api.openai.com 主机”一律当成代理，直接优先走流式，兼容层比较脆

证据：

- `src/video2tasks/vlm/openai_api.py:285` 到 `src/video2tasks/vlm/openai_api.py:289` 的 `_prefer_stream_first_for_text_json()` 只要主机不是 `api.openai.com`，就优先走流式。
- `src/video2tasks/vlm/openai_api.py:611` 到 `src/video2tasks/vlm/openai_api.py:629` 的 `infer_text_json()` 会把这个判断直接用于 merge/summary/export 这类文本结构化请求。
- `src/video2tasks/vlm/openai_api.py:472` 到 `src/video2tasks/vlm/openai_api.py:480` 显示，一旦走这个分支，会先尝试 `chat/completions + stream`，而不是先用 `responses + strict json_schema`。

影响：

- 这会把“是否用严格结构化接口”绑定到域名，而不是绑定到能力。很多兼容服务、企业代理、网关、自建转发域名，其实也支持 `/responses` 或者至少更适合先走非流式严格输出。
- 一旦先走流式，就从“让服务端按 schema 约束输出”退成“本地拼接文本再猜 JSON”。接口一抖，质量和耗时都会更不稳定。
- 这条路径主要影响 `llm_merge` 一类文本结构化功能，不一定直接影响 worker 图片推理，但会影响后处理质量。

建议：

- 不要只看 host 名称。改成显式配置，或者做一次能力探测后缓存。
- 默认还是应先走最严格、最稳的结构化接口，流式更适合做明确的回退，而不是默认首选。

### 3. [中] OpenAI 的兼容提取逻辑比 Gemini 的兼容提取逻辑更窄，同类代理返回会出现“一边能过，一边空返回”

证据：

- `src/video2tasks/vlm/openai_api.py:193` 到 `src/video2tasks/vlm/openai_api.py:240` 的 `_extract_chat_completions_payload_with_reason()` 只看 `message.parsed` 和 `message.content`。
- `src/video2tasks/vlm/gemini_api.py:218` 到 `src/video2tasks/vlm/gemini_api.py:258` 的 `_collect_openai_text_candidates()` / `_extract_openai_compatible_payload()` 还会继续找 `reasoning_content`、`arguments` 和更深层的 `text.value`。
- `tests/vlm/test_gemini.py:265` 到 `tests/vlm/test_gemini.py:307` 明确覆盖了 Gemini 兼容层从 `reasoning_content` 恢复结果；没有找到 OpenAI 后端对应的同类保护用例。

影响：

- 对一些 OpenAI 兼容端点来说，真正的结构化文本不一定放在 `message.content`。如果它放在 `reasoning_content` 或函数参数里，Gemini 兼容层能吃到，OpenAI 后端却会判成空结果。
- 这会让“同一个兼容服务，换个 backend 名字结果就不同”，对排障很不友好，也让兼容层行为不可预测。

建议：

- 统一 OpenAI 与 Gemini 兼容层的文本候选提取规则。
- 给 OpenAI 后端补上 `reasoning_content`、`arguments`、嵌套文本值的测试夹具。

### 4. [中] OpenAI 流式解析假定每个 `data:` 行都是完整 JSON，而且没有显式关闭流连接，遇到兼容网关时有误判风险

证据：

- `src/video2tasks/vlm/openai_api.py:330` 到 `src/video2tasks/vlm/openai_api.py:347` 逐行读取 SSE，只认 `data:` 开头，并把每一行都当作一个完整 JSON 事件来 `json.loads()`。
- 这段逻辑没有处理多行 `data:` 合并，也没有使用 `event:` 信息。
- `src/video2tasks/vlm/openai_api.py:306` 到 `src/video2tasks/vlm/openai_api.py:313` 用了 `stream=True`，但整段函数没有显式 `close()` 响应对象。
- 现有测试覆盖了基本流式和编码问题，见 `tests/vlm/test_openai_api.py:271` 到 `tests/vlm/test_openai_api.py:384`，但没有覆盖多行 SSE 或代理自定义事件格式。

影响：

- 官方 OpenAI 常见返回大多是一行一个事件，所以这里平时可能没事；但兼容网关更容易在 SSE 细节上不同。
- 一旦网关把一个事件拆成多行，当前实现会把它们当坏 JSON 跳过，最后变成 `parse_failure` 或 `content_missing`。
- 流连接不显式关闭，在长时间跑的服务里可能慢慢堆连接，进一步放大延时问题。

建议：

- 用真正的 SSE 事件拼装，而不是逐行猜。
- 流式请求用上下文管理或显式关闭。
- 给兼容网关补多行 SSE、带 `event:`、空心跳包等测试。

### 5. [中] Gemini 和 RemoteAPI 的失败信息太薄，遇到空返回时不够定位；RemoteAPI 还缺测试保护

证据：

- `src/video2tasks/vlm/gemini_api.py:302` 到 `src/video2tasks/vlm/gemini_api.py:339` 在很多失败路径上最后只输出 `Error: status=...`、`Empty structured payload`、`curl fallback still empty` 这类日志，没有保留失败原因、返回体形状、顶层字段、哪一轮成功或失败。
- `src/video2tasks/vlm/remote_api.py:70` 到 `src/video2tasks/vlm/remote_api.py:95` 只做一次请求；非 200 只记状态码，JSON 解码失败只打印异常，不带响应片段、顶层字段、请求耗时之外的更多信息。
- `src/video2tasks/worker/runner.py:357` 到 `src/video2tasks/worker/runner.py:367` 上层最终也只会打印 `Empty or invalid VLM JSON`，没有把接入层的更细原因带出来。
- 测试里没有找到 `RemoteAPIBackend` 的覆盖；`rg -n "remote_api|RemoteAPIBackend" tests` 无结果。

影响：

- 空返回、空消息、被服务端拦截、JSON 被截断、字段名漂移，这几类问题现在很容易都落成同一类“空结果”。
- 这样会让排障只能靠反复抓包或复现，线上定位成本高。
- RemoteAPI 是最开放的一层，却也是目前保护最少的一层，兼容新端点时风险最大。

建议：

- 参考 OpenAI 的 `last_text_json_diagnostics`，给 Gemini 和 RemoteAPI 也补统一诊断对象。
- 日志里至少带上：端点名、模型名、HTTP 状态、失败原因、是否收到 JSON、顶层字段、重试轮次、总耗时。
- 给 RemoteAPI 补最基本的测试：200 正常、非 200、坏 JSON、`text` 包 JSON、`vlm_json` 包 JSON、网络异常。

## 开放问题 / 假设

- 我看到 `llm_merge`、summary、export subtitle 这类文本结构化功能当前直接实例化的是 `OpenAIBackend`，见 `src/video2tasks/server/llm_merge.py:1215`、`src/video2tasks/server/llm_merge.py:1453`、`src/video2tasks/server/llm_merge.py:1564`。所以第 2、3、4 条主要影响这些文本后处理，不一定直接影响 worker 的图片推理主链路。
- OpenAI 这边已有较多测试，Gemini 次之，RemoteAPI 基本没有。所以下面的建议里，测试补齐优先级应当不低。

## 泛化优化建议

1. 统一一套接入层诊断字段  
   OpenAI 现在已经有雏形，Gemini 和 RemoteAPI 也应输出同样的字段，方便横向比较。

2. 把“是否重试”从“结果为空”改成“原因驱动”  
   网络错误、限流、服务端 5xx 可以重试；解析失败、字段漂移、空消息应更快停下并上报。

3. 给每次推理加总预算  
   不只是单次 `timeout_sec`，还要有整次调用总时长上限，避免内外层重试相乘。

4. 把兼容策略从“看域名”改成“看能力”  
   是否支持 `/responses`、是否支持严格 schema、是否需要 stream，应做成配置或探测结果，而不是写死域名判断。

5. 补一组最小兼容夹具  
   至少覆盖：空 `message.content` 但有 `reasoning_content`、多行 SSE、200 但空 body、200 但字段形状变化、RemoteAPI 返回 `text/vlm_json` 两种包装。

## 总结

这层接入的主问题不是“完全不能用”，而是“接口一抖就又慢又难查”。OpenAI 文本结构化链路的诊断比 Gemini 和 RemoteAPI 成熟，但兼容策略偏硬编码；Gemini 则是重试过重、空结果原因过少。要提高稳定性，优先级最高的不是再加更多重试，而是先把失败分类、总时长预算和统一诊断补齐。
