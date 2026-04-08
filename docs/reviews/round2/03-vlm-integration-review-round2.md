# Round 2 Audit of 03 VLM Integration Review

## 复核对象

- 首轮文档：`docs/reviews/round1/03-vlm-integration-review.md`
- 本轮回看的代码：
  - `src/video2tasks/vlm/gemini_api.py`
  - `src/video2tasks/vlm/openai_api.py`
  - `src/video2tasks/vlm/remote_api.py`
  - `src/video2tasks/vlm/base.py`
  - `src/video2tasks/worker/runner.py`
  - `src/video2tasks/server/llm_merge.py`
  - `src/video2tasks/config.py`
  - `tests/vlm/test_gemini.py`
  - `tests/vlm/test_openai.py`
  - `tests/vlm/test_openai_api.py`
  - `tests/server/test_llm_merge.py`
  - `tests/server/test_llm_summary.py`

本轮目标不是重做一遍首轮，而是审首轮文档本身：看证据够不够、级别准不准、有没有说大了、有没有漏掉更关键的问题。

## 确认成立的点

### 1. Gemini 的重试确实会把失败放大，首轮把它列为高优先级是成立的

这个判断成立，而且是首轮里最重要的一条。

证据回看：

- `src/video2tasks/vlm/gemini_api.py` 的 `_post_json()` 会对 `408/409/425/429/5xx` 做最多 4 次 HTTP 尝试，并在前几轮之间 sleep。
- 同文件的 `_request_with_payload_retries()` 会在 `HTTP 200` 但结构化结果仍为空时，再做 5 轮 payload 级尝试，之后还有 2 轮显式 `curl` 回退。
- `src/video2tasks/worker/runner.py` 里 worker 外层还有本地重试。
- `tests/vlm/test_gemini.py` 已经覆盖了“空 payload 后继续重试”和“重复空 payload 后进入 curl 回退”这两段行为。

为什么这条重要：

- 这不是只影响可选后处理，而是直接挂在 worker 主推理链路上。
- 只要响应长期落在“200 但结构化结果为空”这种状态，当前实现就会继续消耗整段超时预算。
- 结果不只是慢，还会长期占住 worker。

结论：

- 这一条维持高优先级，没有问题。

### 2. OpenAI 文本结构化路径对非 `api.openai.com` 主机先走流式，首轮方向判断成立

首轮对这个行为本身的判断是对的。

证据回看：

- `src/video2tasks/vlm/openai_api.py` 的 `_prefer_stream_first_for_text_json()` 只按 host 是否等于 `api.openai.com` 决定。
- `infer_text_json()` 会把这个判断直接传给 `_post_structured_json()`。
- `tests/vlm/test_openai.py` 明确验证了代理域名下会先发 `stream=True` 的 `/chat/completions` 请求。

另外，首轮这里其实还少说了一点：

- 当前 stream-first 请求不只是“先流式”，而且请求体里没有 `response_format` 或 `json_schema`，也就是先发了一次无 schema 约束的请求。

结论：

- 行为存在，风险方向也对，但影响范围和级别需要修正，见下文。

### 3. OpenAI 与 Gemini 的兼容提取规则不一致，这一点成立

首轮说“两边兼容提取规则不一致”，这点成立。

证据回看：

- `src/video2tasks/vlm/openai_api.py` 的 `_extract_chat_completions_payload_with_reason()` 只看 `message.parsed` 和 `message.content`。
- `src/video2tasks/vlm/gemini_api.py` 的 `_collect_openai_text_candidates()` 还会继续看 `reasoning_content`、`arguments` 和 `text.value`。
- `tests/vlm/test_gemini.py` 有 `reasoning_content` 的回退测试；OpenAI 侧没有对应保护。

结论：

- 首轮关于“同类兼容返回，两边吞吐路径不一致”的方向判断是对的。
- 但首轮对可覆盖范围说得偏大，见下文。

### 4. Gemini / RemoteAPI 的失败诊断偏薄、RemoteAPI 缺测试，这一点成立

这条基本成立，而且首轮抓到了一个真实短板。

证据回看：

- `src/video2tasks/vlm/gemini_api.py` 的空结果路径主要靠打印字符串，缺少像 OpenAI 那样的结构化诊断对象。
- `src/video2tasks/vlm/remote_api.py` 只做一次请求，非 200 和 JSON 解码失败时日志字段很少。
- `src/video2tasks/worker/runner.py` 上层在空结果时只统一打出 `Empty or invalid VLM JSON`。
- 测试目录里没有 `RemoteAPIBackend` 的直接覆盖。

结论：

- 这条保留。

## 需要修正的点

### 1. 首轮第 1 条的问题级别没错，但表述上把“重试次数”和“触发条件”说粗了

需要改的地方：

- `worker` 外层是 `MAX_LOCAL_RETRIES = 4`，表示 4 次总尝试，不是“额外再重试 4 次”。
- Gemini 的 payload 级重试不是对所有失败都生效，而是只在 `status_code == 200` 且结构化结果为空时才继续；如果 `_post_json()` 最终返回非 200，`_request_with_payload_retries()` 会直接停下。
- 首轮给出的“约 23 分钟 / 94 分钟”更适合表述成“特定上界场景”，不能写成一般情况。

更准确的说法应当是：

- 在“单次 HTTP 尝试持续打满超时预算，且每轮最后都落成 `200 + 空结构化结果`”这种上界路径下，一次 `infer()` 可能被拖到二十多分钟；worker 再叠 4 次总尝试后，单任务可能接近一个半小时。

结论：

- 级别仍然是高。
- 但首轮需要把触发条件写细，不然读者容易误以为所有 5xx/超时都会一路叠满到这个上界。

### 2. 首轮第 2 条把影响范围说大了，级别建议从中下调到低

首轮原文容易让人读成“只要是代理域名，就直接放弃严格结构化接口”。代码不是这样。

更准确的事实是：

- 这条逻辑只影响 `infer_text_json()`，也就是 `llm_merge`、`summary`、`export subtitle` 这些文本结构化调用，不影响 worker 的图片推理主链路。
- stream-first 失败后，代码还会继续尝试 `/responses`，再尝试非流式 `/chat/completions`。
- `summary` 和 `export subtitle` 还会通过 `_single_attempt_config()` 强制只走 1 次远端尝试，然后快速回退本地兜底，不会像 worker 主链路那样把耗时放大得很夸张。

真正该强调的风险不是“彻底走错接口”，而是两点：

- 代理域名下会先多打一枪无 schema 约束的流式请求。
- 如果这一枪恰好返回了可解析 JSON，代码会直接接受它，不会再去更严格的 `/responses`。

结论：

- 问题存在，但首轮给到“中”的级别偏高。
- 更合适的是低：它主要带来额外延迟和更弱的输出约束，且只在可选文本后处理路径上生效。

### 3. 首轮第 3 条对“函数参数”这部分说得过头了，级别建议从中下调到低

首轮证据足以支持：

- Gemini 兼容提取会多看 `reasoning_content`
- 也会多看直接出现的 `arguments`
- 也会递归进 `text.value`

但首轮没有足够证据支持这句更宽的说法：

- “如果它放在函数参数里，Gemini 兼容层能吃到”

原因很简单：

- 当前 `Gemini` 的 `_collect_openai_text_candidates()` 不是泛化遍历，它只会沿着 `text`、`content`、`reasoning_content`、`arguments` 这些固定键继续找。
- 代码和测试里都没有证明它能吃到常见的 `tool_calls[].function.arguments` 这种路径。
- 现有测试只覆盖了 `reasoning_content`，没有覆盖 `arguments` 或 `tool_calls`。

结论：

- “两边提取规则不一致”成立。
- 但“Gemini 能覆盖函数参数”这句话证据不足，应删掉或缩成“Gemini 额外处理了 `reasoning_content`、直接 `arguments` 和 `text.value`”。
- 级别建议从中下调到低。

### 4. 首轮第 4 条把“没有显式 close”直接推到“连接会慢慢堆积”，证据不够

这条需要拆开看。

成立的部分：

- 当前流式解析确实是逐行读 `data:`，没有做标准 SSE 事件拼装。
- 没有覆盖多行 `data:`、`event:`、心跳包等代理更常见的情况。

证据不足的部分：

- “没有显式 `close()`，长时间运行会慢慢堆连接”这个推断现在没有复现证据，也没有测试或监控数据支撑。
- 这是合理担心，但还不能在审计文档里写成已经坐实的影响。

另外，首轮漏掉了比“没显式 close”更直接的问题：

- `response.iter_lines()` 外面没有 `try/except`，如果流在中途断开，异常会直接冒出去。

结论：

- SSE 拼装问题保留。
- “连接堆积”要降成猜测，不能当成已证实结论。
- 首轮这里的重点放偏了，级别也不宜按“连接泄漏”去定。

### 5. 首轮第 5 条方向对，但还少了一层更直接的行为问题

首轮抓住了“诊断太薄、测试太少”，但没有指出：

- `RemoteAPIBackend.infer()` 里的 `requests.post()` 没有 `try/except`。

这意味着：

- 一旦远端抛 `RequestException`，这里不会产出自己的失败诊断，而是直接异常上抛。
- 在 worker 场景里，这个异常会被 `runner.py` 的外层 `except Exception` 吃掉，最后只留下通用日志，端点细节还是丢了。

结论：

- 首轮这条应当保留。
- 但“未捕获请求异常”应明确写出来，不能只停留在“日志不够细”。

## 新增发现

### 1. [中] OpenAI 流式读取中的中途断流异常没有被接住，首轮漏掉了

证据：

- `src/video2tasks/vlm/openai_api.py` 只在 `requests.post()` 外面包了 `try/except`。
- 之后的 `response.iter_lines()` 没有异常处理。
- 测试里没有覆盖流式读取过程中抛异常的情况。

影响：

- 代理或网关如果在 SSE 中途断流，这里会直接抛异常。
- `llm_merge` 会把它当请求失败重试或降级，但诊断对象可能来不及完整记录流式阶段的失败细节。
- `summary` 和 `export subtitle` 因为被压成单次远端尝试，遇到这种问题会更快走回退，但定位信息仍然偏弱。

建议级别：

- 中。它比“没有显式 close”更直接，也更容易在兼容网关上触发。

### 2. [中] RemoteAPI 的网络异常是未捕获异常，不只是“日志太薄”

证据：

- `src/video2tasks/vlm/remote_api.py` 的 `requests.post()` 直接调用，没有 `try/except requests.RequestException`。
- 测试目录里也没有对应覆盖。

影响：

- 一旦远端网络错误、握手失败、超时，这里直接抛异常，不会生成统一诊断。
- worker 最终只会记录通用失败信息，端点级上下文丢失。

建议级别：

- 中。它已经不只是排障体验问题，而是错误分类在接入层就断了。

## 重新排序后的建议

1. 先处理 Gemini 主链路的总时长预算和原因驱动重试  
   这是唯一已经明确挂在 worker 主链路、并且会把队列占满的问题。

2. 补上 OpenAI 流式读取异常和 RemoteAPI 请求异常的接入层捕获  
   先把失败留在接入层并写清原因，后面才有资格谈优化策略。

3. 给 Gemini 和 RemoteAPI 补统一诊断对象，并把关键原因往上带  
   至少统一：端点名、模型名、HTTP 状态、失败原因、是否收到 JSON、顶层字段、尝试序号、总耗时。

4. 把 OpenAI 的 stream-first 从“看域名”改成显式配置或能力探测  
   如果还保留 stream-first，也应把它明确当作回退或兼容模式，而不是默认优先路径。

5. 补测试，不要只补 happy path  
   最少应覆盖：RemoteAPI 的非 200、坏 JSON、网络异常；OpenAI 的多行 SSE、mid-stream 异常；OpenAI/Gemini 的 `reasoning_content` 与 `arguments` 差异。

## 二轮结论

首轮文档抓到了最重要的问题：Gemini 主链路重试放大，这一点判断准确，优先级也对。需要调整的主要有三处：一是把上界耗时的触发条件写细；二是把 OpenAI stream-first 的影响范围收窄到文本后处理链路；三是不要把“未显式 close”直接写成已经证实的连接堆积。

首轮真正漏掉的，是两个更直接的异常路径：OpenAI 流式 `iter_lines()` 中途断流没有捕获，RemoteAPI 的 `requests.post()` 也没有捕获。这两条都比“日志不够看”更靠近真实故障面，应该进入修正版文档的前排。
