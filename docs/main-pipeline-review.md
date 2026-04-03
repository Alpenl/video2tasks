# Video2Tasks 主流程 Review

更新时间：2026-04-03  
分支：`master`

本文只看当前主流程这一条链路：

1. 视频切分窗口
2. 抽帧并拼接为 contact sheet
3. worker 上传图片给 Gemini 兼容接口
4. 解析返回的 `transitions` / `instructions`
5. server 回写窗口结果并生成最终 `segments.json`

## 1. 当前主流程代码入口

关键文件：

- `src/video2tasks/server/app.py`
  - 负责建任务队列、切窗口、抽帧、分发 job、接收结果、最终落盘 `segments.json`
- `src/video2tasks/server/windowing.py`
  - 负责读取视频、构造窗口、抽帧、拼接 contact sheet、把窗口级 `transitions` 合成为全局切分
- `src/video2tasks/worker/runner.py`
  - 负责从 server 拉取 job、解码图片、组 prompt、调用 VLM、提交结果
- `src/video2tasks/vlm/gemini_api.py`
  - 负责 Gemini native / OpenAI-compatible 请求、结构化 JSON 提取、网络重试、curl 回退
- `src/video2tasks/prompt.py`
  - 负责边界检测 prompt、boundary refinement prompt、segment label prompt
- `src/video2tasks/config.py`
  - 默认配置和环境变量覆盖逻辑

## 2. 当前关键配置

下面是当前 `Config.load("config.yaml")` 实际生效的关键值。这里包含了 `config.yaml` 中显式值和 `config.py` 中补上的默认值。

### 2.1 Server

- `server.max_queue = 32`
- `server.inflight_timeout_sec = 300.0`
- `server.max_retries_per_job = 50`
- `server.auto_exit_after_all_done = true`

### 2.2 Worker / Gemini

- `worker.backend = "gemini"`
- `worker.gemini.model = "gemini-3.1-pro-preview"`
- `worker.gemini.api_mode = "openai_compatible"`
- `worker.gemini.base_url = "https://api.duckcoding.ai"`
- `worker.gemini.timeout_sec = 60.0`
- `worker.gemini.max_output_tokens = 2048`

当前默认接口路径不是 native Gemini，而是 OpenAI-compatible：

- 基础地址会先规范化为 `https://api.duckcoding.ai/v1`
- 实际请求地址是 `https://api.duckcoding.ai/v1/chat/completions`

### 2.3 Windowing / Segmentation

- `window_sec = 12.0`
- `step_sec = 6.0`
- `frames_per_window = 128`
- `boundary_prompt_mode = "freeform"`
- `segment_labeling_mode = "inline"`
- `enable_refinement_pass = false`
- `enable_boundary_refinement = false`
- `target_width = 720`
- `target_height = 480`
- `png_compression = 0`
- `use_contact_sheets = true`
- `contact_sheet_rows = 4`
- `contact_sheet_cols = 4`
- `adaptive_merge_guard = true`
- `adaptive_merge_min_segments = 8`
- `adaptive_merge_collapse_ratio = 0.6`
- `boundary_support_threshold = 0.9`
- `refine_final_instructions = true`

### 2.4 这些值意味着什么

当前主流程的默认行为可以直接概括成：

- 每个窗口长度 `12s`
- 窗口滑动步长 `6s`
- 每个窗口抽 `128` 个 logical frames
- 因为开启了 `4 x 4` contact sheet，所以每张大图承载 `16` 帧
- `128 / 16 = 8`，所以每个窗口最终会上传 `8` 张拼图
- 每张拼图在内部先做成 `PNG`
- worker 侧再重新编码成 `JPEG data URL`
- 再通过 OpenAI-compatible `chat/completions` 发给 Gemini

## 3. 主流程实际链路

### 3.1 窗口切分

入口在 `src/video2tasks/server/windowing.py` 的 `build_windows(...)`。

处理方式：

- 用 `window_sec * fps` 算窗口长度
- 用 `step_sec * fps` 算滑动步长
- 对每个窗口用 `np.linspace(...)` 均匀采样 `frames_per_window`
- 每个窗口保存：
  - `window_id`
  - `start_frame`
  - `end_frame`
  - `frame_ids`

当前配置下，一段视频会被切成一串重叠窗口，每个窗口有 128 个逻辑索引位置，模型返回的 `transitions` 也是这些逻辑索引，不是原始视频帧号。

### 3.2 抽帧与拼图

入口在 `src/video2tasks/server/windowing.py` 的 `FrameExtractor.get_many_b64(...)`。

当前开启 `use_contact_sheets=true` 后，处理方式是：

- 把 `frame_ids` 按 `rows * cols = 16` 一组切块
- 优先用 `_build_contact_sheet_b64_via_ffmpeg(...)` 直接从视频抽帧并拼图
- 如果 ffmpeg 失败，再回退到 `_build_contact_sheet_b64_via_cv2(...)`
- 每个 tile 左上角会画出逻辑帧索引，例如 `0..127`

关键点：

- 这里拼图时画上的数字不是原始视频帧号，而是当前窗口内的逻辑索引
- prompt 也明确告诉模型：`transitions` 里要填这些 tile index

### 3.3 Job 入队

入口在 `src/video2tasks/server/app.py` 的 producer loop。

server 会：

- 扫描样本目录里的 `Frame_*.mp4`
- 读取视频信息，构造窗口
- 调用 `extractor.get_many_b64(...)`
- 直接把编码后的 `images` 和 `meta` 一起塞进 `job_queue`

每个窗口 job 的核心字段：

- `task_id`
- `images`
- `meta.subset`
- `meta.sample_id`
- `meta.window_id`
- `meta.frame_ids`
- `meta.fps`
- `meta.window_start_frame`
- `meta.window_end_frame`
- `meta.use_contact_sheets`
- `meta.contact_sheet_rows`
- `meta.contact_sheet_cols`

### 3.4 Worker 拉取、解码、组 prompt

入口在 `src/video2tasks/worker/runner.py`。

worker 的行为：

1. `GET /get_job`
2. 拿到 `images` 后逐张 base64 解码成 `numpy BGR`
3. 按 `job_type` 选择 prompt
   - `window_boundary` -> `prompt_switch_detection(...)`
   - `boundary_refinement` -> `prompt_boundary_refinement(...)`
   - `segment_label` -> `prompt_segment_instruction(...)`
4. 调用 `backend.infer(images, prompt)`
5. `POST /submit_result`

当前默认主流程没有开 refinement 和 deferred labeling，所以主路径基本是：

- `window_boundary` prompt
- 直接返回 `transitions` 和 `instructions`
- 然后 server 汇总成最终段落

### 3.5 Worker 到 Gemini 的请求

当前 backend 是 `GeminiBackend`，走 `openai_compatible` 分支：

- 代码在 `src/video2tasks/vlm/gemini_api.py`
- 入口是 `GeminiBackend._infer_openai_compatible(...)`

实际发送内容：

- `system` 消息：只约束严格 JSON 输出格式
- `user` 消息：
  - 第一段是主 prompt 文本
  - 后面依次追加 8 张图片，格式是 `image_url` + `data:image/jpeg;base64,...`

注意这里有一个格式转换：

- server 里内部队列存的是 `PNG`
- worker 解码为 `numpy`
- 发送给 Gemini 兼容接口时重新编码为 `JPEG`

### 3.6 返回结果与重试层次

当前链路里有四层重试：

#### 第 1 层：HTTP 请求级重试

在 `gemini_api.py` 的 `_post_json(...)`：

- 对 `408/409/425/429/500/502/503/504` 做最多 `4` 次请求重试
- `requests` 失败后，最终会回退到 `curl`

#### 第 2 层：结构化 payload 空返回重试

在 `GeminiBackend._request_with_payload_retries(...)`：

- 如果 HTTP 200 但解析不出有效结构化 JSON
- 会先重试 `5` 次普通请求
- 再做 `2` 次 `curl` 回退尝试

#### 第 3 层：worker 本地重试

在 `worker/runner.py`：

- 对同一个 job，如果 `backend.infer(...)` 仍然给空 JSON
- worker 会再做 `4` 次本地 retry

#### 第 4 层：server 重新入队

在 `server/app.py`：

- 如果 worker 最后提交的是空 `vlm_json`
- server 会把这个 job 重新塞回队尾

这意味着当前一条坏窗口的真实尝试次数会非常高，延迟也会被显著放大。

### 3.7 最终 `segments.json` 如何生成

入口在 `src/video2tasks/server/windowing.py` 的 `build_segments_via_cuts(...)`。

它做的事情：

1. 把每个窗口里的 `transition` 逻辑索引映射回该窗口采样到的真实 `frame_ids`
2. 对多个重叠窗口投出来的 cut 做聚类
   - 聚类间隔：`2.5 * fps`
   - 权重：优先用窗口中心位置投票，边缘帧权重低
3. 构造逐帧 `instruction_timeline`
4. 根据 cut 切出 `raw_segments`
5. 再做：
   - `split_long_raw_segments_on_instruction_drift(...)`
   - `cleanup_auxiliary_segments(...)`
   - `merge_task_level_segments(...)`
6. 如果语义 merge 合并过猛，会触发 `adaptive_merge_guard`
7. 最终写出 `segments.json`

当前默认：

- 不开 `enable_refinement_pass`
- 不开 `enable_boundary_refinement`
- 不开 `segment_labeling_mode = deferred`

所以最终质量主要由：

- 第一阶段窗口 prompt
- 模型返回的 `transitions`
- `build_segments_via_cuts(...)` 的聚类与 merge

共同决定。

## 4. 当前实际提示词

当前真正用于主流程的是：

- `prompt_switch_detection(...)`
- `mode = "freeform"`
- `n_images = 128`
- `contact_sheet_rows = 4`
- `contact_sheet_cols = 4`
- `sheet_count = 8`

此外，OpenAI-compatible 请求还会再加一层 system prompt：

> Return JSON only with keys thought, transitions, and instructions. transitions must be integer frame indexes. instructions must be an array of strings. The thought field must be one short sentence under 20 words. Do not use markdown fences.

中文意思：

- 只能返回 JSON
- 只能有 `thought`、`transitions`、`instructions` 这三个 key
- `transitions` 必须是整数帧索引
- `instructions` 必须是字符串数组
- `thought` 必须是一句不超过 20 个词的短句
- 不要用 Markdown code fence

### 4.1 主 prompt 中文翻译

下面是当前 `freeform` 边界检测 prompt 的中文整理版。

#### 角色与输入映射

- 你是一个机器人视觉分析器
- 你正在看一个包含 `128` 帧的家庭操作视频片段
- 索引范围是 `0..127`
- 这个片段被打包成 `8` 张 contact sheet
- 每张图是 `4 列 x 4 行`
- 每个 tile 左上角已经标好逻辑帧索引
- 阅读顺序是从左到右、从上到下，先看第一张再看下一张
- `transitions` 返回的是这些 tile 索引，而不是上传图片张数

#### 总目标

- 检测对 VLA 训练有用的任务级边界
- 每个 segment 都应该对应一条从开始到结束的连贯机器人指令
- 不能漏掉真实的新任务目标
- “切换”指的是进入一个新的、持续若干帧的操作阶段，而不是短暂碰一下新物体

#### 边界优先级

- 漏掉真实边界比多切一个边界更糟
- 这是预合并阶段，允许过度切分，后面可以再合并
- 倾向于切成更小、更客观的可见持续步骤，而不是整个活动的大总结
- 拿不准时，在“大段”与“更细两段”之间，优先选更细切分
- 重复加入、重复倒入、重复撒料、重复摆放、重复取走，只要能看到新的 committed round，就应当拆开
- 边界要放在新持续操作第一次已经可见的那一帧
- 不要等到新动作“很明显进行了很多”再切
- 不要把加料、换工具、换容器、移除、装盘等不同操作压成一个大 instruction
- 只有在动作 genuinely uninterrupted，没有停顿、复位、撤回、再进入时，才保留成一个段

#### 最早 onset 锚定

- 边界锚定到新步骤第一个 committed frame
- 倒液体：切在第一帧开始倒的时候，不是液体积起来后
- 撒粉末：切在第一帧开始释放的时候，不是堆积变明显后
- 放入新物体：切在它第一次开始进入目标空间或容器的时候，不是完全进去之后
- 换新工具或新容器：切在它开始新持续操作的第一帧
- 如果同一工具或容器停下、离开、之后又回来开始下一轮，应切成新段
- 如果同类大动作出现两次，但来源对象、材料轮次、目标阶段不同，要拆成不同步骤，例如 `First pour` / `Second pour`
- 但也不能编造还没出现的物体，必须以第一帧可见证据为准

#### 核心逻辑

1. 这一轮把每个新的持续操作阶段都当成切换，哪怕它们属于同一个大任务
2. 如果机器人仍在追求同一个结果，只是碰了被搬运物、容器、支撑物、最终堆叠目标，这不算切换
3. 微调、稳定、再抓、放下后的细小校正，不算新任务
4. 为主动作服务的 `reach / hover / align / partial grasp`，仍属于同一任务
5. 只要材料、工具、容器、工件、来源容器、目标容器、操作阶段变了，并且新步骤持续了若干帧，就倾向于新边界
6. 同一工作区内，只要切到不同工具用法、不同对象焦点、不同工作区域，且新目标持续，就应切
7. 多个相似物体的重复批量家务，在同一工作区内可以保持一个段，前提是高层目标确实相同
8. 对镜头讲话、展示、停顿、站着不动，不算任务边界，只要底层操作任务仍在延续
9. Recall 优先；只要新操作持续了几帧，不确定时宁愿保留边界

#### 标注规则

- 不要输出 `Wait`、`Stand by`、`Describe the scene` 这类空泛标签
- instruction 必须锚定到可见操作目标
- 优先使用具体动词 + 可见对象
- 如果物品身份不确定，宁可写客观粗粒度标签，比如：
  - `Dispense granular material`
  - `Pour dark liquid`
  - `Place a flat item`
  - `First pour`
  - `Second pour`
- 不要自信猜测具体品牌、具体调料名称
- 忽略任务结束后的衣物调整、搓手、姿势复位等尾巴动作

#### 输出格式

- 必须返回合法 JSON
- 只包含 `thought`、`transitions`、`instructions`
- `thought` 必须极短，一句话，不超过 20 个词
- `instructions` 要简洁、任务级、适合机器人训练

#### 示例导向

prompt 附带了 11 个示例，主要想强化这些偏好：

- 同一目标下的准备动作不要切开
- 明确新目标时要切
- 同类批量家务可以不切
- 同区域但不同持续目标要切
- 讲话镜头不要切
- 边界要锚在最早 onset
- 材料不清楚时要用客观粗标签
- 同类动作重复两次要拆成 `First / Second`
- 预合并阶段更细切分是允许的
- 同一工具 `stop-and-restart` 也要拆成新一轮

## 5. Review 发现的问题 / 风险

下面这些问题只针对当前主链路，不涉及训练策略本身。

### 高优先级

#### 5.1 refinement pass 打开后，refinement window 的结果不会真正进入最终切分

位置：

- `src/video2tasks/server/windowing.py`

现象：

- `build_segments_via_cuts(...)` 里是用 `enumerate(windows)` 得到的顺序索引 `wid` 去查 `by_wid.get(wid)`
- refinement window 的 `window_id` 是一大串派生 id，例如 `1000000 + ...`
- 这意味着 refinement pass 的结果会被静默忽略

影响：

- 这是一个潜伏 bug
- 当前默认 `enable_refinement_pass = false`，所以默认链路暂时不会触发
- 但一旦把 refinement pass 打开，后处理结果会不符合预期

#### 5.2 结果回写不是幂等的，晚到或重复结果可能非确定性覆盖窗口结果

位置：

- `src/video2tasks/server/app.py`

现象：

- `/submit_result` 收到非空 `vlm_json` 后会直接 append 到 `windows.jsonl`
- 它不要求这个 task 此时仍然在 `inflight`
- finalize 再读文件时，会按 `window_id` 取“最后一条记录”

影响：

- 同一窗口的重复提交、晚到提交都可能被接受
- 最终 `by_wid[d["window_id"]] = d` 的效果取决于文件写入顺序
- 这会让同一轮跑出来的结果存在竞态和非确定性

#### 5.3 空结果重排没有真正受 `max_retries_per_job` 约束

位置：

- `src/video2tasks/server/app.py`

现象：

- `submit_result(...)` 收到空 `vlm_json` 后，会直接 `_requeue_empty_result(...)`
- 这个路径只会计数并重新入队
- 但不会像 inflight timeout 那样检查 `max_retries_per_job`

影响：

- 当前实现里，`max_retries_per_job = 50` 并不能限制“空 JSON 重试”
- 一个长期空返的窗口可以无限拖住样本完成
- 这也是最近尾部只剩一个窗口时会长时间不退出的直接原因

说明：

- 这和最近“空 JSON 持续重试”的要求是一致的
- 但从工程行为上看，它仍然是一个重要操作风险，至少需要在文档层面明确

#### 5.4 worker 提交结果没有确认与重试

位置：

- `src/video2tasks/worker/runner.py`

现象：

- worker 推理成功后直接 `requests.post("/submit_result", ...)`
- 不检查返回状态
- 也没有失败重试或 ack 校验

影响：

- 如果模型推理已经成功，但提交结果时网络抖动或 server 短暂不可达
- 这次昂贵推理结果会丢失
- server 只能等 inflight timeout 以后再把 job 重新排队，导致重复推理和额外延迟

### 中优先级

#### 5.5 上传图一旦解码失败，会静默替换成黑图继续推理

位置：

- `src/video2tasks/worker/runner.py`

现象：

- `decode_b64_to_numpy(...)` 失败时返回 `None`
- worker 会塞一张 `224x224` 全黑 dummy image

影响：

- 上游抽帧、拼图、base64 传输异常时，不会显式打断
- 模型会在混入黑图的情况下继续推理
- 这会把链路问题伪装成“模型边界判断差”

#### 5.6 队列里存放的是完整编码图片，而不是轻量引用

位置：

- `src/video2tasks/server/app.py`
- `src/video2tasks/server/windowing.py`

现象：

- server 在入队前就把窗口里的所有图片 base64 编好，直接塞进 `job_queue`

影响：

- 当前每个窗口是 `8` 张 contact sheet
- 每张是 `720x480` PNG，再 base64
- `max_queue = 32` 时，内存占用会被显著放大
- 样本数、worker 数、分辨率再升高时，这个设计会比较吃内存

#### 5.7 当前链路的重试层次过深，单窗口延迟会被显著放大

位置：

- `src/video2tasks/vlm/gemini_api.py`
- `src/video2tasks/worker/runner.py`
- `src/video2tasks/server/app.py`

现象：

- 请求级 retry
- payload 空结构 retry
- curl fallback retry
- worker 本地 retry
- server 重新入队 retry

影响：

- 一旦接口波动或结构化空返频繁
- 单个窗口的总耗时会非常长
- 最终表现就是“整体很慢、尾部卡住、跑完才发现分数也没变好”

#### 5.8 主提示词和最终 merge 逻辑之间存在天然张力

位置：

- `src/video2tasks/prompt.py`
- `src/video2tasks/server/windowing.py`

现象：

- 当前 prompt 是明显的 `recall-first / over-segmentation` 倾向
- 但最终仍然要经过 cut 聚类、cleanup、semantic merge、adaptive merge guard

影响：

- 如果前面鼓励尽量切碎、后面又在 merge 阶段回收得太狠
- 就会出现“prompt 在推过切，后处理在拉回去”的拉扯
- 这不是单点 bug，但它是当前质量分析必须一直盯着的结构性张力

### 低优先级

#### 5.9 默认主流程没有 refinement 和 deferred labeling

位置：

- `config.py`
- `config.yaml`

现象：

- `enable_refinement_pass = false`
- `enable_boundary_refinement = false`
- `segment_labeling_mode = inline`

影响：

- 当前产出的边界质量几乎完全依赖第一阶段大窗口 prompt
- 如果第一阶段锚点偏了，后面没有额外默认修正步骤兜底

这不一定是 bug，但必须在分析效果时牢记。

## 6. 当前链路里已经做得比较好的点

- contact sheet 已经优先使用 ffmpeg 拼图，并保留 cv2 fallback
- `base_url` 已经会自动规范化，裸域名也能补成 `/v1`
- OpenAI-compatible 响应已经做了更鲁棒的 JSON 提取
- HTTP 异常已经有 `curl` 回退
- 空结构化响应已经有多层 retry
- 最终切分不是简单拼接，而是做了跨窗口投票聚类和 merge guard

## 7. 一页结论

如果只看当前主流程，最重要的事实是：

- 当前默认运行是 `12s / 6s / 128 logical frames / 8 张 contact sheet / Gemini openai-compatible`
- 实际 API 是 `https://api.duckcoding.ai/v1/chat/completions`
- 当前主提示词是 `freeform` 的过切分倾向 prompt
- 当前默认没有 refinement pass，也没有 boundary refinement
- 因此主效果几乎完全取决于第一阶段 prompt + 模型输出 + `build_segments_via_cuts(...)`

如果只看当前链路里最值得警惕的问题：

- refinement pass 一旦打开，当前实现会忽略 refinement windows 的结果
- 空 JSON 重排目前是事实上的无限重试，能拖死单个样本收尾
- 结果回写不是幂等的，晚到结果可能覆盖前面的窗口结果
- 推理成功后的结果提交没有重试，容易造成高成本重复推理
- 解码失败被黑图吞掉，容易把链路问题误判成“纯模型问题”
