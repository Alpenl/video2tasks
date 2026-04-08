# 第二轮复核 02：窗口切分、抽帧、联系图、中间产物与上传前准备

日期：2026-04-08  
方式：静态读码 + 测试回看 + 最小本地复现；只做审计，不改业务代码

## 复核对象

- 第一轮文档：[02-windowing-and-artifacts-review.md](/home/alpen/DEV/video2tasks/docs/reviews/round1/02-windowing-and-artifacts-review.md)
- 直接复核的代码：
  - [app.py](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py)
  - [windowing.py](/home/alpen/DEV/video2tasks/src/video2tasks/server/windowing.py)
  - [task_artifacts.py](/home/alpen/DEV/video2tasks/src/video2tasks/server/task_artifacts.py)
  - [runner.py](/home/alpen/DEV/video2tasks/src/video2tasks/worker/runner.py)
  - [config.py](/home/alpen/DEV/video2tasks/src/video2tasks/config.py)
  - [openai_api.py](/home/alpen/DEV/video2tasks/src/video2tasks/vlm/openai_api.py)
  - [gemini_api.py](/home/alpen/DEV/video2tasks/src/video2tasks/vlm/gemini_api.py)
  - [README.md](/home/alpen/DEV/video2tasks/README.md)
- 回看的测试：
  - [test_windowing.py](/home/alpen/DEV/video2tasks/tests/server/test_windowing.py)
  - [test_task_artifacts.py](/home/alpen/DEV/video2tasks/tests/server/test_task_artifacts.py)
  - [test_runner.py](/home/alpen/DEV/video2tasks/tests/worker/test_runner.py)

## 确认成立的点

### 1. “服务端默认落盘，任务交接依赖共享文件系统”成立，而且仍然是最高优先级问题

第一轮这里判断准确，证据也够。

- 服务端创建 app 时无条件创建 `TaskArtifactWriter`，并没有走 `VIDEO2TASKS_DUMP_INTERMEDIATE` 这条开关逻辑，见 [app.py#L173-L174](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L173) 和 [windowing.py#L635-L639](/home/alpen/DEV/video2tasks/src/video2tasks/server/windowing.py#L635)。
- 构任务时，`return_images` 的条件是 `artifact_writer is None`；而 server 传入的 `FrameExtractor` 总是带 writer，所以默认走“写本地文件，再把 `image_paths` 放进 job”，见 [app.py#L449-L478](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L449)。
- worker 侧如果看到 `image_paths`，会直接按本地路径 `Path(path).read_bytes()`，见 [runner.py#L109-L132](/home/alpen/DEV/video2tasks/src/video2tasks/worker/runner.py#L109)。

这说明第一轮的核心结论没有问题：当前主链路不是纯 HTTP 传图，而是 “HTTP 传任务元数据 + 共享本地路径补图”。这在单机或共享卷模式下可行，但对多机、多容器、多挂载点都很脆弱。

### 2. “坏图会先落盘，再在 worker 端失败，最后表现成 timeout/requeue”成立，且是实 bug

第一轮这里不只是方向对，故障链也基本说准了。

- `FrameExtractor` 在读不到帧或两种联系图生成都失败时，会留下空字节 payload，见 [windowing.py#L887-L907](/home/alpen/DEV/video2tasks/src/video2tasks/server/windowing.py#L887) 和 [windowing.py#L918-L953](/home/alpen/DEV/video2tasks/src/video2tasks/server/windowing.py#L918)。
- `TaskArtifactWriter` 会把空字节照样写盘，只把 `decode_ok` 记成 `False`，见 [task_artifacts.py#L120-L142](/home/alpen/DEV/video2tasks/src/video2tasks/server/task_artifacts.py#L120)。
- server 构任务时没有看 `decode_ok` 或 `byte_size`，只要有记录就直接把路径塞进 `image_paths`，见 [app.py#L473-L475](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L473)。
- worker 读到空文件或解码失败后会直接抛异常；这个异常不会提交空结果，只会落到外层循环打印 `[Error] Loop crashed`，见 [runner.py#L109-L132](/home/alpen/DEV/video2tasks/src/video2tasks/worker/runner.py#L109) 和 [runner.py#L306-L389](/home/alpen/DEV/video2tasks/src/video2tasks/worker/runner.py#L306)。
- server 对没有回执的 inflight 任务会按超时重排队，直到耗尽预算，见 [app.py#L620-L650](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L620)。

所以这里应继续维持高风险评级。问题不只是“报错位置偏后”，而是服务端已知的坏产物被放进了正式任务流，最终以超时重试的形式表现出来。

## 需要修正的点

### 1. “tmp 目录命名容易覆盖”这个问题存在，但要把风险主轴改准

第一轮把这件事讲成了“重跑和多实例下容易覆盖”，方向没错，但对“普通重试”说得偏重。

- 同一次超时重试并不会重新生成图片，而是把原来的 `base_job` 重新放回队列，见 [app.py#L625-L633](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L625)。
- 真正容易互相覆盖的是跨运行和跨实例场景，因为 artifact 路径只包含 `subset/sample_id/task_id`，不含 `run_id`、`dispatch_id` 或其他运行实例标识，见 [task_artifacts.py#L77-L81](/home/alpen/DEV/video2tasks/src/video2tasks/server/task_artifacts.py#L77)。
- server 构造 artifact metadata 时也只传了 `subset`、`sample_id`、窗口/边界相关信息和 `task_id`，没有 `run_id`，见 [app.py#L775-L792](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L775)。

我的修正建议：

- 这条应保留，但级别更适合放在中风险。
- 表述应改成“`tmp` 与 run 生命周期脱钩，跨 `run_id`、双实例、共享 `tmp` 时有覆盖和串读风险”，而不是把普通 timeout retry 当成主要风险来源。

### 2. “切窗默认合理但缺护栏、短窗会重复帧”成立，但应拆成两件事，级别略降

代码层面的事实成立：

- `build_windows()` 没有约束 `step_sec <= window_sec`，见 [windowing.py#L55-L80](/home/alpen/DEV/video2tasks/src/video2tasks/server/windowing.py#L55)。
- 默认配置确实是 `12s / 6s / 24帧`，见 [config.py#L184-L186](/home/alpen/DEV/video2tasks/src/video2tasks/config.py#L184)。
- `np.linspace(...).astype(int)` 会在短窗里制造重复帧，这一点我本地做了最小复现：`build_windows(30.0, 5, 12.0, 6.0, 24)` 的唯一帧只有 5 个，但逻辑帧槽位是 24 个。

但第一轮把这两件事捏成一条“中风险”问题，力度略大：

- “配置没有护栏”是实打实的配置风险，建议保留在中风险。
- “短窗重复帧”更像信息密度下降和提示词计数失真，属于质量/效率问题，建议降到中低风险。
- 默认参数本身并没有直接表现出覆盖缝隙；覆盖缝隙需要把 `step_sec` 调大到超过 `window_sec` 才会出现。

### 3. “联系图和上传链路有明显重复编码”需要缩窄适用范围，不能一概而论

第一轮这一条抓到了成本问题，但把不同后端混在了一起，导致表述偏重。

- OpenAI 路径确实会把 worker 里的图重新编码成 JPEG data URL，见 [openai_api.py#L631-L636](/home/alpen/DEV/video2tasks/src/video2tasks/vlm/openai_api.py#L631)。
- Gemini 原生路径如果拿到的是 `raw_bytes`，会直接把原始字节做 base64 包装，不走 numpy 解码再 JPEG 重编码，见 [gemini_api.py#L341-L363](/home/alpen/DEV/video2tasks/src/video2tasks/vlm/gemini_api.py#L341)。
- Gemini OpenAI 兼容路径也优先复用 `raw_bytes`，只有拿不到原始字节时才退回 JPEG 重编码，见 [gemini_api.py#L404-L417](/home/alpen/DEV/video2tasks/src/video2tasks/vlm/gemini_api.py#L404)。
- 另外，server 默认主链路因为直接落盘并走 `image_paths`，并不会在构任务时保留大批 base64 字符串；因此“服务端统一 base64 再传下去”也不是所有路径都成立，见 [app.py#L457-L478](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L457)。

我的修正建议：

- 这条应改成“OpenAI 与部分远端 API 路径存在明显重复解码/重编码成本；Gemini 路径已经做了原始字节直传优化”。
- 总级别更适合放中风险，而不是把所有后端一起上提。

## 新增发现

### 1. `VIDEO2TASKS_DUMP_INTERMEDIATE` 在 server 主链路上基本失效，这一点第一轮没有单独点透

这不是简单的“文档没写清”，而是行为和开关语义不一致。

- `FrameExtractor` 的默认 writer 构造会尊重 `VIDEO2TASKS_DUMP_INTERMEDIATE`，见 [windowing.py#L635-L639](/home/alpen/DEV/video2tasks/src/video2tasks/server/windowing.py#L635)。
- 但 server 在 app 初始化时已经直接构造了 writer，并在所有任务生成处显式传给 `FrameExtractor`，见 [app.py#L173-L174](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L173) 和 [app.py#L724-L801](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L724)。

结果就是：这个环境变量只对“独立使用 `FrameExtractor`”的路径有效，对 server 主链路无效。第一轮提到“服务端启动时不看该变量”，但我认为这里值得单独升级成一个明确结论，因为它会误导部署者和排障者。

### 2. 共享文件系统依赖和 README 的“多机 worker”叙述是直接冲突的

第一轮讲了结构性风险，但没有把它和对外叙述冲突这件事说透。

- README 明确写了“一个 server + 10 台 worker 并行”，见 [README.md#L156-L160](/home/alpen/DEV/video2tasks/README.md#L156)。
- 实际代码却要求 worker 能按 server 写出的本地路径直接读图，见 [app.py#L449-L478](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L449) 和 [runner.py#L109-L132](/home/alpen/DEV/video2tasks/src/video2tasks/worker/runner.py#L109)。

这意味着当前最危险的地方，不只是“实现有耦合”，而是“实现约束和部署叙事已经不一致”。这会让第一轮的高风险判断更站得住。

### 3. 测试已经覆盖了 `image_paths` 的 happy path，但没有覆盖它的坏路径

第一轮说“worker 能从 `image_paths` 读图已覆盖”，这没错，但遗漏了更关键的负例缺口。

- 已有测试只验证了正常路径和 dispatch_id 提交，见 [test_runner.py#L432-L481](/home/alpen/DEV/video2tasks/tests/worker/test_runner.py#L432)。
- 已有 artifact 测试也主要验证 manifest 与来源字段，见 [test_task_artifacts.py#L13-L106](/home/alpen/DEV/video2tasks/tests/server/test_task_artifacts.py#L13)。

缺的恰好是最需要的场景：

- `image_paths` 指向空文件或不存在路径时，worker 会不会提交明确错误，还是只会 crash 并等待 server timeout。
- server 在构任务前是否应该拦截 `decode_ok=False`。

这部分应该补进第二轮建议里，因为它和第 2 条高风险问题是同一条故障链。

## 重新排序后的建议

### 1. 先把“任务传图方式”说清楚并固定下来

优先级最高。先明确当前只支持共享盘模式，还是要支持网络分离模式。  
如果短期不改实现，至少应同时修正 README、部署文档和环境变量语义，不要继续给出“任意多机 worker”这种会误导的叙述。

### 2. 在 server 入队前拦住坏图，不要把抽帧错误伪装成 worker timeout

这是最直接的故障止损点。  
最小动作是 server 构任务时检查 artifact record 的 `decode_ok` 和 `byte_size`，一旦发现空图就直接记为抽帧失败，不进入 job queue。

### 3. 给 `tmp` 路径补运行身份，并补最基本的清理策略

建议把 `run_id` 放进 artifact 目录层级，必要时再加实例级唯一标识。  
如果不这样做，跨 run、双实例或共享 `tmp` 时，复核和排障都会变得很难。

### 4. 把切窗问题拆开治理：先加配置护栏，再决定是否做去重采样

先做 `step_sec <= window_sec` 这类硬约束，再评估短窗允许“少于目标帧数”是否比重复帧更稳。  
这件事重要，但不应排在共享存储和坏图前置校验之前。

### 5. 编码链路优化放在后面，而且要按后端分别做

OpenAI/远端 API 路径值得优化，Gemini 路径已经有原始字节直传能力。  
这件事能省成本，但它是第二梯队问题，前提是先把传输契约和失败语义理顺。
