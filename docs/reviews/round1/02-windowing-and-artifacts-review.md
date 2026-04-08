# 第一轮审计 02: 窗口切分、抽帧、联系图、中间产物与上传前准备

## 范围

本轮只看下面这些内容：

- [windowing.py](/home/alpen/DEV/video2tasks/src/video2tasks/server/windowing.py)
- [task_artifacts.py](/home/alpen/DEV/video2tasks/src/video2tasks/server/task_artifacts.py)
- [app.py](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py) 中和抽帧、落盘、发任务有关的部分
- [runner.py](/home/alpen/DEV/video2tasks/src/video2tasks/worker/runner.py) 中图片读取与上传前准备的部分
- [config.py](/home/alpen/DEV/video2tasks/src/video2tasks/config.py) 中 windowing 配置
- 相关测试： [test_windowing.py](/home/alpen/DEV/video2tasks/tests/server/test_windowing.py), [test_task_artifacts.py](/home/alpen/DEV/video2tasks/tests/server/test_task_artifacts.py), [test_runner.py](/home/alpen/DEV/video2tasks/tests/worker/test_runner.py)

## 总结

这条链路现在能跑，但它明显偏向“单机、本地磁盘共享”的工作方式。最大的问题不是单点函数，而是整条链路默认会把图片先写到 `tmp`，再把本地路径通过 HTTP 任务发给 worker 读取。这样做在单机上简单，但在多机、多容器、重试和长期运行时会放大路径耦合、磁盘增长、重复编码和失败重试成本。

窗口切分本身在默认参数下基本合理：`12s` 窗口，`6s` 步长，`24` 帧，覆盖较稳，边界也有重叠。但这套策略缺少配置保护，短窗和短段又会出现重复帧，联系图默认 `4x4` 时单格只有 `180x120`，对细小动作和接触点并不宽裕。

## 主要问题

### 1. 高风险：服务端默认总是落 `tmp`，任务交接强依赖“共享文件系统”

证据：

- 服务端启动时不看 `VIDEO2TASKS_DUMP_INTERMEDIATE`，直接创建 `TaskArtifactWriter`： [app.py#L173-L174](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L173)
- 构任务时优先走 `get_many_b64_with_artifacts(...)`，拿到 `artifact_batch` 后直接把 `image_paths` 塞进任务： [app.py#L449-L474](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L449)
- worker 收到任务后按本地路径读文件： [runner.py#L109-L131](/home/alpen/DEV/video2tasks/src/video2tasks/worker/runner.py#L109)

影响：

- 这要求 server 和 worker 看到的是同一套磁盘路径。
- 只要 worker 和 server 不在同一台机器、同一容器卷、同一挂载点，任务就会直接失败。
- 从系统形态上看，这条链路已经不是“HTTP 传任务”，而是“HTTP 传任务 + 本地共享存储补数据”。泛化能力比较差。

判断：

这是当前链路里最重的结构性风险。单机没问题，但一上容器、K8s、远端 worker 或混合部署，就会暴露。

建议：

- 如果必须支持分布式，任务里要么直接带图片字节，要么给共享对象存储 URL，不要给本地磁盘路径。
- 如果短期仍按单机跑，至少要把“必须共享 `VIDEO2TASKS_TMP_DIR`”写成硬约束，避免部署时踩坑。

### 2. 高风险：坏图会先落盘，再在 worker 端失败，最后变成超时重试

证据：

- 抽帧失败或联系图生成失败时，仍会把空字节写入文件： [task_artifacts.py#L120-L142](/home/alpen/DEV/video2tasks/src/video2tasks/server/task_artifacts.py#L120)
- 这些记录只要存在，就会进入 `image_paths`： [app.py#L473-L474](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L473)
- worker 读取到空文件会直接报错： [runner.py#L121-L131](/home/alpen/DEV/video2tasks/src/video2tasks/worker/runner.py#L121)

影响：

- 问题发生在服务端抽帧阶段，但真正报错发生在 worker 读图阶段。
- 这会把“图没准备好”的问题伪装成“worker 任务失败”或“任务超时”，排查路径会变长。
- 每次重试还会重复走调度、拉任务、读盘和推理前准备，浪费 CPU 和时间。

判断：

这是个真实 bug，不只是体验问题。失败信号传递得太晚，而且位置不对。

建议：

- 服务端在入队前就应该拦住空图、坏图。
- 至少要在构任务时过滤 `decode_ok=False` 或 `byte_size=0` 的记录，并把错误变成明确的抽帧失败。

### 3. 中高风险：`tmp` 路径按 `subset/sample_id/task_id` 固定命名，重跑和多实例下容易覆盖

证据：

- 任务目录只由 `subset`、`sample_id`、`task_id` 组成： [task_artifacts.py#L77-L81](/home/alpen/DEV/video2tasks/src/video2tasks/server/task_artifacts.py#L77)
- 图片文件名也固定为 `kind_0000.png` 这种形式： [task_artifacts.py#L120-L124](/home/alpen/DEV/video2tasks/src/video2tasks/server/task_artifacts.py#L120)
- 服务端构任务时把业务 `task_id` 直接塞给 artifact 元数据： [app.py#L465-L466](/home/alpen/DEV/video2tasks/src/video2tasks/server/app.py#L465)

影响：

- 同一个任务只要被重新生成一次，就会写回同一个目录。
- 单次进程内的普通超时重试大多复用原来的 `job`，问题不一定立刻出现；但服务重启、重复跑同一数据、双实例共用 `tmp`、或不同原始 ID 清洗后变成同名时，都可能互相覆盖。
- 覆盖后最难排查，因为 manifest 和图片名都看起来“合法”，只是内容被后一次写入替换了。

建议：

- 路径层至少要带上一次真正唯一的运行标识，比如 dispatch、attempt、run_id 或时间戳。
- 如果要保留“按任务聚合”的可读目录，也建议再包一层唯一子目录，避免覆盖。

### 4. 中风险：切窗默认合理，但配置没有护栏，抽帧在短窗上会制造重复帧

证据：

- 默认切窗参数是 `12s` 窗口、`6s` 步长、`24` 帧： [config.py#L184-L186](/home/alpen/DEV/video2tasks/src/video2tasks/config.py#L184)
- `build_windows(...)` 不校验 `step_sec <= window_sec`，也不校验步长和覆盖关系： [windowing.py#L55-L84](/home/alpen/DEV/video2tasks/src/video2tasks/server/windowing.py#L55)
- 抽帧直接用 `np.linspace(...).astype(int)`，短窗、尾窗、短段都会产生重复帧： [windowing.py#L70-L72](/home/alpen/DEV/video2tasks/src/video2tasks/server/windowing.py#L70), [windowing.py#L144-L148](/home/alpen/DEV/video2tasks/src/video2tasks/server/windowing.py#L144)
- 当前测试没有覆盖 `build_windows` 和短窗重复帧场景。

影响：

- 默认参数下，主流程覆盖还可以，`12s/6s` 的重叠对边界检测是够用的。
- 但一旦把 `step_sec` 调大，可能出现覆盖缝隙；这类配置错误现在不会被拦。
- 对很短的视频、尾窗、边界细化窗、最终 segment 抽帧，重复帧会占掉图片位，降低信息密度。
- worker 侧仍按 `len(frame_ids)` 计算逻辑帧数，提示词会把这些重复位也算进去，等于把“同一张图”当成多个时间点。

建议：

- 配置层最好明确限制 `step_sec <= window_sec`，并优先保证尾部覆盖而不是默认依赖经验参数。
- 对短窗和短段，可以去重后再抽样，或者允许“实际帧数小于目标帧数”。

### 5. 中风险：联系图和上传链路存在明显的重复编码与资源放大

证据：

- 非联系图路径：服务端先编码成 PNG： [windowing.py#L887-L897](/home/alpen/DEV/video2tasks/src/video2tasks/server/windowing.py#L887)
- 联系图路径：服务端先生成 PNG，再写本地文件： [windowing.py#L918-L953](/home/alpen/DEV/video2tasks/src/video2tasks/server/windowing.py#L918), [task_artifacts.py#L112-L150](/home/alpen/DEV/video2tasks/src/video2tasks/server/task_artifacts.py#L112)
- OpenAI worker 会再把读回来的图重新编码成 JPEG data URL： [openai_api.py#L14-L26](/home/alpen/DEV/video2tasks/src/video2tasks/vlm/openai_api.py#L14), [openai_api.py#L631-L636](/home/alpen/DEV/video2tasks/src/video2tasks/vlm/openai_api.py#L631)
- Gemini 虽然支持直接吃原始字节，但仍会把 PNG bytes 再做一次 base64 包装： [gemini_api.py#L37-L55](/home/alpen/DEV/video2tasks/src/video2tasks/vlm/gemini_api.py#L37), [gemini_api.py#L341-L356](/home/alpen/DEV/video2tasks/src/video2tasks/vlm/gemini_api.py#L341)
- `png_compression` 默认是 `0`： [config.py#L224-L226](/home/alpen/DEV/video2tasks/src/video2tasks/config.py#L224)

影响：

- 典型链路是“视频帧 -> PNG -> 落盘 -> 读盘 -> 解码 -> JPEG/base64 -> 上传”。
- 对 OpenAI、Qwen、remote API 这类路径，重复编码非常明显。
- 这会同时放大磁盘写入、磁盘读取、CPU 编解码、内存拷贝和网络传输。
- 长视频再叠加 `window_repeat_count`、refinement、boundary refinement、segment labeling，成本会按阶段叠上去。

建议：

- 如果最终上传端只吃 JPEG，就没必要在中间统一落无压缩 PNG。
- 可以把“落盘格式”和“上传格式”统一，减少一次解码和一次重编码。
- 现在这条链路更像“为了调试方便而保留了最重的中间层”，适合排障，不适合长期高吞吐。

## 切窗与联系图单独评价

### 切窗策略

优点：

- 默认 `12s/6s` 是稳妥的，窗口重叠足够，边界不容易刚好落在缝里。
- 主窗口之外还有两层补救：歧义窗口 refinement 和边界局部 refinement，思路是对的。

问题：

- 这套策略建立在“配置别乱调”的前提上，代码没有把这个前提固定住。
- 尾窗用“小于半窗就丢弃”的规则： [windowing.py#L76-L80](/home/alpen/DEV/video2tasks/src/video2tasks/server/windowing.py#L76)
- 这个规则在默认重叠下通常没问题，但它并不是通用规则。如果以后改成更稀疏的步长，尾部覆盖可能出问题。

建议：

- 保留默认值没问题，但最好把它当成受保护配置，而不是任意可调参数。

### 联系图质量与数量

现状：

- 默认配置里是 `4x4` 联系图： [config.py#L227-L232](/home/alpen/DEV/video2tasks/src/video2tasks/config.py#L227)
- 在 `720x480` 总图下，每格约 `180x120`。

判断：

- 对大动作、明显阶段切换，这个尺寸通常够用。
- 对手部细动作、夹取接触、器具边缘变化、遮挡下的轻微状态变化，这个尺寸偏紧。
- `24` 帧配 `4x4` 时通常会变成 2 张联系图。数量控制住了，但每格信息量不高。

建议：

- 如果目标是抓边界，建议把“每张联系图最多塞多少格”作为硬约束，而不是只看行列数。
- 从泛化角度看，单张不宜超过 `12` 格会更稳，尤其是动作小、目标物细的时候。

## `tmp` 中间产物策略评价

优点：

- 目录结构清楚，可按 `subset/sample/task` 回查。
- manifest 里带了 `frame_ids`、`source`、`byte_size`，对排障有帮助。

问题：

- 没看到清理策略，也没看到保留上限。
- 服务端默认总写 `tmp`，这让 `tmp` 从“调试辅助”变成了“主链路依赖”。
- 路径命名偏稳定，可读性好，但唯一性不够。

成本判断：

- 非联系图模式下，默认每个窗口要写 24 张图。
- 联系图模式下，默认 24 帧大约会落成 2 张图，但 refinement、boundary refinement、segment labeling 会继续写。
- 这意味着磁盘成本不是单次窗口成本，而是“窗口数 × 阶段数 × 重试数”的累积成本。

## 测试覆盖缺口

已覆盖：

- 联系图顺序保持。
- ffmpeg 失败后回退到 cv2。
- artifact manifest 基本写入。
- worker 能从 `image_paths` 读图。

未覆盖但我认为应该补：

- `build_windows(...)` 的尾窗覆盖和异常配置。
- 短窗、短段、boundary refinement 下的重复帧。
- 空字节图片进入 `image_paths` 后的服务端处理。
- `task_id` 重跑或多实例共享 `tmp` 时的覆盖问题。
- server 和 worker 不共享文件系统时的行为。

## 结论

如果只在单机跑，这套实现短期可用，默认窗口策略也基本说得过去；但它把“中间产物落盘”从调试手段做成了主通路，导致路径耦合、失败位置后移、重复编码和磁盘成本都被放大。

我会优先把问题排序成下面这样：

1. 先解决 `image_paths` 对共享文件系统的硬依赖。
2. 再把空图、坏图在服务端前置拦截，不要把抽帧错误拖到 worker。
3. 然后处理 `tmp` 目录唯一性和清理策略。
4. 最后再细调联系图格数、抽帧去重和编码格式，压成本、提泛化。
