# Round 2 配置与运行复核 06

更新时间：2026-04-08  
主审文档：`docs/reviews/round1/06-config-and-ops-review.md`  
复核方式：只复核第一轮文档的证据链、问题级别和遗漏项；不改代码，只新增本复核文档。

## 1. 复核对象

本轮实际回看了下面这些文件和路径：

1. 主审文档：`docs/reviews/round1/06-config-and-ops-review.md`
2. 配置定义与加载：`src/video2tasks/config.py`
3. 运行配置：`config.example.yaml`、`config.g3flash.yaml`、`config.yaml`
4. README 与 CLI：`README.md`、`src/video2tasks/cli/server.py`、`src/video2tasks/cli/worker.py`、`src/video2tasks/cli/cluster.py`、`src/video2tasks/cli/validate_config.py`
5. 运行目录与中间产物：`src/video2tasks/server/app.py`、`src/video2tasks/server/windowing.py`、`src/video2tasks/server/task_artifacts.py`、`src/video2tasks/server/exporter.py`、`src/video2tasks/worker/runner.py`
6. 相关测试与忽略规则：`tests/test_config.py`、`tests/cli/test_server.py`、`.gitignore`

## 2. 确认成立的点

1. 高：`config.g3flash.yaml` 已经是受 Git 跟踪的真实密钥暴露点，这一点成立。  
证据：`config.g3flash.yaml:46-53` 和 `config.g3flash.yaml:80-86` 直接写了真实 `api_key`；`git ls-files` 显示该文件已受版本控制。第一轮把它列为最高优先级是对的。

2. 高：`config.example.yaml` 不能被称为“all available options”，这一点成立。  
证据：`README.md:285-293` 直接把它写成“all available options”；但 `src/video2tasks/config.py:192-250`、`src/video2tasks/config.py:294-307` 里定义了多组字段，示例文件没有覆盖，例如 `windowing.boundary_prompt_mode`、`windowing.enable_refinement_pass`、`llm_merge.summary_levels`、`llm_merge.repeat_count`、`llm_merge.boundary_vote_threshold`。

3. 高：示例值和代码默认值大量漂移，已经影响对真实默认行为的判断，这一点成立。  
证据：  
`worker.count` 在 `src/video2tasks/config.py:165` 默认是 `7`，但 `config.example.yaml:42` 是 `4`。  
`server.max_retries_per_job` 在 `src/video2tasks/config.py:92` 默认是 `5`，但 `config.example.yaml:36` 是 `50`。  
`windowing.frames_per_window` 在 `src/video2tasks/config.py:186` 默认是 `24`，但 `config.example.yaml:86` 是 `128`。  
`windowing.use_contact_sheets` 在 `src/video2tasks/config.py:227-230` 默认是 `false`，但 `config.example.yaml:91` 是 `true`。  
除第一轮已列出的几项外，还存在 `worker.gemini.max_output_tokens`、`llm_merge.protected_boundary_support_threshold`、`llm_merge.protect_duplicate_tail_anchor`、`llm_merge.coarse_min_output_ratio` 等额外漂移。

4. 高：`tmp/` 不是单纯调试目录，而是正常运行路径的一部分，清理不当会打坏任务，这一点成立。  
证据：`src/video2tasks/server/app.py:173-174` 创建了 `TaskArtifactWriter(root_dir=tmp)`；`src/video2tasks/server/app.py:457-478` 优先把持久化后的 `image_paths` 放进 job；`src/video2tasks/worker/runner.py:101-121` 会直接从这些路径读取图片；`src/video2tasks/server/task_artifacts.py:77-124` 把文件写到 `tmp/<subset>/<sample>/<task>/images/...`。第一轮对此的核心判断是正确的。

5. 中：配置优先级注释写反了，真实行为是 `env > yaml > defaults`，这一点成立。  
证据：`src/video2tasks/config.py:469-470` 注释写的是 `file > env > defaults`；但实现是 `src/video2tasks/config.py:698-700`，环境变量在 merge 时覆盖 YAML；测试 `tests/test_config.py:192-241` 也明确验证了环境变量覆盖文件值。

6. 中：README 对环境变量入口、输出目录和运行目录职责说明不足，这一点成立。  
证据：`README.md:285-293` 只给了很粗的章节表；`README.md:371-375` 只明确举了 `OPENAI_API_KEY`；`README.md:392-416` 的项目结构也没有解释 `runs/`、`tmp/`、`exports/`、`.DONE`、`.FAILED` 这些真实运行产物。

## 3. 需要修正的点

1. 第一轮把 `max_empty_retries_per_job=0` 主要写成了“示例配置风险”，这不够准确。  
更准确的说法是：这是代码默认值本身的风险，不只是示例文件写坏了。  
证据：`src/video2tasks/config.py:93-96` 默认就是 `0`，含义也是“无限”；`config.example.yaml:37` 只是沿用了这个默认；`src/video2tasks/server/app.py:542-570` 会按这个默认无限回队列。  
结论：这个问题应当保留，而且级别比第一轮写得更高，因为它不是文档漂移，而是实际默认行为。

2. 第一轮对 `VIDEO2TASKS_DUMP_INTERMEDIATE` 的表述容易让人误解成“server 是否写 tmp 由这个开关控制”，这需要修正。  
更准确的说法是：这个开关只控制 `FrameExtractor` 在“没有显式传入 artifact_writer”时是否自动创建 writer，见 `src/video2tasks/server/windowing.py:635-639`、`src/video2tasks/server/windowing.py:645-656`。  
而 server 路径在 `src/video2tasks/server/app.py:173-174` 和 `src/video2tasks/server/app.py:750` 已经显式传入 `TaskArtifactWriter`，所以正常 server 运行会写 `tmp/`，与 `VIDEO2TASKS_DUMP_INTERMEDIATE` 无关。  
结论：问题成立，但第一轮前半段证据链混用了两个不同入口，建议改写。

3. 第一轮对 `v2t-validate` 的描述有一半对，一半不对。  
对的部分：它确实要求显式传 `--config`，见 `src/video2tasks/cli/validate_config.py:9-15`。  
不对的部分：它并不是“只校验文件入口、不看环境变量”。因为它调用的是 `Config.from_yaml(config)`，见 `src/video2tasks/cli/validate_config.py:18-20`，而 `Config.from_yaml()` 内部仍会合并环境变量，见 `src/video2tasks/config.py:452-461`、`src/video2tasks/config.py:698-700`。  
结论：真正的问题是“不能验证纯 env-only 启动”和“不能复现 `Config.load()` 的当前目录自动发现逻辑”，不是“env 覆盖被忽略”。

4. 第一轮把 `server.host=0.0.0.0` 放得过高，级别建议下调到中。  
理由：它确实值得提醒，尤其当前 `/get_job` 和 `/submit_result` 没有鉴权，见 `src/video2tasks/server/app.py:489-578`；但和“已入库真实密钥”“实际默认无限空结果重试”“示例文件严重漂移”相比，它更像部署硬化项，而不是当前最紧急的配置真相问题。  
结论：问题存在，但建议排位下移。

5. 第一轮对根目录运行目录的判断还可以再收紧。  
`bak/`、`runs/`、`exports/`、`tmp/` 的确没有在 README 里解释清楚，这一点成立；但它们已经被 `.gitignore` 忽略，见 `.gitignore:24`、`.gitignore:34`、`.gitignore:58`、`.gitignore:81`。  
结论：这里的主问题是“运维语义不清”和“清理风险”，不是“容易被误提交”。

## 4. 新增发现

1. 高：第一轮漏掉了另一个真实且正在生效的密钥入口：根目录 `config.yaml`。  
证据：`config.yaml:35-60` 也写了真实 `openai`、`gemini`、`llm_merge` 密钥；`src/video2tasks/config.py:474-476` 在未传 `--config` 时会优先自动发现当前目录的 `config.yaml`；`.gitignore:49-50` 说明它是被忽略的本地文件，不是已入库泄露点。  
复核结论：这不是“又一个已提交泄露文件”，但它是当前实际运行最容易命中的配置源，第一轮完全没提，属于明显遗漏。

2. 中：README 完全没有 Gemini backend 章节，这一点比第一轮写得还要缺。  
证据：`README.md:297-377` 只讲了 Dummy、Qwen3-VL、Remote API、OpenAI、Custom，没有 Gemini；但 `config.example.yaml:71-80` 和 `config.g3flash.yaml:42-53` 都把 Gemini 当作正式 backend 使用。  
复核结论：这会直接放大“示例能跑、README 不会配”的问题，应该进入建议清单前排。

3. 中：环境变量支持不是全量映射，而是“部分支持、命名不对称”。  
证据：`src/video2tasks/config.py:498-693` 虽然提供了不少 env 入口，但没有 `SERVER_HOST`、`SERVER_MAX_QUEUE`、`INFLIGHT_TIMEOUT_SEC`、`SERVER_AUTO_EXIT_AFTER_ALL_DONE`、`LOG_LEVEL`，也没有任何 `windowing.*` 的 env 映射。  
复核结论：第一轮只写了“环境变量入口很多”，但没指出更关键的一点：env-only 启动是可用的，却不是和 YAML 一一对等的。

4. 中：`Config.load()` 对当前工作目录敏感，这个运维风险第一轮漏掉了。  
证据：`src/video2tasks/config.py:474-476` 只查 `Path("config.yaml")`；`src/video2tasks/cli/server.py:16-23`、`src/video2tasks/cli/worker.py:16-23`、`src/video2tasks/cli/cluster.py:15-31` 默认都走 `Config.load(config)`。  
复核结论：同一台机器上从不同 cwd 启动，可能读到不同配置，或者根本读不到 repo 里的 `config.yaml`。这对 systemd、容器入口脚本和手工排障都很不友好。

5. 中：Gemini 的真实密钥回退链比第一轮写得更复杂。  
证据：`src/video2tasks/worker/runner.py:187-196` 会把 `config.worker.gemini.api_key` 或 `GEMINI_API_KEY` 传给 backend；而 `src/video2tasks/vlm/gemini_api.py:272-274` 还会再回退到 `GOOGLE_API_KEY`。  
复核结论：第一轮提到了多层 fallback，但漏掉了 `GOOGLE_API_KEY` 这一层，导致“到底用了哪份 key”仍然说得不够全。

## 5. 重新排序后的建议

1. 先处理真实密钥。  
立即轮换 `config.g3flash.yaml` 中的已入库密钥，并把仓库内受跟踪配置文件里的真实密钥全部移除。与此同时，补一条 README 或运维说明，明确 `config.yaml` 只是本地忽略文件，不应再作为“可分享模板”。

2. 立即收紧空结果重试。  
把 `max_empty_retries_per_job` 当成真实默认行为问题处理，而不是示例文件问题。至少要在文档里明确 `0` 代表无限，并给批量跑一个有上限的建议值。

3. 统一“配置真相”的来源。  
README 里应明确写出 `环境变量 > YAML > 代码默认值`。同时要决定 `config.example.yaml` 的角色：要么做成完整默认镜像，要么明确标注“这是调优示例，不代表默认值”，不能继续两头都像。

4. 把 `tmp/` 的语义写清楚。  
要明确说明它是 worker 读取的实际输入目录之一，不是纯缓存；运行中不能随意清理；`VIDEO2TASKS_DUMP_INTERMEDIATE` 也不能再被写成 server 是否落盘的总开关。

5. 补齐 README 的 backend 和目录说明。  
至少补上 Gemini backend、输出目录结构、`.DONE` / `.FAILED`、`tmp/`、`exports/` / `clips/` 的位置与用途，再补一张环境变量覆盖表，特别注明哪些配置没有 env 映射。

6. 运维侧默认要求显式 `--config` 和绝对路径。  
这可以绕开当前工作目录敏感问题，也能降低 `run.base_dir`、`VIDEO2TASKS_TMP_DIR` 相对路径带来的目录漂移。

7. `server.host=0.0.0.0` 作为第二梯队处理。  
建议在 README 里把它写成“容器/远端部署常用值”，同时给本机调试单独推荐 `127.0.0.1`；但它不应排在密钥、无限重试和配置漂移之前。

## 6. 复核结论

第一轮文档的主判断大体靠谱，尤其是“已入库密钥暴露”“示例配置不再可信”“`tmp/` 不是普通临时目录”这三点，方向是对的。  
但它有三处需要明显修正：一是把 `max_empty_retries_per_job=0` 写轻了，二是把 `VIDEO2TASKS_DUMP_INTERMEDIATE` 和 server 正常落盘路径混在了一起，三是把 `v2t-validate` 说成了“只看文件、不看环境变量”。  
另外，`config.yaml` 这个本地忽略但会被默认自动加载的真实配置源，是第一轮最值得补上的遗漏项。
