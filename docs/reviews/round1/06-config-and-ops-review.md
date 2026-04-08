# Round 1 配置与运行审计 06

更新时间：2026-04-07  
范围：`config.example.yaml`、现有 `config.*.yaml`、`README.md`、CLI、运行目录组织（`runs/`、`tmp/`、`bak/`、`exports/`）和环境变量入口。  
约束：本次只做审计，不改代码。

## 1. 先说结论

这套项目现在能跑，但配置层和运行层有明显“文档说一套、代码实际另一套”的问题。最需要先处理的不是模型参数，而是下面四件事：

1. 仓库里已经出现真实密钥，风险最高。
2. `config.example.yaml` 不是当前真实的“完整示例”，而且多个默认值和代码默认行为不一致。
3. `tmp/`、`runs/`、`exports/`、`bak/` 在仓库根目录同时存在，但 README 没把职责边界讲清楚，批量跑时很容易混。
4. 环境变量入口很多，优先级也比较绕，README 只讲了很少一部分，线上排障会很慢。

## 2. 最关键的配置项

下面这些项最影响“能不能稳定跑”和“跑完产物会不会混”。

| 配置项 | 现在的情况 | 审计判断 |
| --- | --- | --- |
| `run.base_dir` | 默认 `./runs`，相对当前工作目录，见 `src/video2tasks/config.py:20` | 能用，但不够稳。换目录启动时，结果会落到不同地方。 |
| `run.run_id` | 默认 `default`，见 `src/video2tasks/config.py:21` | 不安全。多人或多轮批跑时很容易把结果混进同一目录。 |
| `server.host` | 默认 `0.0.0.0`，示例配置也是这样，见 `src/video2tasks/config.py:88`、`config.example.yaml:32` | 对公开机器不安全。单机本地跑更适合默认 `127.0.0.1`。 |
| `server.max_retries_per_job` | 代码默认 `5`，但示例配置写 `50`，见 `src/video2tasks/config.py:92`、`config.example.yaml:36` | 文档漂移，容易让人误判重试强度。 |
| `server.max_empty_retries_per_job` | 示例里 `0` 表示无限，见 `config.example.yaml:37` | 对批量跑不安全。坏窗口可能长时间反复打转。 |
| `worker.count` | 代码默认 `7`，示例配置写 `4`，README 也写默认 `7`，见 `src/video2tasks/config.py:165`、`config.example.yaml:42`、`README.md:267` | 文档不一致，容易导致机器负载判断错误。 |
| `worker.backend` + 各后端 key/base_url | 支持 YAML 和环境变量双入口，见 `src/video2tasks/config.py:545-601` | 关键，但现在说明不够完整。 |
| `windowing.frames_per_window` | 代码默认 `24`，示例配置写 `128`，见 `src/video2tasks/config.py:186`、`config.example.yaml:86` | 这会直接改变速度、成本和结果，不该让用户靠猜。 |
| `windowing.use_contact_sheets` | 代码默认 `false`，示例配置写 `true`，见 `src/video2tasks/config.py:227-230`、`config.example.yaml:91` | 会直接影响上传图片数和耗时，必须写清。 |
| `llm_merge.enabled` 及其 API 配置 | 影响二次合并和额外模型调用，见 `config.example.yaml:96-122`、`src/video2tasks/config.py:271-380` | 对成本和最终结果影响很大，但 README 基本没讲。 |
| `VIDEO2TASKS_TMP_DIR` | 控制中间产物目录，见 `src/video2tasks/server/app.py:173` | 这是运维关键项，但 README 没说。 |

## 3. 主要问题

### 3.1 高优先级

#### 3.1.1 现有运行配置文件里出现了真实密钥

`config.g3flash.yaml` 里直接写了 Gemini 和 LLM merge 的真实 `api_key`，还带了实际接口地址，见：

- `config.g3flash.yaml:47-53`
- `config.g3flash.yaml:81-86`

这不是“示例不够好”的问题，而是已经进入密钥泄露范围。只要这个文件被提交、同步、打包或截图，风险就已经发生。

建议：

- 立刻停用并轮换这些密钥。
- 后续只允许把密钥放环境变量，不再放进仓库内 YAML。
- 运行配置文件里只保留占位符和注释。

#### 3.1.2 `config.example.yaml` 不是当前真实完整配置

README 明确写“See `config.example.yaml` for all available options”，见 `README.md:285`。但实际不是这样。

代码默认配置里有、示例文件里没有的项，至少包括：

- `windowing.boundary_prompt_mode`
- `windowing.segment_labeling_mode`
- `windowing.enable_refinement_pass`
- `windowing.enable_boundary_refinement`
- `windowing.boundary_refinement_window_sec`
- `windowing.boundary_refinement_frames_per_window`
- `windowing.boundary_refinement_abstain_merge_max_support`
- `windowing.refinement_frames_per_window`
- `windowing.adaptive_merge_guard`
- `windowing.adaptive_merge_min_segments`
- `windowing.adaptive_merge_collapse_ratio`
- `windowing.boundary_support_threshold`
- `windowing.refine_final_instructions`
- `llm_merge.summary_levels`
- `llm_merge.repeat_count`
- `llm_merge.boundary_vote_threshold`

这些字段都在 `src/video2tasks/config.py` 里有默认值，见：

- `src/video2tasks/config.py:192-250`
- `src/video2tasks/config.py:294-307`

另外，现有实跑配置 `config.g3flash.yaml` 里也已经用了很多示例文件里没有的 `windowing.*` 项，见 `config.g3flash.yaml:55-78`。

这会导致两个直接问题：

1. 用户以为自己已经“看完全部配置”了，其实没有。
2. 用户复制示例文件后，运行结果和当前项目常用配置差别很大。

#### 3.1.3 示例值和代码默认值不一致，默认行为很难判断

几个最明显的冲突如下：

- `worker.count`：示例 `4`，代码默认 `7`，见 `config.example.yaml:42`、`src/video2tasks/config.py:165`
- `server.max_retries_per_job`：示例 `50`，代码默认 `5`，见 `config.example.yaml:36`、`src/video2tasks/config.py:92`
- `windowing.frames_per_window`：示例 `128`，代码默认 `24`，见 `config.example.yaml:86`、`src/video2tasks/config.py:186`
- `windowing.window_repeat_count`：示例 `2`，代码默认 `1`，见 `config.example.yaml:87`、`src/video2tasks/config.py:187-190`
- `windowing.use_contact_sheets`：示例 `true`，代码默认 `false`，见 `config.example.yaml:91`、`src/video2tasks/config.py:227-230`
- `worker.gemini.api_mode`：示例 `openai_compatible`，代码默认 `native`，见 `config.example.yaml:76`、`src/video2tasks/config.py:145-149`
- `worker.gemini.base_url`：示例是第三方兼容地址，代码默认是 Google native 地址，见 `config.example.yaml:77`、`src/video2tasks/config.py:146-149`
- `llm_merge.granularity`：示例 `coarse`，代码默认 `guarded`，见 `config.example.yaml:108`、`src/video2tasks/config.py:309-312`

这不是小差异。它会直接影响：

- 单机负载
- 请求成本
- 重试次数
- 输出切分粒度
- 实际调用的 API 形态

#### 3.1.4 `tmp/` 会默认持续写中间产物，批量跑容易把磁盘打满

从命名上看，`VIDEO2TASKS_DUMP_INTERMEDIATE` 像是“打开后才写调试中间产物”，见 `src/video2tasks/server/windowing.py:635-639`。  
但 server 在创建应用时会直接创建 `TaskArtifactWriter(root_dir=tmp)`，见 `src/video2tasks/server/app.py:173-174`，而后续真正建 job 时总是把这个 writer 传给 `FrameExtractor`，见 `src/video2tasks/server/app.py:750`，并且抽帧时默认 `persist_artifacts=True`，见 `src/video2tasks/server/windowing.py:879`、`src/video2tasks/server/windowing.py:945-952`。

同时，job 构建逻辑会优先把这些落盘图片路径发给 worker，而不是直接传内存里的 `images`，见 `src/video2tasks/server/app.py:457-478`。

这意味着：

- 正常运行就会写 `tmp/`
- `tmp/` 不只是调试目录，而是当前默认主路径的一部分
- 批量跑时会不断生成 `tmp/<subset>/<sample>/<task>/images/*.png`

而 README 没有说明这个目录会持续增长，也没有说该怎么清理。

### 3.2 中优先级

#### 3.2.1 配置加载优先级容易误解

`Config.load` 的注释写的是 “file > env > defaults”，见 `src/video2tasks/config.py:469-470`。  
但真实实现是 `_deep_merge_dicts(base_data, env_overrides)`，也就是环境变量覆盖 YAML，见 `src/video2tasks/config.py:698-700`。测试也明确验证了 YAML 会被环境变量覆盖，见 `tests/test_config.py:192-241`。

这会让排障时出现经典问题：

- 用户以为自己改了 `config.yaml`
- 实际仍然被 shell 里的环境变量覆盖

建议至少把优先级写成一句明确的话：

`env > yaml > defaults`

#### 3.2.2 环境变量入口太多，而且名字风格不统一

环境变量入口很多，见 `src/video2tasks/config.py:498-693`。问题不只是数量多，更在于命名不统一：

- `RUN_BASE` 对应 YAML 的 `run.base_dir`
- `PORT` 很泛，容易跟其他服务共用环境冲突
- `MODEL_PATH` 实际只给 `qwen3vl` 用
- `SERVER_URL` 实际是 worker 访问 server 的地址
- Gemini 同时支持 `GEMINI_BASE_URL` 和 `GOOGLE_GEMINI_BASE_URL`，见 `src/video2tasks/config.py:583-585`

运行时还有第二层 API key fallback：

- OpenAI：`src/video2tasks/worker/runner.py:174-185` 和 `src/video2tasks/vlm/openai_api.py:256-258`
- Gemini：`src/video2tasks/worker/runner.py:187-196` 和 `src/video2tasks/vlm/gemini_api.py:272-274`

好处是更耐用，坏处是更难解释“当前到底用的是哪份值”。

#### 3.2.3 `v2t-validate` 和真实启动路径不完全一致

`v2t-server`、`v2t-worker`、`v2t-cluster` 都走 `Config.load()`，所以支持：

- `--config config.yaml`
- 当前目录自动发现 `config.yaml`
- 纯环境变量启动

这点在测试里也被覆盖了，见：

- `tests/cli/test_server.py:6-22`
- `tests/cli/test_worker.py:6-22`
- `tests/cli/test_cluster.py:80-92`

但 `v2t-validate` 必须显式传 `--config`，而且只校验文件入口，见 `src/video2tasks/cli/validate_config.py:9-19`。  
这会造成“验证通过的方式”和“实际运行的方式”不完全一致，尤其是 env-only 启动时。

#### 3.2.4 目录命名容易让人误会

当前代码实际产物目录是：

- 最终结果：`run.base_dir/<subset>/<run_id>/samples/<sample_id>/...`，见 `src/video2tasks/server/app.py:57-59`
- 导出视频：`run_dir/exports/<sample_id>/...` 或 `run_dir/clips/<sample_id>/...`，见 `src/video2tasks/server/exporter.py:113`、`src/video2tasks/server/exporter.py:301`
- 中间产物：`tmp/<subset>/<sample_id>/<task_id>/...`，见 `src/video2tasks/server/task_artifacts.py:77-82`

但仓库根目录现在同时存在：

- `runs/`
- `tmp/`
- `exports/`
- `bak/`

README 的项目结构却没有把这些运行目录放进去，见 `README.md:392-416`。  
用户很容易误以为：

- 根目录 `exports/` 是程序默认导出目录
- `bak/` 是程序自动使用的归档目录
- `tmp/` 是可随时删的临时缓存，但实际可能正在被 worker 读

从代码搜索看，`bak/` 没有被程序直接引用，更像是人工归档目录；问题不是它存在，而是没有任何约定说明。

## 4. README 和操作文档缺口

### 4.1 README 只讲了很少一部分环境变量

README 只明确举了 `OPENAI_API_KEY`，见 `README.md:358-375`。  
但实际还有大量运行入口没有写清：

- `DATASETS`
- `RUN_BASE`
- `RUN_ID`
- `PORT`
- `SERVER_URL`
- `WORKER_COUNT`
- `BACKEND`
- `GEMINI_API_KEY`
- `GEMINI_BASE_URL` / `GOOGLE_GEMINI_BASE_URL`
- `LLM_MERGE_*`
- `VIDEO2TASKS_TMP_DIR`

### 4.2 README 没把输出目录结构讲清楚

README 快速开始教的是“复制配置然后运行”，见 `README.md:252-277`，但没有告诉用户：

- 结果最终落在哪里
- `.DONE` 和 `.FAILED` 在哪里
- `tmp/` 是什么
- `exports/` 和 `clips/` 是相对 `run_dir`，不是仓库根

对第一次批量跑的人，这个缺口非常大。

### 4.3 README 没提醒默认值漂移风险

README 写了 `worker.count` 默认 `7`，见 `README.md:267`。  
但既没有提示“示例文件里可能不是代码默认值”，也没有提示“环境变量会覆盖 YAML”。这会让 README 看上去像唯一真相，实际不是。

## 5. 上线或批量跑时的风险

### 5.1 结果目录容易串

如果不改 `run.run_id`，默认就是 `default`，见 `src/video2tasks/config.py:21`。  
多人、多轮实验、重跑同数据集时，结果很容易堆在一起。尤其当前根目录已经有不少历史 `runs/`、`tmp/`、`exports/`、`bak/`，说明这类混用已经很容易发生。

### 5.2 空结果无限重试会拖死整批任务

示例配置把 `max_empty_retries_per_job` 设成了 `0`，也就是无限，见 `config.example.yaml:37`。  
server 侧空结果会继续回队列，见 `src/video2tasks/server/app.py:542-570`。  
这在单个窗口坏掉时，会造成：

- 队列持续被坏任务占住
- 总完成时间不可预估
- 成本和日志量失控

### 5.3 单机默认暴露到全网

`server.host` 默认 `0.0.0.0`，见 `src/video2tasks/config.py:88`。  
如果用户只是本机调试，或者在共享机器上跑，这个默认不够保守。

### 5.4 `tmp/` 清理时机很危险

当前 worker 拿到的 job 可能是 `image_paths`，见 `src/video2tasks/server/app.py:473-475`。  
也就是说，`tmp/` 里的图片不只是“落地副本”，而是 worker 的实际输入源之一。运维如果按“临时目录”习惯做自动清理，可能会把正在处理的任务直接打坏。

## 6. 优化建议

下面的建议按优先顺序排。

### 6.1 先做的

1. 立刻轮换 `config.g3flash.yaml` 里的密钥，并停止把真实密钥写进仓库文件。
2. 把 `config.example.yaml` 修成“当前完整可配项的真实镜像”，不要再让它和代码默认值长期漂移。
3. 在 README 增加一段很直白的“配置优先级”：`环境变量 > YAML > 代码默认值`。
4. 在 README 增加“运行产物目录说明”，明确写出：
   - `runs/<subset>/<run_id>/samples/<sample_id>/segments.json`
   - `runs/<subset>/<run_id>/exports/<sample_id>/...`
   - `runs/<subset>/<run_id>/clips/<sample_id>/...`
   - `tmp/<subset>/<sample_id>/<task_id>/...`
5. 明确说明 `tmp/` 默认会增长，不能在任务运行时随意清理。

### 6.2 批量跑建议

1. 每次批跑都强制使用唯一 `run_id`，建议包含日期、模型名、批次名。
2. `run.base_dir` 和 `VIDEO2TASKS_TMP_DIR` 都用绝对路径，不要依赖当前工作目录。
3. 单机本地跑时，把 `server.host` 固定成 `127.0.0.1`。
4. 不要使用无限空结果重试；给 `max_empty_retries_per_job` 一个明确上限。
5. 把 `bak/` 当人工归档目录使用时，单独写清规则，不要让它和程序默认输出目录混在 README 里。

### 6.3 文档建议

README 至少补三段：

1. “最小可用配置”  
2. “环境变量覆盖表”  
3. “目录说明与清理建议”  

这样第一次接手这个项目的人，才不会一上来就在 `config.example.yaml`、实际默认值、环境变量和历史目录之间反复猜。

## 7. 一句话结论

当前最大问题不是某个参数调得不够好，而是“配置真相分散在示例 YAML、代码默认值、环境变量和历史运行目录里”。先把这四处统一，后面的批量跑和上线稳定性才会明显提升。
