# 从 pi05_base 训练自己的 pi05_libero

目标：学习统计量生成、LoRA 微调、检查点保存及恢复，随后接入 LIBERO 仿真评估。

已有记录中，batch 1、2、4、8、16 通过短测，32 报 OOM。正式训练采用实际 batch 16，命令需显式传 `--batch-size 16`；首段训练与阶段评估记录、恢复模板和预算见 [FORMAL_TRAINING.md](docs/FORMAL_TRAINING.md)，容量实测见 [BATCH_BENCHMARK.md](docs/BATCH_BENCHMARK.md)。尚未实现梯度累积，模型效果须以对应检查点的评估为准。

早期最小闭环已完成全量统计、100 步 LoRA 试跑、保存、恢复到累计 120 步、真实检查点推理，以及 spatial 十个任务的闭环仿真，该阶段成功率为 0/10。该阶段验收见 [SMOKE_ACCEPTANCE.md](docs/SMOKE_ACCEPTANCE.md)，操作见 [INFERENCE_EVAL.md](docs/INFERENCE_EVAL.md)，历史验证记录位于 [docs/validation](docs/validation/SERVER_VALIDATION.json)。后续正式阶段结果单独记录，不用早期试跑结果代表当前检查点表现。

本管道以官方 OpenPI 的 pi05_libero 配置为基础，固定版本为 215abfb217dbac7d5f1273282331b9b1866c0479。统计量和训练均调用官方入口；单张 RTX 3090 使用 LoRA 配置。显存是否足够仍需服务器实测，也不承诺复现官方全参数训练的分数。

本机 WSL 另外兼容已审查的本地提交 `13a95ea16adeee33b7efd49f8f3dadc6aebe20b9`，其中模型、策略、训练和评估主代码与上述基线一致。保存本次部署配置后的提交 `eacc6a831df78bd8de7ef0804f48ef32f6de0052` 也已加入明确兼容列表，该提交仅修改 Compose 和查看器文档。未知提交和本地版本中未审查的核心代码修改会被拒绝；运行清单分别记录基线与实际提交。本机非训练运行验证见 [RUNTIME_VALIDATION.md](docs/RUNTIME_VALIDATION.md)。

## 文件与目录

诊断、推理检查与评估默认在终端显示结果，只有显式指定输出路径才持久保存报告：Q5 check_config/evaluate_validation、LIBERO check_policy 和两种策略服务使用 --output <JSON文件>；check_model、check_simulation、eval_libero、check_policy_transport 和 benchmark_batch 使用 --output-dir <新目录>。仿真图片、评估视频、batch 子进程日志也只在指定目录时保留。通信测试和 batch 测试需要的中间文件使用系统临时目录并在退出时清理，默认仍会打印最终指标及失败日志摘要。

数据集、归一化统计及其来源记录、训练/恢复配置和检查点是后续程序使用的产物，仍按原流程保存。local_docker_serve 重建的 config.json 供客户端核验服务身份使用，也仍保留；可选的是额外服务报告。

将完整 pipeline 文件夹放在工作区下，与 openpi、datasets 和 openpi_cache 并列。所有 Python 命令在工作区根目录使用 `python -m pipeline.…` 执行，选择训练或仿真对应的解释器。公共配置从包所在位置解析工作区，不依赖当前终端路径，也不依赖 Windows/WSL/服务器的固定用户名。

| 目录 | 用途 | 主要入口 |
|---|---|---|
| `train/` | 训练、恢复与 batch 容量测试 | `pipeline.train.run`、`pipeline.train.benchmark_batch` |
| `inference/` | 策略服务、客户端通信与推理检查 | `pipeline.inference.serve_policy`、`pipeline.inference.local_docker_serve`、`pipeline.inference.check_policy` |
| `eval/` | LIBERO 仿真检查与评估 | `pipeline.eval.check_simulation`、`pipeline.eval.eval_libero` |
| `data/` | 归一化统计与训练输入检查 | `pipeline.data.compute_norm_stats`、`pipeline.data.check_config` |
| `common/` | 共享配置、工作区路径与参数校验 | `config.py`、`validation.py` |
| `ops/` | 独立环境安装、依赖约束与部署说明 | `setup_libero_env.sh`、[部署说明](ops/README.md) |
| `docs/` | 操作说明、历史验证与归档 | [重构与迁移](docs/REFACTOR.md)、`validation/`、`archive/` |

`data/` 只存处理代码；数据集、资产、检查点和运行产物仍放在工作区原目录。现有运行检查保留在 `checks/`，自动化回归测试位于工作区 `tests/`。

旧的 `python pipeline/train.py …` 等调用已迁移；入口对照、回归测试方法和 WSL 当前 Compose 的说明见 [REFACTOR.md](docs/REFACTOR.md)。文档中的历史训练结果仍以原验证记录为准，本次重构不更新模型效果结论。

预期结构：

~~~text
openpi_ws/
  openpi/                 # 固定版本官方仓库，已有 .venv
  pipeline/               # 本目录全部文件
  datasets/libero/        # 已验收的数据
  openpi_cache/           # pi05_base 权重与 tokenizer
  assets/pi05_libero/libero/
  checkpoints/pi05_libero/<实验名>/<步骤>/
  runs/pi05_libero/<实验名>/
  logs/
~~~

## 与官方配置的关系

| 项目 | 官方 pi05_libero | 当前管道 |
|---|---|---|
| 初始权重 | pi05_base | 相同，读取本地 params |
| 动作窗口/内部维度 | 10 / 32 | 相同 |
| 状态输入/额外 delta 转换 | discrete_state_input=False / False | 相同 |
| 参数更新 | 全参数微调 | 两个 Gemma 模块启用 LoRA，配套冻结规则 |
| 训练批大小 | 256 | CLI 默认 1，正式命令显式使用 16；容量实测见对应记录 |
| EMA | 0.999 | 关闭 |
| 训练步数 | 30000 | 正式 30000，试跑 100 |
| 学习率 | 峰值 5e-5，预热 10000，末值 5e-5 | 峰值 5e-5，末值 5e-6；试跑预热 10，正式预热 1000 |
| 保存间隔 | 1000 | 试跑 50，正式 1000 |
| DataLoader 工作进程 | 2 | 默认 0，可通过 --num-workers 设置 |
| 统计批大小 | 256 | 32，沿用官方聚合和末批丢弃行为 |
| 数据集标识 | physical-intelligence/libero | libero，仅作为本地路径别名 |

本地别名 libero 配合 HF_LEROBOT_HOME=<工作区>/datasets，直接读取已有 datasets/libero，便于复用 Arrow 缓存。数据来源仍记录为 physical-intelligence/libero；服务器下载镜像为 buxiangqimingzi/physical-intelligence-libero。脚本不修改官方仓库源文件；配置注册只在当前 Python 进程生效，且依赖上面固定版本。

脚本使用本地权重、tokenizer 和离线数据，不永久修改终端环境。WSL 可通过 --cache-dir /home/xiaodan/.cache/huggingface/datasets 复用已有缓存；服务器默认复用工作区 .cache/huggingface/datasets。

## 多进程读取与速度测试

统计、配置检查和训练脚本都支持 --num-workers N。服务器先测试 2 个工作进程，观察内存与速度后再尝试 4；不要按 CPU 核数直接设置 128。工作进程数控制读取并行度，各计算库线程仍限制为 2，避免每个进程各自创建大量线程。

已运行的统计任务不会因为上传新脚本而改变配置。全量统计没有断点恢复，重启将从头计算。先保留现有任务，可另运行下面的小规模探测，验证服务器上的多进程是否正常并查看速度：

~~~bash
cd /mnt/sda/xd/openpi_ws
PYTHONDONTWRITEBYTECODE=1 openpi/.venv/bin/python -m pipeline.data.compute_norm_stats \
  --num-workers 2 --max-frames 1024
~~~

探测输出与正式统计目录隔离，不影响正在运行的全量任务。探测沿用官方随机抽样逻辑，且包含子进程启动开销，不能直接按总耗时推算全量顺序遍历的耗时；同时运行也会竞争 CPU、磁盘及内存。若系统资源紧张，等当前任务完成后再测试。

确认并行效果后，未来全量任务可加 --num-workers 4；正式数据检查也可先用相同参数验证，再用于训练。改变工作进程数不会改变完整统计任务的样本顺序、批大小和聚合算法。

新增本地验证：在真实 episode 0 上对比 0 和 2 个工作进程，各读取 128 帧；state/actions 的 mean、std、q01、q99 逐项完全一致。全量 Dataset 多进程测试在 WSL 意外退出，未报告 Python 异常，未通过；服务器全量数据下的内存及速度仍待验证。

## 第一步：服务器生成全量官方统计量

上传整个 pipeline 文件夹后，在服务器执行：

~~~bash
cd /mnt/sda/xd/openpi_ws
mkdir -p logs
nohup env PYTHONDONTWRITEBYTECODE=1 openpi/.venv/bin/python -m pipeline.data.compute_norm_stats \
  >> logs/official_norm_stats.log 2>&1 < /dev/null &
~~~

查看日志：

~~~bash
tail -c 4000 /mnt/sda/xd/openpi_ws/logs/official_norm_stats.log | tr '\r' '\n' | tail -n 25
~~~

同一时间只运行一个统计任务。不传 --max-frames 即使用完整数据集。官方入口会丢弃不完整末批：默认批大小 32，处理 273440/273465 帧，丢弃 25 帧，对应 2734400 个动作向量。

输出位于 assets/pi05_libero/libero/：

| 产物 | 内容 |
|---|---|
| norm_stats.json | 官方入口生成的正式统计量 |
| pipeline_config.json | 来源配置、路径、模型及显式调整 |
| statistics_run.json | 完成状态、处理帧数、批大小及统计量 SHA256 |

前一阶段 checks/output/pi05_libero_h10 中的候选统计量保留用于对照。它覆盖所有帧、按 episode 聚合，与官方入口的批次遍历方式不同，哈希不要求一致。

可选的小规模软件测试命令如下；WSL 已通过此测试，服务器可直接进行全量统计：

~~~bash
openpi/.venv/bin/python -m pipeline.data.compute_norm_stats --max-frames 128
~~~

探测统计量单独写入 checks/output/official_norm_smoke/assets/pi05_libero/libero/，不会用于正式训练。

## 第二步：检查正式配置的数据输入

全量统计完成并出现 PASS 后执行：

~~~bash
cd /mnt/sda/xd/openpi_ws
PYTHONDONTWRITEBYTECODE=1 openpi/.venv/bin/python -m pipeline.data.check_config
~~~

预期 PASS、两个真实 CPU 批次：state 为 (1,32)，actions 为 (1,10,32)，tokens 为 (1,200)，各摄像头为 (1,224,224,3)。

--probe-stats 仅供测试探测统计量的软件路径。带该参数通过，不能作为正式统计量已就绪的证据。

## 第三步：GPU 试跑

先确认分配给你的空闲 GPU，将 GPU_ID 替换成对应编号或完整 GPU UUID。完成前两步验收后再执行：

~~~bash
nohup env PYTHONDONTWRITEBYTECODE=1 openpi/.venv/bin/python -m pipeline.train.run \
  --gpu GPU_ID --exp-name libero_smoke_01 --profile smoke \
  >> logs/libero_smoke_01.log 2>&1 < /dev/null &
~~~

默认试跑 100 次更新。脚本只暴露指定的一张 GPU，不回退到 CPU，不覆盖已有实验。Wandb 外部日志关闭。实际权重加载、显存、前向/反向计算和检查点保存需在服务器验收。

官方训练循环的检查点目录使用从零开始的循环编号；100 次更新通常保存在 99 目录中。选择检查点时以实际目录和日志为准。

## 第四步：恢复训练、验证推理评估与正式训练

在试跑完成后，验证从同一实验恢复：

~~~bash
openpi/.venv/bin/python -m pipeline.train.run \
  --gpu GPU_ID --exp-name libero_smoke_01 --profile smoke --steps 120 --resume
~~~

--steps 指累计更新上限，不是额外增加的步数。恢复前会核对原实验记录中的模型、学习率、统计量等设置。

恢复验收通过后，先按照 INFERENCE_EVAL.md 接通检查点策略服务与 LIBERO 评估。batch 与加载测试现已完成，正式首段选定 batch 16、num_workers 2、1000 次更新。详细计划见 FORMAL_TRAINING.md；当前尚未实现梯度累积：

~~~bash
nohup env PYTHONDONTWRITEBYTECODE=1 openpi/.venv/bin/python -m pipeline.train.run \
  --gpu GPU_ID --exp-name libero_lora_01 --profile finetune \
  --batch-size 16 --num-workers 2 --steps 1000 \
  >> logs/libero_lora_01.log 2>&1 < /dev/null &
~~~

正式 profile 的学习率计划仍为 30000 步；这里先运行 1000 次更新验收持续训练与保存，再按 FORMAL_TRAINING.md 恢复到暂定累计 30000 步。效果由后续评估确认。

## 当前验证范围与后续验收

WSL 已通过官方统计入口 128 帧探测，以及使用探测统计量的真实 LoRA 配置 DataLoader 两个批次，详见 docs/validation/WSL_VALIDATION.json。服务器已通过全量正式统计量、LoRA 配置真实批次、单张 3090 的 100 步更新与保存、恢复到累计 120 步、九观测真实推理及 spatial 十任务仿真。详见 docs/validation/SERVER_VALIDATION.json；本轮任务成功率 0/10，效果尚未通过验收。

服务器按顺序验收：

1. 全量官方统计量完成并报告 PASS。
2. check_config.py 不带 --probe-stats 通过。
3. GPU 试跑出现有限的 loss/grad_norm，完成更新并保存检查点。
4. --resume 恢复训练状态并继续推进。
5. 使用试跑检查点接通策略服务和 LIBERO 仿真，再正式微调并评估正式检查点。

策略服务和评估启动文件已经提供，操作见 INFERENCE_EVAL.md。加载当前检查点时必须使用相同的 LoRA 模型结构及对应统计量。
