# Q5：38 维训练、验证与服务器检查

正式训练入口现支持双卡 `--gpus 2,3 --fsdp-devices 2`，全局 batch 必须能被卡数整除；原单卡 `--gpu` 用法保留。保存恢复的服务器命令、运行记录及验收标准见 [双卡训练与保存恢复](FSDP_TRAINING.md)。模型、冻结和学习率默认配置不变，完整模型 GPU 验收由服务器执行。

沿用 LIBERO 的 π0.5 LoRA 冻结规则，视觉编码器继续参与训练；冻结视觉编码器的对照后续再做。当前工作流在服务器执行完整模型加载、推理和梯度更新，本机进行数据处理和CPU回归。历史16GB本机容量尝试见下文，不属于已通过的本机训练配置。

Q5的GPU入口支持显式 `--memory-fraction`，默认0.75。覆盖范围为train、check_model、evaluate_validation和独立benchmark_fsdp；双卡时是对每卡总显存分别设置，并非两卡共享一份预算。该CLI值优先于启动环境中的两种fraction变量，初始化时统一设置，避免被写死常量覆盖。报告/训练清单记录实际fraction及预分配/allocator设置。本次没有调整是否预分配，LIBERO入口仍使用原0.75设置。

显存比例可显式覆盖，不传此参数时使用0.75预算。比例较低只缩小池预算，不降低模型本身需求；服务器双卡全局batch16在0.65下OOM、0.75下通过。恢复仍校验训练配置、数据和上游版本；本地代码指纹用于溯源，差异会打印并记录。新测试使用独立进程。

## 数据与模型

Q5入口仅过滤torchvision关于0.22→0.24视频API弃用的指定UserWarning，保留其他警告和错误；继续使用PyAV。该过滤不改变模型、训练策略或已记录的训练实现指纹。

| 集合 | 轨迹 | 帧数 | 用途 |
|---|---:|---:|---|
| train v2 | 50 | 17,382 | 梯度更新和归一化统计 |
| validation v2 | 5 | 1,817 | 调参和检查点选择 |
| test v2 | 10 | 3,490 | 最终评估，身份与 v1 完全相同 |

五条 validation 按采集顺序分层，从原 55 条训练候选中以 seed=43 抽取。原十条测试轨迹锁定。三份副本各自从零编号、物理隔离；旧数据和统计保留作历史记录。validation/test 都使用 train 统计。轨迹隔离不代表跨场景泛化，相邻采集仍可能相关。

state=(38,)，actions=(10,38)，顺序为左臂7、右臂7、左手12、右手12。目标为30Hz的绝对关节位置，不额外做delta转换。期望单位rad，依据既有HybridJointCommand文档；数据元数据没有单位字段，源bag及执行端仍需现场核实。

一路头部RGB缩放至224×224，缺失腕部图像填零并掩码。discrete_state_input=True，完整任务/状态token上限200。PyAV设置保留。训练统计目录为 assets/pi05_q5/q5_pick_orange_to_basket_v2_train，文件SHA-256为 cc2a1440c128003812bbb07994b1bcff270c86ad525283e624cecd24cc23e2e5。

## 权重及冻结

本地pi05_base的动作投影为32维，Q5为38维。当前选择两个完整动作投影层（含偏置）重新初始化，其他非LoRA权重要求存在且形状匹配，新增LoRA使用模型初始化。完整投影重建是选定的迁移策略，并非唯一方法。

直接调用与LIBERO相同的 model.get_freeze_filter()：LLM非LoRA参数（含动作专家）冻结；LoRA、视觉编码器、完整动作投影层、时间MLP参与训练。

CPU检查会审计真实基础检查点全部元数据路径。服务器训练初始化会生成 weight_loading_report.json；模型检查入口仅在指定 --output-dir 时保存该报告，逐参数记录来源、形状、dtype和可训练状态。只允许动作投影及新增LoRA重新初始化。CPU元数据检查不能替代完整权重加载。

## 更新部署

当前服务器已具备结构化pipeline、v2数据和训练统计，代码来自双卡更新包及警告过滤补丁。日常更新以当前Git代码为准；双卡入口和验收命令见[FSDP_TRAINING.md](FSDP_TRAINING.md)。`q5_dual_training_v1.zip`不包含后发的`q5_video_warning_filter_v1.zip`，旧包内文档也只是打包时的快照。

以下仅为尚无v2数据时的历史数据迁移流程；不要在已更新的服务器上用旧包覆盖当前代码。若需从旧环境迁移，应先导入数据，再部署当前结构化代码。

历史v1/PyAV/基线修复版导入v2数据的包为q5_validation_update_v1.zip：

```bash
cd /mnt/sda/xd/openpi_ws
unzip -o q5_validation_update_v1.zip
.venv/bin/python -B -m pipeline.q5.check_config
```

包内包含Q5代码、必要依赖/测试、新v2数据和统计，不包含基础权重、Python环境和训练检查点。旧LIBERO入口及v1数据保留。新目录：

```text
datasets/q5/pick_orange_to_basket_v2_train/
datasets/q5/pick_orange_to_basket_v2_validation/
datasets/q5/pick_orange_to_basket_v2_test/
datasets/q5/pick_orange_to_basket_v2_split.json
assets/pi05_q5/q5_pick_orange_to_basket_v2_train/
```

准备包可直接使用而不运行prepare_data。持有清理后的65条源数据且尚无v2产物时，可运行 python -B -m pipeline.q5.prepare_data；程序读取旧v1清单锁定测试身份，拒绝覆盖已有输出。--stats-only用于已划分、清单完整但尚无新统计时。运行时路径按工作区解析，JSON中旧绝对路径只用于溯源。

训练前核对各集合 `meta/episodes.jsonl` 的轨迹编号和长度与清单一致，并验证 Parquet、视频及时间记录指纹；`check_files=False` 仅跳过文件哈希比较，仍检查轨迹边界。新清单不生成 `metadata_sha256`，已有清单中的该字段直接忽略；无需修改旧 split 或重算统计。恢复训练仍要求实验记录中的配置、数据和上游版本一致。本地实现指纹差异不再直接阻断，而是记录到 `resume_config.json` 的 `implementation_changes`；原始 `config.json` 保持不变。配置相同不代表代码语义相同，涉及预处理或训练逻辑的修改仍需审查。

`cleaning_manifest.json` 是可选的自定义历史记录，不是 LeRobot 标准数据文件。有清单时校验并保留清理前轨迹编号；没有时仍可按源数据当前的 episode_index 完成 50/5/10 划分及训练统计，划分清单中 cleaning_history_available=false、original_episode_index=null。程序不会猜测此前删除过哪些轨迹，也不会重复删除 episode 0。已有但不完整或编号重复的清单会报错。

CPU检查：

```bash
.venv/bin/python -B -m unittest tests.test_q5_pipeline tests.test_pipeline_entrypoints -v
.venv/bin/python -B -m pipeline.q5.check_config
```

检查包括三集合隔离、train/validation真实样本、全状态token未截断、同一训练统计下的归一化、PyAV/图像掩码、无delta转换、数值/时间/QC记录及真实基础权重元数据冻结。默认只在终端显示结果；需要报告时加 --output checks/output/q5/check_config.json，另保存同目录 check_config_base_parameter_metadata.json。

## 数据加载与连续吞吐测试

第一阶段通过官方 DataLoader 测试并行加载，不修改 OpenPI 子模块。`pipeline.q5.benchmark_fsdp` 新增 `--num-workers` 和 `--timing-mode`；默认仍为 workers=0、diagnostic，保留已有容量测试的计时方式。正式训练入口已有 `--num-workers` 参数。

- `diagnostic`：每批数据和每次更新同步，分别报告数据准备（包括解码、变换和传输等待）及计算时间。数据时间占比不能当作正式训练的 GPU 空闲比例。
- `throughput`：编译与预热完成后连续执行测量窗口，仅在窗口首尾同步，不逐步读取 GPU 指标或打印进度，允许 CPU 加载、设备传输与 GPU 计算重叠。报告 `measurement_seconds`、`mean_step_seconds`、`total_samples_per_second`，每步 loss/grad 在计时结束后读取并校验。

两种模式分别报告 `timing_mode`、`num_workers` 及实际 DataLoader 的预取与持久 worker 设置。多进程加载沿用上游的 spawn、persistent_workers 和 PyTorch 默认预取；默认不另加预取队列。主机数据准备放在 CPU worker，模型仍在指定 GPU 上运行。

在服务器同一张 GPU、相同 batch/fraction 下，分别将下面的 `--num-workers` 替换为 0、2、4，再根据 CPU、RAM 和共享内存余量测试 8。使用较长测量窗口降低预取队列已填满对短测的影响；每组单独启动进程，比较 throughput 模式的结果。以下采用当前用户服务器已短测通过的单卡 batch16、fraction0.95：

```bash
./.venv/bin/python -B -u -m pipeline.q5.benchmark_fsdp \
  --gpus 0 --fsdp-devices 1 --memory-fraction 0.95 \
  --batch-size 16 --num-workers 2 \
  --timing-mode throughput --warmup-steps 5 --measure-steps 100
```

如需分段诊断，将 `--timing-mode` 改为 `diagnostic`。默认结果只在终端显示，显式 `--output <新JSON文件>` 才保存报告，不保存训练检查点。两种计时均排除初始化/编译和预热，连续吞吐还排除逐步日志与指标回传，不代表含检查点保存的整场训练耗时。模型、统计、采样和训练更新调用沿用原实现。

先验收真实服务器上的多进程吞吐，再决定是否增加图像缓存或调整解码后端。当前 PyAV 视频路径存在随机取帧时重复解码的成本；本机同批数据抽查 3 个视频的关键帧间隔为 250 帧，不能据此代替服务器文件核验或各阶段耗时分析。

2026-10-30 本地验证：16 项 CPU 回归通过，包括连续吞吐仅在窗口首尾同步、计时结束后读取指标、拒绝非有限指标和 CLI 参数校验。另用真实训练副本在 CPU 上分别运行 workers=0、2，各读取两批 batch=2，比较全部输出张量的 dtype、形状和字节哈希，结果一致；workers=2 的实际 prefetch_factor=2、persistent_workers=True。服务器 GPU 连续吞吐仍待执行上述命令验收。

## 服务器完整模型检查

等分配的GPU空闲后，检查完整权重、完整参数树及真实前向：

```bash
.venv/bin/python -B -u -m pipeline.q5.check_model --gpu 0
```

默认只在终端显示结果，不创建报告目录或文件。此入口会创建完整模型和优化器，不是轻量CPU检查，不保存检查点。三步真实更新也可直接运行：

```bash
.venv/bin/python -B -u -m pipeline.q5.check_model --gpu 0 --updates 3
```

需要保留报告时，显式指定 --output-dir；仅此时保存 weight_loading_report.json、model_check.json，以及检查期间的 INCOMPLETE 标记。指定的目录已存在则拒绝覆盖：

```bash
.venv/bin/python -B -u -m pipeline.q5.check_model \
  --gpu 0 --updates 3 --output-dir checks/output/q5/model_updates_v2
```

三步检查用于观察连续真实更新。当前官方余弦调度初始学习率为peak_lr/(warmup_steps+1)，不是零；至少三步是检查入口的要求。检查loss/梯度有限，每个参数张量最多分布抽查32个元素，确认可训练组件更新、冻结参数抽样值不变；该抽样不等于全张量逐值证明。完整参数清单另由参数树报告提供。保存恢复另用smoke验收。

三步检查器的训练JIT必须与官方trainer保持一致，包括 donate_argnums=(1,)、输入/输出sharding和mesh上下文。donation允许新训练状态复用已不再使用的旧状态显存。初版检查器漏了这一设置，在本机16GB卡首次更新时出现输入/输出内存超限和OOM，不能用那次结果直接判定正式训练的最低显存。当前已对齐官方设置并释放单独前向检查的编译缓存；CPU小型回归验证了旧状态buffer被复用，服务器完整模型三步更新也已通过。

后续本机16GB重测仍在真实更新阶段OOM；完整加载和前向通过。修复保证检查器与官方内存处理一致，不保证16GB能容纳当前训练配置。用户提供的24GB服务器结果已验证更新、单卡保存恢复，以及双卡FSDP的100→120步保存恢复。

指定 --output-dir 后，失败输出目录保留 INCOMPLETE 记录。保留报告重测时请换新目录，例如 --output-dir checks/output/q5/model_updates_v2_retry1；不保存报告时可直接重新执行。早期q5_check_model_memory_fix_v1.zip是donation修复的历史包，当前代码已包含修复，不应再用旧包覆盖新入口。

## 学习率和样本预算

| 参数 | smoke | finetune默认参考 |
|---|---:|---:|
| 首段上限 | 100 | 1000 |
| warmup | 10 | 1000 |
| schedule | 1000 | 30000 |
| peak/end LR | 5e-5/5e-6 | 5e-5/5e-6 |
| 保存/长期保留 | 50/50 | 1000/5000 |

AdamW、梯度裁剪1.0、无EMA/W&B/梯度累积、seed42。batch1/workers0仅为服务器试跑起点。schedule=30000不是承诺训练30000步。

finetune默认首段全在预热，入口要求显式 --continue-after-segment，否则报错。--steps是累计执行上限，至少为2（上游一步训练最终保存的标签0不满足恢复要求）；--schedule-steps是调度长度，--warmup-steps可显式指定。启动时打印窗口样本预算及相对训练起点比例。1000步batch1约处理1000个窗口样本，相对17,382个起点约5.75%，不等于独立起点覆盖率，不适合判断最终效果。

以下是保留的单卡用法示例，不代表当前双卡验收配置；已执行的batch16双卡命令与结果见[FSDP_TRAINING.md](FSDP_TRAINING.md)。正式总预算和调度尚未选定，下面finetune命令只是默认配置的用法。

单卡短跑保存示例：

```bash
mkdir -p logs
nohup .venv/bin/python -B -u -m pipeline.q5.train \
  --gpu 0 --profile smoke --exp-name q5_v2_smoke_01 \
  --batch-size 1 --num-workers 0 --steps 100 \
  > logs/q5_v2_smoke_01.log 2>&1 < /dev/null &
```

同一smoke配置加 --steps 120 --resume 验证恢复。正式配置另起实验；若明确首段后继续同一计划：

```bash
.venv/bin/python -B -u -m pipeline.q5.train \
  --gpu 0 --profile finetune --exp-name q5_v2_lora_01 \
  --batch-size 1 --num-workers 0 --steps 1000 --continue-after-segment
```

如决定总共只训练1000步，显式选择较短预热和匹配长度，例如 --warmup-steps 100 --schedule-steps 1000 --steps 1000。该选项不会自动替换与LIBERO对齐的默认计划。后续总预算结合服务器吞吐、实际batch和验证结果确定。

恢复需保持模型、数据、统计、调度、全局batch、卡数、分片数量和上游版本一致；不能从旧55/10或不同冻结策略直接恢复为v2，使用新实验名。恢复前检查已提交检查点的步数及优化器计数器；训练结束校验最终保存并写入运行目录的 training_result.json。

上游 resume 会重建数据迭代器；固定 batch 的保存恢复测试验证参数和优化器状态接续，不保证真实数据训练与不中断运行逐步等同。

## 验证集比较

已有v2服务器检查点后：

```bash
.venv/bin/python -B -u -m pipeline.q5.evaluate_validation \
  --gpu 0 --exp-name q5_v2_lora_01 --step 999
```

仅从五条validation中各等间距抽最多20个起点，固定种子和noise，使用训练统计，按轨迹等权报告双臂/双手的目标MAE与首动作MAE（rad），排除窗口末尾补齐帧。默认只在终端显示结果；需要保存时加 --output checks/output/q5/validation/q5_v2_lora_01/999.json。未指定输出时，已有旧报告不会阻止重复评估。相同采样规则用于不同检查点；关注右臂/右手，避免左手常零稀释误差。它是离线误差，不是闭环成功率；test不用于选择检查点。

## 验证边界

本机已完成v2数据处理、CPU数据/元数据检查和单元测试。用户提供的服务器结果已验证单卡完整加载/前向/更新、单卡smoke保存恢复与验证预测，以及双卡benchmark。2026-10-29，双卡正式入口在物理卡1、2、全局batch16、fraction0.75下完成100→120步保存恢复，训练步数及Adam/调度计数器均按100→120接续；119检查点单卡验证推理通过，五条validation共100个起点，未读取test。右臂/右手窗口MAE分别为0.4032109/0.4762642rad；这是短跑离线误差，不能当作机器人闭环成功率。

原始bag和实时机器人不可达。图像、反馈和指令的独立时序及单位、执行端关节映射和绝对目标语义仍待核实。转换旁路最大状态年龄约8.86ms、指令年龄约20.55ms，不能替代原bag重验，也不应据此移动标签。
