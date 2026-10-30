# Q5 双卡训练与检查点验收

`pipeline.q5.train` 现在支持单卡或双卡，继续调用固定版本的官方 `scripts/train.py`。模型、冻结规则、AdamW、学习率调度、训练数据和归一化统计均不变。没有修改 OpenPI 源码。

## 参数和记录

- 原来的 `--gpu 0` 单卡用法保留，默认 `fsdp_devices=1`。
- 双卡显式指定 `--gpus 2,3 --fsdp-devices 2`。`--gpu` 与 `--gpus` 互斥，支持物理 index 或 UUID。只将指定 GPU 暴露给当前进程；JAX 内部会重新编号为 0、1。
- `--batch-size` 为全局 batch。双卡 batch 16 时，每卡处理 8 个窗口。
- 使用官方二维 mesh `(batch=1, fsdp=2)` 和 `fsdp_sharding`。达到大小阈值且维度可整分的数组分片，小数组复制；不会使所有显存占用都恰好减半。
- `--memory-fraction` 默认 0.75，是每卡总显存的池预算。保留原来的预分配设置；不代表进程显存硬上限或共享 GPU 的资源隔离。
- 启动终端与 `runs/pi05_q5/<实验名>/config.json` 记录 GPU 选择、设备数、mesh、全局/每卡 batch、分片数、allocator 设置和代码指纹。

## 更新与 CPU 检查

先在没有该项目训练任务运行时，从工作区根目录解压更新包：

```bash
cd /mnt/sda/xd/openpi_ws
unzip -o q5_dual_training_v1.zip
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m pipeline.q5.check_config
```

该包包含打包时的Q5代码及依赖、CPU回归测试与说明，不包含OpenPI源码、环境、数据、权重或检查点。后发的torchvision警告过滤需另装q5_video_warning_filter_v1.zip，或使用当前Git代码。前提是已经安装结构化pipeline和v2的50/5/10数据与训练统计。不会清理服务器已有文件。

本次历史验收使用新实验名，保存和恢复之间代码未变。当前入口将本地代码指纹差异打印并记录到 resume_config.json，不再仅凭文件差异拒绝恢复；配置、数据和上游版本仍必须一致，代码语义变化需单独审查。

## 服务器双卡保存与恢复

以下使用2026-10-29服务器验收的物理卡1、2、全局 batch 16、fraction 0.75配置。再次执行前确认1、2号卡已经分配给你；这里不会自动抢卡或停止其他进程。单卡的参数和训练策略保持原样。

以下命令保留实际验收的实验名用于对照；该实验已达到120步，不能原样再次运行首次100步或恢复至120步命令。新一轮验收应统一替换实验名和日志名，已完成实验若继续则累计步数必须大于120。

第一次跑 100 步（历史验收命令）：

```bash
cd /mnt/sda/xd/openpi_ws
mkdir -p logs
nohup env PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -u -m pipeline.q5.train \
  --gpus 1,2 --fsdp-devices 2 \
  --profile smoke --exp-name q5_v2_fsdp2_b16_smoke_01 \
  --batch-size 16 --num-workers 0 --memory-fraction 0.75 --steps 100 \
  > logs/q5_v2_fsdp2_b16_smoke_01.log 2>&1 < /dev/null &
```

查看日志：

```bash
tail -n 80 -f logs/q5_v2_fsdp2_b16_smoke_01.log
```

必须等进程完成、最后输出 `status: PASS` 后，才启动恢复。`Ctrl+C` 只停止前台的 tail，不停止后台训练。

同一实验恢复到累计 120 步：

```bash
nohup env PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -u -m pipeline.q5.train \
  --gpus 1,2 --fsdp-devices 2 \
  --profile smoke --exp-name q5_v2_fsdp2_b16_smoke_01 \
  --batch-size 16 --num-workers 0 --memory-fraction 0.75 --steps 120 --resume \
  > logs/q5_v2_fsdp2_b16_smoke_01_resume120.log 2>&1 < /dev/null &
```

`--steps 120` 是累计更新上限，只增加 20 步，不是额外跑 120 步。保存目录沿用官方的从零计数标签：100 步完成时目录为 `99`，其中 `state.step=100`；恢复后的最终目录为 `119`，其中 `state.step=120`。

smoke 的 warmup=10、schedule=1000 在两次命令间保持一致，恢复不重置学习率。此处只验收完整训练链路；正式训练总步数和预热安排需要另外确定。

## 如何确认恢复正确

入口在正式恢复前，使用 Orbax 只读查找已提交的最新检查点，只读取训练步数与优化器计数器，不把完整模型或动量再次加载一份。找不到有效检查点、统计哈希不符、计数器不一致或 `--steps` 没有超过已保存步数时直接失败，避免官方入口退回重新初始化。

恢复还要求模型、冻结规则、学习率、优化器、全局 batch、seed、GPU 数量、分片数量、数据、统计与上游版本一致；本地代码指纹差异仅记录。可以换到另外两张已分配的物理卡，但不能在这个验收实验中改变卡数或训练策略。

两次完成后检查 `runs/pi05_q5/q5_v2_fsdp2_b16_smoke_01/training_result.json`：

| 字段 | 首次完成 | 恢复完成 |
|---|---|---|
| `status` | PASS | PASS |
| `parallelism.device_count` / `fsdp_devices` | 2 / 2 | 2 / 2 |
| `parallelism.global_batch` / `per_device_batch` | 16 / 8 | 16 / 8 |
| `resume_checkpoint.state_step` | 无 | 100 |
| `final_checkpoint.checkpoint_label` | 99 | 119 |
| `final_checkpoint.state_step` | 100 | 120 |
| `final_checkpoint.optimizer_counts` 的各计数器 | 均为 100 | 均为 120 |

原始启动配置保留在 `config.json`，恢复请求另记 `resume_config.json`。学习率由保存的 optimizer schedule counter 接续，报告的 `next_learning_rate` 表示下一次更新将使用的值。完成结果会在训练器等待异步保存结束、并校验最终检查点之后写入；失败时不要把此前已有的完成报告当作当前请求成功。

官方检查点恢复参数、优化器状态和训练步数，但没有持久化数据迭代器的取样进度；真实数据续跑不保证逐样本重现连续运行顺序。CPU 的固定 batch 回归只验证训练状态与保存恢复逻辑。

## 从双卡检查点做单卡验证推理

恢复验收完成后，用一张已分配的卡加载 `119/params` 做验证集推理：

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -u -m pipeline.q5.evaluate_validation \
  --gpu 1 --exp-name q5_v2_fsdp2_b16_smoke_01 --step 119 --memory-fraction 0.75
```

评估入口保持单卡，仅使用五条 validation，每条最多 20 个起点；默认只打印终端，不写评估报告。它加载推理参数，不恢复训练优化器。离线 MAE 不等于机器人闭环成功率，test 仍保留。

## 当前验证边界

本地仅用两个模拟 CPU 设备、小模型及官方训练器执行保存恢复回归；比较全部参数、Adam 动量和计数器，检查 FSDP 布局与单设备推理权重恢复。

2026-10-29用户提供的服务器日志已完成真实Q5验收：实验`q5_v2_fsdp2_b16_smoke_01`，物理GPU1、2，FSDP=2，全局batch16，每卡8，fraction0.75。首次最终检查点99于04:41:43完成保存，state.step及两个optimizer计数器均为100；恢复加载99后继续更新，119于04:47:59完成保存，最终检查均为120并输出PASS。恢复前后的下一步学习率分别为4.908859e-5、4.864308e-5。99按保留策略被删除，50和119保留。

119的单卡验证推理PASS：seed123、五条validation各20个起点、test_observations_read=0。右臂/右手窗口MAE为0.4032109135/0.4762641740rad，首动作MAE为0.3667975898/0.4780299199rad。归一化SHA保持cc2a1440c128003812bbb07994b1bcff270c86ad525283e624cecd24cc23e2e5。这些结果验证短跑链路和离线误差，不代表长期稳定性或闭环抓取成功率。
