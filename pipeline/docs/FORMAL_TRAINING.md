# 正式 LoRA 训练：batch 16，先跑 1000 步

实测 batch 1、2、4、8、16 均通过短程真实训练，batch 32 报 OOM。选择已测试的实际 batch 16 开始正式训练。当前不使用梯度累积。

正式训练首段已完成，检查点 999 保存成功。根据用户下载的 libero_lora_01.log，累计 1000 次更新，实际 batch 16，日志指标均为有限值；loss 从 step 0 的 0.0915 到 step 900 记录的 0.0342。日志开始至最终保存完成约 1 小时 38 分钟，稳定更新约 5.7 秒/步。本段已有初步评估结果，后续恢复训练仍待启动。

首段随后已完成 spatial 评估：十个任务各一次、seed 7，成功 2/10，无运行异常，耗时约 15 分 43 秒。成功任务是 cookie box 旁、plate 旁的黑碗搬至盘子。视频抽样出现抓取与搬运行为，部分失败任务仍未达到最终成功条件。与 119 的 0/10 相比为初步正向变化，但两实验的 batch、学习率计划、样本预算及策略采样状态不同，不能作单因素或稳定成功率结论。关闭服务状态尚待用户确认，再继续安排恢复训练。

此前已补测 batch 16、num_workers=2，预热三步、计时二十步通过，纯训练/含加载吞吐约 3.00/2.98 样本/秒；全机可用 RAM 从 94.6 GiB 降到采样最低 72.3 GiB。正式配置使用两个加载进程。与此前十步的 0 进程测试总吞吐相近，不认定为明确加速或全局最优。

batch 16 含数据加载吞吐约 3.04 个观测样本/秒，纯训练约 3.57 个/秒。这是十个计时步骤的同步短测结果；长程稳定性、保存时资源峰值和任务效果仍需验证。nvidia-smi 的约 22 GiB 数字包含 JAX 预留池，不应当作实际训练缓冲区峰值。

## 正式配置

| 项目 | 设置 |
|---|---|
| 初始权重 | 本地 pi05_base，新实验从头微调 |
| 实验名 | libero_lora_01 |
| 模型 | pi05，两个 Gemma 模块的 LoRA 配置，horizon 10 |
| 实际/有效 batch | 16 / 16，单 GPU，无梯度累积 |
| 总预算 | 暂定累计 30000 次优化器更新 |
| 第一段 | 1000 次更新，用于检查训练指标和正式 batch 的保存 |
| 学习率 | 始终使用 30000 步的余弦计划，预热 1000，峰值 5e-5，末值 5e-6 |
| EMA / workers | 关闭 / 2 |
| 指标 / 保存间隔 | 每 100 步 / 每 1000 步及本段结束 |
| 保留规则 | 官方 max_to_keep=1，另保留满足 keep_period=5000 的检查点 |
| 数据统计量 | 使用已验收的正式统计量，SHA256 7e19abf21b2f8837e1e27c468f4021efe359e38a9f6158acb83b6f978dad6b5a |

现有 `pipeline.train.run` 支持所有这些设置，无需修改运行代码或重新安装环境。每条正式启动/恢复命令都显式传 --batch-size 16，避免当前命令行默认值 1 覆盖它。

30000×16=480000 个训练中心观测样本，约为本数据集帧数的 1.76 倍。这与官方全参数 batch 256×30000 的样本预算不同，不能据此承诺官方成绩。

## 1. 启动正式训练第一段

所有命令在 openpi_ws 根目录执行。GPU 0 仍空闲且可用时，沿用已测试的 UUID；如已分配其他卡，替换 --gpu。确保没有另一个同名训练实验正在运行。第一次使用新实验名，不加 --resume。

~~~bash
mkdir -p logs
nohup env PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -u -m pipeline.train.run \
  --gpu GPU-cca35824-7a74-52fc-77f8-e9a254b966f6 \
  --exp-name libero_lora_01 --profile finetune \
  --batch-size 16 --num-workers 2 --steps 1000 \
  >> logs/libero_lora_01.log 2>&1 < /dev/null &
~~~

~~~bash
tail -n 60 -f logs/libero_lora_01.log
~~~

阶段目标：指标为有限值，参数继续更新，1000 次更新正常结束，最终完成检查点 999 的写入。先发启动与前几个指标记录，结束后再发保存结果。仅凭训练 loss 下降不能验收任务成功率。

按短测含加载吞吐，第一段约 1.5 小时，完整 30000 步约 44 小时，均为粗估；CPU/GPU 流水重叠、编译、存盘及共享服务器负载会改变实际耗时。正式运行数百步后再校准。

产物：

    checkpoints/pi05_libero/libero_lora_01/999/
    runs/pi05_libero/libero_lora_01/config.json
    logs/libero_lora_01.log

## 2. 第一段验收后恢复

以下为后续恢复模板，先不要与第一段同时运行：

~~~bash
nohup env PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -u -m pipeline.train.run \
  --gpu GPU-cca35824-7a74-52fc-77f8-e9a254b966f6 \
  --exp-name libero_lora_01 --profile finetune \
  --batch-size 16 --num-workers 2 --steps 30000 --resume \
  >> logs/libero_lora_01_resume.log 2>&1 < /dev/null &
~~~

--steps 是累计优化器更新上限。第一段 --steps 1000 只限制执行长度，不会把学习率衰减计划缩为 1000 步；恢复后继续同一计划。

检查点目录按从零开始的循环编号命名：累计 1000 次更新通常是 999，累计 30000 次通常是 29999，以实际日志为准。根据保留规则，恢复后保存新检查点可能删除未满足长期保留周期的旧检查点；评估前核对目标目录是否仍存在。

## 3. 模型效果评估

将策略服务和评估程序的 --exp-name 改为 libero_lora_01，并指定实际存在的正式检查点。优先沿用 smoke 的 spatial、seed 7、每任务一次做同条件初步对照，记录动作变化和成功率。再决定是否调整训练预算、有效 batch、学习率或增加评估次数。

不要仅因 batch 16 能运行，就把它当成已经验证效果的最优参数。正式训练的任务表现仍需评估。
