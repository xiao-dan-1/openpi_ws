# 单卡 LoRA 的 batch 容量和吞吐测试

正式训练参数尚待确定。batch 1 已在服务器跑通过流程，但不能据此认定它适合正式训练。此工具比较实际 batch 1、2、4、8，当前不实现梯度累积。

当前脚本版本 2。版本 1 的服务器测试在显式 lower().compile() 时触发 TrainState/ArgInfo 类型检查错误，没有得到 batch 容量结论。版本 2 改用官方训练已采用的普通 JIT 调用，WSL 已用真实 TrainState 复现旧错误并验证修复路径。

当前更新为版本 3：支持 --num-workers，新增系统 MemAvailable 最低值记录；候选进程使用独立会话，终止时同时清理其 DataLoader 子进程。版本 2 的 GPU 容量测试结果仍作为 workers=0 的历史基线。

## 测试内容

每个 batch 单独启动一个 Python 子进程，从本地 pi05_base 初始化，使用正式统计量、真实 LeRobot 数据和 finetune LoRA 配置。直接调用固定版本官方 scripts/train.py 的 init_train_state 与 train_step，使用同样的单卡 sharding/JIT。没有检查点保存，也不创建训练实验目录。

每个 batch：
1. 加载模型和数据，第一次真实更新触发 JIT 编译，单独记录编译加首步更新总耗时。
2. 共执行 3 次真实梯度更新用于预热（包含首步更新）。
3. 执行 20 次真实梯度更新用于计时，检查 loss、梯度等指标有限且测得非零梯度。
4. 进程退出并释放显存，再开始下一个 batch。

所有候选沿用相同的单卡、两路图像、horizon=10、LoRA 架构、正式统计量和学习率配置，DataLoader workers=0。它用于选择实际 batch，不判断模型任务效果。

版本 3 可显式设置 --num-workers，默认仍为 0。固定 batch 16，对比 workers=2 与此前 workers=0 的约 3.04 样本/秒含加载吞吐。若差异很小，需在相近系统负载下用相同步数重测 0/2 才能判断，不把一次短测当作精确性能排名。

~~~bash
nohup env PYTHONDONTWRITEBYTECODE=1 \
  openpi/.venv/bin/python -u -m pipeline.train.benchmark_batch \
  --gpu GPU_ID --batches 16 --num-workers 2 \
  --warmup-steps 3 --measure-steps 10 \
  >> logs/benchmark_workers2.log 2>&1 < /dev/null &
~~~

同时观察 batch_16.json 中 system_available_before_mib 与 system_available_min_mib，了解共享系统的内存变化；差值也会包含模型和其他进程影响，不能全部归因于 DataLoader。先验证两个 worker，再考虑四个。

遇到 OOM 或任一候选失败时，停止更大的候选，保留已经通过的结果。选中显卡已有计算进程或较高显存占用时拒绝启动；测试期间发现另一计算进程使用同卡时，该候选标记 INVALID 并停止。脚本不会终止其他人的进程。

## 服务器执行

部署完整 pipeline 目录，容量测试入口为 `pipeline.train.benchmark_batch`。在 openpi_ws 根目录执行；先确保只运行一次测试，选择一张分配给你的空闲卡：

~~~bash
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
~~~

把 GPU_ID 替换为显卡编号或 UUID：

~~~bash
nohup env PYTHONDONTWRITEBYTECODE=1 \
  openpi/.venv/bin/python -u -m pipeline.train.benchmark_batch \
  --gpu GPU_ID --batches 1 2 4 8 \
  --warmup-steps 3 --measure-steps 20 \
  >> logs/benchmark_batch.log 2>&1 < /dev/null &
~~~

~~~bash
tail -n 50 -f logs/benchmark_batch.log
~~~

总日志每十秒报告当前候选仍在运行；每个候选的完整初始化、编译和逐步指标在 batch_N.log 中，总日志会给出文件路径。第一次编译耗时不代表稳定训练速度。

单个候选默认超时 1200 秒；可通过 --timeout-seconds 调整。父进程收到正常终止信号时会清理自己启动的测试子进程。若使用强制 KILL，仍需单独确认测试子进程是否已退出。

## 输出与指标

目录：checks/output/batch_benchmark/<时间>/。

| 文件/字段 | 含义 |
|---|---|
| summary.json | GPU 信息、各候选结果、通过的 batch 列表、停止位置 |
| batch_N.json | 对应 batch 的配置、指标、显存数据和失败阶段 |
| batch_N.log | 模型初始化、编译及运行日志 |
| compile_and_first_update_seconds | 编译加首次更新的总耗时，未计入稳定吞吐；不是纯编译耗时 |
| median_compute_seconds | 批数据就绪之后的一次同步训练更新耗时，含小量指标读取开销 |
| median_total_seconds | 数据加载/传输加同步更新耗时 |
| compute_samples_per_second | 排除加载和编译的观测样本吞吐 |
| total_samples_per_second | 包含加载/传输的观测样本吞吐 |
| sampled_process_memory_peak_mib | nvidia-smi 对测试进程的显存峰值采样，可能包含 JAX 保留内存 |
| memory_after_init / memory_after_measurement | JAX 分配器统计；若提供 peak_bytes_in_use，可用于辅助判断内存峰值 |
| system_available_before_mib / system_available_min_mib | 测试前及采样期间最低的全机可用系统内存，MiB 单位 |

使用 90% 显存份额和预分配设置。nvidia-smi 可能对多个 batch 显示接近的预留量，因此不能仅靠它判断真实额外显存。短测通过也不能保证长程训练和保存检查点时永远不会 OOM。

计时使用显式同步，将加载与计算分开，便于比较；它不能完整模拟正式训练中 CPU/GPU 的重叠，也不包含检查点保存时间。

选择时同时看是否稳定通过、样本吞吐和显存余量，不仅看最大 batch。若 batch 8 通过，可再用 --batches 16 32 继续探测。若实际 batch 仍较小，再设计、实现并验证梯度累积，之后重新决定训练预算和学习率。

## 当前验证范围

WSL 已通过语法、计划模式、无效参数拒绝检查，以及四项使用模拟子进程的调度测试：显卡占用拒绝、OOM 后停止更大候选、非零退出不可作为 PASS、缺少报告不可作为 PASS。

版本 2 另通过两项真实 OpenPI TrainState 的 CPU 回归测试：显式 lowering 复现 ArgInfo 错误，普通 JIT 调用连续五次优化器更新且 loss 下降。测试使用很小的参数状态，不加载 π0.5，不代表 GPU 容量通过。

版本 3 通过六项调度/进程测试，覆盖原四项、worker 参数确实传给隔离子进程、终止时清理实际衍生进程。另在真实 episode 0 上用 batch 16 对比 workers 0/2，各读取两个经过归一化、tokenizer、图像变换的 CPU 训练批次，输入逐项一致；使用本地探测统计量，不包含全量 Dataset 或 GPU 训练测试。

这些测试没有加载真实模型或测试 GPU。真实初始化、编译、前向/反向、显存与吞吐需要在服务器上验收。

服务器后续实测：版本 2 已完成 batch 1、2、4、8 的真实训练短测，各预热 3 步、计时 10 步，全部通过。含加载吞吐依次约 1.76、2.16、2.44、2.43 个观测样本/秒，4 与 8 的小幅差异不能在十步测试下认定为确定优势。nvidia-smi 的约 22 GiB 读数包含预留池。详细记录见 validation/SERVER_VALIDATION.json。batch 16/32、长程稳定性、保存峰值及模型效果仍待验证。

继续实测：batch 16 通过，纯训练/含加载吞吐约 3.57/3.04 个样本/秒；batch 32 报 OOM，未产生稳定吞吐。16 是目前通过的最大候选，不等于精确容量上限。正式训练第一轮选择实际 batch 16，先跑 1000 步验证长程更新与保存，见 FORMAL_TRAINING.md。
