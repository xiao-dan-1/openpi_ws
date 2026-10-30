# 双卡 FSDP 容量与吞吐短测

本入口仅用于服务器实验，不是正式训练启动器。复用当前 Q5 模型、训练数据、统计、LIBERO 冻结规则和官方初始化/train_step；不改 OpenPI 源码，不改变模型/优化器/学习率，不保存检查点。默认仅输出终端，可显式用 --output 保存报告。

测试背景：单卡batch4已跑通，单卡batch8在75%显存配置下OOM。通信测试确认2+3可用，NCCL路径为SHM，尚不能据此预测完整FSDP性能。

## 安装

当前完整Q5更新包为`q5_dual_training_v1.zip`，包含本benchmark、正式双卡入口及相关依赖；前提是服务器已有结构化pipeline和Q5 v2数据/统计：

```bash
cd /mnt/sda/xd/openpi_ws
unzip -o q5_dual_training_v1.zip
```

包含独立benchmark_fsdp入口、显式多GPU支持、通信查询辅助和CPU测试。正式train也已支持`--gpus`和`--fsdp-devices`，check_model及evaluate_validation保持单卡。本benchmark根据本次命令设置1或2卡分片。训练统计和75%默认显存配置保持原值。不要在旧训练实验中绕过代码指纹校验恢复；旧实验若需resume应保留其原代码。

当前代码和双卡更新包已包含`--memory-fraction`，默认0.75；`q5_memory_fraction_cli_v1.zip`只是早期单独升级显存参数的历史包，不要再覆盖较新的双卡训练代码。为两张24GiB卡指定0.65时，每张池预算约15.6GiB；这不是实际活跃内存或整个进程硬上限。下面是已失败配置的复现示例，不是建议的训练启动参数：

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m pipeline.q5.benchmark_fsdp \
  --gpus 1,2 --fsdp-devices 2 --batch-size 16 --memory-fraction 0.65
```

这是可选择的容量复测命令。服务器实测双卡全局batch16在fraction0.65时OOM，而0.75通过；不要把较小比例当成已通过的配置。其余参数保持相同才便于比较；结束时核对报告memory_fraction与allocator_settings。新参数也支持Q5 train/check_model/evaluate_validation；不修改LIBERO设置。报告默认只打印终端。

## 三组可比较实验

只使用已分配/允许使用的GPU，逐条执行，不要并行跑这三组。输出开始/结束时的其他计算进程，仅描述端点状态，不排除中途发生竞争；共享负载可能严重影响结论。

```bash
# 单卡基线：全局batch4，每卡4
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m pipeline.q5.benchmark_fsdp \
  --gpus 2 --fsdp-devices 1 --batch-size 4

# 双卡FSDP：全局batch4，每卡2
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m pipeline.q5.benchmark_fsdp \
  --gpus 2,3 --fsdp-devices 2 --batch-size 4

# 上一组成功后：全局batch8，每卡4
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m pipeline.q5.benchmark_fsdp \
  --gpus 2,3 --fsdp-devices 2 --batch-size 8
```

首轮先执行双卡batch4也可以，确认容量与分片正常后再补测基线。任何OOM/异常时该进程退出，不自动扩大batch、不修改显存比例或冻结策略。

每组先执行3次真实更新（首步包含JIT编译），再计时10次真实更新。--warmup-steps控制计时前预热次数，不是更改学习率预热长度；smoke学习率预热仍为10。想获得较稳的数字，可相同配置设置 --measure-steps 20，比较时所有组使用同样次数。更新结束进程退出，参数丢弃；不生成runs/checkpoints或任务质量结论。

## 看哪些输出

- state_storage：参数和优化器的logical_bytes、per_device_bytes、sharded_array_leaves。双卡必须看到真实分片；小张量等仍可能复制，不能承诺每卡内存减半。这些是常驻数组大小，不含所有激活和编译/通信临时缓冲区。
- memory_after_init / memory_after_measurement：JAX每卡分配器统计，若存在bytes_in_use、peak_bytes_in_use可辅助分析。预留池大小不等于实际活跃内存。
- first_update_compile_included_seconds：首步编译加更新，不计入稳定吞吐。
- median_compute_seconds：数据已到设备后，完整更新的同步中位耗时。
- median_total_seconds / total_samples_per_second：包含加载与更新的同步时序指标。加载与计算被串行计时，这不是官方训练主循环充分流水重叠的吞吐。
- shared_load_observed：其他任务存在时不能当作独占卡性能；即使false也仅表示前后快照未发现其他任务。

先比较相同全局batch4的单/双卡存储和吞吐，再看更大batch的容量。不同batch的短测loss不能直接用于比较训练效果。正式多卡保存恢复的100→120步验收已经通过，见[FSDP_TRAINING.md](FSDP_TRAINING.md)；较长稳定性仍需另外验证。

## 本机检查范围

用两个模拟CPU设备验证状态张量真实分片、全局batch拆分和donated训练状态复用；不加载完整模型，不进行本机GPU训练。相关CPU测试：

```bash
.venv/bin/python -B -m unittest tests.test_q5_fsdp_benchmark tests.test_gpu_communication \
  tests.test_q5_pipeline tests.test_pipeline_entrypoints -q
```

用户提供的服务器结果已验证：双卡fraction0.75下全局batch4、8、16通过；全局batch32在0.75下OOM、0.90下通过；全局batch16在0.65下OOM。不同NUMA卡对及共享负载会影响耗时，不能视为统一硬件性能保证。实验使用官方OpenPI的JAX分片实现，不能将PyTorch FSDP某个具体版本的通信次数直接当作本入口的行为。
