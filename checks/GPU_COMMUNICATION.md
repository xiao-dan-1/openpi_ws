# GPU 通信测试

使用现有 OpenPI Python 环境中的 JAX，在单机两张卡之间执行合成 AllReduce、AllGather、ReduceScatter。只测通信相关路径，不加载模型、不执行训练、不修改训练配置，不保存报告。它是 JAX 应用级 collective 微基准，并非 NVIDIA 官方 nccl-tests，也不是纯硬件链路测速。

CPU 使用两个模拟设备验证数值、消息尺寸、带宽换算及父子进程协议。用户已提供服务器0+1、2+3、0+2的通信测试结果，三种collective数值检查均通过；部分测试存在其他计算进程，性能结果只代表当时负载。

修复版v2兼容NCCL在结果JSON之后继续刷新原生日志的情况，保留前后诊断文本，避免 `JSONDecodeError: Extra data`。6项CPU检查包括该回归及真实模拟collective；服务器已有v1可直接覆盖脚本。已收到的2+3诊断显示 `intraNodeP2pSupport 0`、`via SHM/direct/direct` 与 `/dev/shm/nccl-*` 主机缓冲区，说明本轮GPU间通道实际使用SHM。启动时的 `Using network Socket` 不能单独作为GPU数据走网卡的证据。

## 安装与执行

上传修复结果解析的gpu_communication_check_v2.zip至服务器工作区，解压（当前完整Q5更新包也已包含此脚本）：

```bash
cd /mnt/sda/xd/openpi_ws
unzip -o gpu_communication_check_v2.zip
```

在可使用的两张卡上逐组执行，不要同时启动多组，以免相互争用。结果直接输出终端：

```bash
env PYTHONDONTWRITEBYTECODE=1 openpi/.venv/bin/python -u checks/check_gpu_communication.py --gpus 0,1
env PYTHONDONTWRITEBYTECODE=1 openpi/.venv/bin/python -u checks/check_gpu_communication.py --gpus 2,3
env PYTHONDONTWRITEBYTECODE=1 openpi/.venv/bin/python -u checks/check_gpu_communication.py --gpus 0,2
```

0+1、2+3是同NUMA的NODE路径；0+2是跨NUMA的SYS路径。脚本显示两张物理卡的UUID、测试前后显存/利用率/计算进程。如果有其他计算进程，会标注shared_load，测试结果只能代表共享负载当时的表现。它不预订显卡、不终止其他进程，也不会自动禁用P2P/SHM来改变通信路线。

默认测试4KiB、1MiB、16MiB、64MiB的每卡float32输入。每种通信预热5次、测量50次，每次等待结果完成后计时。JIT编译、预热和CPU传入/传出检查不进入计时。每对卡使用独立进程，超时默认300秒，超时终止该测试子进程。独立测试进程关闭预分配并设置10%池预算，减小合成测试占用；这些设置不会修改Q5训练的75%配置。此预算不是实际占用或整个进程硬上限。

测试输出的median_ms和p90_ms越低越好。normalized_bus_GB_per_s越高越好，它依据NVIDIA nccl-tests相同的消息尺寸归一化公式换算，单位为十进制GB/s，不是直接测得的PCIe总线吞吐。两卡AllReduce按输入字节/time计算；AllGather按每卡输入字节/time；ReduceScatter按每卡输入字节/(2*time)。AllGather的输出是每卡输入的两倍，ReduceScatter是输入的一半。

小消息结果包含主机调度/同步开销，应比较大消息（16/64MiB）下的趋势，而非把4KiB时间称为纯链路延迟。所有三种操作都检查输出的形状与rank相关数值，并确认lowered graph包含对应collective。PASS表示完成且数值正确，不表示性能达标。没有固定通用的GB/s达标阈值，应结合任务计算时间和不同卡对的实测比较。

## 确认通信路径

若要查看NCCL实际选择的传输方式，额外做一次小测试：

```bash
env PYTHONDONTWRITEBYTECODE=1 openpi/.venv/bin/python -u checks/check_gpu_communication.py \
  --gpus 0,1 --sizes-mib 16 --iterations 10 --debug-transport
```

它仅对这个测试进程设置NCCL_DEBUG=INFO和对应子系统。日志可能包含SHM/NET/P2P等路径信息，具体输出取决于安装的NCCL版本；不能只凭CNS或测得带宽确定运行时路线。诊断日志可能影响计时，此轮不与无debug的正式计时直接比较。

## 如何据此决策

先比较同NUMA与跨NUMA卡对的16/64MiB结果和波动，再估计FSDP所需频繁AllGather/ReduceScatter的通信成本。微基准能排除明显通信异常，但不能承诺完整FSDP训练速度。后续真实Q5双卡benchmark及100→120步保存恢复已验收，见[双卡记录](../pipeline/q5/FSDP_TRAINING.md)。通信测试本身不修改训练入口或冻结规则。

参考：

- [JAX 异步调度与计时](https://docs.jax.dev/en/latest/async_dispatch.html)
- [NVIDIA nccl-tests 性能与带宽定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- [NVIDIA NCCL 通信排查](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
