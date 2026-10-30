# 检查点推理与 LIBERO 仿真评估

本阶段先使用 libero_smoke_01/119 验证完整流程。完成 120 次更新的检查点只用于流程验收，不预设任务成功率。正式 30000 步训练在流程跑通后安排。

## 文件与环境

| 文件 | 用途 | 运行环境 |
|---|---|---|
| inference/serve_policy.py | 校验训练记录、LoRA 结构及检查点统计量，启动官方策略服务 | .venv，Python 3.11，单 GPU |
| inference/policy_rpc.py | 服务健康检查与实验、步骤、模型、统计量身份校验 | 两个环境均兼容 |
| inference/check_policy.py | 发送真实数据中的九个观测，检查返回动作 | .venv，CPU 客户端 |
| eval/check_simulation.py | 独立检查单场景、初始化、十个稳定动作、两路图像及状态 | tools/libero-env，CPU OSMesa |
| ops/setup_libero_env.sh | 按固定仓库中的官方依赖表安装独立仿真环境 | tools/libero-env，Python 3.8 |
| eval/eval_libero.py | 调用官方仿真评估入口，可选保存视频和结构化结果 | tools/libero-env；默认 CPU OSMesa 渲染 |

将整个 pipeline 文件夹更新到 /mnt/sda/xd/openpi_ws/pipeline。推理服务会使用所选检查点的 assets/libero/norm_stats.json，并核对其 SHA256 与原训练记录一致。不会用官方完整模型配置替代当前 LoRA 结构。

## 1. 启动检查点策略服务

先确认有一张可供使用的 GPU。下面沿用恢复训练时的 GPU 2 UUID；如分配不同，替换 --gpu 的值。

~~~bash
cd /mnt/sda/xd/openpi_ws
nohup env PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -u -m pipeline.inference.serve_policy \
  --gpu GPU-93947930-8f97-38bc-fcea-0b9d168bc871 \
  --exp-name libero_smoke_01 --step 119 --port 8000 \
  >> logs/serve_libero_smoke_01_119.log 2>&1 < /dev/null &
~~~

检查加载进度：

~~~bash
tail -n 60 logs/serve_libero_smoke_01_119.log
~~~

出现 server listening 后，另一个终端可确认健康状态：

~~~bash
curl --noproxy '*' -fsS http://127.0.0.1:8000/healthz
~~~

预期输出 OK。模型加载完成后服务会持续运行；首次动作请求还需要编译推理计算。服务绑定本机地址，评估程序在同一服务器连接。若 8000 已被占用，请为服务和客户端使用相同的其他端口。

## 2. 真实数据观测的推理检查

服务保持运行，执行：

~~~bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m pipeline.inference.check_policy \
  --exp-name libero_smoke_01 --step 119 --port 8000
~~~

读取 episode 0、846、1692 的首/中/末帧，共九个真实观测。每次发送两路 RGB 图像、8 维状态及任务文本；期望每个响应为有限数值的 (10,7) 动作。原始数据图像不再旋转，仿真图像旋转遵循官方评估代码。

默认在终端显示服务身份、统计量 SHA256、各请求耗时和动作范围。需要报告时加 --output runs/pi05_libero/libero_smoke_01/inference_probe_119.json。第一次请求包含编译耗时，不代表稳定推理速度。此检查不比较任务成功率。

## 3. 准备独立 LIBERO 仿真环境

保持训练环境原样，另建 Python 3.8 环境。安装采用固定仓库中的 examples/libero/requirements.txt、third_party/libero/requirements.txt 以及 openpi-client 的依赖声明，先合并解析完整依赖、保存 tools/libero_requirements.lock，再同步安装并安装本地 openpi-client 与 LIBERO。这个 compile → sync 步骤遵循 [uv 官方环境锁定流程](https://docs.astral.sh/uv/pip/compile/)。

本机尚未运行安装；此步骤在服务器验收。服务器需要编译工具和渲染系统库，缺少时安装：

~~~bash
sudo apt-get update
sudo apt-get install -y build-essential libosmesa6 libegl1 libgl1 libglib2.0-0
~~~

安装 Python 依赖：

~~~bash
nohup bash pipeline/ops/setup_libero_env.sh \
  >> logs/install_libero_env.log 2>&1 < /dev/null &
~~~

查看日志：

~~~bash
tail -n 40 logs/install_libero_env.log
~~~

安装脚本使用工作区 uv、Python 下载目录和包缓存，网络模式只作用于下载子进程。旧依赖的构建环境使用 setuptools<75.4，以兼容 Python 3.8。安装后执行依赖一致性检查，并将实际安装版本保存到 tools/libero_installed.txt。首次安装会下载 Python 和较大的 PyTorch 包；进程结束后仍需通过下面的真实仿真验证。

网络设置需要按服务器实际下载站测速确认，不能从 Google/魔搭的速度推断 GitHub、PyPI 或 PyTorch。当前 Python 与 PyTorch 下载站尚未完成服务器测速。脚本提供：

| 设置 | 作用 |
|---|---|
| LIBERO_NETWORK_MODE=direct | 包下载绕过代理，默认值；仅参考此前直连更快的经验，尚未逐站验证 |
| LIBERO_NETWORK_MODE=inherit | 包下载保留终端已有的代理与 NO_PROXY 设置 |
| LIBERO_PYTHON_NETWORK_MODE=inherit | 可让 Python 解释器下载单独继承终端网络设置 |
| UV_DEFAULT_INDEX | 指定经服务器验证的 PyPI 镜像；不改全局配置 |
| UV_HTTP_TIMEOUT / UV_CONCURRENT_DOWNLOADS | 默认 120 秒 / 4，可按网络调整 |

国内普通包源可测试 [TUNA PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)。PyTorch 1.11.0+cu113 的 cp38 Linux 包仍使用 [PyTorch 官方索引](https://download.pytorch.org/whl/cu113/torch/)，不能假定普通 PyPI 镜像包含这类专用 wheel；Python 解释器下载也不受 PyPI 索引设置影响。先验证可达性与速度，再选择环境变量，不承诺镜像一定更快。

默认仿真使用 OSMesa 在 CPU 上渲染，并隐藏仿真进程的 CUDA 设备。GPU 用于另一个进程中的模型推理。--render-backend egl 为可选路径，其显卡映射与驱动依赖需要另行核对。

Python 依赖安装完成后，先独立验证渲染；此检查不需要策略服务或空闲 GPU：

~~~bash
cd /mnt/sda/xd/openpi_ws
nohup env PYTHONDONTWRITEBYTECODE=1 \
  tools/libero-env/bin/python -u -m pipeline.eval.check_simulation \
  >> logs/check_simulation.log 2>&1 < /dev/null &
~~~

~~~bash
tail -n 50 logs/check_simulation.log
~~~

预期 PASS：一个真实任务完成 reset、加载首个初始状态、执行十个稳定动作；两路图像为非均匀的 uint8 (256,256,3)，机器人状态为有限数值的 (8,)；环境正常关闭。默认只打印结果；加 --output-dir checks/output/libero_simulation/<新目录> 才保存报告和按官方方向旋转后的 agentview.png、wrist.png。此检查只验证仿真，不使用学习策略，不计算任务成功率。

## 4. 先跑一个任务集合，每任务一次

确认独立环境安装完成，策略服务及九观测推理检查通过后：

~~~bash
nohup env PYTHONDONTWRITEBYTECODE=1 \
  tools/libero-env/bin/python -u -m pipeline.eval.eval_libero \
  --exp-name libero_smoke_01 --step 119 --port 8000 \
  --suite libero_spatial --trials 1 \
  >> logs/eval_libero_smoke_01_119_spatial.log 2>&1 < /dev/null &
~~~

~~~bash
tail -n 50 logs/eval_libero_smoke_01_119_spatial.log
~~~

沿用官方评估：十个任务各一次，固定 seed 7，先等待十步稳定物体，使用 256 像素渲染、224 像素模型输入、每五步重新规划。图像旋转、状态构造、动作调用及任务时长均由官方入口处理。

包装脚本会在工作区 .cache/libero 中生成路径配置，避免 LIBERO 首次导入时交互询问；不改用户主目录中的 LIBERO 配置。

默认只在终端输出评估结果，官方评估生成的中间视频放在自动清理的临时目录。需要报告和回放时，命令加 --output-dir runs/pi05_libero/libero_smoke_01/eval_119/libero_spatial_01；以下产物说明适用于显式保存模式。

| 产物 | 内容 |
|---|---|
| result.json | 检查点身份、任务集合、试验次数、种子、完成 episode 数及任务成功率 |
| videos/*.mp4 | 官方评估生成的任务回放 |
| logs/eval_libero_smoke_01_119_spatial.log | 仿真和评估日志 |

result.json 的 PASS 表示评估流程完整执行；task_success_rate 单独表示模型任务成功率。官方循环内部捕获的运行异常会被包装脚本记录为 FAIL，不会当成普通任务失败混入已通过的流程结果。

## 5. 正式训练与评估

先完成上述服务、真实推理和仿真验收，再按 BATCH_BENCHMARK.md 测试实际 batch 容量与吞吐，确定正式配置。现有默认 batch 1、30000 步不等于已验证的效果方案。之后用新实验名 libero_lora_01 从 pi05_base 训练，命令模板见 [README.md](../README.md)。

正式模型评估时，将服务和客户端的 --exp-name、--step 同时替换成正式检查点。支持 libero_spatial、libero_object、libero_goal、libero_10 四套，每套默认十个任务。完整评估可将 --trials 设置为 50；耗时显著增加。

当前官方评估入口的视频文件名只包含任务与成功/失败，多个试验的同类视频可能覆盖。默认每任务一次的流程验收不受此影响。成功率仍由全部试验计数，查看重复试验视频时需注意这个官方命名行为。

## 验证范围

- 服务器：根据用户提供日志，全量统计、LoRA 100 步训练、保存、恢复到累计 120 步、单场景 CPU 渲染、119 检查点真实推理和 spatial 十任务闭环均已通过执行验收。spatial 成功率为 0/10。详见 validation/SERVER_VALIDATION.json 和 SMOKE_ACCEPTANCE.md。
- WSL：新文件语法/CLI 检查；真实图像通过官方 WebSocket 服务端和客户端传输；错误检查点身份会被拒绝。传输测试使用模拟策略，不代表真实模型推理通过。
- 后续：查看失败回放、正式训练、模型效果验收、其他任务集合及重复试验。
