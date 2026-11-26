# 检查点推理与 LIBERO 仿真评估

本阶段先使用 libero_smoke_01/119 验证完整流程。完成 120 次更新的检查点只用于流程验收，不预设任务成功率。正式 30000 步训练在流程跑通后安排。

## 文件与环境

| 文件 | 用途 | 运行环境 |
|---|---|---|
| inference/serve_policy.py | 校验训练记录、LoRA 结构及检查点统计量，启动官方策略服务 | .venv，Python 3.11，单 GPU |
| inference/policy_rpc.py | 服务健康检查与实验、步骤、模型、统计量身份校验 | 两个环境均兼容 |
| inference/check_policy.py | 发送真实数据中的九个观测，检查返回动作 | .venv，CPU 客户端 |
| eval/check_simulation.py | 独立检查单场景、初始化、十个稳定动作、两路图像及状态 | `tools/libero-env`（Python 3.8，默认 CPU OSMesa） |
| eval/eval_libero.py | 调用官方仿真评估入口，按 `--host`/`--port` 连接策略服务，可选保存视频和结构化结果 | `tools/libero-env`（Python 3.8，默认 CPU OSMesa） |

将整个 pipeline 文件夹更新到 /mnt/sda/xd/openpi_ws/pipeline。推理服务会使用所选检查点的 assets/libero/norm_stats.json，并核对其 SHA256 与原训练记录一致。不会用官方完整模型配置替代当前 LoRA 结构。

## 1. 启动检查点策略服务

先确认有一张可供使用的 GPU。下面沿用恢复训练时的 GPU 2 UUID；如分配不同，替换 --gpu 的值。

~~~bash
cd /mnt/sda/xd/openpi_ws
nohup env PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -u -m pipeline.inference.serve_policy \
  --gpu GPU-93947930-8f97-38bc-fcea-0b9d168bc871 \
  --policy.config pi05_libero \
  --policy.dir checkpoints/pi05_libero/libero_smoke_01/119 --port 8000 \
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

## 3. 准备本地 LIBERO 仿真环境

Ubuntu 或 WSL 上按需安装编译工具和系统渲染库。OpenCV 还需要 GLib，Ubuntu 22.04 的包名为 `libglib2.0-0`，Ubuntu 24.04 及之后使用 `libglib2.0-0t64`。

~~~bash
sudo apt-get update
sudo apt-get install -y build-essential libosmesa6 libegl1 libgl1
~~~

在 Ubuntu 26.04 上也安装 `patchelf`（0.18 或以上）。如果 glibc 拒绝加载旧版 PyTorch，安装脚本会清除仿真虚拟环境中对应共享库的可执行栈标志，并重新验证导入。 [glibc 2.41 变更说明](https://sourceware.org/pipermail/libc-announce/2025/000045.html)

在工作区根目录构建仿真环境并检查渲染。

~~~bash
git submodule update --init --recursive
bash scripts/setup_libero_env.sh
python3 -B -u -m pipeline.eval.check_simulation
~~~

安装脚本使用上游 LIBERO 和 OpenPI 客户端的依赖声明，生成 `tools/libero_requirements.lock` 并同步到独立的 `tools/libero-env`。缺少 `uv` 时安装到 `tools/bin`。下载默认直连，可设 `LIBERO_NETWORK_MODE=inherit` 沿用当前终端的代理设置，或用 `LIBERO_PYTHON_NETWORK_MODE` 单独设置 Python 下载路径。首次安装会下载 Python 和 PyTorch 等依赖。

系统 `python3` 会自动使用工作区的 Python 3.8 环境，也可直接运行 `tools/libero-env/bin/python -B -u -m pipeline.eval.check_simulation`。默认 CPU OSMesa 渲染，支持显式 `--render-backend egl`。如果已经将 OSMesa 库解包到 `tools/osmesa/usr/lib/` 下对应的架构目录，入口会自动加载。需要保留相机图和报告时传 `--output-dir checks/output/libero_simulation/<新目录>`。

预期 `PASS`。一个真实任务完成 reset、加载首个初始状态并执行十个稳定动作，两路图像为非均匀的 `uint8 (256, 256, 3)`，机器人状态为有限数值的 `(8,)`。这一步可以独立于策略服务执行。

如需容器运行，给任一 eval 入口加 `--docker`，默认使用 `libero:latest` 和 EGL。`--image <镜像>` 可显式选择 Docker 镜像，`--render-backend osmesa` 可在容器中使用 CPU 渲染。

## 4. 连接策略服务运行 LIBERO 仿真评估

策略服务运行后，在仿真端（本机或跨机 WSL）直接指定 `--host` 与 `--port`：

~~~bash
# 快速单任务冒烟验证（libero_spatial task 0，1 次试验）
python3 -B -u -m pipeline.eval.eval_libero \
  --host 10.110.230.112 --port 8000 \
  --suite libero_spatial --task-id 0 --trials 1

# 整套 10 任务各跑 1 次，并保存视频与结果
python3 -B -u -m pipeline.eval.eval_libero \
  --host 10.110.230.112 --port 8000 \
  --suite libero_spatial --trials 1 \
  --output-dir runs/pi05_libero/eval_spatial_01
~~~

沿用官方评估：固定 seed 7，先等待十步稳定物体，使用 256 像素渲染、224 像素模型输入、每五步重新规划。图像旋转、状态构造、动作调用及任务时长均由官方入口处理；服务端返回的 `metadata` 会直接记录到结果 JSON 的 `server_metadata` 中，不依赖本机 `runs/.../config.json`。

LIBERO 配置和 Numba 缓存写入临时目录，运行结束后自动清理。默认只在终端输出评估结果，中间视频也随临时目录清理。显式指定 `--output-dir` 时保存 `result.json` 与 `videos/*.mp4`。

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
