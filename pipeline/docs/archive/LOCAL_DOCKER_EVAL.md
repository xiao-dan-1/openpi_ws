> 历史归档：正文保留当时的命令与结果，旧路径不再作为当前入口。`local-eval-6999.env` 已删除。当前使用 WSL `openpi/examples/libero/compose.yml`；重构说明见 [REFACTOR.md](../REFACTOR.md)。

# 本地 Docker 评估：libero_lora_01 / 999

> 当前统一入口已迁移到 WSL 的 `/home/xiaodan/openpi_ws/openpi/examples/libero/compose.yml`
> 与同目录固定 `.env`，详细指令见 `VIEWER_README.md`。下文按检查点区分的
> `compose.local-eval.yml`、`local-eval-6999.env` 仅保留作历史运行记录。
> 日常更换同结构 LoRA 检查点只修改固定 `.env` 的 CHECKPOINT_PATH 与 CHECKPOINT_STEP。

已使用本机现有的 `openpi_server:latest` 和 `libero:latest` 镜像完成评估，无需重建镜像。推理使用 RTX 4070 Ti SUPER，仿真使用 CPU OSMesa。原下载检查点只读挂载。

实测结果：`libero_spatial`，10 个任务，每个任务 1 次，seed=7，成功 2/10，无运行错误，评估循环约 5 分 19 秒。与服务器相同的两个任务成功；一次测试不能代表稳定成功率。

## 文件和路径

| 内容 | 位置 |
| --- | --- |
| Compose 配置 | `pipeline/compose.local-eval.yml` |
| 本地推理入口 | `pipeline/local_docker_serve.py` |
| 原下载检查点 | `C:/Users/xiaodan/Downloads/frp-lab.com/202610241653/999` |
| WSL 项目根目录 | `/home/xiaodan/openpi_ws` |
| 输出根目录 | `D:/home/06_projects/openpi_ws/checks/output/local_docker_eval_999` |
| 本次评估 | 输出根目录下 `runs/pi05_libero/libero_lora_01/eval_999/libero_spatial_20261024T135219478671Z` |
| 结果和录像 | 本次评估目录下 `result.json`、`videos/*.mp4` |
| 运行记录 | 输出根目录下 `local_eval_report.json`、`eval.log`、`policy.log` |

推理入口针对这次导入的 999 检查点：按已验证的 LoRA 配置重建推理清单，核对归一化统计 SHA256 和检查点提交标记，然后通过官方 `create_trained_policy` 加载参数。清单并非从服务器下载的原始 `config.json`。

WSL 只有约 10 GiB RAM，参数恢复并发预算设为 3 GB，JAX 显存池比例设为 0.70。仅在当前进程中调整 Orbax 检查点处理器，不修改已安装的软件包。仿真容器内存上限为 3 GiB。既有仓库的 Docker 和 Orbax 补丁没有用于本次 Compose。

## 重跑

以下命令均在 WSL Ubuntu-24.04 中、项目根目录执行。Windows 的配置文件副本须先同步到 WSL 的 `pipeline/`。保持第一个终端的前台命令运行，避免 WSL 空闲退出使 Docker 服务结束。下载的检查点和数据目录须仍然存在。

终端一：

```bash
cd /home/xiaodan/openpi_ws
docker compose -f pipeline/compose.local-eval.yml up --no-build policy
```

终端二：确认健康检查返回 `OK` 后再启动仿真。

```bash
cd /home/xiaodan/openpi_ws
curl --noproxy '*' -fsS --connect-timeout 3 --max-time 5 http://127.0.0.1:8100/healthz
docker compose -f pipeline/compose.local-eval.yml run --rm --no-deps simulator
```

每次评估生成带新时间戳的结果目录。推理服务的首次请求可能进行编译。

关闭本项目的容器：

```bash
docker compose -f pipeline/compose.local-eval.yml down --timeout 20
```

本次完成后已关闭本项目的推理服务，仿真容器已自动移除。没有操作服务器上的进程。

## 导入 6999 并评估

`202610260907/6999_compress_202610260818.tar.gz` 已解压到 Windows 项目下
`checkpoints/imported/202610260907/6999`。原压缩包保留。

本地入口现支持 `--step`，Compose 通过 `LOCAL_STEP` 选择检查点，默认仍为 999。
6999 首次加载遇到文件读取 ENOMEM；入口增加 `--file-io-concurrency`（默认 8），
在恢复期间为 TensorStore 的文件读写与数据复制设置较低并发，原镜像和权重不变。
`pipeline/local-eval-6999.env` 指定 6999 的只读挂载和独立输出目录
`checks/output/local_docker_eval_6999`。以下命令均在 WSL 项目根目录执行，
保持策略服务的前台终端打开。

终端一：

```bash
docker compose --env-file pipeline/local-eval-6999.env -p openpi-local-6999 \
  -f pipeline/compose.local-eval.yml up --no-build policy
```

终端二：健康检查返回 OK 后运行仿真，不先调用额外的推理探测，以便保持与上次相同的策略随机数起点。

```bash
curl --noproxy '*' -fsS --connect-timeout 3 --max-time 5 http://127.0.0.1:8100/healthz
docker compose --env-file pipeline/local-eval-6999.env -p openpi-local-6999 \
  -f pipeline/compose.local-eval.yml run --rm --no-deps simulator
```

评估完成后关闭：

```bash
docker compose --env-file pipeline/local-eval-6999.env -p openpi-local-6999 \
  -f pipeline/compose.local-eval.yml down --timeout 20
```

6999 本次实测已完成：成功 7/10（70%），999 为 2/10（20%）。
先前成功的两个任务均保持成功，另有五个任务由失败变为成功。
评估循环约 4 分 49 秒，无评估运行错误，10 个 MP4 均通过全片解码检查。
结果位于输出根目录下
`runs/pi05_libero/libero_lora_01/eval_6999/libero_spatial_20261026T023459859941Z`；
逐任务对照见 `COMPARISON.md`，运行报告见 `local_eval_report.json`。
完成后本项目容器已关闭并移除，原压缩包与检查点保留。
