# Pipeline 目录重构与迁移

2026-10-27：按训练、推理、评估、数据、公共代码、环境运维和文档划分职责。Python 入口统一使用包模块调用；模型配置、训练参数默认值、统计量校验、检查点及输出目录规则沿用重构前实现。

## 新目录

```text
pipeline/
  __init__.py
  README.md
  common/
    config.py               # 工作区、官方配置、运行环境、资产与清单
    validation.py           # 跨模块参数校验
  train/
    run.py
    benchmark_batch.py
  inference/
    serve_policy.py
    local_docker_serve.py
    policy_rpc.py
    check_policy.py
  eval/
    eval_libero.py
    check_simulation.py
  data/
    compute_norm_stats.py
    check_config.py
  ops/
    README.md
    setup_libero_env.sh
    libero_build_constraints.txt
  docs/
    FORMAL_TRAINING.md
    BATCH_BENCHMARK.md
    INFERENCE_EVAL.md
    SMOKE_ACCEPTANCE.md
    REFACTOR.md
    validation/             # 原验证 JSON，保留内容
    archive/                # 旧本地部署说明与 Compose
```

Python 子目录均包含 `__init__.py`，包初始化不加载模型或仿真依赖。`common` 不依赖训练、推理或评估入口；实验名校验已从训练脚本提取到 `common.validation`。

`data/` 存处理代码，实际数据继续使用工作区 `datasets/`。`assets/`、`checkpoints/`、`runs/`、`logs/`、`checks/output/` 的位置不变。

## 入口对照

在工作区根目录执行，按任务选用 `openpi/.venv/bin/python` 或独立仿真环境的解释器。下面列的是入口，实际参数见 `--help` 和各操作文档。

| 原脚本 | 新调用 |
|---|---|
| `pipeline/train.py` | `python -m pipeline.train.run` |
| `pipeline/benchmark_batch.py` | `python -m pipeline.train.benchmark_batch` |
| `pipeline/serve_policy.py` | `python -m pipeline.inference.serve_policy` |
| `pipeline/local_docker_serve.py` | `python -m pipeline.inference.local_docker_serve` |
| `pipeline/check_policy.py` | `python -m pipeline.inference.check_policy` |
| `pipeline/eval_libero.py` | `python -m pipeline.eval.eval_libero` |
| `pipeline/check_simulation.py` | `python -m pipeline.eval.check_simulation` |
| `pipeline/compute_norm_stats.py` | `python -m pipeline.data.compute_norm_stats` |
| `pipeline/check_config.py` | `python -m pipeline.data.check_config` |
| `bash pipeline/setup_libero_env.sh` | `bash pipeline/ops/setup_libero_env.sh` |

`policy_rpc.py` 和 `config.py` 是公共模块，分别通过 `pipeline.inference.policy_rpc` 和 `pipeline.common.config` 导入。旧入口没有兼容包装层，部署时替换完整 `pipeline/` 并删除旧的同名平铺脚本，避免混用。服务器上的调度脚本也需按表更新；本次只更新当前 Windows 工作区与本机 WSL。

batch 测试的隔离子进程也使用 `-m pipeline.train.benchmark_batch`，并显式从解析出的工作区启动。工作区路径从 `common/config.py` 的真实位置解析，兼容 Windows、WSL 和容器 `/workspace`。

## 本机 WSL 部署

日常入口仍为 `/home/xiaodan/openpi_ws/openpi/examples/libero/compose.yml` 和同目录固定 `.env`。策略容器的启动命令改为：

```text
/.venv/bin/python -u -m pipeline.inference.local_docker_serve
```

容器工作目录为 `/workspace`，继续只读挂载完整 `/workspace/pipeline`。固定 `.env` 的检查点选择规则沿用现有部署。本次只更新启动配置，不自动重启服务、不启动训练或评估。

`docs/archive/LOCAL_DOCKER_EVAL.md` 保留历史命令和结果，其中提到的 `local-eval-6999.env` 已删除。归档 Compose 的 Python 入口已适配新包结构，但不作为日常入口。

## 回归验证

自动化回归测试放在工作区 `tests/`；真实数据及仿真运行检查继续保留在 `checks/` 和相应业务目录。

不加载模型的入口与调度测试：

```bash
python -B -m unittest tests.test_pipeline_entrypoints tests.test_batch_benchmark -v
```

在固定 OpenPI 训练环境中运行全部回归测试，包括小型 CPU TrainState 测试：

```bash
openpi/.venv/bin/python -B -m unittest discover -s tests -v
```

没有 JAX/Flax/Optax/OpenPI 时 TrainState 模块会报告跳过；入口及调度测试仅使用标准库。测试覆盖模块帮助、非法参数提前拒绝、独立工作目录下的路径解析、计划模式、batch 子进程参数与进程组清理。TrainState 使用小型 CPU 状态，不代表真实 GPU 训练或模型效果通过。

部署配置可在 WSL 中只解析、不启动容器：

```bash
cd /home/xiaodan/openpi_ws/openpi
docker compose --env-file examples/libero/.env -f examples/libero/compose.yml config --quiet
```

本次验证结果见 [REFACTOR_VALIDATION.json](validation/REFACTOR_VALIDATION.json)。原服务器和 WSL 验证记录保留在同目录，代表各自记录时的验证范围。
