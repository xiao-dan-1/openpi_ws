# 环境与部署

独立 LIBERO 环境安装入口：在工作区根目录执行 `bash pipeline/ops/setup_libero_env.sh`。`--help` 只显示说明；不传该参数会按原流程安装 Python 和依赖。构建约束与安装脚本放在同一目录，工作区从脚本上两级解析。

当前 WSL 日常部署仍使用：

- `/home/xiaodan/openpi_ws/openpi/examples/libero/compose.yml`
- 同目录固定 `.env`
- 同目录 `run_eval_viewer_wsl.sh`

Compose 中策略服务通过 `python -u -m pipeline.inference.local_docker_serve` 启动，工作目录为 `/workspace`。完整 `pipeline` 包需挂载为 `/workspace/pipeline`。查看器继续使用现有官方评估适配入口。

2026-10-27 本机 6999 检查点实测：使用 `RESTORE_CONCURRENT_GB=3`、`FILE_IO_CONCURRENCY=1`、`GPU_MEMORY_FRACTION=0.70`。I/O 并发 8 曾触发 TensorStore 读取时宿主内存分配失败，已将本地推理 CLI 默认值、当前 Compose 默认值及固定 `.env` 的 I/O 并发统一为 1。最大的单次恢复分配约为 2.42 GB，恢复预算不能降到 1 或 2 GB。

安装脚本接受固定官方基线和已审查的 WSL 本地提交；使用已有容器镜像进行运行验证时，无需重新执行独立环境安装。保存本次 Compose 与查看器文档后的 OpenPI 提交为 `eacc6a831df78bd8de7ef0804f48ef32f6de0052`，已同步加入兼容列表。

历史 `compose.local-eval.yml` 位于 [docs/archive](../docs/archive/compose.local-eval.yml)。目录迁移与命令对照见 [REFACTOR.md](../docs/REFACTOR.md)。
