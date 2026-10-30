# 环境与部署

## GPU 与进程监控

在目标 Linux GPU 服务器的工作区根目录运行（Python 3.9+，只需标准库与已有的 `nvidia-smi`）：

```bash
python3 -m pipeline.ops.gpu_watch                  # 每 2 秒刷新全部 GPU
python3 -m pipeline.ops.gpu_watch --gpus 0,2 --interval 1
python3 -m pipeline.ops.gpu_watch --once           # 单次紧凑表格
python3 -m pipeline.ops.gpu_watch --once --details | less -S # 完整详情；方向键横向滚动，q 退出
python3 -m pipeline.ops.gpu_watch --json           # 单次结构化快照
python3 -m pipeline.ops.gpu_watch --plain          # 持续紧凑文本输出，不启用交互分页
python3 -m pipeline.ops.gpu_watch --details        # 以完整详情视图启动
```

也可以单独复制 `gpu_watch.py` 到服务器，执行 `python3 gpu_watch.py`，无需安装本管线或训练依赖。
常用 Bash 快捷命令（将路径替换为实际服务器路径，加入 `~/.bashrc`）：

```bash
alias gpu='python3 /path/to/openpi_ws/pipeline/ops/gpu_watch.py'
```

顶部显示 GPU 编号、名称、已用/总显存、温度和功耗/上限，数值列右对齐。
默认进程区为紧凑表格：PID、用户、类型、显存和 TASK 摘要，每张卡仍按显存占用降序排序。
Python 任务摘要缩短解释器和脚本目录，保留 server/client、--worker、--exp-name 等任务参数，只省略组标题已表达的 --gpu/--gpus。
Python 模块保留 `-m 模块名`，Xorg 简写为 `Xorg`；未知启动器、`python -c` 或无法解析的命令回退到原命令。
交互表格宽度不足时 TASK 用 `...` 标记省略，可按 `d` 查看完整详情；非交互输出不截断 TASK。
权限错误在紧凑视图按唯一 PID 汇总提示一次；完整 CWD、EXE、PPID、运行时长和逐字段原因保留在详情视图中。
交互式持续监控固定顶部 GPU 概览，进程区按终端大小分页；紧凑表格跨页时重复 GPU 标题和表头。
按 `d` 切换紧凑/详情视图并返回第一页；详情视图自动换行，完整命令不截断。
`n/p` 或 PgDn/PgUp 翻页，空格暂停/恢复，`q` 或 Ctrl+C 退出；暂停时冻结整帧数据，仍可翻页及切换视图。
终端缩放后重新分页；高度不足以容纳 GPU 概览和详情时显示明确提示，可扩大窗口或使用 `--once`。
持续模式的采集在后台进行，慢查询期间仍可操作键盘；同时最多一条采集查询。
临时查询超时、驱动错误或 XML 解析错误会自动重试，显示 `STALE`、最后成功采样时间及失败详情，保留上一帧。
首次采集失败时显示 `NO DATA`；恢复成功后清除错误状态。失败后至少等 1 秒再重试，避免快速空转。
无效 `--gpus` 和缺失 `nvidia-smi` 仍直接报错退出；`--once`、`--json` 失败也直接退出，不重试。
非交互终端、`TERM=dumb`、缺少标准库可选模块 `curses` 时使用文本输出；也可显式指定 `--plain`，追加 `--details` 可输出完整详情。
单次输出省略底部操作提示；完整选项通过 `--help` 查看。
终端中的显存统一用 `G`（GiB，即 1024 MiB）显示，整卡与进程显存、功耗和温度都保留一位小数。
例如 `1852 MiB → 1.8G`、`15.08 W → 15.1W`、`31 C → 31.0C`；小于 0.05G 的显存会四舍五入显示为 `0.0G`，未知值仍为 `N/A`。
`--json` 保留驱动原始数值和单位，供程序进一步处理。
下方按 GPU 编号升序分组，每组按该卡的原始显存占用降序排列，相同占用按 PID 升序，未知显存排在最后。
排序在四舍五入之前完成；同一 PID 跨卡时出现在每张相关卡的组内，显示该卡的显存占用，但进程元数据只读取一次。
空卡仍保留分组并注明没有返回记录。每条详情展示进程类型、用户、父 PID、运行时长、完整启动命令、工作目录和解释器/可执行文件。
同时展示计算进程 C、图形进程 G 及驱动报告的其他类型。JSON 维持原有结构，进程元数据按 PID 去重。
显存占用不等同于正在计算。
进程列表为空时仅表示 `nvidia-smi` 未返回记录，不能据此认定 GPU 空闲，尤其需要注意 WSL 的进程可见性限制。

工具通过 `nvidia-smi -q -x` 读取 GPU 和进程，再从服务器 `/proc/<PID>` 读取进程详情；详情视图命令不截断，终端宽度不足时自然换行。
必须在对应 Linux 主机运行：本地 Windows 无法解析远程 Linux PID，容器内也可能无法看到宿主机 PID。
权限不足、进程退出或 PID 不可见时显示原因，不会用可执行文件名冒充完整命令；需要宿主机视角时在宿主机执行。
`N/A` 保留为未知值；驱动报告的进程显存之和不保证等于整卡已用显存。
终端较短时用 `--once --details` 配合滚动或 `less` 查看，或用 `--gpus` 缩小范围。默认只读且不保存报告。

## 环境安装与当前部署

独立 LIBERO 环境安装入口：在工作区根目录执行 `bash pipeline/ops/setup_libero_env.sh`。`--help` 只显示说明；不传该参数会按原流程安装 Python 和依赖。构建约束与安装脚本放在同一目录，工作区从脚本上两级解析。

当前 WSL 日常部署仍使用：

- `/home/xiaodan/openpi_ws/openpi/examples/libero/compose.yml`
- 同目录固定 `.env`
- 同目录 `run_eval_viewer_wsl.sh`

Compose 中策略服务通过 `python -u -m pipeline.inference.local_docker_serve` 启动，工作目录为 `/workspace`。完整 `pipeline` 包需挂载为 `/workspace/pipeline`。查看器继续使用现有官方评估适配入口。

2026-10-27 本机 6999 检查点实测：使用 `RESTORE_CONCURRENT_GB=3`、`FILE_IO_CONCURRENCY=1`、`GPU_MEMORY_FRACTION=0.70`。I/O 并发 8 曾触发 TensorStore 读取时宿主内存分配失败，已将本地推理 CLI 默认值、当前 Compose 默认值及固定 `.env` 的 I/O 并发统一为 1。最大的单次恢复分配约为 2.42 GB，恢复预算不能降到 1 或 2 GB。

安装脚本接受固定官方基线和已审查的 WSL 本地提交；使用已有容器镜像进行运行验证时，无需重新执行独立环境安装。保存本次 Compose 与查看器文档后的 OpenPI 提交为 `eacc6a831df78bd8de7ef0804f48ef32f6de0052`，已同步加入兼容列表。

历史 `compose.local-eval.yml` 位于 [docs/archive](../docs/archive/compose.local-eval.yml)。目录迁移与命令对照见 [REFACTOR.md](../docs/REFACTOR.md)。
