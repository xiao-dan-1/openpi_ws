# 重构后非训练脚本运行验证

日期：2026-10-27。环境：WSL Ubuntu-24.04、RTX 4070 Ti SUPER 16 GB、现有 `openpi_server` 与 `libero` 镜像。使用 `libero_lora_01/6999` 检查点；本轮没有启动训练、恢复训练或真实 batch 容量测试。

本机非训练流程运行通过。除入口和回归检查外，实际执行了数据处理、模型加载、动作推理、场景渲染和十任务闭环评估。

## 实测结果

| 项目 | 实际执行范围 | 结果 |
|---|---|---|
| 回归测试 | 入口、参数、路径、batch 调度与进程清理、版本兼容校验，共 16 项 | 全部通过 |
| `data.compute_norm_stats` | 官方统计入口，真实数据 128 帧、1280 个动作向量，4 个批次 | PASS；探测统计，不代表全量统计 |
| `data.check_config` | finetune LoRA 数据配置，真实 CPU 数据批次 2 个 | PASS；使用探测统计量，模型未初始化 |
| 数据多进程 | episode 0，0/2 个工作进程；统计各 128 帧，输入各 2 个 batch 16 批次 | 统计与变换后输入逐项一致 |
| `inference.policy_rpc` | 官方 WebSocket 服务/客户端、真实图像、模拟策略；3 个观测及错误检查点拒绝 | PASS |
| `inference.local_docker_serve` | 真实恢复 6999 的约 6.3 GiB 参数，GPU 策略服务启动 | PASS |
| `inference.check_policy` | episode 0、846、1692 的首/中/末帧，共 9 个真实观测 | 全部返回有限的 `(10, 7)` 动作 |
| `inference.serve_policy` | 通用入口真实加载同一模型；本地内存适配启动，再发送 episode 0 的 3 个观测 | PASS；适配方式与清单来源见下文 |
| `eval.check_simulation` | 真实场景加载、reset、初始状态、10 个稳定动作、两路相机与机器人状态 | PASS；相机 `(256, 256, 3)`，状态 `(8,)` |
| `eval.eval_libero` | `libero_spatial`，10 个任务，每任务 1 次，seed 7，CPU OSMesa 渲染 | 完成 10/10；成功 9/10；运行错误 0；视频 10 段 |
| `ops/setup_libero_env.sh` | Bash 语法、`--help` 与路径检查 | 通过；复用已有镜像，没有重新安装依赖 |

评估阶段耗时约 195 秒，不含检查点加载及前置真实观测推理。`9/10` 是单次流程检查的结果，不能代表稳定成功率或完整基准成绩。

## 运行中发现并修复的问题

### WSL 本地版本被原始提交校验拒绝

WSL OpenPI 当前提交为 `13a95ea16adeee33b7efd49f8f3dadc6aebe20b9`，在原始固定基线 `215abfb217dbac7d5f1273282331b9b1866c0479` 上增加了本地数据路径、Orbax I/O、依赖构建与 WSLg 查看器支持。已核对模型、策略、训练主脚本、统计主脚本及评估主脚本相对基线没有修改。

`common.config` 接受原始基线和明确审查过的本地提交，同时验证祖先关系并拒绝本地版本中未审查的核心代码改动。未知提交仍会被拒绝。运行清单同时记录预期基线和实际提交；安装脚本同步接受明确列出的版本。新增 5 项版本校验回归测试。

本次运行验证使用的是 `13a95ea16adeee33b7efd49f8f3dadc6aebe20b9`。随后 Git 保存部署配置生成 `eacc6a831df78bd8de7ef0804f48ef32f6de0052`，仅涉及 Compose 和查看器文档，也已加入明确兼容列表，并核对版本校验、Bash 语法和回归测试。原运行记录保留实际测试时的提交。

### 检查点读取并发导致宿主内存分配失败

恢复预算 3 GB、I/O 并发 8 时，TensorStore 从 Windows 挂载目录读取参数触发 `RESOURCE_EXHAUSTED / ENOMEM`。进一步实测发现最大单次恢复分配约 2.42 GB，因此 1 GB 和 2 GB 预算也不足。

采用以下设置后，检查点恢复、真实推理和整轮评估均通过：

```text
GPU_MEMORY_FRACTION=0.70
RESTORE_CONCURRENT_GB=3
FILE_IO_CONCURRENCY=1
```

已更新本地推理 CLI 默认值，以及 WSL 实际 Compose 的默认值、固定 `.env` 和查看器文档中的 I/O 并发设置。检查点和原始模型参数没有修改。

## 通用推理入口的验证边界

`serve_policy.main()` 使用测试目录中的 `check_generic_server.py` 启动。该启动器仅施加与本地服务相同的显存和 Orbax/TensorStore 内存限制；模型、参数、归一化、身份校验与通用入口逻辑均实际执行。检查点真实恢复后，3 个真实观测推理通过。

训练清单由本地推理入口按已有验证配置重建，带有清单来源说明；本轮没有取得服务器原始 `config.json`，也没有验证未施加内存限制的通用服务在本机的资源表现。服务器默认启动方式仍按服务器环境验收。

## 产物与清理

- [机器可读汇总](validation/RUNTIME_VALIDATION.json)
- [本次全部日志、命令、探测产物和测试启动器](../../checks/output/refactor_runtime_20261027T062046Z)
- [局部统计与运行记录](../../checks/output/refactor_runtime_20261027T062046Z/generated_probe_assets)
- [本地服务九观测推理结果](../../checks/output/refactor_runtime_20261027T062046Z/local_inference_probe_6999.json)
- [通用服务三观测推理结果](../../checks/output/refactor_runtime_20261027T062046Z/generic_inference_probe_6999.json)
- [十任务闭环结果](../../checks/output/refactor_runtime_20261027T062046Z/runs/pi05_libero/libero_lora_01/eval_6999/libero_spatial_20261027T063133121404Z/result.json)
- [十段任务回放](../../checks/output/refactor_runtime_20261027T062046Z/runs/pi05_libero/libero_lora_01/eval_6999/libero_spatial_20261027T063133121404Z/videos)

本次临时策略容器已停止并移除，测试端口已关闭；原有探测统计目录中的文件已按测试前备份恢复。生成的探测资产另外归档。原检查点只读使用，未运行训练或重新生成正式全量统计量。Windows 和 WSL 的代码及文档修复同步保存。
