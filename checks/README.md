# 检查脚本

这里保存数据审计、读取器检查和通信诊断。脚本不会启动正式训练。
Q5训练配置、完整模型检查和吞吐测试位于`pipeline/q5/`。下表中的LIBERO检查均不适用于Q5，包括没有`libero`前缀的workers和transport脚本。

## 脚本职责与执行边界

| 脚本 | 适用对象与职责 | 前置条件、产物或副作用 |
| --- | --- | --- |
| `check_q5_quality.py` | Q5转换数据、视频、统计与时序审计 | 新建外部报告目录；源数据只读 |
| `check_gpu_communication.py` | 两张GPU的合成通信微基准 | JAX和两张可用GPU；临时子进程，只输出终端 |
| `check_libero_sample.py` | LIBERO原始数据抽样 | 本地数据；无缓存或报告文件 |
| `check_libero_loader.py` | LIBERO样本读取与显式输入变换 | 临时读取缓存，退出清理 |
| `check_libero_all_episodes.py` | 全部数值行和窗口索引规划 | 逐episode读取；不构建全量Arrow缓存 |
| `check_libero_dataset.py` | 完整Dataset构建及9个窗口抽样 | 复用或生成`--cache-dir`缓存，首次约35 GB |
| `check_libero_full_loader.py` | OpenPI全量读取及所有episode边界 | 工作区`.cache`内持久化数据链接与Arrow缓存 |
| `check_libero_normalization.py` | 生成并验证候选归一化统计 | 数据集外新目录；失败保留`INCOMPLETE` |
| `check_libero_training_input.py` | 候选统计下的tokenizer和模型输入变换 | 上一步产物、已校验本地tokenizer；复用或生成读取缓存 |
| `check_stats_workers.py` | 对比0/2个worker的统计结果 | LIBERO episode 0；工作区读取缓存 |
| `check_training_workers.py` | 对比0/2个worker的训练输入 | 先运行官方探测统计；工作区读取缓存 |
| `check_policy_transport.py` | mock策略的WebSocket内容往返 | LIBERO数据及官方服务端/客户端；临时本地服务，默认清理产物 |

LIBERO各项检查验证的层次不同，不要求全部串行执行。候选统计和训练输入检查必须按以下顺序运行，重测统计使用新目录：

```bash
./.venv/bin/python -B checks/check_libero_normalization.py \
  --data-root datasets/libero --output-dir checks/output/pi05_libero_h10_01
./.venv/bin/python -B checks/check_libero_training_input.py \
  --data-root datasets/libero --stats-dir checks/output/pi05_libero_h10_01
```

训练输入检查会重读数值列，核对统计所对应的数据摘要；候选统计不替代正式训练资产。`check_training_workers.py`使用另一套官方探测统计，先运行`python -B -m pipeline.data.compute_norm_stats --max-frames 128`，产物位于`checks/output/official_norm_smoke/assets/`。

## Q5数据审计

依赖PyAV、NumPy、PyArrow和Pillow；读取整个数据集及视频，生成JSON、CSV和复核图片。输出必须是数据集之外的**新目录**，不覆盖已有结果：

```bash
./.venv/bin/python -B checks/check_q5_quality.py \
  datasets/q5/pick_orange_to_basket_v2_train \
  checks/output/q5/quality_train_01
```

- 退出码0、`status=PASS`：结构、数值和必需元数据统计检查通过。视觉提示仍可能出现在`warnings`，不代表抓取成功。
- 退出码1、`status=FAIL`：审计完成但发现错误；细节保存在`quality_metrics.json`，统计不一致也在终端摘要展示。
- 退出码1并保留`INCOMPLETE`：审计运行失败，目录内产物不完整，应查看终端错误。
- 退出码2：输入路径或输出目录不符合要求。重测使用新目录，源数据始终只读。

本审计检查转换后的数据和时序旁路记录，不重读原始bag，也不替代机器人执行接口检查。
来源QC的缺失字段表示未知，不能按零异常解读；报告同时提供字段覆盖情况。

Q5审计、LIBERO产物保护和transport内容校验的回归测试使用临时样本，不需要真实数据或GPU：

```bash
./.venv/bin/python -B -m unittest \
  tests.test_q5_quality tests.test_libero_check_artifacts tests.test_policy_transport -v
```

这些测试使用真实小型Parquet/视频验证Q5路径，使用小型Parquet验证LIBERO数据摘要；OpenPI统计计算和WebSocket服务依赖在单元测试中被替代，仍需在目标环境运行对应脚本完成集成验证。

## GPU通信

`check_gpu_communication.py`需要两张可用GPU；新服务器单卡环境无需运行它。用户输入物理编号，脚本通过`nvidia-smi`解析UUID，并以UUID选择CUDA设备。用法和计时边界见[GPU_COMMUNICATION.md](GPU_COMMUNICATION.md)。

回归测试使用临时数据和两个模拟CPU设备，不运行GPU任务：

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu ./.venv/bin/python -B -m unittest \
  tests.test_q5_quality tests.test_gpu_communication -v
```
