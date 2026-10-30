# 检查脚本

这里保存数据审计、读取器检查和通信诊断。脚本不会启动正式训练。
Q5训练配置、完整模型检查和吞吐测试位于`pipeline/q5/`；LIBERO历史检查以`check_libero_`开头，不适用于Q5。

## Q5数据审计

依赖PyAV、NumPy、PyArrow和Pillow；读取整个数据集及视频，生成JSON、CSV和复核图片。输出必须是数据集之外的**新目录**，不覆盖已有结果：

```bash
./.venv/bin/python -B checks/check_q5_quality.py \
  datasets/q5/pick_orange_to_basket_v2_train \
  checks/output/q5/quality_train_01
```

- 退出码0、`status=PASS`：结构、数值和已有元数据统计检查通过。视觉提示仍可能出现在`warnings`，不代表抓取成功。
- 退出码1、`status=FAIL`：审计完成但发现错误；细节保存在`quality_metrics.json`，统计不一致也在终端摘要展示。
- 退出码1并保留`INCOMPLETE`：审计运行失败，目录内产物不完整，应查看终端错误。
- 退出码2：输入路径或输出目录不符合要求。重测使用新目录，源数据始终只读。

本审计检查转换后的数据和时序旁路记录，不重读原始bag，也不替代机器人执行接口检查。

## GPU通信

`check_gpu_communication.py`需要两张可用GPU；新服务器单卡环境无需运行它。用户输入物理编号，脚本通过`nvidia-smi`解析UUID，并以UUID选择CUDA设备。用法和计时边界见[GPU_COMMUNICATION.md](GPU_COMMUNICATION.md)。

回归测试使用临时数据和两个模拟CPU设备，不运行GPU任务：

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu ./.venv/bin/python -B -m unittest \
  tests.test_q5_quality tests.test_gpu_communication -v
```
