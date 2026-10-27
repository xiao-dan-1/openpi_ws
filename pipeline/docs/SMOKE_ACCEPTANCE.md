# 最小训练与评估闭环验收记录

根据用户提供的服务器日志验收。初始权重为 pi05_base；实验 libero_smoke_01 使用单张 RTX 3090、LoRA、批大小 1，先训练 100 次更新，再恢复到累计 120 次更新。官方配置为 pi05_libero，固定提交 215abfb217dbac7d5f1273282331b9b1866c0479。

| 环节 | 验收结果 | 关键产物/脚本（路径相对于服务器 openpi_ws） |
|---|---|---|
| 数据验证 | 1693 episodes、273465 frames、40 tasks；原始数据、窗口、变换检查通过 | checks/check_libero_*.py |
| 官方统计量 | 处理 273440 帧，按批丢弃末尾 25 帧 | pipeline/data/compute_norm_stats.py；assets/pi05_libero/libero/norm_stats.json |
| 正式配置输入 | 两个真实 CPU 批次通过 | pipeline/data/check_config.py |
| LoRA 训练与保存 | 100 次更新，loss/梯度有限，检查点保存完成 | pipeline/train/run.py；logs/libero_smoke_01.log |
| 恢复训练 | 恢复 99，继续更新并保存 119 | logs/libero_smoke_01_resume120.log；checkpoints/pi05_libero/libero_smoke_01/119/ |
| 仿真依赖与渲染 | Python 3.8.20，125 个包兼容；OSMesa 单场景检查通过 | pipeline/ops/setup_libero_env.sh；pipeline/eval/check_simulation.py |
| 检查点服务 | LoRA 参数及检查点统计量加载通过 | pipeline/inference/serve_policy.py；logs/serve_libero_smoke_01_119.log |
| 真实观测推理 | 九个观测通过，动作均为有限值 (10,7) | pipeline/inference/check_policy.py；runs/pi05_libero/libero_smoke_01/inference_probe_119.json |
| 闭环仿真 | spatial 十个任务各一次，完整执行，无运行异常 | pipeline/eval/eval_libero.py；下列评估目录 |

评估输出目录：

    runs/pi05_libero/libero_smoke_01/eval_119/libero_spatial_20261023T163031661428Z/

其中 result.json 为结构化结果，videos/ 为回放。日志为 logs/eval_libero_smoke_01_119_spatial.log。

本轮运行约 17 分 17 秒，10 个 episode 全部完成，任务成功率为 0/10。PASS 表示评估程序完整执行，模型任务表现尚未通过验收。当前结果覆盖 spatial 一套、每任务一个固定初始状态；尚未评估其余三套任务或多次重复试验。

模型只完成 120 次更新，训练预算很小；仍需检查失败回放和更长训练的结果，不能仅凭有限训练步数解释所有失败。

下一阶段先查看失败回放，核对机器人响应与抓取行为，再开展新实验 libero_lora_01 的正式 LoRA 训练和独立评估。默认正式训练 30000 步，模型效果仍需实测。当前服务持续占用 GPU，正式训练时需安排显卡资源，或先正常停止自己启动的策略服务。

正式统计量 SHA256：

    7e19abf21b2f8837e1e27c468f4021efe359e38a9f6158acb83b6f978dad6b5a

详细日志验收数据见 validation/SERVER_VALIDATION.json。WSL 的模拟策略通信测试与服务器上的真实检查点推理分开记录，见 validation/WSL_VALIDATION.json。
