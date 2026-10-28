# Q5：38 维训练、验证与服务器检查

沿用 LIBERO 的 π0.5 LoRA 冻结规则，视觉编码器继续参与训练；冻结视觉编码器的对照后续再做。所有完整模型加载、推理和梯度更新均在服务器执行，本机只做数据处理和 CPU 检查。

## 数据与模型

| 集合 | 轨迹 | 帧数 | 用途 |
|---|---:|---:|---|
| train v2 | 50 | 17,382 | 梯度更新和归一化统计 |
| validation v2 | 5 | 1,817 | 调参和检查点选择 |
| test v2 | 10 | 3,490 | 最终评估，身份与 v1 完全相同 |

五条 validation 按采集顺序分层，从原 55 条训练候选中以 seed=43 抽取。原十条测试轨迹锁定。三份副本各自从零编号、物理隔离；旧数据和统计保留作历史记录。validation/test 都使用 train 统计。轨迹隔离不代表跨场景泛化，相邻采集仍可能相关。

state=(38,)，actions=(10,38)，顺序为左臂7、右臂7、左手12、右手12。目标为30Hz的绝对关节位置，不额外做delta转换。期望单位rad，依据既有HybridJointCommand文档；数据元数据没有单位字段，源bag及执行端仍需现场核实。

一路头部RGB缩放至224×224，缺失腕部图像填零并掩码。discrete_state_input=True，完整任务/状态token上限200。PyAV设置保留。训练统计目录为 assets/pi05_q5/q5_pick_orange_to_basket_v2_train，文件SHA-256为 cc2a1440c128003812bbb07994b1bcff270c86ad525283e624cecd24cc23e2e5。

## 权重及冻结

本地pi05_base的动作投影为32维，Q5为38维。当前选择两个完整动作投影层（含偏置）重新初始化，其他非LoRA权重要求存在且形状匹配，新增LoRA使用模型初始化。完整投影重建是选定的迁移策略，并非唯一方法。

直接调用与LIBERO相同的 model.get_freeze_filter()：LLM非LoRA参数（含动作专家）冻结；LoRA、视觉编码器、完整动作投影层、时间MLP参与训练。

CPU检查会审计真实基础检查点全部元数据路径。服务器训练初始化会生成 weight_loading_report.json；模型检查入口仅在指定 --output-dir 时保存该报告，逐参数记录来源、形状、dtype和可训练状态。只允许动作投影及新增LoRA重新初始化。CPU元数据检查不能替代完整权重加载。

## 更新部署

服务器已有v1/PyAV/基线修复版时，上传 q5_validation_update_v1.zip 到工作区：

```bash
cd /mnt/sda/xd/openpi_ws
unzip -o q5_validation_update_v1.zip
openpi/.venv/bin/python -B -m pipeline.q5.check_config
```

包内包含Q5代码、必要依赖/测试、新v2数据和统计，不包含基础权重、Python环境和训练检查点。旧LIBERO入口及v1数据保留。新目录：

```text
datasets/q5/pick_orange_to_basket_v2_train/
datasets/q5/pick_orange_to_basket_v2_validation/
datasets/q5/pick_orange_to_basket_v2_test/
datasets/q5/pick_orange_to_basket_v2_split.json
assets/pi05_q5/q5_pick_orange_to_basket_v2_train/
```

直接使用准备包不需运行prepare_data。持有清理后的65条源数据且尚无v2产物时，可运行 python -B -m pipeline.q5.prepare_data；程序读取旧v1清单锁定测试身份，拒绝覆盖已有输出。--stats-only用于已划分但尚无新统计时。运行时路径按工作区解析，JSON中旧绝对路径只用于溯源。

CPU检查：

```bash
openpi/.venv/bin/python -B -m unittest tests.test_q5_pipeline tests.test_pipeline_entrypoints -v
openpi/.venv/bin/python -B -m pipeline.q5.check_config
```

检查包括三集合隔离、train/validation真实样本、全状态token未截断、同一训练统计下的归一化、PyAV/图像掩码、无delta转换、数值/时间/QC记录及真实基础权重元数据冻结。报告在 checks/output/q5/check_config.json，完整元数据清单为同目录 check_config_base_parameter_metadata.json。

## 服务器完整模型检查

等分配的GPU空闲后，检查完整权重、完整参数树及真实前向：

```bash
openpi/.venv/bin/python -B -u -m pipeline.q5.check_model --gpu 0
```

默认只在终端显示结果，不创建报告目录或文件。此入口会创建完整模型和优化器，不是轻量CPU检查，不保存检查点。三步真实更新也可直接运行：

```bash
openpi/.venv/bin/python -B -u -m pipeline.q5.check_model --gpu 0 --updates 3
```

需要保留报告时，显式指定 --output-dir；仅此时保存 weight_loading_report.json、model_check.json，以及检查期间的 INCOMPLETE 标记。指定的目录已存在则拒绝覆盖：

```bash
openpi/.venv/bin/python -B -u -m pipeline.q5.check_model \
  --gpu 0 --updates 3 --output-dir checks/output/q5/model_updates_v2
```

三步越过首步零学习率。检查loss/梯度有限，每个参数张量最多分布抽查32个元素，确认可训练组件更新、冻结参数抽样值不变；该抽样不等于全张量逐值证明。完整参数清单另由参数树报告提供。保存恢复另用smoke验收。

三步检查器的训练JIT必须与官方trainer保持一致，包括 donate_argnums=(1,)、输入/输出sharding和mesh上下文。donation允许新训练状态复用已不再使用的旧状态显存。初版检查器漏了这一设置，在本机16GB卡首次更新时出现输入/输出内存超限和OOM，不能用那次结果直接判定正式训练的最低显存。当前已对齐官方设置并释放单独前向检查的编译缓存；CPU小型回归验证了旧状态buffer被复用，但完整GPU更新仍需重测。

后续本机16GB重测仍在真实更新阶段OOM；完整加载和前向通过。修复保证检查器与官方内存处理一致，不保证16GB能容纳当前训练配置；24GB服务器的更新和保存恢复仍待实测。

指定 --output-dir 后，失败输出目录保留 INCOMPLETE 记录。保留报告重测时请换新目录，例如 --output-dir checks/output/q5/model_updates_v2_retry1；不保存报告时可直接重新执行。已有服务器版本可叠加 q5_check_model_memory_fix_v1.zip；不改变模型、冻结、学习率、batch或显存预留比例。

## 学习率和样本预算

| 参数 | smoke | finetune默认参考 |
|---|---:|---:|
| 首段上限 | 100 | 1000 |
| warmup | 10 | 1000 |
| schedule | 1000 | 30000 |
| peak/end LR | 5e-5/5e-6 | 5e-5/5e-6 |
| 保存/长期保留 | 50/50 | 1000/5000 |

AdamW、梯度裁剪1.0、无EMA/W&B/梯度累积、seed42。batch1/workers0仅为服务器试跑起点。schedule=30000不是承诺训练30000步。

finetune默认首段全在预热，入口要求显式 --continue-after-segment，否则报错。--steps是累计执行上限，--schedule-steps是调度长度，--warmup-steps可显式指定。启动时打印窗口样本预算及相对训练起点比例。1000步batch1约处理1000个窗口样本，相对17,382个起点约5.75%，不等于独立起点覆盖率，不适合判断最终效果。

短跑保存：

```bash
mkdir -p logs
nohup openpi/.venv/bin/python -B -u -m pipeline.q5.train \
  --gpu 0 --profile smoke --exp-name q5_v2_smoke_01 \
  --batch-size 1 --num-workers 0 --steps 100 \
  > logs/q5_v2_smoke_01.log 2>&1 < /dev/null &
```

同一smoke配置加 --steps 120 --resume 验证恢复。正式配置另起实验；若明确首段后继续同一计划：

```bash
openpi/.venv/bin/python -B -u -m pipeline.q5.train \
  --gpu 0 --profile finetune --exp-name q5_v2_lora_01 \
  --batch-size 1 --num-workers 0 --steps 1000 --continue-after-segment
```

如决定总共只训练1000步，显式选择较短预热和匹配长度，例如 --warmup-steps 100 --schedule-steps 1000 --steps 1000。该选项不会自动替换与LIBERO对齐的默认计划。后续总预算结合服务器吞吐、实际batch和验证结果确定。

恢复需保持模型、数据、统计、调度和代码一致；不能从旧55/10或不同冻结策略直接恢复为v2，使用新实验名。

## 验证集比较

已有v2服务器检查点后：

```bash
openpi/.venv/bin/python -B -u -m pipeline.q5.evaluate_validation \
  --gpu 0 --exp-name q5_v2_lora_01 --step 999
```

仅从五条validation中各等间距抽最多20个起点，固定种子和noise，使用训练统计，按轨迹等权报告双臂/双手的目标MAE与首动作MAE（rad），排除窗口末尾补齐帧。相同采样规则用于不同检查点；关注右臂/右手，避免左手常零稀释误差。它是离线误差，不是闭环成功率；test不用于选择检查点。

## 验证边界

本机已完成v2数据处理、CPU数据/元数据检查和单元测试。完整模型加载/前向、真实更新、保存恢复、验证预测尚未在服务器执行，不能标为通过。

原始bag和实时机器人不可达。图像、反馈和指令的独立时序及单位、执行端关节映射和绝对目标语义仍待核实。转换旁路最大状态年龄约8.86ms、指令年龄约20.55ms，不能替代原bag重验，也不应据此移动标签。
