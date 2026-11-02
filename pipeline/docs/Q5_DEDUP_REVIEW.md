# Q5 公共能力复用：修改与验收

日期：2026-11-02（Asia/Shanghai）。本次范围为工作区源码优化、独立代码审查与本地回归验收。

## 修改、依据与产物

| 修改 | 依据 | 代码产物与行为 |
|---|---|---|
| 提取公共运行机制 | LIBERO/Q5 重复实现版本白名单、离线环境、缓存与线程设置 | `common/runtime.py`；两个入口包装器保留各自数据、缓存和设备约束。LIBERO 保留 evaluator 与祖先检查，Q5 保留精确 SHA 白名单的浅克隆兼容性 |
| 共享基础资产与官方脚本加载 | tokenizer 路径/SHA、基础 Orbax 文件和模块位置核对重复 | `common/runtime.py`；Q5 继续检查 32 维基础投影形状，统计来源校验独立保留 |
| 统一 GPU 校验和实际设备解析 | benchmark 曾接受 `2,02`；仅放宽解析还不足以让监控识别 UUID | `common/validation.py`、`q5/parallelism.py`、`q5/benchmark_fsdp.py`、`checks/check_gpu_communication.py`；支持索引、完整 UUID、唯一 UUID 前缀，拒绝歧义和同卡别名，保持选择顺序 |
| 每次预检只读取一次数据载荷 | 原训练调用链对每个训练 Parquet 哈希四次、视频两次 | `q5/runtime.py`、`q5/config.py`、`q5/train.py`、`q5/check_config.py`、`q5/evaluate_validation.py`；已验证 manifest 的摘要与统计记录比较，结果在同次调用中传递。无跨调用缓存；生成训练清单前仍轻量核对 split 和 norm_stats 身份 |
| 严格核对数据与统计指纹的一对一关系 | 独立审查发现重复路径可让两条记录引用同一 Parquet | split 的三个角色和统计指纹各自检查路径唯一性，再核对路径集合与逐项摘要 |
| 合并 Q5 训练计划解析 | CLI 与配置工厂分别维护 segment/warmup/decay 默认值 | `q5/schedule.py:resolve_schedule`；两个入口共享解析函数。保留 finetune 默认 1,000 步训练段、30,000 步衰减计划与 continuation 检查 |
| 复用官方 token 编码 | 三处手抄分桶与 prompt 格式，并访问私有编码器 | `common/tokenization.py`、`q5/check_config.py`；使用公开 `PaligemmaTokenizer(max_len=limit+1)` 和 mask 判定超限；合格样本长度保持精确。训练/验证各读自己的任务表，避免同 task ID 对应不同 prompt 时误报 |
| 使用官方可训练参数过滤器 | 审计手写过滤器与实际优化器的配置属性重复 | `q5/config.py` 的 weight loader 直接取得官方 `config.trainable_filter`；保留逐参数冻结审计 |
| 共享 QC 缺失语义 | 质量报告将缺字段视为 unknown，CPU 数据契约检查原来将其计为零 | `common/qc.py`、`checks/check_q5_quality.py`、`q5/data_contract.py`；报告保留 unknown/覆盖度，CPU 契约检查拒绝必需字段缺失、非有限值、负数与非整数计数 |

## 审查与保留边界

按用户上一轮指定的 GPT-6 Astra / high 进行独立审查。发现并修复了 UUID 监控链路、预检结果与训练清单身份的一致性、重复数据路径、验证集任务表四项问题；最后两项另外执行了独立回归测试。最终补扫未发现新的高优先级回归。

保留 38 维输入输出、单相机掩码、32→38 维投影迁移、LoRA 冻结规则、50/5/10 数据隔离、精确加权分位数、验证指标以及检查点标量计数核验。实际训练仍委托固定版本官方训练器。跨 LIBERO/Q5 的相同学习率与保存策略数值未强行合并；Q5 自身的重复默认值已收敛。

`prepare_data` 的统计计算前后仍分别校验文件，以发现计算期间的数据变化。本次优化的是同次训练/检查入口中重复的文件扫描，没有引入长期校验缓存。

QC 严格检查位于 `python -m pipeline.q5.check_config` 的 CPU 数据契约审计。直接执行 `pipeline.q5.train` 不会自动追加全量数值与 QC 扫描。

## 验收产物与复现

| 验收项 | 结果 |
|---|---|
| WSL 固定依赖环境全量回归 | 195/195 通过；0 失败、0 错误、0 跳过 |
| Windows 标准库与轻依赖补充检查 | 共 39 项，34 项通过；5 项官方依赖测试在此环境跳过，已在 WSL 全部执行通过 |
| 每个训练 Parquet 的启动校验哈希次数 | 4 → 1；每个训练视频 2 → 1，timing sidecar 同样只读取一次 |
| 代码差异空白检查 | `git diff --check` 通过 |
| 独立审查收尾 | 最后两项修正独立测试通过；补扫未发现新的高优先级回归 |

| 产物 | 用途 |
|---|---|
| `../common/runtime.py`、`../common/qc.py`、`../common/tokenization.py` | 新增公共实现；GPU 参数和通用断言加入现有 `common/validation.py` |
| `../../tests/test_q5_refactor.py` | 新增针对性回归：一次哈希、来源变化、重复路径、UUID 设备快照、QC 缺失、token 边界与各 split 独立任务表、官方过滤器和训练计划 |
| `validation/Q5_DEDUP_VALIDATION.json` | 最终状态、测试数量、环境版本、通过的测试 ID、实测哈希次数、修改源码 SHA256 与验收限制 |
| `validation/Q5_DEDUP_TESTS.txt` | 完整测试日志，包含官方双 CPU 设备 FSDP 和小型训练状态保存/恢复检查 |

在具备固定 OpenPI 依赖的工作区根目录执行：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/openpi/src" python -B -m unittest discover -s tests -v
git diff --check
```

实际验收使用现有 WSL Ubuntu-xd 的 Python 3.11 / JAX / Flax 环境，并从当前 Windows 工作区挂载路径读取修改后的源码。未安装新依赖、未改官方 OpenPI 源码。测试使用临时数据夹具与小型 CPU 模型；本次未加载真实 Q5 数据集和完整模型做 GPU 训练，也不代表机器人效果或任务成功率验收。
