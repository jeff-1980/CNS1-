# 对象 04 登记：mingzhangPHD/Transfer-Learning-for-Fault-Diagnosis（WDMAN）

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、缺失数据规则 f2dcd3c。本登记在任何训练之前提交。登记日期 2026-10-20。

## 论文与代码

- 论文：Zhang, Wang, Lu, Yang, Li, Liang，*A Deep Transfer Model With Wasserstein Distance Guided Multi-Adversarial Networks for Bearing Fault Diagnosis Under Different Working Conditions*，IEEE Access 7 (2019) 65303–65318，DOI 10.1109/ACCESS.2019.2916935（开放获取；审计方经 IEEE Xplore 网页读到正文与 Table 6）。
- 仓库：`mingzhangPHD/Transfer-Learning-for-Fault-Diagnosis`，commit `90ff8fe55cc88f20b7eb95cebed1a5093f8c3b02`（2021-12-16）；本对象只用其中 `WDMAN/` 目录。本地 `~/audit_ext/main_objects/04_mingzhangPHD__Transfer-Learning-for-Fault-Diagnosis/WDMAN`。仓库同时含另一篇（MAAN，PHM-Qingdao 2019）的代码，不在本对象范围。
- 代码与论文时间：作者 README 写明代码对应本文；仓库最后提交 2021-12，晚于论文约 2.5 年（GitHub 创建日期另以 search_raw 为准），按 461bc21 规则照常纳入并记协变量。

## 头条表格

论文 Table 6 “Deep Models Accuracy (%) for CWRU Dataset”，12 个跨负载任务 × 5 个深度方法（CNN、DDC、DAN、JAN、WDMAN），WDMAN 平均 99.39%。

本对象选用任务 **A→B**（负载 0 hp → 1 hp）：论文正文用 Table 7 单独给出 A→B 的逐类准确率，仓库默认参数（`CaseDE12K_dataset_0` → `CaseDE12K_dataset_1`）与作者附带的示例结果目录也都是 A→B。Table 6 中 A→B 的值：

| 方法 | 论文 Table 6，A→B (%) |
|---|---|
| CNN | 87.93 |
| DDC | 91.53 |
| DAN | 98.93 |
| JAN | 95.60 |
| WDMAN | 99.73 |

## 方法组（1 个）

仓库只实现了 WDMAN（`FD_DCTAN_model.py` 中 `class WDMAN`）；CNN、DDC、DAN、JAN 的代码不在仓库里。故只有 **1 个方法组**：`WDMAN|CWRU A->B`。H-B2 对本对象无组对可算。

## 配置（作者默认，`run_main_TAN.py` 的 flags）

- 数据：作者预处理好的 h5（Mendeley Data ykbc8hntzx v1，作者 README 给出），`CaseDE12K_dataset_0_nx1200_class10.h5`（源，A）与 `CaseDE12K_dataset_1_nx1200_class10.h5`（目标，B），键 `X`（原始波形，长度 1200）与 `Y`，10 类。每个文件内 `train_test_split(test_size=0.3, random_state=8)` 分为 70% / 30%。
- 训练：预训练分类器 `epoch_pre = 1500` 次迭代（论文写 1000，按代码默认），对抗阶段 `epoch = 20000` 次迭代（与论文一致），batch 64，Adam lr 1e-4、β1 0.5，两个域判别器 `n_critic=2`、初始 `n_critic_init=5`。目标域训练部分（70%，无标签）参与对抗。
- 评估：每 10 次迭代（`epochWriteThre`）在源域训练 / 测试、目标域训练 / 测试 / 全部上算准确率，并算 Wasserstein 距离。
- 运行方式：每个种子独立的 `--checkpoint_dir`（作者代码若在 checkpoint 目录里找到旧的预训练模型就会跳过预训练，必须每次新目录）。

## 种子方案

- 作者的种子写死：`tf.set_random_seed(1)`（建图与训练开始各一次），小批量打乱 `random_mini_batches_CNN1D(..., seed=3)`（两个阶段各一次）；数据划分 `random_state=8`。
- 本对象用种子 **0–9**：补丁把这两处写死的训练种子（TF 图种子与小批量打乱种子）改为由命令行传入，**数据划分 `random_state=8` 保持作者的固定值**（它决定的是数据集本身的划分，而不是训练随机性）。逐条写入偏离记录。

## 作者的选模规则

- 每次评估后：若**目标域测试集**（30%）准确率高于此前最好值，且源域训练与测试准确率都达到门槛（预训练阶段 ≥ 0.99，对抗阶段 ≥ 0.94），就保存模型并写 `BestAccuracy.txt`（预训练阶段）/ `BestAccuracy_DA.txt`（对抗阶段）。两阶段共用同一个“最好值”。作者附带的示例结果与论文报告的即为这个最好值（示例：第 2860 次迭代，Test_T = 0.984）。
- `author` 协议 = 满足源域门槛条件下，目标域测试集准确率第一个严格最大值所在的评估点。**选模集 = 报告集 = 目标域测试集**（源域门槛是附加条件）。
- 补丁：记录单位是“评估点”（每 10 次迭代一次，预训练 150 个 + 对抗 2000 个，共 2150 个，按顺序编号为 epoch 1…2150）；每个评估点记作者算出的目标域测试准确率（报告集）与目标域测试交叉熵（补算，只做前向推理，不改图、不改随机状态）；作者更新最好值时 `author_pick`；结束时 `finish(作者的最好值)`。`select_split=None`。

## 数据来源

- 下载自 Mendeley Data ykbc8hntzx v1（作者给出的公开链接），只取 A、B 两个文件（各 48 MB），按 Mendeley 的 SHA-256 核对，放在 `~/audit_ext/data/wdman/`。
- 这些 h5 是作者从 CWRU 预处理得到的，原始 `.mat` 读取哪个键不可见；99.mat（2 hp，负载 C）不在 A→B 任务中。

## 泄漏编码（只记录，不修复）

- 选模在报告集（目标域测试集）上做。
- 目标域的“训练部分”（无标签）与“测试部分”来自同一批切片的随机划分；切片之间是否重叠取决于作者的预处理，h5 中不可见。
