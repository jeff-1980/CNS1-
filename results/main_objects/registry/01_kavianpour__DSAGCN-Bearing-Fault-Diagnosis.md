# 对象 01 登记：kavianpour/DSAGCN-Bearing-Fault-Diagnosis

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）。本登记在任何训练之前提交。登记日期 2026-10-20。

## 论文与代码

- 论文：Ghorvei, Kavianpour, Beheshti, Ramezani，*Spatial graph convolutional neural network via structured subdomain adaptation and domain adversarial learning for bearing fault diagnosis*，Neurocomputing 517 (2023) 44–61，DOI 10.1016/j.neucom.2022.10.057（2022 年 10 月上线）。
- 仓库：`kavianpour/DSAGCN-Bearing-Fault-Diagnosis`，commit `0ec074aa6af4e4a7e0070b90027fc184830f8057`（检索时 HEAD）；本地 `~/audit_ext/main_objects/01_kavianpour__DSAGCN-Bearing-Fault-Diagnosis`。
- **代码晚于论文**（按 DEVIATIONS 2026-10-20 通用规则，commit 461bc21）：仓库创建于 2026-05-22，最后推送 2026-06-05，比论文上线晚约 3.6 年。README 没说代码与论文结果不同，按规则照常纳入。文档与注释的措辞像按论文事后重写（config.py “follow Section 5.1 of the paper”，methods.py “faithfully reproduces the paper's definitions”，docs/method.md “paraphrased technical summary”）。协变量：代码晚于论文 ≈ 1310 天；进入“代码晚于论文 ≥ 1 年”敏感性分析的剔除集。

## 头条表格

- 论文全文审计方无访问权限（Elsevier 付费）；头条表格取自仓库 `assets/fig6_cwru_results.png`（作者从论文 Fig. 6 截取，图下附逐任务数值表）：CWRU 12 个跨负载任务 × 11 个方法，平均列 Proposed 99.65。
- 本对象选用任务 **B→D**（1 hp → 3 hp）：论文的 t-SNE 图（Fig. 7，仓库 README 图注 “t-SNE of the B→D CWRU task”）与仓库 quick start 都以 B→D 为例。Fig. 6 中 B→D 的值：

  | 方法组 | 论文 Fig. 6，B→D (%) |
  |---|---|
  | DSAGCN（Proposed Method） | 100 |
  | GC-MMD | 97.38 |
  | GC-MKMMD | 98.98 |
  | GC-CORAL | 99.32 |

- 记录：README 正文写“on one of the hardest tasks (B→D) it reaches 99.03% while the weakest comparison method sits near 57.95%”，但按 Fig. 6 的数值表，99.03 与 57.95（SVM）属于 **D→B**，不是 B→D。README 的这句话与图不一致；本登记以图中数值为准。

## 方法组（4 个）

正文强调的 LMMD 消融（README “swapping LMMD for MMD/MKMMD/CORAL (the GC-* variants) lowers accuracy”），同一骨干（CNN + 图分支 + 对抗判别器），只换第三项损失：

| 组 | `methods.py` 设定 |
|---|---|
| DSAGCN | use_graph=True, use_adversarial=True, da_loss=lmmd |
| GC-MMD | use_graph=True, use_adversarial=True, da_loss=mmd |
| GC-MKMMD | use_graph=True, use_adversarial=True, da_loss=mkmmd |
| GC-CORAL | use_graph=True, use_adversarial=True, da_loss=coral |

组名（Recorder group）：`<方法>|CWRU B->D`。

## 配置（全部作者默认，`config.py` 与 `train.py`）

- 数据集 CWRU 12 kHz DE，10 类；源 = 负载 B（1 hp，`--source 1`），目标 = 负载 D（3 hp，`--target 3`）。
- 预处理：窗长 1024、步长 475、逐窗 mean-std 归一化；每类每负载取前 250 个窗，随机打乱后 200 训练 / 50 测试（`np.random.default_rng(seed)`）。
- 训练：100 轮（`epochs_cwru`），batch 128，Adam lr 1e-3、weight_decay 1e-5，β=0.5（第三项损失），μ=1.0（对抗），GRL α=10、max_iter=10000，第 1 轮只做源域监督（warmup_epochs=1）。
- 调用：`train_one_task(cfg, ds, DEEP_METHODS[m], source=1, target=3, epochs=100, seed=s)`，与 `run_method` 内部一致。

## 种子方案

作者有种子机制（`set_seed(seed)`：numpy、torch、cuda；数据划分用 `default_rng(seed)`）。用种子 **0–9**（与 `run_method` 的 `seed0 + r`，r=0..9 一致；论文 §5.1 写每任务重复 10 次）。

## 作者的选模规则

- 每轮结束 `evaluate(model, target_test)`，并维护 `best_acc = max(...)`，但 `best_acc` 只在单任务模式打印，`run_method`（全部任务 / 基准表）报告的是训练结束后的 `final_acc = evaluate(model, target_test)`，即**最后一轮**。
- `author` 协议 = 最后一轮（第 100 轮）；本对象 author = last。
- **选模集**：无（作者没有验证集，也不按任何划分挑 epoch）。**报告集**：目标域测试集 `target_test`（负载 D，每类 50 个窗，共 500）。
- 补丁里 `select_split=None`，`minloss` 协议按 `protocols.py` 规则退回报告集损失。

## 数据来源

- `cfg.data_root = /home/jeffwork/论文CNS/data/cwru_12k_de`（只读；40 个实体 .mat）。
- 每个文件按“键名含 DE_time 的第一个键”读取。B→D 用到的 20 个文件逐个核对：每个文件只有一个 DE 键，读到的就是本文件编号的键（如 98.mat → `X098_DE_time`，100.mat → `X100_DE_time`）。
- **99.mat 问题**：99.mat 含 `X098_DE_time` 与 `X099_DE_time` 两个键，按本加载器会读到 `X098_DE_time`（即负载 B 的正常数据）。但 99.mat 属于负载 C（2 hp），**B→D 任务不读它**，本对象不受影响；只记录。

## 泄漏编码（只记录，不修复）

- 目标域的训练窗（无标签，用于对齐）与测试窗来自同一信号、窗长 1024 步长 475，相邻窗重叠约 54%，打乱后分到两边：目标测试窗与目标训练窗有样本级重叠。
- 每轮都在目标测试集上评估，`best_acc` 是在报告集上挑的（只在单任务模式打印，不进入作者报告的数）。
