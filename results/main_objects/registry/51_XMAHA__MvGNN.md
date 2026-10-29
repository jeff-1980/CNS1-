# 对象 51 登记：XMAHA/MvGNN

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-23。任务按 jeff 2026-10-23 裁定取 SEU（另一任务 XJTU Spurgear 为纯齿轮数据）。

## 论文与代码

- 论文：C. Li, L. Mo, C. K. Kwoh, X. Li, Z. Chen, M. Wu, R. Yan，*Noise-robust multi-view graph neural network for fault diagnosis of rotating machinery*，Mechanical Systems and Signal Processing 224 (2025) 112025，DOI 10.1016/j.ymssp.2024.112025（OpenAlex：closed，全文未取得）。
- 仓库：`XMAHA/MvGNN`，commit `2121f88486cd250385de53ad118a70d74075035c`（2026-10-13），README 列论文作者与引用。本地 `~/audit_ext/main_objects/51_XMAHA__MvGNN`。入口 `train.py`（README 命令）。

## 头条表格

- 论文全文不可得，README 无结果数值，**头条表格的数值无法登记**；S6 对本对象记为“无法对照”。

## 方法组（1 个）

- 仓库只发布 MvGNN：`MvGNN|SEU 9-class`。

## 配置（README 命令 + `configs/seu_paper.json`）

- 数据：作者公开 Google Drive 上处理好的 `117_small_20-50_multi_1024_TD_ordered.h5`（SEU 机械数据集，8 个传感器，长 1024 的窗口），用作者 `scripts/prepare_seu.py` 默认参数（`--start-window 0 --max-windows-per-condition 1000`）转为 `seu.npz`：去掉 4 个 `bearing_health` 工况，9 类（齿轮健康、4 种轴承故障、4 种齿轮故障）× 4 个转速，每类 4000 个样本，共 36000；并用 `scripts/validate_prepared_data.py` 核对。
- 训练：分层 80% / 10% / 10% 划分，Adam lr 0.005，StepLR(3, 0.5)，batch 16，最多 50 轮，早停（验证准确率，耐心 5），kNN 图 k = 1，ChebNet 1 层 K = 2，宽核 104。

## 种子方案

- 作者有种子机制（`--seed`，`set_seed` 设 random / numpy / torch，**同一种子也决定数据划分**），README 写“seeds 0–9 十次试验”。用 **种子 0–9**，经作者的 `--seed` 传入。

## 作者的选模规则

- 每轮在验证集上算准确率，`val_acc > best_val` 时保存 checkpoint；耐心 5 轮早停；结束后加载该 checkpoint 在测试集上算准确率并报告。**选模集**：验证集；**报告集**：测试集。
- 补丁：`train.py` 加模块级 `AUDIT_HOOK = AUDIT_FINAL = None`，在每轮 `val_acc` 算出后调用钩子（记录作者的验证准确率，并在 `isolate` 中补算验证交叉熵、测试准确率与交叉熵），在最终 `test_acc` 算出后交给 `AUDIT_FINAL`（共 4 行）；按作者 `>` 规则标 `author_pick`。

## 环境说明

- `torch_cluster` 只有对应特定 torch 版本的预编译包；本对象的环境由对象 02 的 venv（Python 3.10、torch 2.1.2+cu121）复制，补装 `torch_geometric`、`torch_cluster`（PyG 官方 torch-2.1.0+cu121 轮子）、`h5py`。

## 泄漏编码（只记录，不修复）

- 窗口来自同一条连续记录，按样本随机分层划分训练 / 验证 / 测试。
