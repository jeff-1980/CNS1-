# 对象 18 登记：Onward-AI/torch_isonet

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-21。

## 论文与代码

- 论文：Q. Xiang, X. Wang, Y. Song & L. Lei，*ISONet: Reforming 1DCNN for aero-engine system inter-shaft bearing fault diagnosis via input spatial over-parameterization*，Expert Systems with Applications 277 (2025) 127248，DOI 10.1016/j.eswa.2025.127248（仓库 `pdf/` 内附全文）。
- 仓库：`Onward-AI/torch_isonet`，commit `af5037e6025e2bd38e3db9684894c5e0eef7cbcd`（2025-05-12）。本地 `~/audit_ext/main_objects/18_Onward-AI__torch_isonet`。入口 `isonet_run.py` → `ljp/engine.py` 的 `Trainer.fit`。

## 头条表格

- 论文 Table 3（与 SOTA 比较，加权平均）：ISONet Precision 98.924 / Recall 98.922 / **F1 98.921**；1DCNN 98.419 / 98.414 / 98.412。正文：C5 通道、核长 9、batch 32 时 ISONet 准确率 98.92%（Table 1）。论文设置：HIT 数据随机分为训练 2412、测试 9648，200 轮，“每轮后保存测试准确率最高的参数”。
- Table 4（限样本，取 20% 为验证集、30 轮、5 次平均）另报 ISONet 采样比 0.1 时 99.39%。

## 方法组（1 个）

- 仓库只实现 ISONet（`ljp/models/isonet.py`），没有 1DCNN、CA-1DCNN、MSGF-1DCNN、Transformer 等对比模型。只有 **1 个方法组**：`ISONet|HIT`。H-B2 对本对象无组对可算。

## 配置（作者代码默认，`isonet_run.py` 的 `__main__`）

- 数据：仓库自带 `hit_data/`（`train_x.npy` 9648×2048、`test_x.npy` 2412×2048，3 类：健康、内圈、外圈）。`HIT(train_frac=0.1)`：每类按 `random_state=2024` 抽 10% 训练样本（约 965 个），测试集为 `test_x.npy` 全部 2412 个。
- 模型：`layer_cfg='4l064'`、`kernel_size=9`、`activator='mish'`。
- 训练：200 轮，batch 32，Adan lr 5e-4，`StepLR(step_size=60, gamma=0.1)`，交叉熵；`monitor='val_acc'`、`monitor_mode='max'`；`cudnn.benchmark=True`。
- **与头条表格设置不一致（只记录）**：仓库数据的训练/测试规模与论文相反（论文 2412 训练 / 9648 测试，仓库文件为 9648 / 2412），代码默认再取训练文件的 10%。本审计照作者代码默认运行，不交换文件；S6（复现）记为“设置与头条表格不同，仅作参考对照”。
- 唯一改动：`save_dir` 原为作者机器路径 `/works/xiang/test/`，改为审计工作目录（偏离记录）。

## 种子方案

- 作者代码不设任何种子（数据抽样用固定 `random_state=2024`，训练不设种子）。按预注册做 **10 次不设种子的重复**，记录里的 `seed` 字段为重复序号 0–9。

## 作者的选模规则

- `Trainer.fit` 每轮训练后在 `data.testloader`（即测试集，作者称 `val_`）上评估；`val_acc >= 历史最好` 时更新 `best_net` 与 `best_epoch`（平局取较晚的轮次），结束时报告 `best_val_acc` 与该轮的全部指标。论文正文同样写明保存测试准确率最高的参数。**作者口径 = 在报告集上取最好（author ≈ best_rep）**。
- **选模集**：无（没有独立验证集）。**报告集**：测试集 2412 个。
- 补丁：不改作者源码。审计驱动脚本照抄 `isonet_run.py` 的 `__main__`（只改 `save_dir`），`fit` 返回后由作者自己的 `history` 逐轮写入记录：`select_split=None`、`report_split="test"`，每轮记 `val_acc`、`val_loss`，加权 F1（`val_f1`）另存；`author_pick(best_epoch)`、`finish(best_val_acc)`；作者 `best_history` 中的 F1 另行核对。

## 数据来源

- 仓库自带 `hit_data/*.npy`（原始数据来自 Hou et al. 2023，https://github.com/HouLeiHIT/HIT-dataset）。

## 泄漏编码（只记录，不修复）

- 每轮在测试集上评估并按测试准确率保存最好参数，作者报告即为测试集最好值（选模与报告同一集合）。
- 仓库训练/测试文件如何从原始数据切分未给出代码，无法核查片段重叠。
