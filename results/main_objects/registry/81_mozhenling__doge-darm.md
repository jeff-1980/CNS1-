# 对象 81 登记：mozhenling/doge-darm

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d 及 jeff 2026-10-24 的 6 小时规则；按 jeff 2026-10-25 两次裁定。本登记在任何训练之前提交。登记日期 2026-10-25。

## 论文与代码

- 论文：Mo, Zhang, Tsui, *Distance Aware Risk Minimization for Domain Generalization in Machine Fault Diagnosis*，IEEE IoTJ 11(22):37287–37301, 2024，DOI 10.1109/JIOT.2024.3441253（全文未取得；摘要无数值）。
- 仓库：`mozhenling/doge-darm`，commit `8cc91743734e465c479e8c88026c94b5956aacea`；DomainBed 风格代码，入口 `main_train.py`（单次运行）/ `main_sweep.py`（超参搜索）。

## 头条表格

- 全文不可得，**头条数值无法登记**（S6 记为“无法对照”）。

## 数据（jeff 裁定）

- 作者 Google Drive 公开包 `DARM-datasets.rar`（73 MB，`unrar t` 通过，sha256 `aaf73a57…cd59b7` 记于 `D:\audit_data\doge81`）。包内只有 CWRU_Bearing、CU_Actuator、PHM_Gear、UBFC_Motor，**没有 PU_Bearing 与 UO_Bearing**，因此 jeff 裁定由“只跑 PU”改为**跑包内的 CWRU_Bearing**。
- CWRU_Bearing：作者预处理的 `CWRU_DE_{0hp_1797rpm,2hp_1750rpm,3hp_1730rpm}_seg.mat`，4 类（正常、滚动体、内圈、外圈），段长 1000，每类每环境 150 段；无原始数据到 .mat 的转换脚本（原始 `.mat` 变量键无法核对）。训练时经软链接从 D 盘读取。

## 方法组（4 个，jeff 裁定）

- `DARM|CWRU_Bearing DG test env 0hp`、`ERM|…`、`IRM|…`、`Mixup|…`；`--test_envs 0`（0 hp 为测试域，2 hp、3 hp 为训练域）。
- 默认超参（`hparams_seed_id 0`），**不做超参搜索**：1000 步，每域 batch 32，Adam lr 1e−3，每 100 步一次 checkpoint（另含第 0 步与最后一步），holdout 0.2。

## 种子方案

- 作者 `--seed`（`seed_everything_update`：random / numpy / torch / cudnn deterministic）→ **种子 0–9**；划分种子 `trial_seed` 取默认 0（各域 80/20 划分固定）。

## 作者的选模规则

- 作者的 `IIDAccuracySelectionMethod`（训练域验证集），`is_end_acc=True` → **报告最后一个 checkpoint**；报告的测试数为测试域 in-split（80 %）准确率 `env0_in_acc`。超参搜索中由训练域 out-split 均值选超参（本次不搜索）。
- 作者在每个 checkpoint 评估全部划分并写入 `results.jsonl`；审计直接读取作者算出的数：**报告集** = `env0_in`；**选模集** = 训练域 out-split 准确率均值（`val`）；另记 `env0_out`。作者不计算损失（记 NaN，minloss 为 n/a）。checkpoint 稀疏（11 个），按 jeff 2026-10-24 的稀疏测试规则计算口径。

## 改动

- 无源码改动（driver 读取作者的 `results.jsonl`；`--skip_model_save` 不存权重）。环境复制对象 82 的环境（torch 2.1.2、torchvision 0.16.2）。

## 泄漏编码（只记录，不修复）

- 测试域在每个 checkpoint 都被评估（作者只记录，不用于选模）。
