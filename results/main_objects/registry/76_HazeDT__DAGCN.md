# 对象 76 登记：HazeDT/DAGCN

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d 及 jeff 2026-10-24 的 6 小时规则。本登记在任何训练之前提交。登记日期 2026-10-25。

## 论文与代码

- 论文：*Domain Adversarial Graph Convolutional Network for Fault Diagnosis Under Variable Working Conditions*，IEEE TIM 2021，DOI 10.1109/TIM.2021.3075016（IEEE 9410617，全文未取得）。
- 仓库：`HazeDT/DAGCN`，commit `9be9099c1d2e5d04b2a6de8229b07c8ea86d2980`；README 链接该论文并给出引用，代码基于 UDTL 框架。入口 `DAGCN/train_advanced.py`（README 命令）。

## 头条表格

- 论文全文不可得，README 与摘要均无数值，**头条数值无法登记**（S6 记为“无法对照”）。

## 方法组（1 个）

- 仓库只实现 DAGCN（CNN + MRF-GCN + 域对抗 + MMD 结构损失），论文中的对比方法不在仓库中：`DAGCN|CWRU 12k DE 3hp->0hp`。

## 配置（README 命令，其余参数取代码默认）

- 命令：`--model_name DAGCN_features --data_name CWRU --transfer_task [3],[0] --last_batch True`。
- 数据：CWRU 12k 驱动端，10 类（正常 + 3 种故障 × 3 种尺寸），文件 `100, 108, 121, 133, 172, 188, 200, 212, 225, 237.mat`（3 hp，源域）与 `97, 105, 118, 130, 169, 185, 197, 209, 222, 234.mat`（0 hp，目标域），变量键 `X<编号>_DE_time`（编号 < 100 时为 `X0<编号>`），不重叠 1024 点切段、mean-std 归一化。文件取自对象 77 仓库附带的 CWRU 原始目录（目录结构与代码一致），抽查 5 个文件与本机另一份 CWRU 副本 md5 完全一致。
- 训练：300 轮，batch 64，Adam 1e−3、weight decay 1e−5，MultiStepLR(150, 250, 0.1)；`middle_epoch` 1（第 2 轮起加入对抗与结构损失）；源域、目标域各按 `random_state=40` 分层 80/20。

## 种子方案

- 代码不设 torch / numpy 种子（划分固定）→ **10 次不设种子的重复**，`seed` = 重复序号 0–9。

## 作者的选模规则

- 每轮在 `target_val`（目标域 20%）上评估；**`target_val` 准确率严格创新高时（从第 2 轮起）保存“best model”**，最后一轮也强制保存。作者口径 = 目标域测试集准确率第一个严格最大的轮（等于 best_rep 口径的定义，选模集即报告集）；`AUDIT_FINAL` 取作者最后一次严格创新高时保存的准确率。
- **选模集**：无独立选模集（作者直接用目标测试集）。**报告集**：`target_val`。另记 `source_val`（源域 20%）的准确率与损失。

## 依赖适配与补丁

- `iter_target.next()` → `next(iter_target)`（当前 PyTorch 迭代器无 `.next()`）；`np.float(` → `float(`（NumPy ≥ 1.24 已移除）。各 1 行。
- 钩子：模块级 `AUDIT_HOOK` / `AUDIT_FINAL`（由 driver 设置），每个阶段的 epoch 日志之后调用一次；作者保存 best model 处交出报告值。
- 环境：复制对象 68 的环境（torch 2.1.2、torch_geometric 2.8.1，`dropout_adj` 仍可用），补装 pandas、scikit-learn、tqdm。

## 泄漏编码（只记录，不修复）

- 目标域测试集同时用于选模（best model）；`dropout_adj` 在评估时也随机丢边（未传 `training`），评估本身带随机性。
