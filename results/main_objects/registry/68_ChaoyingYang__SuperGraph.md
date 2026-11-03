# 对象 68 登记：ChaoyingYang/SuperGraph

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-24。

## 论文与代码

- 论文：C. Yang 等，*SuperGraph: Spatial-temporal graph-based feature extraction for rotating machinery diagnosis*，IEEE TIE 2021，DOI 10.1109/TIE.2021.3075871（不开放，全文未取得）。
- 仓库：`ChaoyingYang/SuperGraph`，commit `044518ffc8197e654971f47329adb3cf506d3067`。本地 `~/audit_ext/main_objects/68_ChaoyingYang__SuperGraph`。入口 `code/Proposed method/CWRU-GCN.py`（无命令行参数）。

## 头条表格

- 论文全文不可得，README 无结果数值，**头条表格的数值无法登记**；S6 对本对象记为“无法对照”。

## 方法组（1 个）

- CWRU 上仓库只有提出方法：`SuperGraph-ChebGCN|CWRU 11-class`。KAT 上另有 Keras CNN 与 KNN（非深度）对比脚本，不在默认任务上，不跑。

## 配置（作者脚本默认）

- 数据：仓库自带 `data/CWRU/12k Drive End Bearing Fault Data/` 的 44 个文件（0–3 hp；内圈 / 滚动体 / 外圈 @6 × 0.007 / 0.014 / 0.021 及 0.028 内圈、滚动体，共 11 类，**无正常类**），读 `X###_DE_time` 键；每个文件取前 30 个不重叠的 2048 点窗口，4 个负载合并，共 1320 个节点；每个窗口经 STFT 构造时空图，以图拉普拉斯特征值为节点特征。
- 训练：300 轮，全批量，Adam lr 0.015，三层 ChebConv（K = 3），逐层 weight decay 6e−4 / 3e−4 / 1e−4；随机划分节点 30% 训练 / 35% 测试 / 35% 验证。
- 运行设备：脚本在无 GPU 时用 CPU；本对象在 CPU 上运行（`CUDA_VISIBLE_DEVICES` 置空，避免与 GPU 队列冲突），只影响速度。

## 种子方案

- 脚本不设任何种子（`np.random.shuffle` 划分与权重初始化均随机）。按预注册做 **10 次不设种子的重复**，记录中 `seed` 为重复序号 0–9。

## 作者的选模规则

- 每轮计算训练 / 验证 / 测试准确率，`val_acc > best_val_acc`（严格 >）时把当轮测试准确率记为 `test_acc`，最后打印的 `test_acc` 即报告值。**作者口径 = 验证准确率首次达到最大值的轮**。**选模集**：验证节点；**报告集**：测试节点。
- 补丁（`code/Proposed method/CWRU-GCN.py`，3 行）：取注入的 `AUDIT_HOOK` / `AUDIT_FINAL`；每轮 `test()` 后调用钩子（记录作者的验证 / 测试准确率，并在 `isolate` 中补算验证 / 测试 NLL）；最后把 `test_acc` 交给 `AUDIT_FINAL`。
- 依赖适配（1 行）：`signal.stft(..., window='hanning')` 在当前 SciPy 中已移除该别名，改为同一窗函数的正式名 `'hann'`。

## 泄漏编码（只记录，不修复）

- **图结构由标签决定**：节点按文件（即类别）顺序排列，每连续 10 个节点两两全连接，测试节点只与同类节点（包括训练节点）相连。
- 同一记录的窗口随机分到训练 / 验证 / 测试。
- 每轮在测试集上评估（报告值按验证集选轮）。
