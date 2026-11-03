# 对象 74 登记：wongsihan/FDR

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-24。按 jeff 2026-10-24 裁定按现状试跑。

## 论文与代码

- 论文：*Few-shot multiscene fault diagnosis of rolling bearing under compound variable working conditions*，IET Control Theory & Applications，DOI 10.1049/cth2.12315（DOAJ 列为开放获取，但 Wiley 页面返回 403，全文未取得）。
- 仓库：`wongsihan/FDR`，commit `c5db9b037acb678a8427b87cfc0b2afb057068d0`。README 第 3 行称 “Official PyTorch implementation”，第 5 行写 “The project has been restructured using a large model. If there are any issues with the code, please contact me.”——**未声明与论文代码不同，按规则不视为 E8**。入口 `train.py`（README “Original Training Script”）。

## 头条表格

- 论文全文不可得，README 无结果数值，**头条数值无法登记**；S6 记为“无法对照”。

## 方法组（1 个）

- 仓库只有提出的注意力编码器关系网络：`FDR-RelationNet|PU N09M07F10->N15M01F10 5-way 1-shot`。
- **记录（jeff 裁定写入偏离）**：论文描述的锚点编码器与因果 / 熵损失在 `train.py` 中被计算（第 176–178 行），但**从未加入总损失**（第 208 行只用 MSE），即论文的这一关键组件在仓库代码中不起作用；按现状运行。

## 配置（README 命令 `-s 1 -b 15 -e 100 -g 0`）

- 数据：PU 原始 `.mat`，`./data/<轴承>/<轴承>/<工况>_<轴承>_<n>.mat`（软链接自本机 `~/data_pu`，只读）。元训练：KA01、KA03、KA05、KA07、KA08（N09_M07_F10）；元测试：K001、KB23、KI14、KB27、KI04（N15_M01_F10）；锚点 KA01（N09）与 KI04（N15）。读取 vibration_1，不重叠 2048 点切段，|FFT| 前 1024 点。
- 训练：100 个 episode，Adam 1e−3，StepLR(1000, 0.5)（100 个 episode 内不衰减），梯度裁剪 0.5；每 5 个 episode 在元测试集上做 1000 个测试 episode。

## 种子方案

- 作者 `set_seed(3)`（random / numpy / torch）、文件顺序 `random.seed(1)`。按预注册改为 **种子 0–9**（替换 `set_seed(3)`，文件顺序不变）。

## 作者的选模规则

- 无验证集；每 5 个 episode 测试一次，另存测试集上最好的 checkpoint；**报告 `Final average accuracy` = 第 85、90、95、100 个 episode 测试准确率的均值**（`finalsum / 4`）。审计以最后一个评估点为 `author_pick`，并另行核对记录中这 4 个点的均值与作者报告值（`author_check_mean4`）。**选模集**：无。**报告集**：元测试集（目标工况）。

## 平台适配与补丁

- `train.py` 中 7 处 `.__iter__().next()` → `next(...)`（当前 PyTorch 已移除，jeff 2026-10-24 记为平台适配）。
- 钩子：取注入的钩子、`set_seed(3)` → 审计种子、每次测试后调用、最终均值交出。

## 泄漏编码（只记录，不修复）

- 无验证集，checkpoint 与报告值都来自目标测试集；BN 动量为 1 且模型从不切换到 eval 模式（关系网络常见写法）。
