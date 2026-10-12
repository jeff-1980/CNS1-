# 判定 C6-P：两个领域都是 INCONCLUSIVE → §6 未预先约定此组合，C6′ 在本试点中既未得到支持也未被否定

> 2026-10-12。预注册 `PREREG_C6_crossdomain_pilot.md`（SHA256 596b04ac…，训练前冻结；`analyze_c6p.py` 运行时校验通过）。数据 `c6p_d1.json`、`c6p_d2.json`（90/90 个实例，每个 3000 步，最后一轮权重），判定 `c6p_verdict.json`。本机 RTX A5000 Laptop：D1 08:56–09:39，D2 09:39–10:29。偏离：`DEVIATIONS_C6P.md`（冒烟先于冻结；训练期间无协议偏离）。

## 1. 结果（−8 dB 主导类别，每格 n = 5 个种子）

**D1 心电（PTB-XL；类 NORM / MI / STTC / CD）**

| 臂 | wdcnn | drsn | lstm |
|---|---|---|---|
| S_a（IMI / STTC / LAFB-LPFB） | MI 3，STTC 2 | NORM 3，STTC 2 | MI 4，STTC 1 |
| S_b（AMI / NST_ / CLBBB） | STTC 4，MI 1 | STTC 3，NORM 1，MI 1 | STTC 4，MI 1 |
| S_ALL | MI 3，STTC 2 | MI 3，STTC 2 | STTC 3，MI 2 |
| 标签 | **UNSTABLE** | **UNSTABLE** | **MIXED** +SUBTYPE_SPECIFIC |
| TV(a,ALL) / TV(b,ALL) / TV(a,b) | 0.0 / 0.4 / 0.4 | 0.6 / 0.4 / 0.4 | 0.4 / 0.2 / 0.6 |
| S_ALL 干净准确率（5 种子均值） | 0.602 | 0.599 | 0.602 |
| 集中度 ≥ 0.60 的实例比例 | 15/15 | 14/15 | 13/15 |

**D2 自然图像（CIFAR-100；类 scenes / fish / flowers / vehicles_1）**

| 臂 | cnn4 | resnet18 | vit_tiny |
|---|---|---|---|
| S_a（aquarium_fish / orchid / bicycle） | vehicles_1 5 | flowers 5 | flowers 3，vehicles_1 2 |
| S_b（flatfish / poppy / bus） | vehicles_1 5 | flowers 3，vehicles_1 2 | vehicles_1 4，flowers 1 |
| S_ALL | vehicles_1 4，flowers 1 | vehicles_1 3，flowers 2 | vehicles_1 3，flowers 2 |
| 标签 | **COVERAGE_INVARIANT** | **UNSTABLE** | **NO_COLLAPSE** |
| TV(a,ALL) / TV(b,ALL) / TV(a,b) | 0.2 / 0.2 / 0.0 | 0.6 / 0.2 / 0.4 | —（−8、−12 dB 均不足 60%） |
| S_ALL 干净准确率 | 0.827 | 0.733 | 0.685 |
| 集中度 ≥ 0.60 的实例比例 | 14/15 | 12/15 | 3/15（−12 dB：4/15） |

- **领域判定**：D1 = **INCONCLUSIVE**（2 个 UNSTABLE、1 个 MIXED，无任何标签达到 2/3）；D2 = **INCONCLUSIVE**（COVERAGE_INVARIANT、UNSTABLE、NO_COLLAPSE 各一）
- **§6 后果**：两个 INCONCLUSIVE 不在 §6 的四行之内（`NOT_COVERED`）。按预注册不作推论；下一步由 Jeff 决定
- 次要口径（§7.2）：−6 dB 下 D1 = MIXED / UNSTABLE / MIXED，D2 = COVERAGE_INVARIANT / UNSTABLE / NO_COLLAPSE；集中度 ≥ 0.90 口径下与主口径相同。两个领域的判定都不变

## 2. 次要（描述性，§7）

1. **纯噪声探针**（§7.1）：探针众数类命中 −8 dB 主导类别的比例 D1 = 0.67（常数预测器"STTC" 0.51），D2 = 0.69（常数预测器"vehicles_1" 0.62）。D2 中 cnn4 为 0.93，但 cnn4 的主导类别本身几乎恒为 vehicles_1
2. **吸引子落在基准类**（§7.4）：D1 4/45（全部为 drsn 的 NORM），D2 0/45。与轴承 CWRU/JNU 的 Normal 0/39 方向一致，与 PU Healthy 14/19 不一致
3. 吸引子只在少数类之间移动：D1 几乎全在 MI 与 STTC 之间（45 个实例中 41 个），D2 全在 vehicles_1 与 flowers 之间（45/45）

## 3. 读法与限制

1. **不支持的表述**：不能写"覆盖决定吸引子在心电或图像中同样成立"，也不能写"跨域不成立"。按 §5 的规则，两个领域都没有达到 SUPPORTS 或 CONTRADICTS
2. **主要障碍是种子间不稳定，不是没有崩溃**：除 vit_tiny 外，强噪声下的集中度都很高（D1 13–15/15、D2 12–14/15），但同一臂 5 个种子的主导类别经常在两个类之间分裂（一致性 0.6）。轴承上同样设计的种子一致性接近 1（E5：189/189 抽样一致、21/21 格子跨抽取稳定）。在这两个领域里，"失效吸引子"本身对种子更敏感
3. **D1 的模型学得较弱**：干净准确率约 0.60（机会水平 0.25），+10 dB 准确率只有 0.21–0.45，说明即使轻度噪声也已明显偏向少数类。在这一学习程度下，吸引子的种子不稳定可能部分来自模型本身欠拟合
4. **D2 唯一清楚的结果是 cnn4 的 COVERAGE_INVARIANT**：三臂都几乎全部落到 vehicles_1，与子类覆盖无关。这与 C6′ 的预测相反，但只有 1 个架构，达不到领域判定 CONTRADICTS 的 2/3
5. 本试点只有 5 种子 × 3 架构，TV 分辨率 0.2，只能发现大效应（§10）
6. 实现上预注册未规定、由本次选定的细节（在 `c6p_train.py` 中注明）：D2 的 ResNet-18 为手写的 CIFAR 版（BasicBlock [2,2,2,2]，3×3 stem，无 maxpool；本机无 torchvision）；ViT-tiny 用 3 个头、MLP 768、无 dropout；CNN4 为 32/64/128/256 通道；纯噪声探针的噪声种子为 7000 + seed
