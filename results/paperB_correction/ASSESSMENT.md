# paperB（Wang et al. 2026, MST, DOI 10.1088/1361-6501/ae9438）：CWRU 99.mat 身份泄漏影响评估

2026-11-03，按 jeff 授权只读评估。**未启动任何训练，未修改任何现有文件**（E4 环境、`~/论文CNS`、`~/论文1`、桌面上的稿件目录都只读）。

## 读过的材料
- 已发表版的最终送审源码（R2）：`桌面/工作档案/论文/论文1my_mamba_paper/paperB-MST-6.16/7.30-MST-R2/MST-R2-1/MST-137114_R2_clean_source_NOT_anonymous/main.tex`
- 补充材料包：`~/论文1/paperB_supplement/`，包括 `README.md`、`code.zip`（`code/data/cwru_loader.py`、`splits_v2.py`）、`results/*.zip`、`hparam_control/report.md`
- 运行日志：`~/论文1/logs/{leakage_fix_cwru,cwru_clean_v2,leakage_fix_ablation,leakage_fix_E3,hparam_control}.log`
- 已有的缺陷审计：`~/论文CNS/results/AUDIT_CWRU_99mat.md`（2026-10-13）、`VERDICT_CINIT_FIX.md`
- paperB 复核报告：`~/论文1/EVIDENCE_REPORT_paperB.md`

## 1. 缺陷与 paperB 的暴露

- **加载器缺陷。** `cwru_loader._extract_de_signal` 返回 .mat 里第一个名字含 `DE_time` 的数组。CWRU 的 99.mat 里有 `X098_DE_time` 和 `X099_DE_time` 两个键，旧加载器读到的是前者，所以读出的“99.mat”与 98.mat 逐字节相同。
- **paperB 的 CWRU 同工况设置。** 4 类（Normal、IR、OR、Ball，0.007″）× 4 个负载，共 16 个文件；用 `splits_v2.build_cwru_splits` 按源文件分组划分，`split_seed = 42`。
  - Normal 的分配：97、99 → 训练，100 → 验证，98 → 测试。`AUDIT_CWRU_99mat.md` §1 核定，16 文件与 40 文件两套的分配相同。
  - 日志核对：`verify_no_leakage` 报告文件数 train/val/test = 8/4/4，窗口数 2844/1656/1653，与上述设置一致。
- **后果。** 测试集里 944 个 Normal 窗口全部来自 98.mat，在训练集里都有逐字节副本（训练用的“99.mat”就是 98 的信号）。Normal 占测试窗口的 944/1653 = 57%。
  - paperB 的 `verify_no_leakage` 按“文件 + 区间”检查，两个文件名不同，所以检查通过，没有发现这个问题。
  - 例：`main_awgn_v2/wdcnn/seed0` 在 −8 dB 时 Normal 行为 944/944 全对，其余类别大量误判。这和“记住了 Normal”相符，但还不能据此下结论。
- **不受影响的部分：**
  - CWRU **跨负载**（训练 0 hp、测试 3 hp，文件 97 与 100，不涉及 98/99）；
  - JNU（Mode A、Mode B）、PU：数据集不同，审计核定无重复。

## 2. 受影响与不受影响的表、图和结论

| 论文中的位置 | 内容 | 数据来源 | 是否受影响 |
|---|---|---|---|
| Table `tab:datasets`、`tab:baselines` | 数据集与架构说明 | — | 否 |
| Fig. `fig:framework` | 流程图 | — | 否（图注里“leakage-audited”的措辞需要加限定） |
| **Table `tab:incidence`** | −8 dB 崩溃发生率（NA−），各数据集 × 架构 | CWRU 行：`cwru_clean_v2`、`ablation_v3/mamba2_no_noise_train` | **是（CWRU 行）** |
| **Fig. `fig:heatmap`** | 跨数据集崩溃热图 | 同上 | **是（CWRU 行）** |
| **Fig. `fig:confusion`** | 三个代表性崩溃格的混淆矩阵 | 面板 1：CWRU 1D-CNN −8 dB | **是（面板 1）**；JNU、PU 面板不受影响 |
| **Table `tab:geom_centrality`** | CWRU Mamba-2 倒数第二层各类中心距离 | `ablation_v3` 的 checkpoint | **是** |
| **Fig. `fig:cosine`(a)** | CWRU Mamba-2 编码器噪声不动点与分类权重的余弦 | `mechanism/E3_v2`（用 CWRU 留出测试集与 checkpoint） | **是（面板 a）**；(b) JNU 不受影响 |
| **Fig. `fig:tsne`** | CWRU Mamba-2 NA−/NA+ 特征可视化 | CWRU 测试集与 checkpoint | **是** |
| Table `tab:or_blind_spot`、Fig. `fig:patch` | OR 盲点 | CWRU 跨负载 0→3 hp | 否 |
| **Table `tab:rescue`、Fig. `fig:rescue`** | 噪声感知训练的挽救效果 | CWRU 格：`main_awgn_v2`（NA+）对 `cwru_clean_v2`（NA−） | **是（CWRU 格）** |
| **§ `sec:hparam`（超参数对照）** | `cwru_mamba2` 18 组超参数 × 3 种子 | `hparam_control`，CWRU `splits_v2` | **是（CWRU 部分）**；JNU 三个对象不受影响 |
| § `sec:hparam` 中 VibrMamba patch 与 ConvEmbedding 的比较 | −4 dB 跨负载 CWRU | 跨负载 | 否 |
| 正文和摘要里的计数 | 25 个崩溃格（CWRU 17 个）、22/25 IR、浓度分布、24/25 种子一致等 | `revision_numbers.zip/collapsed_cells.csv` | **是（CWRU 的 17 格）** |
| 补充材料 | `collapsed_cells.csv`、`confusion_matrices/index.csv` 的 CWRU 行、`threshold_sensitivity` | 由以上结果派生 | **是（CWRU 部分）** |

## 3. 已有的旁证（来自 CNS 项目，不是 paperB 本身）

`~/论文CNS` 的参考协议沿用 paperB 的训练函数，并逐格复现了 paperB `cwru_clean_v2` 公开的混淆矩阵（E4，150/150），所以和 paperB 处于同样的暴露状态。该项目已做过两项检验：
1. **只换测试集**（`rescore_leakfree.py`）：用已有权重，把 Normal 测试窗口换成训练中没见过的 100.mat 或真实的 X099 信号。−8 dB 下的主导类别 180/180 不变，各格崩溃事件数也不变；+10 dB 准确率差 ≤ 0.001。
2. **修复加载器后重训**（`VERDICT_CINIT_FIX.md`，预注册）：wdcnn、drsn 判为 LEAK_ROBUST，崩溃方向逐实例一致 20/20；+10 dB 准确率变化 ≤ 0.002。

**含义：** paperB 的核心现象（强噪声下塌缩到 IR 等吸引子类）大概率不受影响。但 paperB 自己的格子和数字（尤其是 Normal 召回率、高 SNR 准确率、各格的均值 ± 标准差）**没有被直接检验过**，要更正，就需要用 paperB 自己的配置重跑。

## 4. 需要重跑的配置、次数和 GPU 时间

修复方式与 CNS 项目相同：加载器优先取与文件号一致的 `X{nnn}_DE_time` 键。文件到划分的分配和测试窗口都不变，只有训练用的“99.mat”从 X098 换成真实的 X099。每次重跑都要在独立的工作副本里进行，不改 `~/论文1`。

GPU 时间按 paperB 原日志里逐次运行的实际耗时求和，硬件是本机 RTX A5000 Laptop：

| # | 实验 | 配置 | 次数 | 原耗时 | 支撑哪些表图 |
|---|---|---|---|---|---|
| R1 | `main_awgn_v2`（NA+，同工况） | 7 个架构 × 5 个种子，100 轮，SNR 扫描 | 35 | **4.25 h** | `tab:rescue` 的 NA+ 端、`fig:rescue` |
| R2 | `cwru_clean_v2`（NA−） | 6 个架构 × 5 个种子 | 30 | **2.06 h** | `tab:incidence`、`fig:heatmap`、`fig:confusion` 面板 1、`tab:rescue` 的 NA− 端、17 个 CWRU 崩溃格 |
| R3 | `ablation_v3`（Mamba-2：full / no_noise_train / no_conv_embed） | 3 个变体 × 5 个种子 | 15 | **3.03 h** | `tab:incidence` 中的 Mamba-2 NA−、`tab:geom_centrality`、`fig:cosine`(a)、`fig:tsne` 所用的 checkpoint |
| R4 | `hparam_control` 的 `cwru_mamba2` | 18 组超参数 × 3 个种子 | 54 | **6.58 h** | `sec:hparam` 的 CWRU 部分 |
| R5 | 机制与派生量 | E3_v2 余弦、t-SNE、中心距离、混淆矩阵索引、`collapsed_cells`、阈值敏感性 | 只推理 | 估计 < 0.3 h | 对应的图和补充表 |
| | **合计** | | **134 次训练** | **约 15.9 GPU 小时** | |

**分级建议：**
- **第 0 步（不需要训练，几分钟）：** 和 CNS 项目一样，用 paperB 已存的 checkpoint（`~/论文1/results/main_awgn_v2/*/best_seed*.pt` 等）只替换 Normal 测试窗口，重新推理，先看主导类别和准确率会不会变。这一步只回答“测试端记忆影响多大”，不能代替训练端重跑。
- **最小更正集：R2 + R3 + R5，约 5.4 h。** 覆盖 NA− 崩溃发生率、热图、混淆矩阵、机制图，也就是论文的核心证据。
- **完整更正集：R1–R5，约 15.9 h。** 再加上挽救效果（R1）和 CWRU 超参数对照（R4）。

按审计规则（单次运行 ≤ 2 GPU 小时，单个对象合计 ≤ 6 小时，否则停下汇报），R4 单项超过 6 小时。跑之前要你单独确认，或者拆成几批。

## 5. 对更正的影响判断（事实层面）

- **一定要改的：**
  - 方法部分“所有同工况结果都经过泄漏审计、互不重叠”的表述，必须加上 99.mat 的例外；
  - 补充材料里 CWRU 那部分数字。
- **大概率不变的：** 核心定性结论，即强噪声下塌缩到 IR/Ball、噪声感知训练能挽救（依据第 3 节的旁证）。
- **可能变的：** 各 CWRU 格的准确率和标准差（Normal 是测试集的 57%）；浓度刚过或刚低于 90% 阈值的格（如 `cwru_clean_v2/lstm@-8dB`，种子间不稳定），可能让“25 格”这个计数变化。
- **要不要更正、用哪种形式（corrigendum 或 addendum）由作者决定。** 本评估只给出范围和成本。

## 6. 未核实或需要注意的事项
- 已发表的 Version of Record 和 R2 源码是否逐字一致，本次没有核对（`EVIDENCE_REPORT_paperB.md` 记载 R2 是送审终稿）。
- `fig:tsne` 和 `tab:geom_centrality` 用的是哪一批 checkpoint，按图注和 `mechanism.zip` 推断为 `ablation_v3`/`main_awgn_v2`，没有逐行核对绘图脚本。
- 原日志里的耗时是当时的实际耗时；本机 GPU 后来常处于温控降频状态（`VERDICT_CINIT_FIX.md` 有记录），实际可能慢 10%–30%。
