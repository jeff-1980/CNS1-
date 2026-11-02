# NC / SA 子刊评估（按 ncsa-mech-subjournal skill，2026-11-02）

依据：`results/main_analysis/SUMMARY.md`、`RESULTS_DRAFT_v1.md`（3.1–3.8）、`results/static_coding/ai_coding/analysis*.json`。引言、讨论、图尚未写，以下是按现有结果给出的判断和骨架。

> **2026-11-02 更新**：第 4 节的“虚高预测模型”已按事先登记的计划检验（`results/inflation_law/`，90c7b49、c1be41f），未达判据（检验 A R² 0.47 < 0.5，检验 C R² 0.35 < 0.4），NC 定律路线不成立。明确说明选模规则的论文数已更正为 2–10/79（v1.1 范围）。

## 1. 类型与创新级别

**类型 F（纯计算/数据），评估类杠杆**：用大规模真实复跑，检验领域内一个普遍做法（报告“最好的一轮”）在什么条件下成立、什么条件下失效。

**级别：T2 的边缘。** 不是 T1（没有几十年的老问题，也没有推翻普遍认识）；不是 T4（不是把已有方法搬到新对象上）。

一句话卖点：
> 这是一篇 F 类的 T2（边缘）文章，核心卖点是：第一次用预注册的大规模复跑（83 个代码–论文对中可运行的 38 个、880 次训练）证明，“报告最好一轮”在接近饱和的基准上几乎无害，在难任务上会把准确率抬高数个百分点；而 79 篇随机抽样论文中，只有 2–10 篇（3–13%，按两模型分歧取范围）在正文明确说明自己报告的是哪一轮。

**纯计算稿件判定清单**（skill 规定：任何一项为“否”，就不建议直接投 NC/SA）：

| 检查项 | 现状 | 判定 |
|---|---|---|
| 结论能写成“X 在 Y 条件下成立/失效” | 虚高随任务余量增大，ρ = 0.74：h ≤ 0.02 时中位 0.08 pp，h > 0.10 时中位 3.76 pp | 是 |
| 有真实测量数据，不只是合成数据 | 38 个对象中 37 个用真实轴承数据，仿真对象 49 已做敏感性剔除 | 是 |
| 对比对象包含工程上的标准做法 | 对比了最后一轮、验证集选模、固定预算等五种协议 | 是 |
| 有已知真值的检验 | 所有虚高都是“相对最后一轮”，没有已知真实泛化性能的对照 | **否** |
| 结论能推出设计、维护或安全决策 | 还没有展示虚高会导致什么错误的工程选择 | **否** |

**结论：按 skill 的标准，现在不建议直接投 NC。** 补上两个“否”之后可以投 NC；不补的话，改投领域顶刊（MSSP、RESS、IEEE T-Rel）更稳。

### 审稿最可能挑战的点

1. **平均效应小。** 作者实际所用规则造成的虚高，中位数只有 0.37 pp（95% CI 0.04–1.20）。H-B1 方差压缩、H-B2 显著性改变都不成立，S1 几乎没有反转。“问题不大”同样是一种合理读法。
   - 对策：把主角从“平均虚高”换成“虚高随难度增大”（H-A2）。平均值小，是因为样本里饱和的任务多。
2. **领域窄**，只涉及轴承故障诊断。
3. **缺已知真值和工程后果**，即上表中的两个“否”。
4. **静态编码全部由 AI 完成**，没有人工验证。S4a 的 κ 为 0.51，已标为“低可信”。要在方法部分讲清楚，并准备审稿人要求人工抽检时的补做方案。

### 升级到稳妥 T2 要补的（按优先级）

1. **已知真值检验。** 在 3–5 个对象上留出一份从未参与任何选择的独立测试集，或使用多工况数据中未被使用的工况，得到“真实泛化性能”，比较五种协议哪一种偏差最小。这能把“相对最后一轮”的比较变成“相对真值”的比较。
2. **虚高的预测模型。** 用余量 h（以及选模集大小、训练轮数等）拟合一个预测虚高的简单模型，留一对象交叉验证，并给出“什么条件下必须报告最后一轮或验证集规则”的阈值。这就是一条可推广的规律。
3. **工程后果。** 展示两种后果中的一种：
   - 同一组方法在不同协议下的排名变化（现有 14、42、30 三个对象已经出现，可系统统计）；
   - 在难任务上，按“最好一轮”选出的方法在真值下并不更好。
4. **可选：跨领域。** 把同一套记录器用到 1–2 个相邻任务（齿轮、刀具磨损或剩余寿命预测），每个任务 3–5 个对象，说明问题不只出在轴承上。

## 2. 选刊

- **推荐 NC。** skill 规则第 3 条：纯计算稿件优先 NC（样本中 NC 13 篇，SA 4 篇）。读者会因为“这个做法会让指标虚高多少”而想读，属于“能力/指标”型，也指向 NC。
- **SA 不推荐。** SA 机械类偏器件、机器人（42%）和设计原理（22%）。
- **转投路径。** NC 拒稿后，可以走 Nature 旗下转投到 Communications Engineering，不必重写。
- **NC 格式。** 标题 ≤15 词；摘要 ≤200 词、无引用；正文约 5000 词；图表 ≤10；每条图注 ≤350 词；Methods < 3000 词；参考文献约 70 篇；讨论不分小标题；不接受预投稿咨询。

## 3. 骨架

### 标题候选
1. Checkpoint selection inflates deep-learning fault diagnosis on hard tasks（结论式，9 词）
2. Benchmark saturation hides checkpoint-selection bias in machine fault diagnosis（结论式，9 词）
3. A preregistered audit of epoch selection in deep-learning fault diagnosis（名词短语式，10 词）

第 1、2 个是结论式。结论式要求有直接证据，所以补上已知真值检验之后再用；否则用第 3 个。

### 摘要 5 步提纲
1. **Context：** 深度学习故障诊断论文大量报告接近 100% 的准确率，是预测性维护的依据。
2. **Gap：** 这些数字往往取自训练中“最好的一轮”，而选这一轮的规则很少被说明，它对报告值的影响从未被系统测量。
3. **Approach：** Here, we rerun 38 published codebases（880 training runs, preregistered），逐轮记录，比较作者规则与最后一轮、验证集选模等五种协议。
4. **Results（2–3 个数字）：**
   - 虚高随任务余量增大，ρ = 0.74：饱和任务上 0.08 pp，难任务上 3.76 pp，17% 的方法组超过 5 pp；
   - 只有 46% 的代码能运行；
   - 79 篇随机论文中只有 2–10 篇（3–13%）明确说明报告的是哪一轮。
5. **Implication：** 接近饱和的基准掩盖了这种偏差；给出报告规范（报告最后一轮或验证集规则，以及余量 h）。

### 引言 4 步提纲
1. **钩子：** 抽样框里 2019–2025 年的噪声鲁棒性轴承诊断论文有 1344 篇（OpenAlex 检索），头条准确率普遍超过 99%。如果要引用轴承故障占旋转机械故障比例的数字，必须核对来源。
2. **现有路线：**
   - ML 可复现性研究：数据泄漏、测试集重复使用（各一句优点和局限，引用要密集）；
   - PHM 基准批评：CWRU 的样本级随机切分导致信息泄漏；
   - 二者都停留在静态检查或少量案例，没有逐轮复跑。
3. **缺口句：** “How much the choice of reported epoch changes published accuracy, and when, has not been measured.”
4. **本文段：** Here, we … (Fig. 1)，接一句写 ρ = 0.74 和 3.76 pp。

### Results 小节（F 类模板改写）
1. A preregistered rerun of 38 fault-diagnosis codebases（3.1，可运行率和排除原因）
2. Rerun outputs match the authors' own code（3.1 记录器核对，加上 S6 与论文值对比；这是管线的真值检验）
3. Inflation is small on saturated tasks and grows with headroom（3.2 H-A1、H-A2，**主角**）
4. Selection rarely changes statistical conclusions（3.3 H-B1、H-B2、S1，如实报告阴性结果）
5. **待补：** Known-truth test and predicted inflation
6. What the wider literature discloses（3.8 静态编码，AI 双编码）

### Discussion 第一段要点
- In this work, we 复跑了 38 个代码库共 880 次训练；
- 重述三个数字：ρ = 0.74、难任务 3.76 pp、2–10/79 篇明确说明选模规则；
- 第二段写推广到哪些系统；第三段写局限：只涉及轴承、AI 编码、46% 可运行率带来的选择偏差。

## 4. 图序

| 图 | 作用 | 面板 |
|---|---|---|
| Fig. 1 | 概念总图 | (a) 审计流程：1344 篇 → 83 个尝试 → 38 个可运行 → 880 次运行；(b) 一条典型的逐轮曲线，标出 author、last、best_rep、minloss、budget50；(c) 论文报告值与最后一轮值并排对比（如对象 76：99.43 对 93.52） |
| Fig. 2 | 可运行率与复现 | (a) 排除原因；(b) 记录器与作者输出一致；(c) S6：17 个对象论文值与复跑值的差 |
| Fig. 3 | **核心规律** | (a) best_rep − last 对余量 h 的散点图，88 个方法组，ρ = 0.74（现有 `fig_inflation`）；(b) 分档中位数；(c) 作者规则下的虚高 |
| Fig. 4 | 方差与显著性（阴性结果） | (a) log SD 比；(b) 判定改变数与零分布；(c) S2 早饱和 |
| Fig. 5 | 文献层透明度 | S4a、S5、S8 的比例和范围（争议格），两个模型的一致性（κ、PABAK） |
| Fig. 6（末图，待补） | 后果与规范 | (a) 已知真值下各协议的偏差；(b) 虚高预测模型的留一验证；(c) 排名变化；(d) 报告规范示意 |

**自查**
- [x] Fig. 1 有对比面板
- [ ] 核心论断有模型与真实数据的双重证据：H-A2 目前只有相关，缺预测模型和真值检验
- [ ] 末图回答引言第一段的需求（“报告的准确率可信吗”）：要等 Fig. 6 补上
- [x] 图表 ≤10

## 5. 句式框架（交给 nature-writing 成句）

- **摘要 Gap：** "However, the epoch at which accuracy is reported is rarely stated, and its effect on published results has not been measured."
- **摘要 Results：** "… the gap between best-epoch and last-epoch accuracy rose with task headroom (Spearman ρ = 0.74), from 0.08 percentage points on near-saturated tasks to 3.76 on harder ones."
- **引言缺口句：** "… has been limited to static checks of a few benchmarks."
- **Implication：** "Near-saturated benchmarks therefore conceal a selection bias that becomes material exactly where diagnosis is hardest."
- 避免使用 novel、first、unprecedented；“第一次”用事实表达，例如“预注册”“38 个代码库”。

## 6. 投稿前自查

| 项目 | 状态 |
|---|---|
| 一句话卖点写得出来 | ✓ |
| 级别 ≥ T2 | △ 边缘，补已知真值和预测模型后 ✓ |
| 3 篇以内最强对手，并有量化对比 | ✗ 要找：ML 泄漏综述、PHM 基准批评、checkpoint 选择相关研究 |
| 纯计算判定清单 | ✗ 两项未过（已知真值、工程决策） |
| 标题、摘要格式 | 待写 |
| 引言钩子、缺口句、“Here, we” | 待写，见提纲 |
| Fig. 1 概念对比 | 设计已有 |
| 双重证据 | ✗ |
| 与 SOTA 的对比图 | 不适用；改为“与报告值对比”（S6）和协议对比 |
| 末图回答需求 | ✗ 待补 Fig. 6 |
| 统计（n、误差、检验） | ✓ 预注册齐全，建议过一遍 nature-statistics |
| 局限段 | 待写：轴承、AI 编码、可运行率 |
| 数据和代码可用性 | 待写（nature-data）；论文全文不能公开，只公开哈希 |
| AI 使用声明 | **必须写**：静态编码由 Claude 和 DeepSeek 完成，方法里写清模型、日期、提示词哈希 |
| 模拟审稿 | 建议在补实验前做一次（nature-reviewer），重点挑战“平均虚高只有 0.37 pp”这一点 |

## 建议的下一步
1. 先决定是否补“已知真值检验”和“虚高预测模型”（估计 GPU 量要先算）。这决定能否投 NC。
2. 不补的话，按领域顶刊（MSSP）的格式写，现有结果已经够用。
