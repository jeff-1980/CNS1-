# 参考文献 Consensus 核对记录（2026-11-01）

方法：逐条用题名在 Consensus（Semantic Scholar / OpenAlex 库）检索，核对作者、年份、期刊、DOI，并对照摘要检查正文的引用语境。4 条（标准、会议幻灯片、数据集）不在论文库中，另行说明。

## 结果汇总

- 46 条可检索文献全部在 Consensus 中找到，作者、年份、期刊、DOI 均与参考文献表一致，没有发现张冠李戴或错号。
- 发现 2 处引用语境需要调整，已改正文（字数不变，仍为 5,999 词；参考文献仍为 50 条，编号不变）：
  1. 引言 "Reported VG uplifts range from about 2% to 6%" → "about 1% to 6%"。依据：Hwangbo 2017 [20] 摘要写 VG 带来 "moderate 1–5%" 增产；Moon 2023 [19] 为 +1.87% AEP。
  2. 讨论 5.1 节 "delay stall at the angles of attack typical of moderate winds [zhao, moon]" → "[lin02, zhao]"。依据：Moon 2023 摘要说增益在 10 m/s 附近、低风速反而下降，用来支撑"中等风速失速延迟"不够贴切；Lin 2002 是 VG 控制边界层分离的综述，更直接。Moon 仍在引言中引用，故文献数不变。

## 逐条记录

| # | 文献 | Consensus 结果 | 结论 |
|---|---|---|---|
| 1 | Veers 2019 Science | 一致，DOI 10.1126/science.aau2027 | ✅ |
| 2 | Lin 2002 Prog Aerosp Sci | 一致 | ✅（现亦用于 5.1 节） |
| 3 | Zhao 2022 Ocean Eng | 一致 | ✅ |
| 4 | Mueller-Vahl 2012 ASME GT2012-69197 | 一致 | ✅ |
| 5 | Baldacchino 2018 Wind Energy | 一致 | ✅ |
| 6 | Skrzypiński 2014 JPCS 524 | 一致 | ✅ |
| 7 | Astolfi 2018 Energies 11:1300 | 一致 | ✅ |
| 8 | Astolfi 2019 J Energy Resour Technol | 一致 | ✅ |
| 9 | Terzi 2018 JPCS 1102 | 一致 | ✅ |
| 10 | Castellani 2020 JPCS 1452 | 一致 | ✅ |
| 11 | Lee 2015 Wind Energy | 一致 | ✅ |
| 12 | Shin 2018 Ann Appl Stat | 一致 | ✅ |
| 13 | Latiffianti 2022 Wind Energy, 10.1002/we.2722 | 一致；摘要为分离天气与机组本身的能量分解法，与"控制气象协变量的统计方法"语境相符 | ✅ |
| 14 | Wagner 2009 Wind Energy, 10.1002/we.297 | 一致；风廓线/湍流影响功率曲线 | ✅ |
| 15 | St. Martin 2016 WES 1:221 | DOI 一致；Consensus 把 Lundquist 列为第一作者，属数据库排序问题，WES 官网第一作者为 C.M. St. Martin，保留现写法 | ✅ |
| 16 | Gonzalez 2019 Renew Energy 131 | 一致 | ✅ |
| 17 | Astolfi 2021 Energies 14:1105 | 一致 | ✅ |
| 18 | Manolesos 2024 JPCS 2767 | 一致；AEP +5.77% | ✅ |
| 19 | Moon 2023 Renew Energy 212 | 一致；AEP +1.87%，高风速增益、低风速下降 | ✅ 元数据；5.1 节语境已改 |
| 20 | Hwangbo 2017 Renew Energy 113 | 一致；摘要"moderate 1–5%" | ✅ 元数据；引言区间已改为 1%–6% |
| 21 | IEC 61400-12-1:2022 | 标准，不在论文库 | ➖ 不适用（前轮已核对 IEC 版本号） |
| 22 | Clerc et al. 2025 WindEurope 幻灯片 | 会议报告，不在论文库 | ➖ 不适用（前轮已核对 RES 官网 PDF 链接） |
| 23 | Ribrant 2007 IEEE TEC 22 | 一致；齿轮箱单次故障停机时间最长 | ✅ |
| 24 | Tavner 2007 Wind Energy 10 | 一致（Consensus 标 2006 为网络首发年，卷期年 2007） | ✅ |
| 25 | Spinato 2009 IET RPG 3 | 一致 | ✅ |
| 26 | Faulstich 2011 Wind Energy 14 | 一致 | ✅ |
| 27 | Pinar Pérez 2013 RSER 23 | 一致；叶片与齿轮箱停机最长 | ✅ |
| 28 | Carroll 2016 Wind Energy 19 | 一致 | ✅ |
| 29 | Artigao 2018 Appl Energy 228 | 一致 | ✅ |
| 30 | Nejad 2022 WES 7:387 | 一致（Consensus 收录的是 2021 讨论稿；正式版 WES 7 (2022) 387–411 前轮已经 Crossref 核对） | ✅ |
| 31 | Kim 2023 JMST 37 | 一致；摘要：污染叶片加 VG 后叶根挥舞 DEL 增加 4%–7%，与正文一致 | ✅ |
| 32 | Bak 2016 JPCS 753 | 一致 | ✅ |
| 33 | Ji 2026 Machines 14:965 | 一致；摘要明确为示例性数值数据，与正文描述一致 | ✅ |
| 34 | Noppe 2018 WES 3:139 | 一致；用应变测量训练神经网络 | ✅ |
| 35 | Vera-Tudela 2017 Renew Energy 107 | 一致；用实测载荷训练 SCADA 神经网络 | ✅ |
| 36 | Dimitrov 2018 WES 3:767 | 一致；用高保真载荷仿真库训练代理模型 | ✅ |
| 37 | Hill of Towie 数据集（Zenodo） | 数据集，不在论文库 | ➖ 不适用（前轮已核对 Zenodo DOI） |
| 38 | Abadie 2010 JASA 105 | 一致 | ✅ |
| 39 | Abadie 2021 J Econ Lit 59 | 一致 | ✅ |
| 40 | Conley & Taber 2011 Rev Econ Stat 93 | 一致；少数处理单元下的推断 | ✅ |
| 41 | Fleming 2019 WES 4:273 | 一致 | ✅ |
| 42 | Burton 2011 Wind Energy Handbook 2nd ed. | 一致，DOI 10.1002/9781119992714 | ✅ |
| 43 | ISO 281:2007 | 标准，不在论文库 | ➖ 不适用 |
| 44 | Nejad 2014 Int J Fatigue 61 | 一致；齿轮齿根弯曲 S-N 疲劳 | ✅ |
| 45 | Künsch 1989 Ann Stat 17 | 一致；分块 bootstrap | ✅ |
| 46 | Bertrand 2004 QJE 119 | 一致；序列相关导致 DiD 标准误偏小 | ✅ |
| 47 | Bossanyi 2000 Wind Energy 3 | 一致 | ✅ |
| 48 | Johnson 2006 IEEE CSM 26 | 一致 | ✅ |
| 49 | Wang & Vidal 2026 JPCS 3232 | 一致；摘要"增益集中在 10–18 m/s"与正文一致 | ✅ |
| 50 | Staffell 2014 Renew Energy 66 | 一致 | ✅ |
