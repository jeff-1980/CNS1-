import re
s=open('manuscript_v2.md',encoding='utf-8').read()
def rep(a,b,count=1):
    global s
    assert s.count(a)==count, (s.count(a), a[:70])
    s=s.replace(a,b)
rep("> DRAFT v2 (2026-11-01). Items in {{braces}} still need author input or a final check.\n\n","")
ab0=s[s.index("## Abstract\n")+12:s.index("\n\n## Keywords")]
rep(ab0,"Aerodynamic retrofits such as vortex generators are validated by the energy they add, but the extra load they pass to the drivetrain is rarely measured. At the Hill of Towie wind farm (21 Siemens SWT-2.3-VS-82 turbines), one turbine received a vortex-generator retrofit in September 2021 while the other 20 remained unmodified for nine months. Using the farm's open 10-minute SCADA data, we compared the retrofitted turbine with its own pre-retrofit behaviour in the same farm state, corrected for the null drift of 20 placebo turbines, and checked the estimator by injecting a known effect into real data. The retrofit raised energy by 4.8% (95% CI 3.3–6.3%), consistent with the operator's independent 4.3%, time at rated power by 1.7 percentage points, and a quasi-static main-shaft torque damage-equivalent load (DEL) by 2.3% (1.3–3.2%) for an S-N exponent of 3; the retrofitted turbine ranked first of 21 on all three outcomes. Each 1% of extra energy thus cost 0.48% (0.39–0.54) of extra torque DEL, falling to 0.20 for an exponent of 10. This load cost ratio is about 60% of the value implied by scaling power at constant rotor speed, because the variable-speed controller passed one third of the gain into rotor speed (+1.7%) and because the gain arose mainly at part load. The 2022–23 rollout to the other 20 turbines reproduced the time-at-rated increase and a consistent ratio. The ratio can be estimated from routine SCADA data alongside standard upgrade validation.")
rep("C = ΔDEL / ΔE,                                                    (1)","$$C = \\frac{\\Delta \\mathrm{DEL}}{\\Delta E} \\qquad (1)$$")
rep("T = P / ω,  ω = 2πn/60,                                             (2)","$$T = \\frac{P}{\\omega}, \\qquad \\omega = \\frac{2\\pi n}{60} \\qquad (2)$$")
rep("- E, the energy ratio change: ΣP_obs / ΣP_exp − 1;","- $E$, the energy ratio change: $E = \\sum P_\\mathrm{obs} / \\sum P_\\mathrm{exp} - 1$;")
rep("- R, time at rated power:","- $R$, time at rated power:")
rep("- D_m, the torque DEL proxy for S-N exponent m: (mean T_obs^m / mean T_exp^m)^{1/m} − 1, for m = 3","- $D_m$, the torque DEL proxy for S-N exponent $m$: $D_m = \\left(\\overline{T_\\mathrm{obs}^{\\,m}} \\,/\\, \\overline{T_\\mathrm{exp}^{\\,m}}\\right)^{1/m} - 1$, for m = 3")
rep("- W, the relative change in mean rotor speed; SD,","- $W$, the relative change in mean rotor speed; $SD$,")
rep("All analysis steps were specified before the data were examined, in a frozen plan with five pre-registered kill tests; two deviations forced by the data are reported as such (Section 3.6).",
    "The estimator and the tests it had to pass were fixed and time-stamped with SHA-256 hashes before they were run (Section 3.8); two deviations forced by the data are reported as such (Section 3.6).")
rep("main-shaft speed is above 0.3 rad/s. {{Data-quality table: share kept per turbine-year.}}",
    "main-shaft speed is above 0.3 rad/s. Across 2020–2023 the filter keeps 72% of records (Table S1). Records with P ≤ 0 account for most of the loss (15–22% per year), followed by a reduced power set-point or low rotor speed (4–11%) and shutdowns (3–7%). T13 keeps 75–86% per year. Two turbines, T16 (13–25%) and T17 (34–53%), were often run with a reduced set-point; they remain in the placebo fleet, whose median is insensitive to them.")
rep("Coverage of post records by pre cells was ≥ {{99.7}}%.","At least 99.4% of post records fell in a cell populated in the pre window for every turbine (99.95% for T13).")
rep("The baseline estimator and five kill tests (K1 known truth, K2 drift-corrected T13 effect with rank test, K3 replication, K4 seasonal re-weighting, K5 within-10-minute variability) were frozen with SHA-256 hashes before any data were analysed; all five passed. The paper analyses in Sections 3.5 and 3.7, the power-bin decomposition and the event study are descriptive extensions added after the kill tests and are labelled as such. Code: {{GitHub/Zenodo DOI}}.",
    "The study proceeded in fixed stages, each frozen with a SHA-256 hash before it was run. A baseline script (raw T13 effect with placebo checks) was frozen before its first run. After its result was known, a plan with four pass/fail tests and one informational check was frozen before any of them was run: K1 known truth (Section 3.4), K2 drift-corrected T13 effect with rank test (Section 3.3), K3 rollout replication (Section 3.6), K4 seasonal re-weighting, and K5 within-10-minute variability (informational). All four pass/fail tests passed, K3 after the deviations D1 and D2 described in Section 3.6. The plan was time-stamped internally, not publicly registered. The analyses in Sections 3.5 and 3.7, the power-bin and wind-speed decompositions and the monthly contrasts are descriptive extensions added after the tests. The frozen scripts, their hashes and the paper analysis code are publicly available (see Data and code availability).")
rep("The measured C is about 60% of that benchmark at every exponent.","The measured C is about 60% of that benchmark at m = 3 and about 40% at m = 10.")
rep("Two mechanisms account for C being about 0.6 of the naïve benchmark,","Two mechanisms account for C being about 0.6 of the naïve benchmark at m = 3,")
rep("generator torque law of the form T = Kω², so that P = Kω³. An aerodynamic gain that is absorbed by this law raises rotor speed and torque together: to first order, ΔT/T = (2/3)ΔP/P and Δω/ω = (1/3)ΔP/P.",
    "generator torque law of the form $T = K\\omega^2$, so that $P = K\\omega^3$. An aerodynamic gain that is absorbed by this law raises rotor speed and torque together: to first order, $\\Delta T/T = \\tfrac{2}{3}\\,\\Delta P/P$ and $\\Delta\\omega/\\omega = \\tfrac{1}{3}\\,\\Delta P/P$.")
rep("- This load price is about 60% of what a pure power scaling would imply,","- At m = 3 this load price is about 60% of what a pure power scaling would imply,")
# back matter
i=s.index("## Data and code availability"); j=s.index("## Table 2 (draft)")
back="""## CRediT authorship contribution statement
**Yi Wang:** Conceptualization, Methodology, Software, Formal analysis, Data curation, Visualization, Writing – original draft. **Yongqing Tang:** Validation, Writing – review & editing.

## Declaration of competing interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Acknowledgements
We thank RES and the owners of the Hill of Towie wind farm for releasing the SCADA data, shutdown records and retrofit dates as an open dataset.

## Data and code availability
The data are the Hill of Towie wind farm open dataset, version 2.1.0, Zenodo, https://doi.org/10.5281/zenodo.22662930 (CC BY 4.0) [18]. The frozen test scripts with their SHA-256 hashes, the paper analysis scripts and the figure code are available at https://github.com/jeff-1980/aeroup-vg-load-cost.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process
During the preparation of this work the authors used Claude (Anthropic) in order to write analysis code and to draft and edit the text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

"""
s=s[:i]+back+s[j:]
# tables: Table 1 before Table 2
t2=s[s.index("## Table 2 (draft)"):s.index("## Table 1.")]
t1=s[s.index("## Table 1."):s.index("## Figure captions")]
s=s.replace(t2+t1, t1+t2.replace("## Table 2 (draft). ","## Table 2. "))
# supplementary figures out of main file
k=s.index("![](figs/figS1_torque_spectrum.png)"); r=s.index("## References")
supp_figs=s[k:r]; s=s[:k]+s[r:]
open('manuscript_v3.md','w',encoding='utf-8').write(s)
open('/tmp/claude-0/-home-user-CNS1-/3cd31274-fa87-5692-a85b-d421aef7474a/scratchpad/supp_figs.md','w',encoding='utf-8').write(supp_figs)
ab=s[s.index("## Abstract\n")+12:s.index("\n\n## Keywords")]
print("abstract words",len(ab.split()))
for h in s[s.index("## Highlights"):s.index("## Abstract")].splitlines()[1:]:
    if h.startswith("- "): print(len(h)-2,h)
print("{{" in s)
