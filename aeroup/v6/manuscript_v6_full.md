# Energy gain and drivetrain torque exposure of a vortex-generator retrofit: a field assessment at an operating wind farm

Yi Wang

College of Engineering, Lishui University, Lishui, Zhejiang 323000, China
E-mail: jefflsxy@lsu.edu.cn

## Highlights
- One retrofitted turbine and 20 unmodified neighbours form a natural experiment
- Vortex generators raised normal-operation energy by 4.8% and time at rated by 1.7 pp
- A 10-min torque exposure index rose 0.48% per 1% of extra energy (m = 3)
- Part-load gains and a rotor-speed response are consistent with this low ratio
- Placebo turbines, pseudo-retrofit dates and injected effects bound the evidence

## Abstract
Aerodynamic retrofits such as vortex generators are validated by the energy they add, while the extra load they pass to the drivetrain is rarely measured on commercial turbines. At the Hill of Towie wind farm (21 Siemens SWT-2.3-VS-82 turbines), one turbine received vortex generators in September 2021 while the other 20 stayed unmodified for nine months. From open 10-minute operational records of the turbines we estimated the retrofit's effect on energy, time at rated power and a quasi-static main-shaft torque exposure index, comparing the turbine with its own pre-retrofit behaviour in the same farm state and correcting for the drift of 20 placebo turbines. Energy in normal operation rose by 4.8% (95% confidence interval 3.3–6.6%, covering time sampling only), consistent with the operator's independent 4.3%, time at rated power by 1.7 percentage points (0.7–2.8) and the torque exposure index by 2.3% (1.6–3.5%) for a fatigue-curve exponent of 3. The exposure ratio, the relative change in torque exposure per relative change in energy, was 0.48 (0.41–0.60), falling to 0.20 for an exponent of 10. Pseudo-retrofit dates and placebo turbines show that the time-at-rated effect is far outside the null range, whereas the energy and exposure effects exceed about 95% of it. The exposure ratio lies below the value for pure power scaling (0.79), consistent with most of the gain arising at part load and rotor speed absorbing about a third of it. The index measures 10-minute load level, not fatigue damage.

## Keywords
Wind turbine retrofit; vortex generators; drivetrain loads; SCADA; natural experiment; placebo analysis

## 1. Introduction

Raising the output and extending the life of the existing fleet is one of the open problems of wind energy [1], and retrofitting operating turbines with aerodynamic add-ons is now a routine way to do the former. Vortex generators (VGs) delay separation on the thick inboard airfoils, raise lift near stall and are typically sold on an annual energy production (AEP) uplift of a few percent. Lin [2] reviewed how low-profile VGs control boundary-layer separation and Zhao et al. [3] their application to wind turbine blades; wind-tunnel and field parametric studies [4] and a parameter study on a 30%-thick airfoil [5] mapped how VG height and spacing set the gain, and a design study of the DTU 10 MW reference blade [6] estimated it at turbine scale. Field verification of such upgrades has matured into a small methods literature: post-upgrade production is compared with a model of what the turbine would have produced without the upgrade, built from nearby reference turbines or pre-upgrade behaviour: Astolfi et al. set out the neighbour-based approach for power-curve upgrades [7] and refined its precision for a combined aerodynamic and control upgrade [8], and Terzi et al. [9] and Castellani et al. [10] compared alternative assessment methods on test cases. Statistical alternatives match pre- and post-upgrade periods on wind and weather covariates, through a kernel-plus model [11], covariate matching [12] or an energy and power decomposition applied to leading-edge protection [13]. Power-curve-based assessments must contend with the sensitivity of output to shear, turbulence and atmospheric stability [14,15] and with the choice and resolution of the supervisory control and data acquisition (SCADA) variables used [16,17]. Reported VG uplifts range from about 1% to 6%: a moderate 1–5% in an academia–industry field study [18], 1.9% AEP for a MW-class blade in combined numerical and field tests [19], and 5.8% AEP when VGs and Gurney flaps were retrofitted together on a Vestas V52 [20]; and the operator of the Hill of Towie wind farm in Scotland has published an open-source, IEC 61400-12-1-style [21] validation of its own "AeroUp" VG retrofit at +4.3% (90% confidence interval (CI) 3.3–5.3%) for the first retrofitted turbine [22].

The other side of the ledger has received far less attention. More energy captured below rated power means more torque through the main shaft, gearbox and generator for a larger share of the time. The drivetrain, and the gearbox in particular, already dominates downtime per failure in field reliability surveys: in Swedish data from 1997–2005 [23], in large German and Danish turbine populations analysed by subassembly [24,25], in a comparison of onshore downtime with offshore needs [26], in a review of published surveys [27], in offshore failure, repair-time and cost records [28] and in a review aimed at condition monitoring [29]; and its design and failure modes remain an active research topic [30]. Upgrade studies note that a retrofit may affect residual lifetime [8] but do not quantify loads. Load effects of VGs are known mainly from aeroelastic simulation, which reports blade-root flapwise damage-equivalent loads (DELs) rising by 4–7% when VGs are added to contaminated blades [31], and from one instrumented research turbine, where VGs on clean blades increased flapwise blade moments [32]. Multi-criteria retrofit evaluation frameworks call for load metrics but have so far been illustrated with assumed numbers [33]. Methods that estimate loads from SCADA data exist, but they need loads to train the mapping from SCADA to loads: strain measurements for quasi-static thrust from 1 s data [34], measured fatigue loads under wind-farm flow conditions [35], or a high-fidelity aeroelastic load database for site-specific surrogates [36]. To our knowledge no study has measured on commercial turbines how much additional drivetrain torque exposure accompanies the energy gain of an aerodynamic retrofit.

Routine 10-minute SCADA data cannot resolve the load cycles that drive gear and bearing fatigue, but they do record the level of drivetrain torque and how often each level occurs. We therefore ask a narrower question that the data can answer: by how much does a retrofit raise the quasi-static torque exposure of the drivetrain per unit of energy gained? We summarise the answer in an exposure ratio,

*C* = *D*~*m*~ / *E*,          (1)

the relative change in a power-law-weighted mean of the 10-minute torque (Section 3.1) per relative change in energy. C is a measure of load level, not of fatigue damage or remaining life; translating it into damage requires component-level load and stress–life (S-N) information that SCADA data do not contain.

The Hill of Towie open dataset [37] makes the comparison possible. Its 21 identical Siemens SWT-2.3-VS-82 turbines were retrofitted in two stages: one turbine (T13) in September 2021, the remaining 20 between July 2022 and May 2023. For nine and a half months one turbine carried the upgrade while 20 sister turbines did not. This is a natural experiment with one treated unit, the setting for which econometrics developed synthetic-control comparisons with placebo-based inference [38], guidance on their data requirements [39] and inference with few treated units [40]; field campaigns on wake steering at commercial wind farms have used similar toggled or side-by-side designs [41]. The later rollout provides a further, weaker check because no unmodified reference turbines remain.

This paper makes three contributions. (i) We estimate the change in energy, time at rated power and a quasi-static main-shaft torque exposure index caused by the VG retrofit, and we bound the strength of this single-turbine evidence with placebo turbines, pseudo-retrofit dates, injected known effects and a farm-wide joint bootstrap. (ii) We report the exposure ratio C and where in the operating envelope the energy gain and the exposure arise. (iii) We compare the measured C with scenarios that add the measured location of the gain and the controller's rotor-speed response to a pure scaling of power. The estimator and the tests it had to pass were fixed and time-stamped with SHA-256 hashes before they were run (Section 3.9); two deviations forced by the data are reported as such (Section 3.7).

## 2. Data

### 2.1 Site and turbines
Hill of Towie is an onshore wind farm in Moray, Scotland, owned by TRIG and operated by RES. It comprises 21 Siemens SWT-2.3-VS-82 turbines (rated power 2300 kW, rotor diameter 82 m, geared drivetrain, variable speed, pitch regulated). The turbines entered commercial operation in May 2012, so they were about 9.4 years old when the first retrofit was installed. The operator released 10-minute SCADA data from 2016 onward, alarm logs, a shutdown-duration table and the AeroUp installation dates under a CC BY 4.0 licence [37]. We use the yearly archives for 2020–2023.

### 2.2 The retrofit
AeroUp is a vortex-generator retrofit [22]. T13 was retrofitted on 23–29 September 2021. Works on the other 20 turbines started between 14 July and 20 September 2022; twelve were completed within weeks, eight (T06, T07, T10, T11, T12, T15, T16, T20) only in April–May 2023 (Table 1). A separate control upgrade (TuneUp) was installed on nine turbines in March 2024, outside our analysis windows.

### 2.3 Variables
From tables tblSCTurGrid, tblSCTurbine and tblSCTurTemp we use the 10-minute mean active power P, main-shaft speed n, nacelle yaw position, the power reference set-point, the within-period standard deviation of active power, ambient temperature, and all bearing, gearbox-oil and generator temperature channels. Time stamps mark the end of each 10-minute period; the shutdown table, which marks the start, is shifted by 10 minutes before merging.

### 2.4 Normal-operation filter
A record is kept when the shutdown duration is zero, P > 0, the power set-point is ≥ 2290 kW (no curtailment), and main-shaft speed is above 0.3 rad/s. Across 2020–2023 the filter keeps 71% of merged records (3,069,135 of 4,338,962; Table S1). Records with P ≤ 0 account for most of the loss (15–22% per year), followed by a reduced power set-point or low rotor speed (4–11%) and shutdowns (3–7%). Two turbines, T16 (13–25%) and T17 (34–53%), were often run with a reduced set-point; they remain in the placebo fleet, whose median is insensitive to them. T13's kept share fell from 86% (2020) to 81% (2021) and 78% (2022), against 84%, 78% and 71% for the fleet median.

## 3. Methods

### 3.1 Outcomes
For each kept record we form the electrical torque proxy at the main shaft,

*T* = *P* / *ω*,   *ω* = 2π*n* / 60          (2)

which equals the aerodynamic torque minus drivetrain losses and inertial terms; at 10-minute resolution the inertial term averages out. T is an electrical proxy for the main-shaft torque T/η, where η is drivetrain and generator efficiency. Because the retrofit moves the turbine along its efficiency curve, the proxy and the mechanical torque need not change by the same amount. As an illustration, if η rose linearly from 0.90 at zero load to 0.95 at rated (an assumption, not a measurement), a 5% uplift at 0.8 of rated power would raise η by 0.2% and the electrical proxy would overstate the relative increase in mechanical torque by about 0.2 percentage point. Forming T from 10-minute means instead of averaging the instantaneous ratio adds an error that depends on the within-period variance of rotor speed and its covariance with power; without high-rate data we cannot bound it or show that it cancels between pre and post. We therefore report D_m and C for the electrical torque proxy and do not claim a bound on their difference from the mechanical values. Outcomes for a target turbine over a window are:

- *E*, the energy ratio change: *E* = Σ*P*~obs~ / Σ*P*~exp~ − 1, over normal-operation records only, i.e. the energy gain while the turbine runs unconstrained; it is not a change in annual energy production, which also depends on availability and curtailment;
- *R*, time at rated power: the share of records with P ≥ 0.95 × 2300 kW (in percentage points);
- *D*~*m*~, the torque exposure index for exponent *m*: *D*~*m*~ = (mean *T*~obs~^*m*^ / mean *T*~exp~^*m*^)^1/*m*^ − 1, for m = 3, 10/3, 4, 6 and 10;
- *W*, the relative change in mean rotor speed; *T*~1~, the relative change in mean torque; *SD*, the relative change in mean within-10-minute power standard deviation; and temperature residuals for every bearing and gearbox channel.

D_m has the form of a damage-equivalent load [42] computed from 10-minute means: the exponent weights high-torque periods as a power-law S-N curve would, with m = 3 conventional for shafts and welded structures, 10/3 for roller bearings [43], and 6–10 for gear teeth [44]. It describes the distribution of the quasi-static torque level only. It contains no load cycles within the 10-minute period, no dynamic amplification and no component-specific stress, so it is not a fatigue damage estimate, and changes in D_m cannot be converted into damage or remaining life without further information.

### 3.2 Counterfactual from reference turbines
For each 10-minute stamp we form a farm reference state from all other turbines that are operating normally (at least 10 needed): their mean power, binned in 50 kW steps, and their circular-mean yaw, binned in eight 45° sectors. A target turbine's expected outcome in a post-retrofit record is the mean of its own pre-retrofit records in the same cell (Fig. 1b). Effects are observed minus expected, averaged over post records. This "reference-cell" estimator is a discretised version of the side-by-side methods used in upgrade validation [7,22]; it needs no anemometer and no power-curve model, and it treats any change in the target turbine's behaviour relative to its neighbours as the effect. At least 99.4% of post records fell in a cell populated in the pre window for every turbine (99.95% for T13).

### 3.3 Drift correction (T13 window)
Pre window: 1 January 2020 – 22 September 2021; post window: 30 September 2021 – 13 July 2022. T13 is excluded from every reference set. Each of the 20 unmodified turbines is run through the same estimator with the real dates; because nothing changed on them, their effects measure the estimator's null drift over the same calendar windows (seasonal mix, ageing, wake changes). The drift-corrected T13 effect is the T13 effect minus the median of the 20 placebo effects.

### 3.4 Uncertainty and inference
We use three complementary measures, because no single one covers every source of error in a one-turbine design.

*Joint farm bootstrap.* Each of 1000 draws resamples calendar weeks of the pre and post windows with replacement, the same weeks for all turbines, so that the dependence between turbines and the serial correlation within weeks are preserved [45,46]. Within each draw the reference cells, all 21 turbine effects, the placebo median, the corrected E, R, D_m and C, and T13's rank are recomputed. Percentile intervals from these draws are reported as 95% CIs. They describe sampling variability in time; they do not describe how much an individual turbine's drift can differ from the placebo median.

*Placebo calibration.* To cover that second source we subtract each placebo turbine's own corrected effect from T13's corrected effect, which gives 20 values of what T13's estimate would be if its null drift were like that of each unmodified turbine. Their range is reported as a placebo-calibrated interval, in the spirit of inference from the distribution of control-unit effects when only one unit is treated [40].

*Rank and pseudo-retrofit dates.* T13's rank among the 21 turbines is the placebo permutation test of comparative case studies [38], for a design in which any turbine could have been the one retrofitted first; under that assumption of exchangeability, rank 1 corresponds to a one-sided p = 1/21 ≈ 0.048. Exchangeability is plausible for identical turbines in one farm, but T13 was not chosen at random and a turbine-specific trend would violate it. We therefore also apply the full estimator, with the same drift correction, at five pseudo-retrofit dates inside the pre-retrofit period (1 July, 1 September and 1 November 2020, 1 January and 1 March 2021; pre window from 1 January 2020 to eight days before the date, post window up to 290 days after it and no later than 22 September 2021), treating every turbine in turn as the retrofitted one. The resulting 105 turbine-date effects form a null distribution in time and space against which T13's real effect is compared. The shorter pseudo-date windows make this null, if anything, wider than that of the real design.

### 3.5 Known-truth checks
We inject a known effect, post-window power multiplied by 1.04 and capped at 2300 kW with torque scaled at the measured speed, (a) into each of the 20 unmodified turbines over the real windows, and (b) into T13's own data at each pseudo-retrofit date. The estimator, with drift correction, is re-run and the recovered effect is compared with the injected truth computed directly from the altered data. Check (a) tests bias across turbines; check (b) tests recovery on the target turbine itself.

### 3.6 Counterfactual decomposition of C
If a retrofit simply multiplied power by a constant k below the rated cap, at unchanged rotor speed, C would follow from the turbine's own power distribution. Starting from this pure-scaling benchmark (k = 1.05) on T13's pre-retrofit records, we add two features of the observed response one at a time: (b) the measured gain profile, i.e. T13's drift-corrected relative power uplift in each 200 kW bin of farm reference power, applied record by record at unchanged rotor speed; and (c) the same profile with the rotor speed of region-2 records (P < 0.9 × rated and speed below 98% of rated speed) raised by (1 + u)^1/3^, the response of a K*ω*^2^ torque law to a power uplift u. The difference between counterfactual (c) and the measured C is the part that these two features do not explain.

### 3.7 Rollout check (deviation D1) and its null (deviation D2)
The frozen plan specified a replication using turbines not yet retrofitted as references. The installation file showed that works on all 20 turbines started within ten weeks and eight finished only in 2023, so no such reference set exists (deviation D1, recorded before the replication was run). We instead used T13, whose state was constant after 2021, as the single reference turbine: cells are T13 power (50 kW) × T13 yaw (8 sectors); pre = 1 October 2021 – 13 July 2022 (only T13 retrofitted), post = June, July and October–December 2023 (all retrofitted; August–September dropped because the pre window lacks them); pre records are re-weighted to the post month mix. The pooled effect is the mean over turbines with a turbine-level bootstrap CI. Because all 20 turbines share the same reference turbine and the same year-on-year change, these are not 20 independent replications; the check tests whether the direction and size of the effect recur, not its causal attribution.
Because a single reference turbine is noisier and the windows are a year apart, we added a known-null check after the rollout check had been run (deviation D2): the identical estimator applied to 2020 versus 2021 (same months, no turbine changed state; T13 excluded), with T01 and, separately, T21 as reference. Its pooled effects measure the design's year-on-year drift. The gate for this check was written before it was run.

### 3.8 Sensitivity analyses and concurrent-change screen
One factor at a time from the default: power-bin width (25, 100 kW), yaw sectors (4, 12), minimum number of reference turbines (8, 15), rated threshold (0.90, 0.98 × rated), month-of-year re-weighting of the pre window, dropping the first month after the retrofit, and starting the pre window in October 2020. We also replace the placebo median by the mean and by a trimmed mean, and the reference-power cells by cells of reference-turbine mean nacelle wind speed (0.5 m/s × eight yaw sectors). To screen for changes made to T13 at the same time as the retrofit, we track monthly control signatures of every turbine relative to the fleet median (99th percentile and median rotor speed at rated power, mean power at rated, share of records with a reduced power set-point, median pitch at part load, and the ratio of the nacelle anemometer reading to the fleet median), compare the step in each signature across the retrofit date with the steps of the other 20 turbines, and count alarm-log entries around the works.

### 3.9 Staged freezing and code
The study proceeded in fixed stages, each frozen with a SHA-256 hash before it was run. A baseline script (raw T13 effect with placebo checks) was frozen before its first run. After its result was known, a plan with four pass/fail tests and one informational check was frozen before any of them was run: K1 known truth (Section 3.5a), K2 drift-corrected T13 effect with rank test (Sections 3.3–3.4), K3 rollout check (Section 3.7), K4 seasonal re-weighting, and K5 within-10-minute variability (informational). All four pass/fail tests passed, K3 after the deviations D1 and D2. The plan was time-stamped internally, not publicly registered. The joint bootstrap, placebo calibration, pseudo-retrofit dates, injection into T13, the counterfactual decomposition, the power-bin and wind-speed decompositions, the monthly contrasts and the concurrent-change screen are descriptive analyses added afterwards. The frozen scripts, their hashes and all analysis code are publicly available (see Data and code availability).

## 4. Results

### 4.1 Data
After filtering, 3,069,135 turbine-records of normal operation remain from 4,338,962 merged records for 2020–2023 (median 158,635 per turbine). In the T13 window the pre and post periods contain about 71,800 and 33,400 T13 records.

### 4.2 Effects on T13
Relative to its own pre-retrofit behaviour in the same farm state, T13 produced 3.6% more energy after the retrofit, spent 1.35 percentage points (pp) more time at rated power, and its torque exposure index (m = 3) rose by 1.75% (raw estimates). Over the same calendar windows the 20 unmodified turbines drifted in the opposite direction: 17 of 20 show negative energy changes (median −1.2%), and 18 of 20 show negative time-at-rated changes (Fig. 2). Subtracting the placebo median, the drift-corrected effects on T13 are (Table 2; 95% CIs from the joint farm bootstrap):

- energy +4.8% (3.3–6.6%), consistent with the operator's independent +4.3% (90% CI 3.3–5.3%) [22];
- time at rated power +1.67 pp (0.65–2.80);
- torque exposure index +2.3% for m = 3 (1.6–3.5%), +2.1% for m = 10/3 (1.5–3.3%) and +0.96% for m = 10 (0.60–1.44%);
- mean rotor speed +1.66% (1.12–2.26%) and mean torque +4.0% (2.8–5.5%).

T13 ranks first of the 21 turbines on energy, time at rated and exposure in the full sample, and in 79%, 93% and 91% of the joint bootstrap draws, respectively; the corrected effects are positive in every draw.

T13's month-by-month deviation from the fleet has a seasonal pattern of its own (Fig. S4): before the retrofit it ran a few percent below its average in summer and above it in winter. To remove this, Fig. 3 compares each month with the same calendar month one year earlier and subtracts the fleet median of the same difference. In the eight complete pre-retrofit months of 2021 this year-on-year contrast averages −0.7% for energy, −0.42 pp for time at rated and −0.02% for torque exposure, with signs scattered around zero. In all nine post-retrofit months it is positive for energy and exposure, averaging +4.1%, +1.59 pp and +2.2% respectively, which reproduces the window estimate and gives C = 0.54. Single months are noisy (pre-retrofit months range from −5% to +6% in energy); the evidence lies in the consistent sign after the retrofit. Re-weighting the pre window to the post-window month mix changes the window estimate only slightly (energy +4.7%, time at rated +1.42 pp, C = 0.50; Fig. S2).

### 4.3 How strong is the single-turbine evidence?
The three null references in Section 3.4 agree on an ordering: the time-at-rated effect is far outside the null, the energy and exposure effects lie at its upper edge (Fig. 4).

*Placebo calibration.* The corrected effects of the 20 unmodified turbines, which should be zero, have a standard deviation of 1.6% in energy, 0.26 pp in time at rated and 0.67% in the exposure index. T13's corrected effects are 3.0, 6.5 and 3.4 of these standard deviations, respectively. Subtracting each placebo's corrected effect from T13's gives placebo-calibrated intervals of 2.2–7.7% for energy, 1.17–2.14 pp for time at rated and 1.2–3.6% for exposure; none includes zero. Rotor speed is weaker (−0.3% to +3.3%).

*Pseudo-retrofit dates.* Across the 105 turbine-date pairs, the null effects have a 95th percentile of 4.6% for energy, 0.45 pp for time at rated and 2.2% for exposure (Fig. 4). T13's real effects exceed 96% (energy), 100% (time at rated, null maximum 0.65 pp) and 95% (exposure) of them. All four null values above T13's energy effect come from T15, T16 and T17, two of which have by far the lowest share of normal-operation records (Table S1); without them no null value reaches T13's energy effect. T13 itself shows effects between −4.1% and +0.2% in energy at the five pseudo dates.

*Known truth.* Injected into the 20 unmodified turbines, the +4% effect is recovered without median bias (within 0.04 pp for all outcomes), but the error of an individual turbine is its own drift, with a standard deviation of 1.8 pp in energy, 0.27 pp in time at rated and 0.7 pp in exposure (Fig. S3). Injected into T13 at the pseudo dates, the recovery errors (−4.3 to +0.2 pp in energy) mirror the null effects at the same dates.

Taken together, the time-at-rated increase is attributable to the retrofit with high confidence. The energy and exposure increases are consistent with it and with the operator's independent 4.3%, but a turbine-specific drift of up to about 2–3% in energy cannot be excluded for a single turbine; their magnitudes carry that uncertainty in addition to the bootstrap CIs.

### 4.4 Exposure ratio
Each 1% of additional energy came with 0.48% (95% CI 0.41–0.60) of additional torque exposure at m = 3 (Fig. 5). The ratio falls with the exponent: 0.45 (0.37–0.57) at m = 10/3, 0.40 at m = 4, 0.31 at m = 6 and 0.20 (0.14–0.27) at m = 10. C is more stable than its numerator and denominator because drift in energy and exposure is correlated: the placebo-calibrated values of C(m = 3) lie between 0.44 and 0.59 for 18 of the 20 placebo turbines (range 0.28–0.89). The sensitivity specifications give C(m = 3) between 0.47 and 0.52 (Fig. S2), and cells defined by reference wind speed give 0.50.

### 4.5 Where the gain and the exposure arise
The energy gain is a part-load phenomenon (Fig. 6a). T13's net relative power uplift is 10–11% for farm reference power of 200–800 kW, 6–7% at 800–1600 kW, and below 1.2% above 1800 kW. Farm states with reference power between 200 and 1000 kW contribute 58% of the energy gain. The torque-cubed gain is shifted to higher power: the five bins from 600 to 1600 kW contribute 12–15% each and 68% together (Fig. 6b), because the exponent weights higher torque more (see also the torque spectrum in Fig. S1). Time at rated rises because rated is reached in slightly lower winds. Binning on the mean wind speed of the reference turbines gives the same picture: 85% of the net energy gain falls below 10 m/s (Fig. S5).

Mean rotor speed rose by 1.66%, or 0.35 of the energy gain, close to the one third expected when a variable-speed controller absorbs a power gain through a K*ω*^2^ torque law (Δ*ω*/*ω* = (1/3) Δ*P*/*P*). The counterfactuals in Fig. 5 are scenarios under stated assumptions, not a fitted decomposition: they apply assumed changes to T13's pre-retrofit records, carry no uncertainty interval, and their energy gains differ from the measured 4.8% (3.5% for pure scaling by 1.05 with the rated cap, 5.0% for the measured gain profile). With this caveat, C(m = 3) is 0.79 for pure scaling at fixed speed, 0.67 for T13's measured gain profile at fixed speed, 0.58 when the K*ω*^2^ speed response is added, and 0.48 measured. In this sequence the location of the gain lowers C by about 0.12, the speed response by about 0.09, and 0.10 of the gap to pure scaling remains. The pattern is similar at m = 10 (0.50, 0.32, 0.31 and 0.20). The within-10-minute power standard deviation did not change detectably (+1.1%, −3.1 to +5.7%; rank 3), and the mean blade-pitch reference rose by 0.21° (0.08–0.34°; rank 1).

### 4.6 Concurrent changes on T13
None of the control signatures shows a change on T13 at the retrofit date that is unusual within the fleet. Comparing October 2021 – May 2022 with January – August 2021, T13's step relative to the fleet median ranks between 11th and 19th of 21 for rated rotor speed (99th percentile and median), mean power at rated, the share of reduced set-points and part-load pitch. There is therefore no sign of a change in rated speed, power limit or fine-pitch setting made with the retrofit. The alarm log shows nothing specific to T13: its alarm count stays near the fleet median (rank 9–16 of 21), the codes logged during the works week mark the works stop on other turbines too, and codes that became more frequent on T13 were equally frequent on several unmodified turbines. The one signature that did change is the nacelle anemometer: relative to the fleet median, T13's reading rose by about 10 percentage points after the retrofit (from +7–17% to +13–26% depending on the month), the largest step of the 21 turbines. The retrofit may have altered the flow at the anemometer behind the blade roots; a sensor exchange cannot be excluded. The estimator does not use T13's anemometer, but the shift explains why binning on its wind speed misplaces the gain (Section 5.1).

### 4.7 Thermal side evidence
At equal power and ambient temperature, the gearbox high-speed bearing warmed by 0.29 K (rank 1 of 21) and the main bearing by 0.47 K (rank 4); other channels did not change distinctively. One first rank among 30 channels is what chance would produce, so this is consistent with, not confirmation of, a higher drivetrain load.

### 4.8 Rollout check
With T13 as the only reference turbine, the 20 later-retrofitted turbines show pooled changes of +5.2% energy (3.2–7.5%), +1.82 pp time at rated (1.50–2.14; 19 of 20 positive) and +2.8% torque exposure (2.0–3.7%), giving C = 0.54 (0.31–0.85) (Fig. 7). The known-null check of this design (2020 vs 2021, no turbine changed) is not null for energy and exposure: +1.0% energy and +1.05% exposure with T01 as reference, +2.3% and +1.35% with T21. Time at rated is close to null (+0.33 and +0.05 pp). Subtracting the null drift gives rollout effects of about +2.9–4.2% energy, +1.5–1.8% exposure and +1.5–1.8 pp time at rated, and C ≈ 0.42–0.50. This consistency check supports the direction and size of the time-at-rated effect and a C of the same order; its energy and exposure magnitudes depend on a drift correction that we can only bracket.

## 5. Discussion

### 5.1 Why the exposure ratio is below pure scaling
In the scenario sequence of Section 4.5, two features of the response reproduce about two thirds of the gap between the measured C and pure power scaling, and both are visible in the data. First, the location of the gain. Vortex generators act on the thick inboard sections, where they delay stall at the angles of attack typical of moderate winds [2,3]. At Hill of Towie the relative uplift is largest at low and moderate power and vanishes towards rated power, where torque is high and dominates an index that weights torque by T^m^. The steeper the exponent, the stronger this dilution, which is why C falls from 0.48 at m = 3 to 0.20 at m = 10. Second, the controller. Below rated speed, a variable-speed turbine tracks its optimal tip-speed ratio with a generator torque law of the form *T* = *Kω*^2^, as in closed-loop controller design [47], reviews of standard and adaptive variable-speed control [48] and the design handbook [42], so that *P* = *Kω*^3^, and part of an aerodynamic gain leaves the drivetrain as speed rather than torque: to first order Δ*T*/*T* = (2/3) Δ*P*/*P* and Δ*ω*/*ω* = (1/3) Δ*P*/*P*. The measured speed increase of 0.35 times the energy gain matches this, although the single-turbine evidence for the speed change is weaker than for energy (rank 2 of 21). The remaining third has candidate causes (a change in the torque distribution within each power bin, the small pitch change, turbine-specific drift; Section 4.3) that one retrofitted turbine cannot separate.

An independent analysis of the same farm places most of the gain at wind speeds of 10–18 m/s [49], which for this turbine lies near and above rated. We traced the difference to the wind-speed variable used for binning (Fig. S5). When we bin our per-record effects on the mean nacelle wind speed of the reference turbines, which the retrofit cannot affect, 85% of T13's net energy gain falls below 10 m/s and almost none above 13 m/s; the relative uplift peaks at +9–11% at 5–8 m/s and fades towards rated. When we bin the same records on T13's own nacelle anemometer, as is usual in single-turbine power-curve analysis [21], 71% of the gain moves to 10–18 m/s and the uplift turns negative below 6 m/s (−11% to −19% at 3–5 m/s). A retrofit that lowers output at low wind is not plausible; rather, T13's anemometer reading rose by about 10% relative to the fleet at the retrofit (Section 4.6), shifting post-retrofit records to higher bins. The 10–18 m/s concentration is thus a property of the target's own anemometer axis, not a disagreement about the size of the effect; both analyses find about 4–5%. For load assessment the distinction matters, because the farm-state axis places the gain where torque is low.

### 5.2 What the exposure ratio can and cannot inform
C converts a validated energy uplift into a change in the quasi-static torque level that the drivetrain sees: at Hill of Towie, a 4–5% energy gain came with about a 2% higher torque exposure index at m = 3 and about 1% at m = 10. For an owner this is a first-order screening quantity. It shows that the retrofit did not raise the drivetrain's torque level in proportion to the energy, and it can be estimated at no extra cost from the same SCADA data and side-by-side design already used to validate the energy gain [7,22]. It is not a fatigue damage or lifetime estimate. Converting it into damage would require the load cycles within each 10-minute period, the transfer from shaft torque to gear and bearing stresses, and component S-N data, none of which SCADA provides. Turbines with a known gearbox weakness or in life extension warrant such follow-up, for example high-rate torque measurements on a retrofitted and an unmodified turbine.

### 5.3 Limitations
- **Quasi-static exposure, not fatigue.** T = P/ω is an electrical torque estimate averaged over 10 minutes. D_m and C describe the torque level and its distribution, not the cycles inside each 10-minute period, which drive much of the gear-tooth and bearing damage. The within-period power standard deviation did not change detectably, which argues against a large change in fluctuation amplitude, but it is not a substitute for strain-gauge or high-rate torque data.
- **Drivetrain only.** Blade-root and tower loads, the components on which VG studies have reported load increases [31,32], are not observable in SCADA data.
- **One turbine.** The main estimate rests on a single retrofitted turbine. Its rank of 1 in 21 gives the smallest p-value a 21-turbine permutation can produce, and exchangeability is an assumption (Section 3.4). The pseudo-retrofit null places the energy and exposure effects at about the 95th–96th percentile; a turbine-specific drift of the size seen at some pseudo dates would change their magnitudes by up to 2–4 percentage points in energy.
- **Rollout check.** Its single-reference design carries a year-on-year drift that we can only bracket, and its 20 turbines are not independent replications (Section 4.8).
- **Drift of the reference fleet.** The placebo turbines drifted by a median −1.2% in energy; the correction is robust to its summary statistic, but the cause (seasonal mix, wake changes, ageing [50]) is not identified.
- **Concurrent changes.** Maintenance that leaves no trace in SCADA or alarm data cannot be ruled out (Section 4.6).
- **Mechanism.** The scenario sequence attributes about two thirds of the gap to pure scaling to the location of the gain and the speed response, under assumed rather than estimated changes and without uncertainty; the rest is unexplained.
- **Basis of the outcomes.** E is a normal-operation energy gain, not an AEP change; T is an electrical torque proxy whose difference from mechanical torque is illustrated, not bounded.
- **Seasonal heterogeneity of T13.** T13 deviates from the fleet seasonally (Fig. S4). The window estimate, the month-re-weighted estimate and the same-month year-on-year contrast agree (energy 4.1–4.8%, C 0.48–0.54), but a seasonally varying retrofit effect cannot be excluded with one turbine-year of post data.

## 6. Conclusions
- A vortex-generator retrofit at an operating wind farm raised energy in normal operation by 4.8% (95% CI 3.3–6.6%, time-sampling uncertainty only) and time at rated power by 1.7 pp on the first retrofitted turbine, measured against 20 unmodified neighbours. The time-at-rated effect lies far outside the placebo null range; the energy effect lies at its upper edge and agrees with the operator's estimate.
- A quasi-static main-shaft torque exposure index rose by about 0.5% per 1% of extra energy at m = 3 and by 0.2% at m = 10.
- At m = 3 the exposure ratio is about 60% of what pure power scaling implies. In a scenario decomposition under stated assumptions, the concentration of the gain at part load and a rotor-speed response consistent with the controller's torque law reproduce about two thirds of the difference.
- The 2022–23 rollout on 20 further turbines is consistent in direction.
- The exposure ratio can be estimated from routine SCADA data alongside energy-uplift validation. It screens load level; fatigue damage and remaining life require load measurements.

## CRediT authorship contribution statement
**Yi Wang:** Conceptualization, Methodology, Software, Formal analysis, Data curation, Visualization, Validation, Writing – original draft, Writing – review & editing.

## Declaration of competing interest
The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Acknowledgements
We thank RES and the owners of the Hill of Towie wind farm for releasing the SCADA data, shutdown records and retrofit dates as an open dataset.

## Data and code availability
The data are the Hill of Towie wind farm open dataset, version 2.1.0, Zenodo, https://doi.org/10.5281/zenodo.22662930 (CC BY 4.0) [37]. The frozen test scripts with their SHA-256 hashes, the paper analysis scripts and the figure code are available at https://github.com/jeff-1980/aeroup-vg-load-cost (version used for this paper: release v1.0.0, commit dfcc49c).

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process
During the preparation of this work the author used Claude (Anthropic) in order to write analysis code and to draft and edit the text. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article.

## Table 1. AeroUp installation dates (first and last day of works)
| Turbine | First | Last | | Turbine | First | Last |
|---|---|---|---|---|---|---|
| T01 | 2022-07-23 | 2022-08-01 | | T12 | 2022-09-01 | 2023-05-09 |
| T02 | 2022-07-18 | 2022-07-28 | | **T13** | **2021-09-23** | **2021-09-29** |
| T03 | 2022-07-28 | 2022-08-14 | | T14 | 2022-08-30 | 2022-10-03 |
| T04 | 2022-07-31 | 2022-09-10 | | T15 | 2022-07-14 | 2023-05-26 |
| T05 | 2022-07-16 | 2022-07-22 | | T16 | 2022-08-27 | 2023-04-15 |
| T06 | 2022-07-16 | 2023-04-15 | | T17 | 2022-08-10 | 2022-08-27 |
| T07 | 2022-09-20 | 2023-05-25 | | T18 | 2022-08-22 | 2022-09-21 |
| T08 | 2022-08-08 | 2022-09-21 | | T19 | 2022-08-11 | 2022-08-23 |
| T09 | 2022-09-13 | 2022-10-03 | | T20 | 2022-08-13 | 2023-04-20 |
| T10 | 2022-08-04 | 2023-04-26 | | T21 | 2022-08-21 | 2022-09-01 |
| T11 | 2022-09-09 | 2023-05-18 | | | | |

Source: Hill_of_Towie_AeroUp_install_dates.csv in the dataset.

## Table 2. Drift-corrected effects on T13

| Outcome | Estimate | 95% CI (joint farm bootstrap) | Placebo-calibrated interval | Rank of 21 |
|---|---|---|---|---|
| Energy ratio change (normal operation) | +4.81% | 3.28 to 6.59% | 2.22 to 7.66% | 1 |
| Time at rated power | +1.67 pp | 0.65 to 2.80 | 1.17 to 2.14 | 1 |
| Torque exposure, m = 3 | +2.30% | 1.57 to 3.54% | 1.15 to 3.56% | 1 |
| Torque exposure, m = 10/3 | +2.15% | 1.45 to 3.34% | 1.08 to 3.33% | 1 |
| Torque exposure, m = 4 | +1.93% | 1.27 to 2.97% | | 1 |
| Torque exposure, m = 6 | +1.48% | 0.92 to 2.20% | | 1 |
| Torque exposure, m = 10 | +0.96% | 0.60 to 1.44% | 0.51 to 1.53% | 1 |
| Mean rotor speed | +1.66% | 1.12 to 2.26% | −0.28 to 3.27% | 2 |
| Mean torque | +3.98% | 2.75 to 5.46% | | |
| Pitch reference | +0.21° | 0.08 to 0.34°^a^ | | 1 |
| Within-10-min power std | +1.1% | −3.1 to +5.7%^a^ | | 3 |
| C (m = 3) | 0.48 | 0.41 to 0.60 | 0.28 to 0.89^b^ | |
| C (m = 10/3) | 0.45 | 0.37 to 0.57 | | |
| C (m = 10) | 0.20 | 0.14 to 0.27 | 0.09 to 0.46^b^ | |

^a^ Weekly block bootstrap of T13 with the placebo median held fixed. ^b^ 18 of 20 values between 0.44 and 0.59 (m = 3) and between 0.15 and 0.26 (m = 10).

## Figure captions
![](figs/fig1_design.png)
**Fig. 1.** Study design. (a) Retrofit timeline and analysis windows. T13 received vortex generators on 23–29 September 2021; the other 20 turbines were retrofitted between 14 July 2022 and 26 May 2023. Main estimate: T13 pre window 1 January 2020 – 22 September 2021, post window 30 September 2021 – 13 July 2022, with all 20 unmodified turbines as the reference fleet. Rollout check (deviation D1): the 20 later turbines, with T13 as the single reference, comparing October 2021 – July 2022 with June–July and October–December 2023. Known-null check (deviation D2): the same single-reference design applied to the same calendar months of 2020 and 2021, when no turbine changed. (b) Reference-cell estimator. Each 10-minute stamp is assigned to a farm-state cell (reference mean power × yaw sector); the target turbine's expected outcome is its own pre-window mean in that cell; effects compare observed and expected post-window records; the null drift is measured by running the identical estimator on each unmodified turbine.

![](figs/fig2_per_turbine.png)
**Fig. 2.** Raw effects in the T13 window for all 21 turbines, each estimated against the same reference set (all turbines except T13 and the target). Blue: T13 (retrofitted). Grey: unmodified turbines, whose effects measure the null drift of the estimator; dashed line: their median, which is subtracted to obtain drift-corrected effects.

![](figs/fig3_event_study.png)
**Fig. 3.** Same-calendar-month year-on-year contrast. Each bar is T13's monthly effect minus its effect in the same month a year earlier, minus the fleet median of the same difference. Grey band: 10th–90th percentile of the unmodified turbines. Open bar: retrofit month (September 2021).

![](figs/fig6_null.png)
**Fig. 4.** T13's drift-corrected effects (blue line) against the null distribution from pseudo-retrofit dates inside the pre-retrofit period (five dates × 21 turbines, each corrected by the median of the other 20). Dark grey: turbines other than T13 and T15–T17; light grey: T15–T17; blue bars: T13 at the pseudo dates. Dashed line: 95th percentile of all 105 null values. (a) Energy; (b) time at rated power; (c) torque exposure index, m = 3.

![](figs/fig5_cost_ratio.png)
**Fig. 5.** Exposure ratio C = D_m/E against the exponent m. Blue: T13, drift-corrected, with 95% intervals from the joint farm bootstrap. Grey: pure power scaling (×1.05, capped at rated, rotor speed unchanged) applied to T13's pre-retrofit records. Orange: T13's measured gain profile by farm reference power applied at unchanged rotor speed. Green: the same profile with the region-2 rotor-speed response of a Kω² torque law.

![](figs/fig4_power_bins.png)
**Fig. 6.** Where the gain arises. Share of T13's energy gain (a) and of its torque-cubed gain (b) by farm reference power, after subtracting the median of the unmodified turbines.

![](figs/fig7_replication.png)
**Fig. 7.** Rollout check on the 2022–23 retrofits with T13 as the single reference turbine (blue), and the known-null check of the same design on 2020 versus 2021 with T01 or T21 as reference (grey). Intervals: 95% turbine bootstrap. The 20 turbines share one reference and are not independent replications.

## References

[1] P. Veers, K. Dykes, E. Lantz, S. Barth, C.L. Bottasso, O. Carlson, et al., Grand challenges in the science of wind energy, Science 366 (2019) eaau2027. https://doi.org/10.1126/science.aau2027.

[2] J.C. Lin, Review of research on low-profile vortex generators to control boundary-layer separation, Prog. Aerosp. Sci. 38 (2002) 389–420. https://doi.org/10.1016/S0376-0421(02)00010-6.

[3] Z. Zhao, R. Jiang, J. Feng, H. Liu, T. Wang, W. Shen, M. Chen, D. Wang, Y. Liu, Researches on vortex generators applied to wind turbines: a review, Ocean Eng. 253 (2022) 111266. https://doi.org/10.1016/j.oceaneng.2022.111266.

[4] H. Mueller-Vahl, G. Pechlivanoglou, C.N. Nayeri, C.O. Paschereit, Vortex generators for wind turbine blades: a combined wind tunnel and wind turbine parametric study, in: Proc. ASME Turbo Expo 2012: Turbine Technical Conference and Exposition, Vol. 6, Copenhagen, 2012, pp. 899–914. https://doi.org/10.1115/GT2012-69197.

[5] D. Baldacchino, C. Ferreira, D. De Tavernier, W.A. Timmer, G.J.W. van Bussel, Experimental parameter study for passive vortex generators on a 30% thick airfoil, Wind Energy 21 (2018) 745–765. https://doi.org/10.1002/we.2191.

[6] W. Skrzypiński, M. Gaunaa, C. Bak, The effect of mounting vortex generators on the DTU 10MW reference wind turbine blade, J. Phys. Conf. Ser. 524 (2014) 012034. https://doi.org/10.1088/1742-6596/524/1/012034.

[7] D. Astolfi, F. Castellani, L. Terzi, Wind turbine power curve upgrades, Energies 11 (2018) 1300. https://doi.org/10.3390/en11051300.

[8] D. Astolfi, F. Castellani, M.L. Fravolini, S. Cascianelli, L. Terzi, Precision computation of wind turbine power upgrades: an aerodynamic and control optimization test case, J. Energy Resour. Technol. 141 (2019) 051205. https://doi.org/10.1115/1.4042450.

[9] L. Terzi, A. Lombardi, F. Castellani, D. Astolfi, Innovative methods for wind turbine power curve upgrade assessment, J. Phys. Conf. Ser. 1102 (2018) 012036. https://doi.org/10.1088/1742-6596/1102/1/012036.

[10] F. Castellani, D. Astolfi, L. Terzi, Wind turbine power curve upgrades: methods for the assessment and test cases study, J. Phys. Conf. Ser. 1452 (2020) 012004. https://doi.org/10.1088/1742-6596/1452/1/012004.

[11] G. Lee, Y. Ding, L. Xie, M.G. Genton, A kernel plus method for quantifying wind turbine performance upgrades, Wind Energy 18 (2015) 1207–1219. https://doi.org/10.1002/we.1755.

[12] Y. Shin, Y. Ding, J.Z. Huang, Covariate matching methods for testing and quantifying wind turbine upgrades, Ann. Appl. Stat. 12 (2018) 1271–1292. https://doi.org/10.1214/17-AOAS1109.

[13] E. Latiffianti, Y. Ding, S. Sheng, L. Williams, M. Morshedizadeh, M. Rodgers, Analysis of leading edge protection application on wind turbine performance through energy and power decomposition approaches, Wind Energy 25 (2022) 1203–1221. https://doi.org/10.1002/we.2722.

[14] R. Wagner, I. Antoniou, S.M. Pedersen, M.S. Courtney, H.E. Jørgensen, The influence of the wind speed profile on wind turbine performance measurements, Wind Energy 12 (2009) 348–362. https://doi.org/10.1002/we.297.

[15] C.M. St. Martin, J.K. Lundquist, A. Clifton, G.S. Poulos, S.J. Schreck, Wind turbine power production and annual energy production depend on atmospheric stability and turbulence, Wind Energy Sci. 1 (2016) 221–236. https://doi.org/10.5194/wes-1-221-2016.

[16] E. Gonzalez, B. Stephen, D. Infield, J.J. Melero, Using high-frequency SCADA data for wind turbine performance monitoring: a sensitivity study, Renew. Energy 131 (2019) 841–853. https://doi.org/10.1016/j.renene.2018.07.068.

[17] D. Astolfi, F. Castellani, A. Lombardi, L. Terzi, Multivariate SCADA data analysis methods for real-world wind turbine power curve monitoring, Energies 14 (2021) 1105. https://doi.org/10.3390/en14041105.

[18] H. Hwangbo, Y. Ding, O. Eisele, G. Weinzierl, U. Lang, G. Pechlivanoglou, Quantifying the effect of vortex generator installation on wind power production: an academia-industry case study, Renew. Energy 113 (2017) 1589–1597. https://doi.org/10.1016/j.renene.2017.07.009.

[19] H. Moon, J. Jeong, S. Park, K. Ha, J.-H. Jeong, Numerical and experimental validation of vortex generator effect on power performance improvement in MW-class wind turbine blade, Renew. Energy 212 (2023) 443–454. https://doi.org/10.1016/j.renene.2023.04.104.

[20] M. Manolesos, Y. Celik, H. Ramsay, R. Karande, B. Wood, I. Dinwoodie, I. Masters, M. Harrold, G. Papadakis, Performance improvement of a Vestas V52 850kW wind turbine by retrofitting passive flow control devices, J. Phys. Conf. Ser. 2767 (2024) 022027. https://doi.org/10.1088/1742-6596/2767/2/022027.

[21] IEC 61400-12-1:2022, Wind energy generation systems – Part 12-1: Power performance measurements of electricity producing wind turbines, Edition 3.0, International Electrotechnical Commission, Geneva, 2022.

[22] A. Clerc, A. Torner, A. Bamberger, M. Lannic, E. Lingkan, S. Northover-Naylor, G. Powles, K. Watts, T. Young, Open-source dataset and AEP validation from a real-world case study of turbine upgrades worth 5% AEP, presentation at WindEurope Technology Workshop 2025, Istanbul, 20 June 2025. https://www.res-group.com/wp-content/uploads/2026/05/RES-open-source-AEP-validation-of-turbine-upgrades.pdf (accessed 1 November 2026).

[23] J. Ribrant, L.M. Bertling, Survey of failures in wind power systems with focus on Swedish wind power plants during 1997–2005, IEEE Trans. Energy Convers. 22 (2007) 167–173. https://doi.org/10.1109/TEC.2006.889614.

[24] P.J. Tavner, J. Xiang, F. Spinato, Reliability analysis for wind turbines, Wind Energy 10 (2007) 1–18. https://doi.org/10.1002/we.204.

[25] F. Spinato, P.J. Tavner, G.J.W. van Bussel, E. Koutoulakos, Reliability of wind turbine subassemblies, IET Renew. Power Gener. 3 (2009) 387–401. https://doi.org/10.1049/iet-rpg.2008.0060.

[26] S. Faulstich, B. Hahn, P.J. Tavner, Wind turbine downtime and its importance for offshore deployment, Wind Energy 14 (2011) 327–337. https://doi.org/10.1002/we.421.

[27] J.M. Pinar Pérez, F.P. García Márquez, A. Tobias, M. Papaelias, Wind turbine reliability analysis, Renew. Sustain. Energy Rev. 23 (2013) 463–472. https://doi.org/10.1016/j.rser.2013.03.018.

[28] J. Carroll, A. McDonald, D. McMillan, Failure rate, repair time and unscheduled O&M cost analysis of offshore wind turbines, Wind Energy 19 (2016) 1107–1119. https://doi.org/10.1002/we.1887.

[29] E. Artigao, S. Martín-Martínez, A. Honrubia-Escribano, E. Gómez-Lázaro, Wind turbine reliability: a comprehensive review towards effective condition monitoring development, Appl. Energy 228 (2018) 1569–1583. https://doi.org/10.1016/j.apenergy.2018.07.037.

[30] A.R. Nejad, J. Keller, Y. Guo, S. Sheng, H. Polinder, S. Watson, et al., Wind turbine drivetrains: state-of-the-art technologies and future development trends, Wind Energy Sci. 7 (2022) 387–411. https://doi.org/10.5194/wes-7-387-2022.

[31] S. Kim, H. Im, B. Kim, Numerical analysis of the effects of vortex generators attached to contaminated blades on wind turbine power performance and fatigue load, J. Mech. Sci. Technol. 37 (2023) 3515–3524. https://doi.org/10.1007/s12206-023-0617-0.

[32] C. Bak, W. Skrzypiński, M. Gaunaa, H. Villanueva, N.F. Brønnum, E.K. Kruse, Full scale wind turbine test of vortex generators mounted on the entire blade, J. Phys. Conf. Ser. 753 (2016) 022001. https://doi.org/10.1088/1742-6596/753/2/022001.

[33] F. Ji, F. Yang, Z. Wang, S. Liu, D. Xu, W. Zhou, L. Wu, X. Chu, Y. Zhong, Y. Ke, Modular performance testing and comparative evaluation method for wind turbine retrofit schemes, Machines 14 (2026) 965. https://doi.org/10.3390/machines14090965.

[34] N. Noppe, W. Weijtjens, C. Devriendt, Modeling of quasi-static thrust load of wind turbines based on 1 s SCADA data, Wind Energy Sci. 3 (2018) 139–147. https://doi.org/10.5194/wes-3-139-2018.

[35] L. Vera-Tudela, M. Kühn, Analysing wind turbine fatigue load prediction: the impact of wind farm flow conditions, Renew. Energy 107 (2017) 352–360. https://doi.org/10.1016/j.renene.2017.01.065.

[36] N. Dimitrov, M.C. Kelly, A. Vignaroli, J. Berg, From wind to loads: wind turbine site-specific load estimation with surrogate models trained on high-fidelity load databases, Wind Energy Sci. 3 (2018) 767–790. https://doi.org/10.5194/wes-3-767-2018.

[37] A. Clerc, E. Lingkan, Hill of Towie wind farm open dataset, version 2.1.0 [dataset], Zenodo, 2026. https://doi.org/10.5281/zenodo.22662930.

[38] A. Abadie, A. Diamond, J. Hainmueller, Synthetic control methods for comparative case studies: estimating the effect of California's tobacco control program, J. Am. Stat. Assoc. 105 (2010) 493–505. https://doi.org/10.1198/jasa.2009.ap08746.

[39] A. Abadie, Using synthetic controls: feasibility, data requirements, and methodological aspects, J. Econ. Lit. 59 (2021) 391–425. https://doi.org/10.1257/jel.20191450.

[40] T.G. Conley, C.R. Taber, Inference with "difference in differences" with a small number of policy changes, Rev. Econ. Stat. 93 (2011) 113–125. https://doi.org/10.1162/REST_a_00049.

[41] P. Fleming, J. King, K. Dykes, E. Simley, J. Roadman, A. Scholbrock, et al., Initial results from a field campaign of wake steering applied at a commercial wind farm – Part 1, Wind Energy Sci. 4 (2019) 273–285. https://doi.org/10.5194/wes-4-273-2019.

[42] T. Burton, N. Jenkins, D. Sharpe, E. Bossanyi, Wind Energy Handbook, second ed., John Wiley & Sons, Chichester, 2011. https://doi.org/10.1002/9781119992714.

[43] ISO 281:2007, Rolling bearings – Dynamic load ratings and rating life, International Organization for Standardization, Geneva, 2007.

[44] A.R. Nejad, Z. Gao, T. Moan, On long-term fatigue damage and reliability analysis of gears under wind loads in offshore wind turbine drivetrains, Int. J. Fatigue 61 (2014) 116–128. https://doi.org/10.1016/j.ijfatigue.2013.11.023.

[45] H.R. Künsch, The jackknife and the bootstrap for general stationary observations, Ann. Stat. 17 (1989) 1217–1241. https://doi.org/10.1214/aos/1176347265.

[46] M. Bertrand, E. Duflo, S. Mullainathan, How much should we trust differences-in-differences estimates?, Q. J. Econ. 119 (2004) 249–275. https://doi.org/10.1162/003355304772839588.

[47] E.A. Bossanyi, The design of closed loop controllers for wind turbines, Wind Energy 3 (2000) 149–163. https://doi.org/10.1002/we.34.

[48] K.E. Johnson, L.Y. Pao, M.J. Balas, L.J. Fingersh, Control of variable-speed wind turbines: standard and adaptive techniques for maximizing energy capture, IEEE Control Syst. Mag. 26 (2006) 70–81. https://doi.org/10.1109/MCS.2006.1636311.

[49] Q. Wang, Y. Vidal, Neighbor-based estimation of wind turbine power for upgrade performance assessment, J. Phys. Conf. Ser. 3232 (2026) 012001. https://doi.org/10.1088/1742-6596/3232/1/012001.

[50] I. Staffell, R. Green, How does wind farm performance decline with age?, Renew. Energy 66 (2014) 775–786. https://doi.org/10.1016/j.renene.2013.10.041.
