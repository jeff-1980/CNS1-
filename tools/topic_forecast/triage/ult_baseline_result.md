# ULT freezer baseline-first check (refrigeration candidate)

Date: 2026-11-04. Run on jeff's PC (CPU only, no models). Script: `ult_baseline.py` in this folder (rules fixed in its docstring before the full run); local copy in `C:/Users/ThinkPad/claude_project/ult_baseline/`.

## Data

- Huang et al., Sci Data 2023, doi 10.1038/s41597-023-02808-6; figshare 10.6084/m9.figshare.24033324 (CC0). Paper CC BY 4.0.
- Downloaded: processed `<id>_temp.parquet` + `_events.parquet` for the 25 Revco freezers with service visits (7 via the DTU GitLab, 18 extracted member-by-member from the figshare zip with CRC check), `freezer_faults.csv` (46 fault rows), and 85 service/PM visit dates taken from the repository's `service_reports` folder names. Location `D:/audit_data/ult_freezer` (~4.3 GB).
- Columns: Datetime, RTD (chamber), 1st Suc., Cond. Air In, Evap. In, Evap. Out, 2nd Suc., Chil. water In, 2nd Sump, H.E., SetPoint, Mains Voltage, State, event fields. Sampling 30 s to 1 min. **No compressor power / current / duty-cycle column**; compressor features are the stage temperatures.

## Design (frozen)

- Pre-fault window: 14 days before the registered fault date; event usable if >= 10 days with >= 50% minute coverage and >= 60 control days.
- Controls: same-freezer days >= 60 days from every service/PM visit and fault date.
- Features: daily mean and std of each temperature channel and mains voltage, std of the RTD first difference, daily event-log count (22 features).
- Statistic per feature: S = mean over events of |AUC(pre vs control) - 0.5|. Placebo: 1000 draws, each event date replaced by a random eligible date on the same freezer. Pass only if p < 0.05/22 = 0.00227 (Bonferroni).

## Counts

- Freezers with readable data: 24; fault rows: 46; **events usable: 29**; excluded for insufficient pre-fault coverage: 17 (e.g. all 3 of 806023 have no data before the faults).

## Results

| feature | S observed | mean signed AUC | placebo S 2.5% | placebo S 97.5% | p | passes |
|---|---|---|---|---|---|---|
| n_events | 0.096 | 0.566 | 0.038 | 0.076 | 0.003 | no |
| 1st Suc._mean | 0.326 | 0.430 | 0.203 | 0.302 | 0.005 | no |
| Evap. Out_mean | 0.301 | 0.501 | 0.197 | 0.307 | 0.036 | no |
| 2nd Suc._mean | 0.303 | 0.407 | 0.202 | 0.311 | 0.043 | no |
| 2nd Sump_mean | 0.295 | 0.278 | 0.200 | 0.303 | 0.065 | no |
| Cond. Air In_mean | 0.276 | 0.622 | 0.190 | 0.295 | 0.094 | no |
| Cond. Air In_std | 0.249 | 0.445 | 0.170 | 0.265 | 0.105 | no |
| Evap. In_mean | 0.282 | 0.608 | 0.198 | 0.307 | 0.140 | no |
| Evap. Out_std | 0.276 | 0.533 | 0.196 | 0.300 | 0.147 | no |
| 2nd Sump_std | 0.273 | 0.436 | 0.197 | 0.298 | 0.154 | no |
| Chil. water In_std | 0.271 | 0.323 | 0.199 | 0.304 | 0.234 | no |
| RTD_std | 0.257 | 0.554 | 0.189 | 0.296 | 0.326 | no |
| Evap. In_std | 0.261 | 0.535 | 0.200 | 0.304 | 0.378 | no |
| 1st Suc._std | 0.258 | 0.536 | 0.199 | 0.308 | 0.415 | no |
| 2nd Suc._std | 0.253 | 0.459 | 0.200 | 0.312 | 0.518 | no |
| Chil. water In_mean | 0.253 | 0.584 | 0.201 | 0.305 | 0.520 | no |
| RTD_diff_std | 0.241 | 0.500 | 0.194 | 0.301 | 0.559 | no |
| H.E._mean | 0.242 | 0.492 | 0.198 | 0.307 | 0.643 | no |
| H.E._std | 0.239 | 0.430 | 0.198 | 0.306 | 0.713 | no |
| Mains Voltage_std | 0.180 | 0.481 | 0.146 | 0.249 | 0.742 | no |
| Mains Voltage_mean | 0.188 | 0.420 | 0.198 | 0.306 | 0.988 | no |
| RTD_mean | 0.159 | 0.574 | 0.197 | 0.303 | 1.000 | no |

## Verdict

**No simple feature beats the placebo at the frozen threshold (0 of 22).** Random dates on the same freezer already give S of about 0.20-0.30: each freezer drifts a lot over time, so 'the 14 days before a fault look different' is mostly a time effect.

Two features come close but do not pass: the daily event-log count (S 0.096 vs placebo 97.5% 0.076, p = 0.003) and the 1st-stage suction temperature mean (S 0.326 vs 0.302, p = 0.005; signed AUC 0.43, i.e. lower before faults). The event count is likely circular: alarms and door activity rise when staff already notice a problem and call service, so it is not an early warning. The suction-temperature lead is one borderline feature across 29 events and would need a pre-registered confirmatory test on held-out events before it counts.

Caveats: fault dates are service-report dates (the paper notes they may be offset from the true onset); the 46 rows mix fault types; earlier partial runs (4 and 22 usable events) gave the same verdict and are superseded.
