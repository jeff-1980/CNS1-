# h2kit: baseline-first check for operational records

Rule H2 of CNS规划4# says a candidate direction must first beat the simplest baseline on real data. For datasets of
**units × time series × event table** (freezers, turbines, drives, pumps ...), h2kit answers one question before any
model is trained: do simple per-day features separate the days before an event from ordinary days of the same unit
better than random dates on that unit?

Evaluation type: retrospective, real ground truth, within-unit comparison, no model.

## Method

All thresholds are set in the YAML config before the run.

1. Daily features: the mean and std of every numeric channel. Optionally also the std of the first difference of
   chosen channels and the daily count of an event log. Days below `min_coverage` are dropped.
2. Pre window: the `pre_days` days before the event. An event is usable if its pre window has at least `min_pre`
   covered days and the unit has at least `min_ctrl` control days.
3. Controls: days of the same unit at least `gap_days` from every avoid date. Avoid dates are all events, plus
   optional extra dates such as service visits.
4. Statistic: AUC per event and feature, summarised as S = mean |AUC − 0.5|. This ignores direction, because event
   types differ.
5. Placebo: `n_placebo` draws, each moving every event to a random eligible date on the same unit.
   p = (1 + #null ≥ observed) / (n_placebo + 1).
6. A feature passes only if p < alpha / n_features (Bonferroni).

## Use

```bash
~/tfvenv/bin/python h2kit.py --smoke --out /tmp/h2smoke                                  # self-test first
~/tfvenv/bin/python h2kit.py --config configs/ult_revco.yaml --out /tmp/h2_ult            # real data
```

`--smoke` is a known-truth check on synthetic hourly data: 20 units, 1500 days, unit-level random-walk drift and
2 events per unit. The planted case shifts channel `x` by +1.5 sd during the 14 days before each event, and the
tool must find `x_mean` and nothing else. The null case has no shift, and nothing may pass. The command exits with 1
if either check fails. Result on 2026-11-04: **SMOKE PASS**. In the planted case only `x_mean` passed; in the null
case nothing passed.

## Config fields

`timeseries.path` takes a `{unit}` placeholder and accepts parquet or csv. The other fields are
`timeseries.time_col`, an optional `event_log` (path and time_col), and `events` (path, unit_col, date_col,
date_format). `avoid_dates_json` is optional and maps unit → [dates]. The remaining settings are `exclude_cols`,
`diff_std_cols`, `coverage` (`distinct_minutes` or `relative_count`) and `params`.

## First real use case: ULT freezers

`configs/ult_revco.yaml` uses the data in `D:/audit_data/ult_freezer`. It reproduces `../triage/ult_baseline_result.md`
(commit 99e4a10) exactly: max abs difference 0.0 over all 22 features in S, signed AUC, placebo quantiles and p.
The run used 24 units and 29 usable events of 46, and 0 of 22 features passed. Outputs are in `examples/ult_revco/`.
