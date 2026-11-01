# Drivetrain load cost of a vortex-generator retrofit (Hill of Towie)

Code and results for: Y. Wang, Y. Tang, "Energy gain and drivetrain torque exposure of a vortex-generator retrofit: a field assessment at an operating wind farm" (submitted to *Renewable Energy*).

## Data
Hill of Towie wind farm open dataset v2.1.0 (A. Clerc, E. Lingkan), Zenodo, https://doi.org/10.5281/zenodo.22662930, CC BY 4.0. Download the yearly archives `2020.zip` … `2023.zip`, `Hill_of_Towie_ShutdownDuration.zip` and the AeroUp install-date file into one folder (`ROOT`). The data are not redistributed here.

## Layout
| Folder | Content |
|---|---|
| `frozen/` | Scripts and plan frozen with SHA-256 hashes before they were run (see below). |
| `analysis/` | Paper analysis (`paper_aeroup.py`), wind-speed reconciliation (`paper_aeroup_ws.py`) and figure scripts. |
| `results/run_out/` | CSV outputs and log of `paper_aeroup.py` (2000 bootstrap draws) used in the paper. |
| `results/run_out_ws/` | Results of `paper_aeroup_ws.py`, transcribed from its log. |
| `results/run_out_rev/` | Results of `paper_aeroup_rev.py` (joint farm bootstrap, pseudo-retrofit dates, injection into T13, mechanism counterfactuals, control signatures) and `paper_aeroup_alarms.py`, transcribed from their logs. |

## Order of the study
Hashes are checked with `sha256sum -c <file>.sha256.txt`, run inside `frozen/` (baseline, plan, K3, K3b) or `frozen/scripts/` (kill_aeroup_A).
1. `frozen/bl_20261030_aeroup.py` – baseline (raw T13 effect, placebo checks), frozen before its first run (`bl_20261030_aeroup.FROZEN.sha256.txt`).
2. `frozen/PLAN_v1_FROZEN.md` – plan with tests K1–K5 and their pass criteria, frozen after the baseline and before any test (`PLAN_v1_FROZEN.sha256.txt`; written in Chinese). K1 known truth, K2 drift-corrected T13 effect with rank test, K3 rollout replication, K4 seasonal re-weighting, K5 within-10-minute variability (informational).
3. `frozen/scripts/kill_aeroup_A.py` (K1, K2, K4, K5), `kill_aeroup_K3.py` (K3 after deviation D1), `kill_aeroup_K3b.py` (known-null check, deviation D2); each frozen before running (`*.FROZEN.sha256.txt`).
4. `frozen/DEVIATIONS.md` – deviations D1 and D2, each written before the affected script was run; `kill_results_2026-11-01_A_K3.md` – test results.
5. `analysis/` – descriptive paper analyses added after the tests (no new pass/fail criteria), including `paper_aeroup_rev.py` and `paper_aeroup_alarms.py`.

## Reproduce
Python 3.10+, numpy, pandas, matplotlib.
```
python analysis/paper_aeroup.py ROOT out 2000
python analysis/paper_aeroup_ws.py ROOT out_ws 1000
python analysis/paper_aeroup_rev.py ROOT out_rev 1000
python analysis/paper_aeroup_alarms.py ROOT out_alarms
python analysis/make_figs.py out figs out_rev
python analysis/make_fig_null.py out_rev/R2_pseudo_dates_all_turbines.csv figs
python analysis/make_fig1.py figs
python analysis/make_figS5.py results/run_out_ws/ws_bins_from_log.csv figs
python analysis/make_graphical_abstract.py out figs out_rev/R1_joint_bootstrap_summary.csv
```

## License
Code: MIT. Results files: CC BY 4.0.
