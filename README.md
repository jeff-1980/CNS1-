# Training-file composition and the collapse class of deep bearing fault classifiers under strong noise

Code, result records, preregistrations and figure/table generators for the manuscript submitted to *Mechanical Systems and Signal Processing*.

## Layout

| path | content |
|---|---|
| `code/` | training, inference and analysis scripts; `pbx/` data loaders and models |
| `results/*.json` | per-instance result records of every experiment (dominant class, concentration, confusion counts, weight paths and SHA-256) |
| `results/PREREG_*.md`, `B_RULE_FREEZE.md` | preregistrations; their SHA-256 values are in `results/registry/*_sha256.txt` and are checked by each `analyze_*.py` at run time |
| `results/VERDICT_*.md`, `DEVIATIONS_*.md` | preregistered verdicts and logged deviations |
| `results/registry/training_instances.csv` | registry of all 1081 training instances (1074 analysis + 7 smoke): data subset, seeds, steps, weight path and SHA-256 |
| `results/registry/raw_hashes.csv` | SHA-256 of every raw data file used |
| `figures_reference/`, `supplement/MSSP_SUPPLEMENT_reference.md` | the figures and supplementary tables as submitted |

## Regenerate figures, supplementary tables and preregistered verdicts (no GPU, no raw data)

```bash
pip install -r requirements-figures.txt
python code/mssp_figs.py          # figures/fig1–fig5, figS1
python code/build_supplement.py   # supplement/MSSP_SUPPLEMENT.md (Tables S1–S7)
python code/analyze_2x2.py        # likewise analyze_e5, analyze_cinit, analyze_wp2, analyze_wp5, analyze_b, analyze_wp5b2, wp5_f1_paired
python code/analyze_ext.py e1 e2 e3
```

`analyze_e4.py` and `analyze_wp1.py` additionally compare with the published results of the reference benchmark and need `PAPERB_RESULTS` (see below). `collapse_defs.py` rewrites `collapse_defs_sensitivity.json` without the field `non_unanimous_instances`, which `s2_two_instances.py` adds afterwards by re-inference (needs weights and PU data).

## Data

The raw data are not redistributed. Obtain them from the providers and check them with `python code/prepare_data.py`, which compares every file with `results/registry/raw_hashes.csv` and builds the CWRU 16-file subset (symbolic links) used by the reference protocol.

| dataset | source | expected layout (environment variable) |
|---|---|---|
| CWRU, 12 kHz drive end, 40 files | Case Western Reserve University Bearing Data Center | `.mat` files named by CWRU file number, e.g. `100.mat` (`CWRU40_DIR`); file-to-split map in `results/cwru40_file_map.csv` |
| Paderborn (PU) | KAt-DataCenter, Paderborn University [Lessmeier et al. 2016] | bearing folders `K001 … KI21`, any depth (`PU_DIR`) |
| JNU | Jiangnan University bearing data [Li et al. 2013] | 12 `.csv` files, e.g. `ib600_2.csv` (`JNU_DIR`) |

Other paths: `CNS_ROOT` (repository root, default: parent of `code/`), `CWRU16_DIR` (default `data/cwru_16`), `PAPERB_CODE` and `PAPERB_RESULTS` (code package and published result package of Wang and Tang 2026, used only by the reference-protocol scripts `e4_paperB_harness.py`, `wp1_harness.py`, `select_rpm.py`, `analyze_e4.py`, `analyze_wp1.py` and by the Mamba models).

## Trained weights

The 1051 weight files (≈1.1 GB) are attached to release [v1.0](https://github.com/jeff-1980/CNS1-/releases/tag/v1.0) as `CNS1_weights_part1.tar` and `CNS1_weights_part2.tar` (checksums in `SHA256SUMS.txt`). Unpack both into `results/` (giving `results/ckpt_2x2/`, `results/ckpt_e5/`, …) and run `python code/verify_weights.py`. 138 registry rows from early screening runs have no kept weights and are marked `missing`.

## Retraining and re-inference

GPU required. Run the training and re-inference scripts from `code/` (their default output paths are `../results/...`). All entry scripts except `b_dev.py` and `reconstruct_init.py` have a `--smoke` mode that checks data construction and model shapes without training.

| experiment | script | analysis |
|---|---|---|
| 2 × 2 coverage × ratio; E3 (lstm, transformer; `--models lstm transformer --out ../results/arm2x2_lt.json`) | `arm2x2.py` | `analyze_2x2.py` |
| E1 cnn1d factorial, E2 size vs file count | `arm_ext.py` | `analyze_ext.py e1 e2 e3` |
| E5 size identification | `e5_size_id.py` (`run_e5_chain.sh`) | `analyze_e5.py` |
| C-INIT fixed initialization | `cinit.py` | `analyze_cinit.py` |
| reference protocol (E4, WP1) | `e4_paperB_harness.py`, `wp1_harness.py`, `run_wp1.sh` | `analyze_e4.py`, `analyze_wp1.py` |
| PU damage extent (WP2) | `pu_wp2.py`, `run_wp2.sh` | `analyze_wp2.py` |
| decision consequences (WP5), selective classification (WP5b) | `wp5_consequence.py`, `wp5b2_selective.py` (`run_wp5.sh`, `run_wp5b2.sh`) | `analyze_wp5.py`, `analyze_wp5b2.py`, `wp5_f1_paired.py` |
| B line | `b_dev.py`, `b_confirm.py` | `analyze_b.py` |
| initialization reconstruction of the original runs | `reconstruct_init.py` | — |

GPU training is not bitwise deterministic: retraining from identical initial weights gave different weights (max. difference 0.45) with the same dominant class at −8 dB. Re-inference with the archived weights is reproducible.

## Notes

- Result records keep the absolute paths of the machine on which they were produced; only the relative weight paths are used by the scripts.
- Comments and the preregistration / verdict documents are partly in Chinese; the manuscript and its supplementary tables are in English.
