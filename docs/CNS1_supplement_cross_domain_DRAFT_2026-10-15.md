# Supplementary section draft: cross-domain boundary tests (C6-P, T1, T2, T3)

Draft for the IEEE Access supplement (2026-10-15). It defines the four experiments cited in the main-text paragraph "Scope of the coverage–attractor result". All numbers are copied from the verdict files named below; no new claims. Insert as the last supplementary section (numbered after Table S8) and add one sentence to the main-text paragraph: "Designs, preregistrations and full results are given in Supplementary Section S9 and in the repository (results/PREREG_*.md, results/VERDICT_*.md)."

## S9. Cross-domain boundary tests

**Common protocol.** All four experiments reuse the training and noise protocol of the main text (AdamW, learning rate 1e-3, weight decay 1e-4, batch 64, gradient clipping 1.0, inverse-frequency class weights, per-window additive white Gaussian noise). The primary outcome of T1–T3 is seed agreement A: the mean pairwise agreement of the predicted class among models of one cell, evaluated on the same 1000 pure-noise inputs. Confidence intervals are 95% bootstrap intervals over models, excluding self-pairs. Preregistrations were frozen (SHA-256 recorded in results/registry/) before any training; deviations are logged in results/DEVIATIONS_*.md.

**C6-P (preregistered).** Tested whether the coverage effect transfers to single-lead ECG (PTB-XL, lead II; classes NORM, MI, STTC, CD; wdcnn, drsn, lstm) and natural images (CIFAR-100 superclasses scenes, fish, flowers, vehicles_1; cnn4, ResNet-18, ViT-tiny). Three training arms differed only in subclass coverage; 5 seeds per cell; 90 models. Both domains were INCONCLUSIVE: the −8 dB dominant class split across seeds within arms (ECG clean accuracy about 0.60; images 0.69–0.83), so the coverage effect could be neither confirmed nor refuted. Files: PREREG_C6_crossdomain_pilot.md, VERDICT_C6P.md.

**T1 (post hoc, inference only).** Compared the 189 CWRU models of experiment E5 with the 90 C6-P models. Seed agreement separated the domains (CWRU 1.000 for all three architectures; ECG and images 0.39–0.72), whereas the per-model noise margin did not. On CWRU, high-pass noise of equal power reproduced the white-noise dominant class in 0.79–0.87 of models and low-pass noise in 0.21–0.35. File: POSTHOC_T1_margin.md.

**T2 (preregistered).** Domain (CWRU, 12-lead ECG) × learning strength (strong; weakened by fewer steps; weakened by label noise), 10 seeds per cell, 180 models. All three architectures were INDETERMINATE because ECG could not be trained beyond 0.79 clean accuracy (STRENGTH_GAP). Descriptively, CWRU strong cells had A = 1.000 and label-noise weakening reduced it to 0.22–0.33. File: PREREG_T2_domain_strength.md, VERDICT_T2.md.

**T3 (preregistered).** Matched learning strength: CWRU against a non-bearing time series trained to at least 0.95 clean accuracy. Speech Commands (four words) was infeasible (0.92–0.93); StarLightCurves (UCR 2018, K = 3) was selected (0.96–0.97). High tier: 3000 steps; low tier: label noise calibrated to about 0.78; 10 seeds per cell; 150 models.

| Cell | wdcnn | drsn | lstm |
|---|---|---|---|
| CWRU high | 1.000 (0.943) | 1.000 (0.994) | 0.966 (0.979) |
| StarLightCurves high | 0.409 (0.971) | 0.644 (0.968) | 0.359 (0.965) |
| CWRU low | 0.490 (0.770) | 0.358 (0.760) | 0.275 (0.708) |
| StarLightCurves low | 0.388 (0.774) | 0.271 (0.783) | 0.298 (0.767) |
| ECG natural | 0.571 (0.772) | 0.523 (0.784) | 0.793 (0.790) |

Seed agreement A on pure noise; mean clean accuracy of the 10 models in parentheses.

Domain difference at the high tier, ΔD = A[CWRU] − A[StarLightCurves] (equivalence margin 0.15): wdcnn +0.591 [0.257, 0.775], DATA; drsn +0.356 [0.000, 0.641], INDETERMINATE; lstm +0.607 [0.373, 0.794], DATA. Cross-architecture verdict: DATA. Flags: CWRU wdcnn high tier 0.944 < 0.95 (HI_GAP) and K = 3 versus 4 (K_MISMATCH); both bias against a DATA verdict. Files: PREREG_T3_matched_strength.md, VERDICT_T3.md, DEVIATIONS_T3.md.

**Loader note.** The CWRU cells of T1–T3 were trained with the pre-correction loader (Section 3.8). Re-scoring the existing T2/T3 models on Normal windows never seen in training left the clean accuracy of well-trained CWRU models unchanged (Normal recall 1.000); low-tier cells changed by −0.05 to +0.02. Seed agreement was not re-measured on the corrected training pool.
