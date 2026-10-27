# Results (draft v1, 2026-10-27)

Numbers from `results/main_analysis/hypotheses.json` (script `code/audit/main/analysis.py`, commit ebf652f) and `results/main_objects/attempts.csv`. Every deviation cited is in `results/DEVIATIONS_AUDIT.md`. Wording follows prereg §8: "under protocol X, conclusion Y changes from A to B"; no paper is called wrong.

## 3.1 Attempted and runnable objects (S7)

We attempted 83 code-paper pairs in the pre-registered random order: 32 from the original frame and 51 from the expansion search, which was registered before any expansion object was cloned. Of these, 38 (46%) were runnable under the attempt protocol and 45 were excluded. The main exclusion reasons were:
- code that would not run as released (10);
- data that were private or unavailable (5 + 5) or could not be downloaded (2);
- a missing training script (4);
- a headline result produced by a non-deep model (3);
- a task that was not bearing diagnosis (3);
- a single training run over 2 GPU-h (2).

The 38 runnable objects contributed 88 method groups and 880 training runs (10 seeds or unseeded repeats per group). For each run, we compared the value the authors' own code reported with the value our epoch recorder logged. They agreed to within float32 rounding in every run except five Teacher seeds of object 62. There the difference was 4e-5 to 1e-4 in macro-F1, about one window in 25,000, because we ran inference in batches at every epoch while the authors ran it one sample at a time. For seven objects (13, 14, 21, 24, 62, 73, 74), the check result is recorded only in attempts.csv; the per-run protocol files carry no check field. [TODO: confirm the per-run check logs for these seven.]

The authors' checkpoint rule was the last epoch in 18 objects. In the other 20 it was a selection rule:
- the best validation accuracy or lowest validation loss (11 objects);
- the best accuracy on the reported test or target split itself (9 objects).

## 3.2 Inflation (H-A1, H-A2)

**H-A1 (supported).** Among the 20 objects whose authors did not report the last epoch, the object-level inflation I_o = author − last had a median of 0.37 percentage points (bootstrap 95% CI 0.04–1.20; one-sided Wilcoxon p = 0.002, Holm p = 0.006).
- Fourteen objects were inflated, two were exactly zero and four were slightly deflated (largest −1.58 pp, object 21).
- The three largest inflations were 7.05 pp (object 36), 5.90 pp (76) and 5.59 pp (34). All three select on the reported split.

**H-A2 (supported).** Across all 88 method groups, the counterfactual gap best_rep − last grew with task headroom h = 1 − mean last-epoch accuracy (Spearman ρ = 0.74; object-cluster bootstrap 95% CI 0.54–0.88; Holm p < 0.001).
- The median gap was 0.08 pp in the 29 groups with h ≤ 0.02 and 3.76 pp in the 35 groups with h > 0.10.
- Overall, half of the groups (51%) had a gap above 1 pp and 17% had a gap above 5 pp.
- Restricting to the authors' own rule (author − last, 20 objects) gave the same direction with a wider interval (ρ = 0.49, CI −0.01 to 0.83).

## 3.3 Variance and significance (H-B1, H-B2, S1)

**H-B1 (not supported after correction).** The authors' rule narrowed the seed-to-seed spread relative to the last epoch only slightly. The object-level median log(SD_author / SD_last) was −0.06 (n = 19), with one-sided Wilcoxon p = 0.029 before correction and Holm p = 0.058 after.

For comparison, the counterfactual best_rep rule compressed variance clearly: median log ratio −0.46 over 35 objects, p < 0.001, reported as prereg §4 requires. This is the effect the pilot predicted, but the selection rules authors actually used produced much less of it.

**H-B2 (not supported).** Eight objects that do not report the last epoch had two or more method groups, giving 43 group pairs. Paired t-test verdicts under author and under last differed in 5 pairs. The seed-swap null placed 7 changes at its 95th percentile (p = 0.42).

**S1.** Across 86 tests per protocol, no comparison that was significant under the author protocol reversed sign under last, best_rep or minloss, and one reversed under budget50. The pilot's "significance created or erased" pattern therefore appears in this sample only at the rate expected by chance.

## 3.4 Sensitivity analyses

The four verdicts are unchanged in each of the following cases:
- dropping the simulated-data object (49);
- dropping the four objects admitted through a one-line or dead-import repair (13, 42, 77, 82);
- dropping all five.

Only H-B1 weakens further when objects are dropped:

| Excluded | H-A1 Holm p | H-A2 ρ | H-B1 Holm p | H-B2 observed / null q95 |
|---|---|---|---|---|
| none (main) | 0.006 | 0.74 | 0.058 | 5 / 7 |
| 49 | 0.010 | 0.74 | 0.093 | 5 / 7 |
| 13, 42, 77, 82 | 0.010 | 0.76 | 0.093 | 5 / 6 |
| both | 0.015 | 0.76 | 0.148 | 5 / 6 |

## 3.5 Cases worth reporting individually (descriptive, from attempts.csv)

These cases are illustrations, not confirmatory evidence:
- **25:** the last epoch reached about 38% against about 99.5% claimed in the README.
- **76 DAGCN:** the author protocol gave 99.43 against 93.52 for the last epoch.
- **82:** under minloss and budget50 the three method groups AT*, CAT*-TempScaling and CAT*-CPCS cannot be told apart, because the authors' code trains them identically through epoch 151. The ordering CAT > AT exists only from epoch 152 on.
- **81 DARM:** the proposed method did not exceed ERM or Mixup under any protocol.
- **Rankings:** in three objects (14, 42, 30) the method ranking differs from the paper's: in 42 an ablation outranks the full model, and in 30 ablations beat the full model.

## Still open for this section

- **S2 (early saturation):** needs selection-split curves from the raw records on the PC.
- **S3:** static coding still needs a second coder.
- **S4 and S5:** checkpoint provenance and code lineage.
- **S6 (agreement within 2 pp of paper values):** needs a table of paper values, one per object; the attempts notes have them in free text only.
- **Figure:** `fig_inflation.pdf` (a: I_o by object with seed-bootstrap 95% CI; b: best_rep − last against h, coloured by author rule) is done; captions still to write.
