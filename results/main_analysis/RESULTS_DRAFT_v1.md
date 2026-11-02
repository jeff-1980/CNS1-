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

## 3.5 Early saturation (S2)

Seventeen objects (400 runs) have a selection split that is separate from the reported split. In 38 of these runs (9.5%), selection-split accuracy reached ≥ 0.999 within the first 5% of epochs, after which the selection criterion can no longer separate checkpoints. These runs are concentrated in two objects: 60 (27 of 40) and 51 (8 of 10). The reported split saturates this early in 35 of all 880 runs (4.0%). Script: `code/audit/main/s2_saturation.py`; output: `s2.json`.

## 3.6 Agreement with reported values (S6)

Seventeen objects publish a value for their headline method that we could compare with our author-protocol seed mean (`s6_paper_values.csv`). For each one we took the method the paper or README leads with and the metric it reports.
- **All 17:** 11 lie within 2 pp of the reported value.
- **Paper values only:** for the 14 objects where the value comes from the paper itself (dropping the three README-only values), 9 lie within 2 pp.
- **The six misses:**
  - two collapse-prone distillation students, 13 and 62 (17–19 pp below);
  - one model that is far from reproduced, 25 (38 % against 99.5 % in its README);
  - three gaps of 3.8–5.1 pp: 77, 21 and 14. In 14 the two methods also swap rank.

In the other 21 objects the paper was closed, the table was not accessible, or the value was given only in a figure, so no comparison is possible. Each headline-group choice is our reading of the paper and is listed in the CSV.

## 3.7 Cases worth reporting individually (descriptive, from attempts.csv)

These cases are illustrations, not confirmatory evidence:
- **25:** the last epoch reached about 38% against about 99.5% claimed in the README.
- **76 DAGCN:** the author protocol gave 99.43 against 93.52 for the last epoch.
- **82:** under minloss and budget50 the three method groups AT*, CAT*-TempScaling and CAT*-CPCS cannot be told apart, because the authors' code trains them identically through epoch 151. The ordering CAT > AT exists only from epoch 152 on.
- **81 DARM:** the proposed method did not exceed ERM or Mixup under any protocol.
- **Rankings:** in three objects (14, 42, 30) the method ranking differs from the paper's: in 42 an ablation outranks the full model, and in 30 ablations beat the full model.

## 3.8 Disclosure in the wider literature (static coding; S3)

Numbers from `results/static_coding/ai_coding/analysis.json` (codebook v1) and `analysis_v1.1.json` (S4a and S8 recoded under v1.1). All counts are out of the 79 sampled papers with full text (paper 003 had no accessible full text and is coded X). Ranges run from the cells both models agreed on (lower bound) to those cells plus every disputed cell in which either model gave that code (upper bound). These are descriptive results: no human coder checked the AI codes (deviation, 2026-10-29 and 2026-11-02).

**Disclosure of the checkpoint rule (S3; item S4a, v1.1).** Both models agree that 2 papers state in the text which epoch or checkpoint their reported numbers come from; counting every disputed cell where either model coded "explicit", the upper bound is 10 (3–13%). Under codebook v1 the same bounds are 4–14 (5–18%). 48–63 papers (61–80%) say nothing about checkpoint selection, and 10–27 mention it only vaguely.

**Code and artefacts (S3 code rule, S4b, S8).** One paper links its own code in the text, so the code-based checkpoint rule (S3) and the text–code consistency check (S4b) could be assessed for one paper only. Under v1.1, 76–77 papers release no code, weights, training logs or per-run results, and 2–3 release code.

**Evaluation design (S1, S2, S5–S7, v1).**
- Split unit: 26–38 papers split windows or samples at random, 10–15 split by operating condition or domain, and 29–40 do not say. No paper splits by recording file or by bearing, under either model.
- Validation: 6–19 papers select or tune on the test split, 22–29 use no validation split, 11 use a separate one, and 25–35 do not say.
- Repeated runs: 51–55 papers do not report how many times they trained; 48–59 report no spread across runs.
- Metrics: 31–34 report accuracy only.
- Noise protocol: 18–21 run no noise experiment; 6–9 add noise to test data only; 8–13 add it to both training and test data.

**Reliability of the AI coding.** Two models from different vendors (Claude Opus 5.5 via the Anthropic batch API; DeepSeek v4-pro at temperature 0) coded each paper three times from the same frozen prompt and input pack. A model's code for a cell is its majority over the three runs; at least two runs agreed in 96–100% of cells for every item. Quotes were found verbatim in the input for 99.7% (Claude) and 93.1% (DeepSeek) of non-empty quotes. Between models, Cohen's κ was 0.70–1.00 for eight of ten items (PABAK 0.72–1.00). Two items fell below the pre-registered 0.6 threshold under v1: S4a (κ 0.48, PABAK 0.61) and S8 (κ 0.49, PABAK 0.92). After rewriting their definitions (codebook v1.1) and recoding, S8 reached κ 1.00 and S4a κ 0.51 (PABAK 0.67). S4a is therefore reported as a low-confidence item. Of its 19 disputed cells under v1.1, 11 are between "vague" and "not stated" and 8 involve "explicit". 87 of 790 cells (11%) were disputed under v1.

## Still open for this section

- **S4 and S5:** checkpoint provenance and code lineage.
- **Figure:** `fig_inflation.pdf` (a: I_o by object with seed-bootstrap 95% CI; b: best_rep − last against h, coloured by author rule) is done; captions still to write.
