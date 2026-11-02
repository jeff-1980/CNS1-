# topic_forecast v0 — mechanical research-direction forecaster

Predicts which not-yet-studied topic pairs (a mechanical OpenAlex topic x any topic) will emerge
(>= 10 co-tagged articles) within 3 years, Science4Cast-style link prediction with a two-stage
plan (v0 = emergence only; impact stage later). Design doc: claude.ai artifact "科研方向预测模型".

- `fetch_openalex.py` — group_by count queries only (no bulk download); resumable; needs
  `OPENALEX_API_KEY` (free key, ~10k list queries/day). `--probe` makes ~7 requests to check the API.
- `model.py` — builds yearly co-occurrence networks, features, LightGBM; frozen splits
  (train T<=2015, val T=2018, test T=2021, K3 at T=2020, forecast at T=2025); gate K1 vs
  preferential-attachment baselines; K3 = where NC/SA 2023-2026 mechanical papers' topic pairs fall.
- `make_synthetic.py` — offline test data in the same layout.
- `run_wsl.sh probe|fetch|model` — entry point on the WSL machine.
