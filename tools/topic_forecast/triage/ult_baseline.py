#!/usr/bin/env python3
"""ULT freezer baseline-first check (CNS规划4#, refrigeration candidate from the topic-forecast assay).

Data: Huang et al., Sci Data 2023, doi 10.1038/s41597-023-02808-6 (CC0), processed Revco parquet files.
Question: do simple per-day features already separate the 14 days before a registered fault from
ordinary days of the same freezer, beyond what random dates on that freezer give?

Judgement rules (fixed before looking at results):
  * Event = row of freezer_faults.csv (registered_date, M/D/YYYY). Pre-fault window = the 14 calendar days
    [D-14, D-1]. Usable if >= 10 of those days have >= 50% minute coverage and the freezer has >= 60 control days.
  * Control days = days of the same freezer with >= 50% coverage that are >= 60 days from every service or
    preventive-maintenance visit (service_visit_dates.json, from the repository's service_reports folders,
    plus all registered fault dates).
  * Features per day (no models): mean and std of every numeric temperature channel and Mains Voltage,
    std of the chamber temperature (RTD) first difference, and the daily event-log count.
  * Per event and feature: AUC of pre-fault days vs control days (Mann-Whitney). Summary per feature:
    S = mean over events of |AUC - 0.5| (direction-free, because fault types differ), and the mean signed AUC.
  * Placebo: 1000 draws; in each draw every event's date is replaced by a random date on the same freezer
    whose 14-day window has >= 10 covered days and which is >= 60 days from all real visits; controls are
    recomputed excluding +-60 days around the placebo date. p = (1 + #null >= observed) / 1001.
  * A feature "beats the placebo" only if p < 0.05 / n_features (Bonferroni). Otherwise the verdict is
    "no simple feature beats the placebo".
Usage: ult_baseline.py <data_dir> <out_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

PRE_DAYS, MIN_PRE, GAP, MIN_CTRL, COVER = 14, 10, 60, 60, 0.5
N_PLACEBO = int(os.environ.get("ULT_N_PLACEBO", "1000"))  # smaller only for smoke tests
rng = np.random.default_rng(20261103)


def auc(pos, neg):
    """Mann-Whitney AUC of pos vs neg (ties count 0.5)."""
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def daily_features(temp, events):
    t = temp.copy()
    tcol = next(c for c in t.columns if c.lower() in ("datetime", "date_time", "timestamp"))
    t[tcol] = pd.to_datetime(t[tcol])
    t = t.set_index(tcol).sort_index()
    num = [c for c in t.columns if pd.api.types.is_numeric_dtype(t[c]) and c not in ("State", "SetPoint")]
    day = t.index.floor("D")
    g = t[num].groupby(day)
    feats = pd.concat([g.mean().add_suffix("_mean"), g.std().add_suffix("_std")], axis=1)
    if "RTD" in t.columns:
        feats["RTD_diff_std"] = t["RTD"].diff().groupby(day).std()
    # coverage = share of the day's minutes with at least one sample (sampling is 30 s to 1 min)
    cover = pd.Series(t.index.floor("min"), index=t.index).groupby(day).nunique() / 1440.0
    if events is not None and len(events):
        e = events.copy()
        ecol = next(c for c in e.columns if c.lower() in ("datetime", "date_time", "timestamp"))
        feats["n_events"] = pd.to_datetime(e[ecol]).dt.floor("D").value_counts().reindex(feats.index).fillna(0)
    feats = feats[cover.reindex(feats.index).fillna(0) >= COVER]
    return feats


def main():
    data, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rev = os.path.join(data, "dataset", "Revco")
    faults = pd.read_csv(os.path.join(data, "other_data", "freezer_faults.csv"), dtype={"freezer_id": str})
    faults["date"] = pd.to_datetime(faults["registered_date"], format="%m/%d/%Y")
    visits = json.load(open(os.path.join(data, "other_data", "service_visit_dates.json")))
    log = []
    feats_by, avoid_by = {}, {}
    for fid in sorted(faults["freezer_id"].unique()):
        ft = os.path.join(rev, f"{fid}_temp.parquet")
        if not os.path.exists(ft):
            log.append(f"{fid}: temp parquet missing")
            continue
        ev = os.path.join(rev, f"{fid}_events.parquet")
        try:
            feats_by[fid] = daily_features(pd.read_parquet(ft), pd.read_parquet(ev) if os.path.exists(ev) else None)
        except Exception as e:  # noqa  (truncated download etc.)
            log.append(f"{fid}: unreadable ({str(e)[:80]})")
            continue
        avoid = set(pd.to_datetime(visits.get(fid, []))) | set(faults.loc[faults.freezer_id == fid, "date"])
        avoid_by[fid] = np.array(sorted(avoid), dtype="datetime64[D]")
    cols = sorted(set.intersection(*[set(f.columns) for f in feats_by.values()])) if feats_by else []

    def windows(fid, d):
        f = feats_by[fid]
        idx = f.index.values.astype("datetime64[D]")
        d = np.datetime64(d, "D")
        pre = (idx >= d - PRE_DAYS) & (idx < d)
        dist = np.min(np.abs((idx[:, None] - avoid_by[fid][None, :]).astype(int)), axis=1)
        ctrl = (dist >= GAP) & (np.abs((idx - d).astype(int)) >= GAP)
        return pre, ctrl

    events = []
    for _, r in faults.iterrows():
        fid = r.freezer_id
        if fid not in feats_by:
            continue
        pre, ctrl = windows(fid, r.date)
        if pre.sum() >= MIN_PRE and ctrl.sum() >= MIN_CTRL:
            events.append((fid, r.date, r.main_fault, pre, ctrl))
        else:
            log.append(f"{fid} {r.date.date()}: unusable (pre days {pre.sum()}, control days {ctrl.sum()})")
    print(f"freezers with data {len(feats_by)}, events usable {len(events)} of {len(faults)}", flush=True)

    def stats(evs):
        a = np.array([[auc(feats_by[fid][c].to_numpy()[pre], feats_by[fid][c].to_numpy()[ctrl]) for c in cols]
                      for fid, _, _, pre, ctrl in evs])
        return a

    A = stats(events)
    S_obs = np.nanmean(np.abs(A - 0.5), axis=0)
    # candidate placebo dates per freezer
    cand = {}
    for fid, f in feats_by.items():
        idx = f.index.values.astype("datetime64[D]")
        ok = []
        for d in idx:
            pre = (idx >= d - PRE_DAYS) & (idx < d)
            if pre.sum() >= MIN_PRE and np.min(np.abs((avoid_by[fid] - d).astype(int))) >= GAP:
                ok.append(d)
        cand[fid] = ok
    null = np.zeros((N_PLACEBO, len(cols)))
    for k in range(N_PLACEBO):
        evs = []
        for fid, _, _, _, _ in events:
            d = cand[fid][rng.integers(len(cand[fid]))]
            pre, ctrl = windows(fid, d)
            evs.append((fid, d, "", pre, ctrl))
        null[k] = np.nanmean(np.abs(stats(evs) - 0.5), axis=0)
        if k % 100 == 0:
            print(f"placebo {k}/{N_PLACEBO}", flush=True)
    p = (1 + (null >= S_obs).sum(axis=0)) / (N_PLACEBO + 1)
    res = pd.DataFrame({"feature": cols, "S_obs_mean_abs_auc_dev": S_obs,
                        "mean_signed_auc": np.nanmean(A, axis=0),
                        "placebo_S_2.5pct": np.percentile(null, 2.5, axis=0),
                        "placebo_S_97.5pct": np.percentile(null, 97.5, axis=0),
                        "p_placebo": p}).sort_values("p_placebo")
    res["beats_placebo_bonferroni"] = res["p_placebo"] < 0.05 / len(cols)
    res.to_csv(os.path.join(out, "feature_results.csv"), index=False)
    pd.DataFrame(A, columns=cols, index=[f"{e[0]}_{e[1].date()}" for e in events]).to_csv(
        os.path.join(out, "per_event_auc.csv"))
    summary = {"freezers_with_data": len(feats_by), "faults_listed": len(faults), "events_usable": len(events),
               "n_features": len(cols), "bonferroni_alpha": 0.05 / len(cols),
               "n_beating_placebo": int(res["beats_placebo_bonferroni"].sum()),
               "top5": res.head(5).round(4).to_dict("records"), "notes": log}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    print(res.head(12).to_string(index=False), flush=True)
    print("VERDICT:", "some simple feature beats the placebo" if summary["n_beating_placebo"]
          else "no simple feature beats the placebo", flush=True)


if __name__ == "__main__":
    main()
