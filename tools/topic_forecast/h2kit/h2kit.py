#!/usr/bin/env python3
"""h2kit: baseline-first (H2) check for operational records, before any model is trained.

Question: in a dataset of units (machines, freezers, turbines, drives ...) x time series x event table, do
simple per-day features already separate the days before an event from ordinary days of the same unit,
beyond what random dates on that unit give?

Evaluation type: retrospective, real ground truth (labelled events), within-unit comparison, no model.

Method (all thresholds come from the YAML config and are fixed before the run):
  * Daily features: mean and std of every numeric channel (optionally std of the first difference of chosen
    channels, and the daily count of an event log). Days below `min_coverage` are dropped.
  * Event usable if its pre window [D - pre_days, D - 1] has >= min_pre covered days and the unit has
    >= min_ctrl control days.
  * Control days: covered days of the same unit >= gap_days from every avoid date (all events plus optional
    extra dates such as service visits) and from the event itself.
  * Per event and feature: AUC(pre days vs control days). Summary S = mean over events of |AUC - 0.5|.
  * Placebo: n_placebo draws; each event date replaced by a random eligible date of the same unit (pre window
    covered, >= gap_days from all avoid dates); p = (1 + #null >= observed) / (n_placebo + 1).
  * A feature passes only if p < alpha / n_features (Bonferroni).

Usage:
  h2kit.py --config cfg.yaml --out DIR          run on a real dataset
  h2kit.py --smoke --out DIR                    known-truth self-test (planted precursor must pass,
                                                no-precursor data must come out null); exit 1 if not
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

DEFAULTS = dict(pre_days=14, min_pre=10, gap_days=60, min_ctrl=60, min_coverage=0.5, n_placebo=1000,
                alpha=0.05, seed=20261103, coverage="relative_count", exclude_cols=[], diff_std_cols=[])


def auc(pos, neg):
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def daily_features(ts, time_col, cfg, events_log=None, log_time_col=None):
    t = ts.copy()
    t[time_col] = pd.to_datetime(t[time_col])
    t = t.set_index(time_col).sort_index()
    excl = set(cfg["exclude_cols"])
    num = [c for c in t.columns if pd.api.types.is_numeric_dtype(t[c]) and c not in excl]
    day = t.index.floor("D")
    g = t[num].groupby(day)
    feats = pd.concat([g.mean().add_suffix("_mean"), g.std().add_suffix("_std")], axis=1)
    for c in cfg["diff_std_cols"]:
        if c in t.columns:
            feats[f"{c}_diff_std"] = t[c].diff().groupby(day).std()
    if cfg["coverage"] == "distinct_minutes":  # share of the day's minutes with >= 1 sample
        cover = pd.Series(t.index.floor("min"), index=t.index).groupby(day).nunique() / 1440.0
    else:  # samples per day relative to the unit's median day
        cnt = t[num[0]].groupby(day).count()
        cover = cnt / max(1.0, float(cnt.median()))
    if events_log is not None:
        feats["n_events"] = (pd.to_datetime(events_log[log_time_col]).dt.floor("D").value_counts()
                             .reindex(feats.index).fillna(0))
    return feats[cover.reindex(feats.index).fillna(0) >= cfg["min_coverage"]]


def run(units, events, avoid, cfg, out, label=""):
    """units: {unit_id: daily feature DataFrame}; events: DataFrame(unit, date, kind); avoid: {unit: dates}."""
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    P, MINP, GAP, MINC = cfg["pre_days"], cfg["min_pre"], cfg["gap_days"], cfg["min_ctrl"]
    log = []
    avoid_by = {}
    for u in units:
        a = set(pd.to_datetime(avoid.get(u, []))) | set(events.loc[events.unit == u, "date"])
        avoid_by[u] = np.array(sorted(a), dtype="datetime64[D]")
    cols = sorted(set.intersection(*[set(f.columns) for f in units.values()])) if units else []
    idx_by = {u: f.index.values.astype("datetime64[D]") for u, f in units.items()}
    vals_by = {u: {c: f[c].to_numpy() for c in cols} for u, f in units.items()}
    dist_by = {u: (np.min(np.abs((idx_by[u][:, None] - avoid_by[u][None, :]).astype(int)), axis=1)
                   if len(avoid_by[u]) else np.full(len(idx_by[u]), 10 ** 6)) for u in units}
    one = np.timedelta64(1, "D")

    def windows(u, d):
        idx = idx_by[u]
        d = np.datetime64(d, "D")
        pre = (idx >= d - P * one) & (idx < d)
        ctrl = (dist_by[u] >= GAP) & (np.abs((idx - d).astype(int)) >= GAP)
        return pre, ctrl

    evs = []
    for _, r in events.iterrows():
        if r.unit not in units:
            log.append(f"{r.unit}: no data")
            continue
        pre, ctrl = windows(r.unit, r.date)
        if pre.sum() >= MINP and ctrl.sum() >= MINC:
            evs.append((r.unit, r.date, pre, ctrl))
        else:
            log.append(f"{r.unit} {pd.Timestamp(r.date).date()}: unusable (pre days {pre.sum()}, control days {ctrl.sum()})")
    print(f"{label}units with data {len(units)}, events usable {len(evs)} of {len(events)}", flush=True)
    if not evs:
        raise SystemExit("no usable events")

    def stats(ev_list):
        return np.array([[auc(vals_by[u][c][pre], vals_by[u][c][ctrl]) for c in cols] for u, _, pre, ctrl in ev_list])

    A = stats(evs)
    S_obs = np.nanmean(np.abs(A - 0.5), axis=0)
    cand = {}
    for u in units:
        idx = idx_by[u]
        ok = []
        for d in idx:
            pre = (idx >= d - P * one) & (idx < d)
            if pre.sum() >= MINP and (len(avoid_by[u]) == 0 or np.min(np.abs((avoid_by[u] - d).astype(int))) >= GAP):
                ok.append(d)
        cand[u] = ok
    null = np.zeros((cfg["n_placebo"], len(cols)))
    for k in range(cfg["n_placebo"]):
        draw = []
        for u, _, _, _ in evs:
            d = cand[u][rng.integers(len(cand[u]))]
            pre, ctrl = windows(u, d)
            draw.append((u, d, pre, ctrl))
        null[k] = np.nanmean(np.abs(stats(draw) - 0.5), axis=0)
        if k % 200 == 0:
            print(f"{label}placebo {k}/{cfg['n_placebo']}", flush=True)
    p = (1 + (null >= S_obs).sum(axis=0)) / (cfg["n_placebo"] + 1)
    res = pd.DataFrame({"feature": cols, "S_obs_mean_abs_auc_dev": S_obs, "mean_signed_auc": np.nanmean(A, axis=0),
                        "placebo_S_2.5pct": np.percentile(null, 2.5, axis=0),
                        "placebo_S_97.5pct": np.percentile(null, 97.5, axis=0), "p_placebo": p}).sort_values("p_placebo")
    res["beats_placebo_bonferroni"] = res["p_placebo"] < cfg["alpha"] / len(cols)
    res.to_csv(os.path.join(out, "feature_results.csv"), index=False)
    pd.DataFrame(A, columns=cols, index=[f"{u}_{pd.Timestamp(d).date()}" for u, d, _, _ in evs]).to_csv(
        os.path.join(out, "per_event_auc.csv"))
    summary = {"evaluation_type": "retrospective_real_gt_within_unit_no_model", "units_with_data": len(units),
               "events_listed": len(events), "events_usable": len(evs), "n_features": len(cols),
               "bonferroni_alpha": cfg["alpha"] / len(cols), "n_beating_placebo": int(res["beats_placebo_bonferroni"].sum()),
               "passing_features": res.loc[res["beats_placebo_bonferroni"], "feature"].tolist(),
               "config": {k: cfg[k] for k in DEFAULTS}, "top5": res.head(5).round(4).to_dict("records"), "notes": log}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    print(res.head(10).to_string(index=False), flush=True)
    print(f"{label}VERDICT:", "some simple feature beats the placebo: " + ", ".join(summary["passing_features"])
          if summary["n_beating_placebo"] else "no simple feature beats the placebo", flush=True)
    return summary


def load_config(path):
    c = yaml.safe_load(open(path, encoding="utf-8"))
    cfg = {**DEFAULTS, **(c.get("params") or {})}
    cfg["exclude_cols"] = c.get("exclude_cols", cfg["exclude_cols"])
    cfg["diff_std_cols"] = c.get("diff_std_cols", cfg["diff_std_cols"])
    cfg["coverage"] = c.get("coverage", cfg["coverage"])
    ev = c["events"]
    events = pd.read_csv(ev["path"], dtype={ev["unit_col"]: str})
    events = pd.DataFrame({"unit": events[ev["unit_col"]].astype(str),
                           "date": pd.to_datetime(events[ev["date_col"]], format=ev.get("date_format"))})
    avoid = {}
    if c.get("avoid_dates_json"):
        avoid = json.load(open(c["avoid_dates_json"], encoding="utf-8"))
    ts = c["timeseries"]
    units = {}
    for u in sorted(events.unit.unique()):
        f = ts["path"].format(unit=u)
        if not os.path.exists(f):
            print(f"{u}: time series missing ({f})", flush=True)
            continue
        log = None
        if c.get("event_log"):
            fl = c["event_log"]["path"].format(unit=u)
            log = pd.read_csv(fl) if fl.endswith(".csv") else pd.read_parquet(fl) if os.path.exists(fl) else None
        try:
            df = pd.read_csv(f) if f.endswith(".csv") else pd.read_parquet(f)
            units[u] = daily_features(df, ts["time_col"], cfg, log, (c.get("event_log") or {}).get("time_col"))
        except Exception as e:  # noqa  truncated download etc.
            print(f"{u}: unreadable ({str(e)[:80]})", flush=True)
    return units, events, avoid, cfg


def synthetic(planted, n_units=20, days=1500, seed=7):
    """Hourly data with per-unit random-walk drift, daily cycle and noise; 2 events per unit.
    planted=True: channel 'x' shifts by +1.5 sd (of the hourly noise) during the 14 days before each event."""
    rng = np.random.default_rng(seed)
    units, ev_rows = {}, []
    start = pd.Timestamp("2015-01-01")
    for u in range(n_units):
        hours = pd.date_range(start, periods=days * 24, freq="h")
        drift = np.repeat(np.cumsum(rng.normal(0, 0.05, days)), 24)
        x = drift + np.sin(np.arange(days * 24) * 2 * np.pi / 24) + rng.normal(0, 1, days * 24)
        y = 0.5 * drift + rng.normal(0, 1, days * 24)
        evd = sorted(rng.choice(np.arange(200, days - 30), 2, replace=False))
        if planted:
            for d in evd:
                x[(d - 14) * 24:d * 24] += 1.5
        df = pd.DataFrame({"time": hours, "x": x, "y": y})
        cfg = {**DEFAULTS, "coverage": "relative_count"}
        units[f"u{u:02d}"] = daily_features(df, "time", cfg)
        for d in evd:
            ev_rows.append((f"u{u:02d}", start + pd.Timedelta(days=int(d))))
    return units, pd.DataFrame(ev_rows, columns=["unit", "date"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-placebo", type=int, help="override (smoke tests only)")
    a = ap.parse_args()
    if a.smoke:
        cfg = {**DEFAULTS, "n_placebo": a.n_placebo or 300}
        ok = True
        for planted in (True, False):
            units, events = synthetic(planted)
            s = run(units, events, {}, cfg, os.path.join(a.out, "planted" if planted else "null"),
                    label="[planted] " if planted else "[null] ")
            expect = (s["passing_features"] == ["x_mean"]) if planted else (s["n_beating_placebo"] == 0)
            print(("PASS" if expect else "FAIL"), "planted" if planted else "null", s["passing_features"], flush=True)
            ok &= expect
        print("SMOKE", "PASS" if ok else "FAIL", flush=True)
        sys.exit(0 if ok else 1)
    units, events, avoid, cfg = load_config(a.config)
    if a.n_placebo:
        cfg["n_placebo"] = a.n_placebo
    run(units, events, avoid, cfg, a.out)


if __name__ == "__main__":
    main()
