"""fleetdata core: one on-disk layout for every "units x time series x event table" dataset.

Layout of a built dataset (directory DIR):
  meta.json            dataset name, licence, DOI, source files, build time, time convention, row counts
  units.parquet        one row per unit (see UNIT_COLS)
  channels.parquet     one row per time-series column: channel, raw_name, table, unit_of_measure, core (bool)
  ts/<unit_id>.parquet time series of one unit; column "time" (UTC, tz-aware) + channels, float32
  events.parquet       one row per event / alarm / stop / maintenance / inspection (see EVENT_COLS)
  daily.parquet        unit x day: coverage, exposure columns, event counts per kind

Time convention: "time" is the END of the averaging interval for interval data (Kelmarsh, Penmanshiel and Hill of
Towie publish interval starts; the adapters shift them, and meta.json records the shift). Event "start"/"end" are
as published, converted to UTC.

Event kinds (column "kind"), fixed vocabulary so that tools can filter across datasets:
  fault        stop / failure / forced outage attributed to the equipment
  alarm        non-stopping warning or informational alarm
  stop_other   stop not attributed to equipment failure (grid, weather, curtailment, low wind, manual)
  maintenance  scheduled or unscheduled service, repair, part replacement
  intervention retrofit / upgrade / settings change with a known date
  inspection   periodic test with an outcome (e.g. vehicle roadworthiness test)
  anomaly      dataset-provided labelled anomaly / failure event (e.g. CARE event_info)
Anything an adapter cannot classify is "other"; the raw status / category text is always kept.
"""
import json
import os
import time

import numpy as np
import pandas as pd

UNIT_COLS = ["unit_id", "dataset", "site", "manufacturer", "model", "rated_kw", "commission_date", "lat", "lon",
             "meta"]
EVENT_COLS = ["unit_id", "start", "end", "kind", "code", "text", "category", "stopping", "source"]
# adapters may add dataset-specific event columns; they must start with "x_" and are kept as is
KINDS = {"fault", "alarm", "stop_other", "maintenance", "intervention", "inspection", "anomaly", "other"}
SCHEMA_VERSION = "1.0"


def _utc(s):
    s = pd.to_datetime(s, utc=True, format="mixed", errors="coerce")
    return s


def _units_frame(rows):
    df = pd.DataFrame(rows)
    for c in UNIT_COLS:
        if c not in df:
            df[c] = None
    df["unit_id"] = df["unit_id"].astype(str)
    df["meta"] = df["meta"].map(lambda m: m if isinstance(m, str) else json.dumps(m or {}, ensure_ascii=False))
    df["rated_kw"] = pd.to_numeric(df["rated_kw"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["commission_date"] = pd.to_datetime(df["commission_date"], errors="coerce")
    return df[UNIT_COLS]


def events_frame(df):
    """Normalise an event DataFrame to EVENT_COLS; raises on unknown kinds."""
    df = df.copy()
    for c in EVENT_COLS:
        if c not in df:
            df[c] = None
    df["unit_id"] = df["unit_id"].astype(str)
    df["start"] = _utc(df["start"])
    df["end"] = _utc(df["end"])
    for c in ["code", "text", "category", "source"]:
        df[c] = df[c].astype("string")
    df["stopping"] = df["stopping"].astype("boolean")
    bad = set(df["kind"].dropna().unique()) - KINDS
    if bad:
        raise ValueError(f"unknown event kinds {sorted(bad)}")
    df["kind"] = df["kind"].fillna("other").astype("string")
    if df["start"].isna().any():
        raise ValueError(f"{int(df['start'].isna().sum())} events without a parseable start")
    extra = sorted(c for c in df.columns if c.startswith("x_"))
    return df[EVENT_COLS + extra].sort_values(["unit_id", "start"]).reset_index(drop=True)


class Writer:
    """Incremental writer used by adapters. Call add_ts() once per unit (or repeatedly; chunks are concatenated)."""

    def __init__(self, out, dataset, licence, doi=None, sources=None, time_convention="interval end",
                 interval="10min", notes=None, overwrite=False):
        if os.path.exists(os.path.join(out, "meta.json")) and not overwrite:
            raise SystemExit(f"{out} already built; pass --overwrite to rebuild")
        os.makedirs(os.path.join(out, "ts"), exist_ok=True)
        self.out = out
        self.meta = dict(schema_version=SCHEMA_VERSION, dataset=dataset, licence=licence, doi=doi,
                         sources=sources or [], time_convention=time_convention, interval=interval,
                         notes=notes or [], built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._chunks = {}
        self.channels = {}

    def add_channel(self, channel, raw_name, table="", unit_of_measure="", core=False):
        self.channels[channel] = dict(channel=channel, raw_name=raw_name, table=table,
                                      unit_of_measure=unit_of_measure, core=bool(core))

    def add_ts(self, unit_id, df):
        """df: column 'time' (UTC) + numeric channels already renamed. Each call is spilled to a part file so that
        big fleets never sit in memory; flush_unit()/finish() merge the parts."""
        df = df.copy()
        df["time"] = _utc(df["time"])
        df = df.dropna(subset=["time"])
        for c in df.columns:
            if c != "time":
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        if df.empty:
            return
        if df["time"].duplicated().any():  # collapse repeats inside the chunk early (same rule as flush_unit)
            df = df.groupby("time", sort=True).first().reset_index()
        unit_id = str(unit_id)
        d = os.path.join(self.out, "ts", "_parts", unit_id)
        os.makedirs(d, exist_ok=True)
        k = self._chunks.get(unit_id, 0)
        df.to_parquet(os.path.join(d, f"{k:05d}.parquet"), index=False)
        self._chunks[unit_id] = k + 1

    def flush_unit(self, unit_id):
        import shutil
        unit_id = str(unit_id)
        d = os.path.join(self.out, "ts", "_parts", unit_id)
        self._chunks.pop(unit_id, None)
        if not os.path.isdir(d):
            return 0
        parts = [pd.read_parquet(os.path.join(d, f)) for f in sorted(os.listdir(d))]
        df = pd.concat(parts, ignore_index=True)
        n_raw = len(df)
        if df["time"].duplicated().any():
            # several parts for the same timestamp (overlapping exports, or extra channels such as shutdown time):
            # keep, per column, the first non-missing value in part order
            df = df.groupby("time", sort=True).first().reset_index()
        else:
            df = df.sort_values("time", kind="stable").reset_index(drop=True)
        cols = ["time"] + sorted(c for c in df.columns if c != "time")
        df[cols].to_parquet(os.path.join(self.out, "ts", f"{unit_id}.parquet"), index=False)
        shutil.rmtree(d)
        self.meta.setdefault("ts_rows", {})[unit_id] = int(len(df))
        if n_raw != len(df):
            self.meta.setdefault("ts_duplicate_rows_dropped", {})[unit_id] = int(n_raw - len(df))
        return len(df)

    def finish(self, units_rows, events, daily_exposure=None):
        pdir = os.path.join(self.out, "ts", "_parts")
        if os.path.isdir(pdir):
            for u in sorted(os.listdir(pdir)):
                self.flush_unit(u)
            os.rmdir(pdir)
        units = _units_frame(units_rows)
        units.to_parquet(os.path.join(self.out, "units.parquet"), index=False)
        ev = events_frame(events) if events is not None and len(events) else events_frame(pd.DataFrame(columns=EVENT_COLS))
        ev.to_parquet(os.path.join(self.out, "events.parquet"), index=False)
        pd.DataFrame(list(self.channels.values()),
                     columns=["channel", "raw_name", "table", "unit_of_measure", "core"]).to_parquet(
            os.path.join(self.out, "channels.parquet"), index=False)
        self.meta["n_units"] = int(len(units))
        self.meta["n_events"] = int(len(ev))
        self.meta["events_by_kind"] = {k: int(v) for k, v in ev["kind"].value_counts().items()}
        fleet = Fleet(self.out, _allow_partial=True)
        fleet.meta = self.meta
        daily = fleet._compute_daily(daily_exposure)
        daily.to_parquet(os.path.join(self.out, "daily.parquet"), index=False)
        self.meta["daily_rows"] = int(len(daily))
        with open(os.path.join(self.out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=1, ensure_ascii=False, default=str)
        return Fleet(self.out)


class Fleet:
    """Read side. All loaders return pandas DataFrames in the common layout."""

    def __init__(self, path, _allow_partial=False):
        self.path = path
        mp = os.path.join(path, "meta.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                self.meta = json.load(f)
        elif _allow_partial:
            self.meta = {}
        else:
            raise SystemExit(f"{path}: not a built fleetdata dataset (no meta.json)")
        self.units = pd.read_parquet(os.path.join(path, "units.parquet"))
        self.channels = pd.read_parquet(os.path.join(path, "channels.parquet"))
        self._events = pd.read_parquet(os.path.join(path, "events.parquet"))

    def __repr__(self):
        m = self.meta
        return (f"Fleet({m.get('dataset')}: {m.get('n_units')} units, {m.get('n_events')} events, "
                f"interval {m.get('interval')}, licence {m.get('licence')})")

    @property
    def unit_ids(self):
        return list(self.units["unit_id"])

    def events(self, kind=None, unit=None, start=None, end=None):
        e = self._events
        if kind is not None:
            e = e[e["kind"].isin([kind] if isinstance(kind, str) else kind)]
        if unit is not None:
            e = e[e["unit_id"].isin([str(unit)] if not isinstance(unit, (list, tuple, set)) else [str(u) for u in unit])]
        if start is not None:
            e = e[e["start"] >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            e = e[e["start"] < pd.Timestamp(end, tz="UTC")]
        return e.reset_index(drop=True)

    def ts(self, unit, channels=None, start=None, end=None, core=False):
        p = os.path.join(self.path, "ts", f"{unit}.parquet")
        if not os.path.exists(p):
            return pd.DataFrame(columns=["time"])
        if core and channels is None:
            channels = list(self.channels.loc[self.channels["core"], "channel"])
        cols = None if channels is None else ["time"] + [c for c in channels if c != "time"]
        if cols is not None:
            import pyarrow.parquet as pq
            have = set(pq.read_schema(p).names)
            cols = [c for c in cols if c in have]
        df = pd.read_parquet(p, columns=cols)
        if start is not None:
            df = df[df["time"] >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df["time"] < pd.Timestamp(end, tz="UTC")]
        return df.reset_index(drop=True)

    def daily(self):
        return pd.read_parquet(os.path.join(self.path, "daily.parquet"))

    def panel(self, channels, freq="D", agg="mean", units=None):
        """Long panel: unit_id, period, <channel>_<agg>... for quick baselines."""
        out = []
        for u in units or self.unit_ids:
            d = self.ts(u, channels=channels)
            if d.empty:
                continue
            g = d.set_index("time").resample(freq)
            a = g.agg(agg)
            a.columns = [f"{c}_{agg}" for c in a.columns]
            a.insert(0, "unit_id", u)
            out.append(a.reset_index().rename(columns={"time": "period"}))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def _compute_daily(self, exposure=None):
        """coverage = samples per day / expected samples per day; event counts per kind by event start day.
        exposure: optional callable(unit_id, ts DataFrame) -> DataFrame indexed by UTC day with extra columns."""
        interval = pd.Timedelta(self.meta.get("interval") or "10min") if self.meta.get("interval") not in (None, "irregular") else None
        rows = []
        for u in self.unit_ids:
            p = os.path.join(self.path, "ts", f"{u}.parquet")
            if not os.path.exists(p):
                continue
            t = pd.read_parquet(p)
            if t.empty:
                continue
            day = t["time"].dt.floor("D")
            d = pd.DataFrame({"n_samples": t.groupby(day).size()})
            if interval is not None:
                d["coverage"] = d["n_samples"] / (pd.Timedelta("1D") / interval)
            if exposure is not None:
                x = exposure(u, t)
                if x is not None and len(x):
                    d = d.join(x, how="left")
            d.index.name = "date"
            d.insert(0, "unit_id", u)
            rows.append(d.reset_index())
        daily = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["unit_id", "date"])
        ev = self._events
        if len(ev) and len(daily):
            c = ev.assign(date=ev["start"].dt.floor("D")).groupby(["unit_id", "date", "kind"]).size().unstack(
                "kind", fill_value=0)
            c.columns = [f"n_{k}" for k in c.columns]
            daily = daily.merge(c.reset_index(), on=["unit_id", "date"], how="left")
            for col in c.columns:
                daily[col] = daily[col].fillna(0).astype("int32")
        return daily

    def validate(self):
        """Structural checks; returns list of problems (empty = OK)."""
        probs = []
        if self.units["unit_id"].duplicated().any():
            probs.append("duplicate unit_id in units")
        unknown = set(self._events["unit_id"]) - set(self.unit_ids)
        if unknown:
            probs.append(f"events reference unknown units {sorted(unknown)[:5]}")
        bad = self._events[(self._events["end"].notna()) & (self._events["end"] < self._events["start"])]
        if len(bad):  # a quirk of published logs, not a build error: reported, not fatal
            probs.append(f"WARN {len(bad)} events end before start (kept as published)")
        for u in self.unit_ids:
            p = os.path.join(self.path, "ts", f"{u}.parquet")
            if os.path.exists(p):
                t = pd.read_parquet(p, columns=["time"])
                if t["time"].dt.tz is None:
                    probs.append(f"{u}: time not tz-aware")
                if t["time"].duplicated().any():
                    probs.append(f"{u}: duplicate timestamps")
                if not t["time"].is_monotonic_increasing:
                    probs.append(f"{u}: time not sorted")
        missing = [c for c in self.channels["channel"]] if len(self.channels) else []
        if self.unit_ids and missing:
            import pyarrow.parquet as pq
            p = os.path.join(self.path, "ts", f"{self.unit_ids[0]}.parquet")
            if os.path.exists(p):
                have = set(pq.read_schema(p).names)
                lost = [c for c in missing if c not in have]
                if len(lost) == len(missing):
                    probs.append("no declared channel present in first unit's ts file")
        return probs

    def to_h2kit(self, out_dir, event_kinds=("fault",), channels=None, avoid_kinds=("maintenance",),
                 params=None):
        """Write files + YAML config for h2kit (tools/topic_forecast/h2kit, baseline-first H2 check).
        Events of `event_kinds` are the targets; starts of `avoid_kinds` become avoid dates."""
        import yaml
        os.makedirs(os.path.join(out_dir, "ts"), exist_ok=True)
        channels = channels or list(self.channels.loc[self.channels["core"], "channel"])
        for u in self.unit_ids:
            d = self.ts(u, channels=channels)
            if len(d):
                d.to_parquet(os.path.join(out_dir, "ts", f"{u}.parquet"), index=False)
        ev = self.events(kind=list(event_kinds))
        evd = pd.DataFrame({"unit": ev["unit_id"], "date": ev["start"].dt.strftime("%Y-%m-%d"), "kind": ev["kind"],
                            "code": ev["code"]})
        evd.to_csv(os.path.join(out_dir, "events.csv"), index=False)
        av = self.events(kind=list(avoid_kinds)) if avoid_kinds else self._events.iloc[0:0]
        avoid = {u: sorted(set(g["start"].dt.strftime("%Y-%m-%d"))) for u, g in av.groupby("unit_id")}
        with open(os.path.join(out_dir, "avoid.json"), "w") as f:
            json.dump(avoid, f)
        cfg = dict(timeseries=dict(path=os.path.abspath(os.path.join(out_dir, "ts", "{unit}.parquet")), time_col="time"),
                   events=dict(path=os.path.abspath(os.path.join(out_dir, "events.csv")), unit_col="unit",
                               date_col="date", date_format="%Y-%m-%d"),
                   avoid_dates_json=os.path.abspath(os.path.join(out_dir, "avoid.json")),
                   exclude_cols=[], diff_std_cols=[], coverage="relative_count",
                   params=dict(dict(pre_days=14, min_pre=10, gap_days=60, min_ctrl=60, min_coverage=0.5,
                                    n_placebo=1000, alpha=0.05, seed=20261105), **(params or {})))
        p = os.path.join(out_dir, "h2kit_config.yaml")
        with open(p, "w") as f:
            f.write(f"# generated by fleetdata from {self.path} ({self.meta.get('dataset')}); "
                    f"targets {list(event_kinds)}, n={len(evd)}\n")
            yaml.safe_dump(cfg, f, sort_keys=False)
        return p


def open_fleet(path):
    return Fleet(path)
