"""CARE to Compare wind farm datasets (Guck et al. 2024; licence and DOI as stated in the copy in ~/data_care, not re-verified here),
Wind Farms A, B, C. Read-only.

root (default ~/data_care) / "Wind Farm <X>" / datasets/<event_id>.csv (';'-separated, columns time_stamp, asset_id,
id, train_test, status_type_id, <sensor>_{avg,max,min,std}), event_info.csv, feature_description.csv.

Unit = asset (unit_id CARE<X><asset_id>). Several event datasets cover the same asset; their rows are merged by
timestamp (first non-missing value wins). Time stamps are anonymised by the publisher (shifted), used as published.
Channels: every *_avg column plus status_type_id, raw names kept so results stay comparable with CARE papers; core
channels (power_kw, wind_ms, ...) are matched on the feature description.
Events: one row per event dataset: kind "anomaly" for anomaly-labelled events, "other" with category "normal" for
normal-behaviour events; x_dataset_id, x_train_end (last training timestamp) and the free-text description kept.
"""
import os

import pandas as pd

from ..common import CARE_CORE, map_core, wind_exposure
from ..core import Writer

CHUNK = 50000


def build(farm, out, root=None, overwrite=False, log=print):
    root = root or os.path.join(os.path.expanduser("~"), "data_care")
    d = os.path.join(root, f"Wind Farm {farm}")
    fd = pd.read_csv(os.path.join(d, "feature_description.csv"), sep=";", encoding="latin-1")
    info = pd.read_csv(os.path.join(d, "event_info.csv"), sep=";", encoding="latin-1")
    files = sorted(os.listdir(os.path.join(d, "datasets")), key=lambda s: (len(s), s))
    desc = {str(r["sensor_name"]): (str(r["description"]), str(r["unit"])) for _, r in fd.iterrows()}
    w = Writer(out, dataset=f"care_{farm.lower()}", licence="see ~/data_care README (not re-verified)", doi=None,
               sources=[d], time_convention="as published (anonymised, shifted by publisher)", interval="10min",
               overwrite=overwrite, notes=[f"{len(files)} event datasets"])
    with open(os.path.join(d, "datasets", files[0]), encoding="latin-1") as f:
        cols = f.readline().strip().split(";")
    avg = [c for c in cols if c.endswith("_avg")]
    by_desc = {c: desc.get(c[:-4], (c, ""))[0] for c in avg}
    core = map_core([by_desc[c] for c in avg], CARE_CORE)
    core_cols = {c: core[by_desc[c]] for c in avg if by_desc[c] in core}
    rename = {c: core_cols.get(c, c) for c in avg}
    for c in avg:
        w.add_channel(rename[c], c, table="datasets", unit_of_measure=desc.get(c[:-4], ("", ""))[1],
                      core=c in core_cols)
    w.add_channel("status_type_id", "status_type_id", table="datasets", core=True)
    log(f"CARE {farm}: {len(files)} files, {len(avg)} avg channels, core {core_cols}")
    w.meta["notes"].append(f"core mapping {core_cols}")
    ev, assets = [], set()
    for fn in files:
        eid = int(os.path.splitext(fn)[0])
        p = os.path.join(d, "datasets", fn)
        with open(p, encoding="latin-1") as f:
            head = f.readline().strip().split(";")
        use = ["time_stamp", "asset_id", "train_test", "status_type_id"] + [c for c in avg if c in head]
        asset, train_end, n = None, None, 0
        for ch in pd.read_csv(p, sep=";", usecols=use, chunksize=CHUNK, encoding="latin-1", low_memory=False):
            asset = int(ch["asset_id"].iloc[0])
            tr = ch.loc[ch["train_test"].astype(str).str.lower() == "train", "time_stamp"]
            if len(tr):
                train_end = tr.iloc[-1]
            ch = ch.drop(columns=["asset_id", "train_test"]).rename(columns=rename)
            ch["time"] = pd.to_datetime(ch.pop("time_stamp"), utc=True, errors="coerce")
            w.add_ts(f"CARE{farm}{asset:02d}", ch)
            n += len(ch)
        assets.add(asset)
        r = info[info["event_id"] == eid]
        if len(r):
            r = r.iloc[0]
            lab = str(r["event_label"]).strip().lower()
            ev.append(dict(unit_id=f"CARE{farm}{asset:02d}", start=r["event_start"], end=r["event_end"],
                           kind="anomaly" if lab == "anomaly" else "other",
                           category="anomaly" if lab == "anomaly" else "normal", code=str(eid),
                           text=None if pd.isna(r.get("event_description")) else str(r["event_description"]),
                           source="event_info.csv", x_dataset_id=eid, x_train_end=train_end))
        log(f"  dataset {eid}: asset {asset}, {n} rows")
    for a in sorted(assets):
        w.flush_unit(f"CARE{farm}{a:02d}")
    units = [dict(unit_id=f"CARE{farm}{a:02d}", dataset=f"care_{farm.lower()}", site=f"Wind Farm {farm}",
                  meta={"asset_id": a}) for a in sorted(assets)]
    events = pd.DataFrame(ev)
    if len(events):
        events["x_train_end"] = pd.to_datetime(events["x_train_end"], utc=True, errors="coerce")
    return w.finish(units, events, daily_exposure=wind_exposure())
