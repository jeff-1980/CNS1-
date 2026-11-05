"""Ultra-low-temperature freezers (Huang et al., Sci Data 2023, doi 10.1038/s41597-023-02808-6; figshare
10.6084/m9.figshare.24033324, CC0), Revco units. Read-only.

root (default /mnt/d/audit_data/ult_freezer): dataset/Revco/<id>_temp.parquet (irregular sensor log),
dataset/Revco/<id>_events.parquet (controller event log), other_data/freezer_faults.csv,
other_data/service_visit_dates.json, other_data/freezers_model_data.csv.

Time: Datetime as published (no time zone given; stored as UTC labels). Interval irregular, so daily coverage is
not computed (n_samples per day is).
Events: freezer_faults -> fault (main fault text, secondary faults and repair flags in x_ columns); service visits
-> maintenance; controller events containing 'alarm' -> alarm, other controller events -> other.
"""
import glob
import json
import os

import pandas as pd

from ..common import snake
from ..core import Writer

NON_NUMERIC = {"Datetime", "Type", "Event", "main_fault", "index"}
CORE = {"RTD": "cabinet_temp_c", "SetPoint": "setpoint_c", "Mains Voltage": "mains_v", "State": "state"}


def build(out, root=None, overwrite=False, log=print):
    root = root or "/mnt/d/audit_data/ult_freezer"
    rd = os.path.join(root, "dataset", "Revco")
    temps = sorted(glob.glob(os.path.join(rd, "*_temp.parquet")))
    ids = [os.path.basename(p).split("_")[0] for p in temps]
    w = Writer(out, dataset="ult_revco", licence="CC0", doi="10.1038/s41597-023-02808-6", sources=[root],
               time_convention="as published (no zone stated)", interval="irregular", overwrite=overwrite)
    chans = {}
    for uid, p in zip(ids, temps):
        df = pd.read_parquet(p)
        df = df.drop(columns=[c for c in df.columns if c in NON_NUMERIC - {"Datetime"}])
        ren = {c: CORE.get(c, snake(c)) for c in df.columns if c != "Datetime"}
        for raw, nm in ren.items():
            chans[nm] = raw
        df = df.rename(columns=ren)
        df["time"] = pd.to_datetime(df.pop("Datetime"), utc=True, errors="coerce")
        w.add_ts(uid, df)
        w.flush_unit(uid)
        log(f"  {uid}: {len(df)} rows")
    for nm, raw in chans.items():
        w.add_channel(nm, raw, table="temp", core=raw in CORE)
    ev = []
    ff = pd.read_csv(os.path.join(root, "other_data", "freezer_faults.csv"), dtype=str)
    ev.append(pd.DataFrame({"unit_id": ff["freezer_id"], "start": pd.to_datetime(ff["registered_date"], format="mixed",
                                                                                   errors="coerce"),
                            "kind": "fault", "text": ff["main_fault"], "category": "freezer_faults",
                            "source": "freezer_faults.csv", "x_secondary_faults": ff.get("secondary_faults"),
                            "x_refrigerant_filled": ff.get("refrigerant_filled"),
                            "x_insulation_rework": ff.get("insulation_rework"),
                            "x_no_repair_done": ff.get("no_repair_done")}))
    sv = json.load(open(os.path.join(root, "other_data", "service_visit_dates.json")))
    ev.append(pd.DataFrame([dict(unit_id=u, start=d, kind="maintenance", text="service visit",
                                 source="service_visit_dates.json") for u, ds in sv.items() for d in ds]))
    for uid in ids:
        p = os.path.join(rd, f"{uid}_events.parquet")
        if os.path.exists(p):
            e = pd.read_parquet(p)
            txt = e["Event"].astype(str)
            ev.append(pd.DataFrame({"unit_id": uid, "start": e["Datetime"], "code": e["Type"].astype(str),
                                    "text": txt, "kind": ["alarm" if "alarm" in t.lower() else "other" for t in txt],
                                    "category": "controller_event", "source": os.path.basename(p)}))
    events = pd.concat(ev, ignore_index=True)
    known = set(ids)
    dropped = events[~events["unit_id"].isin(known)]
    if len(dropped):
        w.meta.setdefault("notes", []).append(f"{len(dropped)} events for freezers without sensor data dropped")
    events = events[events["unit_id"].isin(known)].dropna(subset=["start"])
    md = pd.read_csv(os.path.join(root, "other_data", "freezers_model_data.csv"))
    rv = md[md["Manufacturer"] == "Revco"].iloc[0].to_dict() if (md["Manufacturer"] == "Revco").any() else {}
    units = [dict(unit_id=u, dataset="ult_revco", site="hospital biobank (anonymised)", manufacturer="Revco",
                  model=rv.get("Model_number"), meta={k: str(v) for k, v in rv.items()}) for u in ids]
    return w.finish(units, events)
