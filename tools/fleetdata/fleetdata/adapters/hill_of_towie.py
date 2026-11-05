"""Hill of Towie open dataset v2.1.0 (RES; Zenodo, CC BY 4.0), 21 x Siemens SWT-2.3-VS-82. Read-only on the zips.

root (default /mnt/d/audit_data/hill_of_towie) holds {YEAR}.zip with monthly CSVs per table, plus
Hill_of_Towie_turbine_metadata.csv, Hill_of_Towie_alarms_description.csv, Hill_of_Towie_ShutdownDuration.zip and
Hill_of_Towie_AeroUp_install_dates.csv.

Time: TimeStamp is the END of the 10-min period (data dictionary), used as is. StationId = turbine number + 2304509.
Time series: 10-min tables tblSCTurbine, tblSCTurGrid, tblSCTurTemp, tblSCTurPress, tblSCTurIntern. "means" keeps
every *_mean column plus wtc_PowerRef_endvalue; "core" keeps the WIND_CORE_HOT channels; "all" keeps every numeric
column. Channel shutdown_s = ShutdownDuration (seconds of the 10-min period the turbine was shut down).
Events: tblAlarmLog (TimeOn/TimeOff/StationNr/Alarmcode). The published alarm description table covers only a few
codes; codes in it with Stopping = 1 are stop_other when the text names wind, ice, grid or availability, else
fault; Stopping = 0 -> alarm; codes not described -> other (raw code kept). AeroUp installation dates ->
intervention.
"""
import glob
import os
import re
import zipfile

import pandas as pd

from ..common import WIND_CORE_HOT, map_core, wind_exposure
from ..core import Writer

TS_TABLES = ["tblSCTurbine", "tblSCTurGrid", "tblSCTurTemp", "tblSCTurPress", "tblSCTurIntern"]
BASE = 2304509
EXTERNAL = re.compile(r"wind|ic(e|ing)|grid|availab|remote|manual|curtail|low|high", re.I)


def _table_members(zf, table):
    return sorted(n for n in zf.namelist() if re.match(table + r"_\d{4}_\d{2}\.csv$", os.path.basename(n)))


def build(out, root=None, channels="means", overwrite=False, units=None, log=print):
    root = root or "/mnt/d/audit_data/hill_of_towie"
    zips = sorted(z for z in glob.glob(os.path.join(root, "*.zip")) if re.match(r"\d{4}\.zip$", os.path.basename(z)))
    if not zips:
        raise SystemExit(f"no yearly zips in {root}")
    stations = set(range(BASE + 1, BASE + 22)) if not units else {BASE + u for u in units}
    w = Writer(out, dataset="hill_of_towie", licence="CC BY 4.0", doi="Hill of Towie open dataset v2.1.0 (Zenodo)",
               sources=[os.path.abspath(z) for z in zips], time_convention="interval end (as published)",
               interval="10min", overwrite=overwrite, notes=[f"channel mode {channels}",
                                                            f"years on disk: {[os.path.basename(z) for z in zips]}"])
    # pass 1: columns per table (first month of first zip)
    colsel = {}
    with zipfile.ZipFile(zips[0]) as zf:
        for t in TS_TABLES:
            m = _table_members(zf, t)
            if not m:
                continue
            with zf.open(m[0]) as f:
                cols = list(pd.read_csv(f, nrows=0).columns)
            colsel[t] = [c for c in cols if c not in ("TimeStamp", "StationId")]
    all_cols = [c for t in colsel for c in colsel[t]]
    core_map = map_core(all_cols, WIND_CORE_HOT)
    keep = {}
    for t, cols in colsel.items():
        if channels == "core":
            k = [c for c in cols if c in core_map]
        elif channels == "all":
            k = [c for c in cols if not c.endswith("_endvalue") or c in core_map]
        else:
            k = [c for c in cols if c.endswith("_mean") or c in core_map]
        keep[t] = k
    rename = {c: core_map.get(c, c[4:] if c.startswith("wtc_") else c) for t in keep for c in keep[t]}
    for t in keep:
        for c in keep[t]:
            w.add_channel(rename[c], c, table=t, core=c in core_map)
    w.add_channel("shutdown_s", "ShutdownDuration", table="ShutdownDuration", unit_of_measure="s", core=True)
    missing_core = [u for u, _ in WIND_CORE_HOT if u not in core_map.values()]
    w.meta["notes"].append(f"core channels not found: {missing_core}")
    log(f"hill_of_towie: {len(zips)} zips; channels kept {len(rename) + 1} (core {len(core_map) + 1}); "
        f"core not found {missing_core}")

    alarms = []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            months = sorted({os.path.basename(n)[-11:-4] for t in keep for n in _table_members(zf, t)})
            for mon in months:
                df = None
                for t, cols in keep.items():
                    if not cols:
                        continue
                    names = [n for n in _table_members(zf, t) if n.endswith(f"_{mon}.csv")]
                    if not names:
                        continue
                    with zf.open(names[0]) as f:
                        head = pd.read_csv(f, nrows=0).columns
                    use = ["TimeStamp", "StationId"] + [c for c in cols if c in head]
                    with zf.open(names[0]) as f:
                        d = pd.read_csv(f, usecols=use, low_memory=False)
                    d = d[d["StationId"].isin(stations)]
                    d = d.drop_duplicates(["TimeStamp", "StationId"])
                    df = d if df is None else df.merge(d, on=["TimeStamp", "StationId"], how="outer")
                if df is None:
                    continue
                df = df.rename(columns=rename)
                df["time"] = pd.to_datetime(df.pop("TimeStamp"), utc=True, format="mixed", errors="coerce")
                for sid, g in df.groupby("StationId"):
                    w.add_ts(f"T{int(sid) - BASE:02d}", g.drop(columns=["StationId"]))
                log(f"  {os.path.basename(z)} {mon}: {len(df)} rows")
            for n in _table_members(zf, "tblAlarmLog"):
                with zf.open(n) as f:
                    a = pd.read_csv(f, low_memory=False)
                a["source"] = os.path.basename(n)
                alarms.append(a)

    sdz = os.path.join(root, "Hill_of_Towie_ShutdownDuration.zip")
    if os.path.exists(sdz):
        with zipfile.ZipFile(sdz) as zf:
            name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(name) as f:
                sd = pd.read_csv(f)
        sd["time"] = pd.to_datetime(sd["TimeStamp_StartFormat"], utc=True, format="mixed") + pd.Timedelta(minutes=10)
        sd["u"] = sd["TurbineName"].astype(str)
        lo = min(pd.to_datetime(f"{os.path.basename(z)[:4]}-01-01", utc=True) for z in zips)
        hi = max(pd.to_datetime(f"{int(os.path.basename(z)[:4]) + 1}-01-01 00:10", utc=True) for z in zips)
        sd = sd[(sd["time"] > lo) & (sd["time"] <= hi)]
        for u, g in sd.groupby("u"):
            if BASE + int(u[1:]) in stations:
                w.add_ts(u, g[["time"]].assign(shutdown_s=g["ShutdownDuration"]))
        log(f"shutdown rows {len(sd)}")
    else:
        w.meta["notes"].append("ShutdownDuration.zip not found")

    for sid in sorted(stations):
        n = w.flush_unit(f"T{sid - BASE:02d}")
        log(f"T{sid - BASE:02d}: {n} rows")

    # events
    desc = {}
    dp = os.path.join(root, "Hill_of_Towie_alarms_description.csv")
    if os.path.exists(dp):
        dd = pd.read_csv(dp)
        desc = {int(r["Alarm Code"]): (str(r["Description"]).strip(" :"), int(r["Stopping"])) for _, r in dd.iterrows()}
    ev = []
    if alarms:
        a = pd.concat(alarms, ignore_index=True)
        a = a[a["StationNr"].isin(stations)]
        txt = a["Alarmcode"].map(lambda c: desc.get(int(c), (None, None))[0] if pd.notna(c) else None)
        stop = a["Alarmcode"].map(lambda c: desc.get(int(c), (None, None))[1] if pd.notna(c) else None)
        kind = []
        for t, s in zip(txt, stop):
            if s is None or pd.isna(s):
                kind.append("other")
            elif int(s) == 1:
                kind.append("stop_other" if EXTERNAL.search(t or "") else "fault")
            else:
                kind.append("alarm")
        ev.append(pd.DataFrame({"unit_id": [f"T{int(s) - BASE:02d}" for s in a["StationNr"]], "start": a["TimeOn"],
                                "end": a["TimeOff"], "code": a["Alarmcode"].astype("Int64").astype(str), "text": txt,
                                "kind": kind, "stopping": stop.map(lambda s: None if pd.isna(s) else bool(s)),
                                "category": "tblAlarmLog", "source": a["source"]}))
        w.meta["notes"].append(f"alarm codes described: {len(desc)}; alarm rows {len(a)}, "
                               f"undescribed share {sum(k == 'other' for k in kind) / max(1, len(kind)):.2f}")
    ip = os.path.join(root, "Hill_of_Towie_AeroUp_install_dates.csv")
    if os.path.exists(ip) and os.path.getsize(ip) > 0:
        ins = pd.read_csv(ip)
        tcol = next((c for c in ins.columns if "turbine" in c.lower() or c.lower() in ("wtg", "name")), ins.columns[0])
        dcols = [c for c in ins.columns if c != tcol and pd.to_datetime(ins[c], errors="coerce", dayfirst=True).notna().any()]
        if dcols:
            st = pd.to_datetime(ins[dcols[0]], errors="coerce", dayfirst=True)
            en = pd.to_datetime(ins[dcols[-1]], errors="coerce", dayfirst=True) if len(dcols) > 1 else None
            uid = ins[tcol].astype(str).str.extract(r"(\d+)")[0].astype(int).map(lambda k: f"T{k:02d}")
            ev.append(pd.DataFrame({"unit_id": uid, "start": st, "end": en, "kind": "intervention",
                                    "text": "AeroUp aerodynamic upgrade installation", "category": "retrofit",
                                    "source": os.path.basename(ip)}).dropna(subset=["start"]))
            w.meta["notes"].append(f"AeroUp install dates from columns {dcols} (start = first, end = last)")
    events = pd.concat(ev, ignore_index=True) if ev else pd.DataFrame()
    if len(events):
        events = events[events["unit_id"].isin({f"T{s - BASE:02d}" for s in stations})]

    rows = []
    mp = os.path.join(root, "Hill_of_Towie_turbine_metadata.csv")
    md = pd.read_csv(mp, encoding="utf-8-sig") if os.path.exists(mp) else pd.DataFrame()
    for sid in sorted(stations):
        uid = f"T{sid - BASE:02d}"
        r = md[md["Turbine Name"] == uid].iloc[0] if len(md) and (md["Turbine Name"] == uid).any() else {}
        rows.append(dict(unit_id=uid, dataset="hill_of_towie", site="Hill of Towie",
                         manufacturer=r.get("Manufacturer") if len(r) else None, model=r.get("Model") if len(r) else None,
                         rated_kw=r.get("Rated power (kW)") if len(r) else None,
                         commission_date=r.get("Commercial Operations Date") if len(r) else None,
                         lat=r.get("Latitude") if len(r) else None, lon=r.get("Longitude") if len(r) else None,
                         meta={"StationId": sid}))
    return w.finish(rows, events, daily_exposure=wind_exposure())
