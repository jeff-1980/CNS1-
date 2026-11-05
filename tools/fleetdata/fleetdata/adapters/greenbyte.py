"""Kelmarsh (Zenodo record 8252025, CC BY 4.0) and Penmanshiel (Zenodo record 8253010, CC BY 4.0)
wind farm SCADA, Greenbyte exports. Read-only on the yearly zips.

Inputs (defaults are jeff's WSL paths):
  zips:   ~/wind/zips/<site>/*.zip   each holds Turbine_Data_<Site>_<n>_*.csv (10-min) and Status_<Site>_<n>_*.csv
  static: ~/wind_d2/<site>/<Site>_WT_static.csv

Channel selection: "core" = the unified channels in common.WIND_CORE_GREENBYTE only; "means" (default) = core plus
every 10-min mean column (no ', Standard deviation/Minimum/Maximum') present in EVERY yearly export of that site, so
that the channel set is stable across years (2023+ exports carry many more columns). "all" = every column.

Time: Greenbyte 'Date and time' labels the START of the 10-min period (stated in the Kelmarsh/Penmanshiel Zenodo
records' readme; flagged as an assumption in meta.json); time is shifted +10 min to the interval end.

Events from Status files: Stop + IEC 'Forced outage' -> fault; Stop + IEC/service category mentioning maintenance or
service -> maintenance; other Stop -> stop_other; Warning -> alarm; Informational -> other. Raw Status, IEC and
service-contract categories are kept in x_ columns.
"""
import glob
import io
import os
import re
import zipfile

import pandas as pd

from ..common import (WIND_CORE_GREENBYTE, canon_raw, coalesce_duplicate_columns, greenbyte_header, is_stat_column,
                      map_core, snake, unique_names, wind_exposure)
from ..core import Writer

SITES = {
    "kelmarsh": dict(Site="Kelmarsh", prefix="KEL", doi="10.5281/zenodo.8252025 (record checked 2026-11-05; zips carry 2016-2024, version not pinned)", n_rx=r"Kelmarsh_(\d+)_"),
    "penmanshiel": dict(Site="Penmanshiel", prefix="PEN", doi="10.5281/zenodo.8253010 (record checked 2026-11-05; zips carry 2016-2024, version not pinned)", n_rx=r"Penmanshiel_(\d+)_"),
}
CHUNK = 20000


def _members(zips, kind, n_rx):
    out = []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                b = os.path.basename(n)
                if b.startswith(kind + "_") and b.endswith(".csv"):
                    m = re.search(n_rx, b)
                    if m:
                        out.append((z, n, int(m.group(1))))
    return out


def _header(z, n):
    with zipfile.ZipFile(z) as zf, zf.open(n) as f:
        lines = []
        for _ in range(40):
            ln = f.readline().decode("utf-8", errors="replace")
            if not ln:
                break
            lines.append(ln)
    return greenbyte_header(lines)


def _status(z, n):
    with zipfile.ZipFile(z) as zf, zf.open(n) as f:
        text = f.read().decode("utf-8", errors="replace")
    lines = text.splitlines(True)
    i, cols = greenbyte_header(lines)
    df = pd.read_csv(io.StringIO("".join(lines[i + 1:])), header=None, names=cols, na_values=["-"],
                     keep_default_na=True, dtype=str)
    return df


def classify(status, iec, svc):
    status, iec_l, svc_l = [x.strip().lower() if isinstance(x, str) else "" for x in (status, iec, svc)]
    if status == "stop":
        if "forced outage" in iec_l:
            return "fault"
        if any(k in iec_l or k in svc_l for k in ["maintenance", "service", "planned corrective"]):
            return "maintenance"
        return "stop_other"
    if status == "warning":
        return "alarm"
    return "other"


def build(site, out, zips_dir=None, static=None, channels="means", overwrite=False, units=None, log=print):
    cfg = SITES[site]
    home = os.path.expanduser("~")
    zips_dir = zips_dir or os.path.join(home, "wind", "zips", site)
    static = static or os.path.join(home, "wind_d2", site, f"{cfg['Site']}_WT_static.csv")
    zips = sorted(glob.glob(os.path.join(zips_dir, "*.zip")))
    if not zips:
        raise SystemExit(f"no zips in {zips_dir}")
    td = _members(zips, "Turbine_Data", cfg["n_rx"])
    st = _members(zips, "Status", cfg["n_rx"])
    if units:
        td = [t for t in td if t[2] in units]
        st = [t for t in st if t[2] in units]
    log(f"{site}: {len(zips)} zips, {len(td)} Turbine_Data files, {len(st)} Status files")

    # pass 1: headers only -> stable channel set. Raw names are compared after canon_raw() (unit spellings differ
    # between exports); several raw columns may feed one channel (coalesced, first non-missing wins).
    headers = {(z, n): _header(z, n)[1] for z, n, _ in td}
    years = {}
    for (z, n), cols in headers.items():
        years.setdefault(os.path.basename(z), set()).update(canon_raw(c) for c in cols)
    all_canon = sorted(set().union(*years.values()) - {"Date and time"})
    in_all = set.intersection(*years.values()) - {"Date and time"}
    core_of = {}  # canon name -> unified core channel (every match of each rule)
    for uni, rx in WIND_CORE_GREENBYTE:
        for c in all_canon:
            if c not in core_of and re.search(rx, c, re.I):
                core_of[c] = uni
    if channels == "core":
        keep = set(core_of)
    elif channels == "all":
        keep = set(all_canon)
    else:
        keep = set(core_of) | {c for c in in_all if not is_stat_column(c)}
    rest = sorted(keep - set(core_of))
    name_of = dict(core_of)
    for c, nm in zip(rest, unique_names([snake(r) for r in rest])):
        name_of[c] = "raw_" + nm if nm in core_of.values() else nm
    missing_core = [u for u, _ in WIND_CORE_GREENBYTE if u not in core_of.values()]
    log(f"channels: {len(set(name_of.values()))} kept ({len(set(core_of.values()))} core); core not found: {missing_core}")

    w = Writer(out, dataset=site, licence="CC BY 4.0", doi=cfg["doi"], sources=[os.path.abspath(z) for z in zips],
               time_convention="interval end (Greenbyte 'Date and time' = interval start, +10 min; assumption)",
               interval="10min", overwrite=overwrite,
               notes=[f"channel mode {channels}", f"core channels not found: {missing_core}",
                      f"raw columns seen: {len(all_canon)}; in every zip: {len(in_all)}"])
    for c, uni in name_of.items():
        if uni not in w.channels:
            w.add_channel(uni, c, table="Turbine_Data", unit_of_measure=(re.findall(r"\(([^()]*)\)\s*$", c) or [""])[0],
                          core=c in core_of)
        else:
            w.channels[uni]["raw_name"] += " | " + c

    by_unit = {}
    for z, n, k in td:
        by_unit.setdefault(k, []).append((z, n))
    for k in sorted(by_unit):
        uid = f"{cfg['prefix']}{k:02d}"
        for z, n in sorted(by_unit[k], key=lambda t: t[1]):
            i, cols = _header(z, n)
            rename = {c: name_of[canon_raw(c)] for c in cols if canon_raw(c) in name_of}
            use = ["Date and time"] + list(rename)
            nrows = 0
            with zipfile.ZipFile(z) as zf, zf.open(n) as f:
                reader = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"), skiprows=i + 1,
                                     header=None, names=cols, usecols=use, chunksize=CHUNK, na_values=["NaN"],
                                     low_memory=False)
                for ch in reader:
                    ch = coalesce_duplicate_columns(ch.rename(columns=rename))
                    ch["time"] = pd.to_datetime(ch.pop("Date and time"), utc=True, errors="coerce") + pd.Timedelta(
                        minutes=10)
                    w.add_ts(uid, ch)
                    nrows += len(ch)
            log(f"  {uid} {os.path.basename(n)}: {nrows} rows")
        rows = w.flush_unit(uid)
        log(f"{uid}: {rows} rows after merge")

    ev = []
    for z, n, k in st:
        uid = f"{cfg['prefix']}{k:02d}"
        df = _status(z, n)
        g = lambda c: df[c] if c in df else pd.Series([None] * len(df))
        e = pd.DataFrame({
            "unit_id": uid,
            "start": g("Timestamp start"),
            "end": g("Timestamp end"),
            "code": g("Code"),
            "text": g("Message"),
            "x_status": g("Status"),
            "x_iec_category": g("IEC category"),
            "x_service_category": g("Service contract category"),
            "x_comment": g("Comment"),
            "x_duration": g("Duration"),
            "source": os.path.basename(n),
        })
        e["kind"] = [classify(a, b, c) for a, b, c in zip(e["x_status"], e["x_iec_category"], e["x_service_category"])]
        e["category"] = e["x_iec_category"]
        e["stopping"] = e["x_status"].str.strip().str.lower().eq("stop")
        ev.append(e)
    events = pd.concat(ev, ignore_index=True) if ev else pd.DataFrame()
    if len(events):
        n0 = len(events)
        events = events.drop_duplicates(["unit_id", "start", "end", "code", "x_status"])
        w.meta["notes"].append(f"status rows {n0}, after de-duplication across overlapping exports {len(events)}")

    srows = []
    if os.path.exists(static):
        s = pd.read_csv(static, encoding="utf-8-sig")
        s = s.loc[:, ~s.columns.str.startswith("Unnamed")]
        for _, r in s.iterrows():
            m = re.search(r"(\d+)\s*$", str(r["Title"]))
            if not m:
                continue
            k = int(m.group(1))
            if k not in by_unit and k not in {t[2] for t in st}:
                continue
            srows.append(dict(unit_id=f"{cfg['prefix']}{k:02d}", dataset=site, site=cfg["Site"],
                              manufacturer=r.get("Manufacturer"), model=r.get("Model"),
                              rated_kw=r.get("Rated power (kW)"),
                              commission_date=pd.to_datetime(r.get("Commercial Operations Date"), dayfirst=True,
                                                             errors="coerce"),
                              lat=r.get("Latitude"), lon=r.get("Longitude"),
                              meta={k2: str(r[k2]) for k2 in ["Title", "Alternative Title", "Identity",
                                                               "Hub Height (m)", "Rotor Diameter (m)", "Elevation (m)"]
                                    if k2 in r}))
    have = {r["unit_id"] for r in srows}
    for k in sorted(set(by_unit) | {t[2] for t in st}):
        uid = f"{cfg['prefix']}{k:02d}"
        if uid not in have:
            srows.append(dict(unit_id=uid, dataset=site, site=cfg["Site"]))
    f = w.finish(srows, events, daily_exposure=wind_exposure())
    return f


def list_headers(site, zips_dir=None, out_path=None):
    """Write every distinct Turbine_Data column with the yearly zips it appears in (fast: headers only)."""
    cfg = SITES[site]
    zips_dir = zips_dir or os.path.join(os.path.expanduser("~"), "wind", "zips", site)
    zips = sorted(glob.glob(os.path.join(zips_dir, "*.zip")))
    seen = {}
    for z, n, k in _members(zips, "Turbine_Data", cfg["n_rx"]):
        for c in _header(z, n)[1]:
            seen.setdefault(c, set()).add(os.path.basename(z))
    core = map_core(list(seen), WIND_CORE_GREENBYTE)
    lines = [f"{len(seen)} columns over {len(zips)} zips"]
    for c in sorted(seen):
        lines.append(f"{len(seen[c])}\t{'CORE:' + core[c] if c in core else ''}\t{'stat' if is_stat_column(c) else ''}\t{c}")
    txt = "\n".join(lines)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    return txt
