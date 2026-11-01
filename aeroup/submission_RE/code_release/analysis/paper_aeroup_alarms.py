"""AeroUp paper, revision R5b: alarm-log screen for concurrent changes on T13 around the retrofit (descriptive).
Counts alarm records per turbine in windows around T13's works (23-29 Sep 2021), lists T13 alarm codes in those
windows with the fleet median count of the same code, and flags codes that occur on T13 but rarely elsewhere.
Usage: python paper_aeroup_alarms.py <ROOT> <OUTDIR>
"""
import io
import os
import sys
import zipfile

import pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
LOGF = f"{OUT}/log_alarms.txt"
if os.path.exists(LOGF):
    os.remove(LOGF)
T13 = 2304522
STATIONS = list(range(2304510, 2304531))


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOGF, "a") as fh:
        fh.write(s + "\n")


out = []
for y in (2021, 2022):
    with zipfile.ZipFile(f"{ROOT}/{y}.zip") as z:
        for n in sorted(n for n in z.namelist() if n.split("/")[-1].startswith("tblAlarmLog_")):
            with z.open(n) as f:
                out.append(pd.read_csv(f, low_memory=False))
a = pd.concat(out, ignore_index=True).drop_duplicates()
log("alarm rows 2021-2022:", len(a), "columns:", list(a.columns))
log("StationNr examples:", a["StationNr"].drop_duplicates().head(25).tolist())
a["t"] = pd.to_datetime(a["TimeOn"], utc=True, format="mixed", errors="coerce")
a["toff"] = pd.to_datetime(a["TimeOff"], utc=True, format="mixed", errors="coerce")
st = pd.to_numeric(a["StationNr"], errors="coerce")
if st.dropna().between(1, 21).all():
    st = st + 2304509
a["st"] = st
a = a[a["st"].isin(STATIONS) & a["t"].notna()]
a["dur_h"] = (a["toff"] - a["t"]).dt.total_seconds() / 3600
desc = None
p = f"{ROOT}/Hill_of_Towie_alarms_description.csv"
if os.path.exists(p):
    raw = open(p, "rb").read()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            desc = pd.read_csv(io.StringIO(raw.decode(enc)))
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    log("alarm description columns:", list(desc.columns) if desc is not None else None)
code_txt = {}
if desc is not None:
    cc = [c for c in desc.columns if "code" in c.lower()]
    tc = [c for c in desc.columns if c not in cc]
    if cc and tc:
        code_txt = dict(zip(desc[cc[0]], desc[tc[0]].astype(str).str[:70]))

W = {"before_60d": ("2021-07-25", "2021-09-22 23:59"), "works": ("2021-09-23", "2021-09-29 23:59"),
     "after_60d": ("2021-09-30", "2021-11-28 23:59"), "jan_aug_2021": ("2021-01-01", "2021-08-31 23:59"),
     "oct21_may22": ("2021-10-01", "2022-05-31 23:59")}
rows = []
for k, (b, e) in W.items():
    sel = a[(a["t"] >= pd.Timestamp(b, tz="UTC")) & (a["t"] <= pd.Timestamp(e, tz="UTC"))]
    cnt = sel.groupby("st").size().reindex(STATIONS, fill_value=0)
    rows.append({"window": k, "T13": int(cnt[T13]), "fleet_median": float(cnt.drop(T13).median()),
                 "T13_rank_of_21": int((cnt >= cnt[T13]).sum())})
    if k in ("works", "after_60d", "before_60d"):
        t13 = sel[sel["st"] == T13].groupby("Alarmcode").agg(n=("t", "size"), hours=("dur_h", "sum"))
        oth = sel[sel["st"] != T13].groupby(["Alarmcode", "st"]).size().unstack(fill_value=0).reindex(columns=[s for s in STATIONS if s != T13], fill_value=0)
        t13["fleet_median_n"] = oth.median(axis=1).reindex(t13.index).fillna(0)
        t13["n_other_turbines_with_code"] = (oth > 0).sum(axis=1).reindex(t13.index).fillna(0).astype(int)
        t13["text"] = [code_txt.get(c, "") for c in t13.index]
        t13 = t13.sort_values("n", ascending=False)
        t13.to_csv(f"{OUT}/R5b_T13_codes_{k}.csv")
        log(f"\nT13 alarm codes, window {k} ({b} .. {e}):")
        log(t13.head(25).round(2).to_string())
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/R5b_alarm_counts.csv", index=False)
log("\nalarm record counts per window:")
log(R.to_string(index=False))
log("\nDONE")
