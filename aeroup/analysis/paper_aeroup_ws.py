"""AeroUp paper, wind-speed reconciliation (descriptive; no gates).
Question: an independent analysis of the same farm (Wang et al. 2026) places most of the AeroUp gain at 10-18 m/s,
while our power-bin decomposition puts most of the energy gain at part load. Two candidate reasons:
 (i) binning variable (farm reference power vs wind speed), (ii) metric (power-curve uplift in kW per bin vs
 frequency-weighted share of energy). This script reports both metrics on wind-speed bins.
Wind speed definitions: refWS = mean nacelle wind speed (wtc_AcWindSp_mean) of the reference turbines (all except
T13 and target), unaffected by T13's retrofit; ownWS = T13's own nacelle anemometer (can be biased by the retrofit
through the nacelle transfer function; reported for comparison only).
Also re-estimates E, R, D3, C with cells = refWS 0.5 m/s x yaw 8 sectors (robustness of the reference variable).
Same data, filter, windows, T = P/omega and drift correction (minus median of 20 unmodified turbines) as paper_aeroup.py.
Usage: python paper_aeroup_ws.py <ROOT> <OUTDIR> [NBOOT=1000]
"""
import os
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]
NBOOT = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
os.makedirs(OUT, exist_ok=True)
RATED = 2300.0
T13 = 2304522
STATIONS = list(range(2304510, 2304531))
PRE = ("2020-01-01", "2021-09-22 23:59")
POST = ("2021-09-30", "2022-07-13 23:59")
WS = "wtc_AcWindSp_mean"
rng = np.random.default_rng(20261105)
LOG = open(f"{OUT}/log_ws.txt", "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def read_table(table, cols, years=(2020, 2021, 2022)):
    out = []
    for y in years:
        with zipfile.ZipFile(f"{ROOT}/{y}.zip") as z:
            names = sorted(n for n in z.namelist() if n.split("/")[-1].startswith(table + "_"))
            for n in names:
                with z.open(n) as f:
                    head = pd.read_csv(f, nrows=0).columns
                use = [c for c in ["TimeStamp", "StationId"] + cols if c in head]
                with z.open(n) as f:
                    out.append(pd.read_csv(f, usecols=use, low_memory=False))
    df = pd.concat(out, ignore_index=True)
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], utc=True, format="mixed")
    return df.drop_duplicates(["TimeStamp", "StationId"])


grid = read_table("tblSCTurGrid", ["wtc_ActPower_mean"])
tur = read_table("tblSCTurbine", ["wtc_MainSRpm_mean", "wtc_YawPos_mean", "wtc_PowerRef_endvalue", WS])
df = grid.merge(tur, on=["TimeStamp", "StationId"], how="inner")
del grid, tur
with zipfile.ZipFile(f"{ROOT}/Hill_of_Towie_ShutdownDuration.zip") as z:
    name = [n for n in z.namelist() if n.endswith(".csv")][0]
    with z.open(name) as f:
        sd = pd.read_csv(f)
sd["TimeStamp"] = pd.to_datetime(sd["TimeStamp_StartFormat"], utc=True, format="mixed") + pd.Timedelta(minutes=10)
sd["StationId"] = sd["TurbineName"].str[1:].astype(int) + 2304509
sd = sd.drop_duplicates(["TimeStamp", "StationId"])
df = df.merge(sd[["TimeStamp", "StationId", "ShutdownDuration"]], on=["TimeStamp", "StationId"], how="left")
ok = (df["ShutdownDuration"].fillna(1) == 0) & (df["wtc_ActPower_mean"] > 0) & (df["wtc_PowerRef_endvalue"].fillna(RATED) >= 2290)
df = df[ok].copy()
df["omega"] = df["wtc_MainSRpm_mean"] * 2 * np.pi / 60
df = df[df["omega"] > 0.3]
df["T"] = df["wtc_ActPower_mean"] * 1e3 / df["omega"] / 1e6
log("normal-operation rows:", len(df), "; WS non-null share:", round(df[WS].notna().mean(), 4))

P = df.pivot_table(index="TimeStamp", columns="StationId", values="wtc_ActPower_mean")
Y = df.pivot_table(index="TimeStamp", columns="StationId", values="wtc_YawPos_mean")
TQ = df.pivot_table(index="TimeStamp", columns="StationId", values="T")
W = df.pivot_table(index="TimeStamp", columns="StationId", values=WS)
del df


def frame(target, cellvar):
    refs = [s for s in P.columns if s not in (target, T13)]
    n = P[refs].notna().sum(axis=1)
    yaw = np.deg2rad(Y[refs])
    ang = np.rad2deg(np.arctan2(np.sin(yaw).mean(axis=1), np.cos(yaw).mean(axis=1))) % 360
    f = pd.DataFrame({"p": P[target], "t": TQ[target], "rp": P[refs].mean(axis=1), "rws": W[refs].mean(axis=1),
                      "ows": W.get(target), "sec": ang // 45, "n": n})
    f = f[(f["n"] >= 10) & f["p"].notna()].dropna(subset=["rp", "rws", "sec", "t"])
    if cellvar == "rp":
        f["cell"] = (f["rp"] // 50).astype(int).astype(str) + "_" + f["sec"].astype(int).astype(str)
    else:
        f["cell"] = (f["rws"] // 0.5).astype(int).astype(str) + "_" + f["sec"].astype(int).astype(str)
    f["rated"] = (f["p"] >= 0.95 * RATED).astype(float)
    f["t3"] = f["t"] ** 3
    f["week"] = f.index.to_period("W").astype(str)
    return f.loc[PRE[0]:PRE[1]], f.loc[POST[0]:POST[1]]


def expected(pre, post):
    cm = pre[["p", "rated", "t3"]].groupby(pre["cell"]).mean()
    po = post[post["cell"].isin(cm.index)]
    ex = cm.loc[po["cell"]].set_index(po.index)
    return po, ex


def eff(pre, post):
    po, ex = expected(pre, post)
    return {"E": po.p.mean() / ex.p.mean() - 1, "R_pp": 100 * (po.rated.mean() - ex.rated.mean()),
            "D3": (po.t3.mean() / ex.t3.mean()) ** (1 / 3) - 1, "cov": len(po) / len(post)}


# ---------------- 1. decomposition by wind-speed bin (default reference-power cells, as in the paper)
log("\n=== 1. Gain by wind-speed bin (cells = reference power x yaw, as in the paper) ===")
frames = {s: frame(s, "rp") for s in STATIONS if s in P.columns}
rows = []
for s, (pre, post) in frames.items():
    po, ex = expected(pre, post)
    for wsname, wsv in [("refWS", po["rws"]), ("ownWS", po["ows"])]:
        b = np.floor(wsv.clip(0, 25))
        g = pd.DataFrame({"bin": b, "n": 1, "p_obs": po.p, "p_exp": ex.p, "t3_obs": po.t3, "t3_exp": ex.t3}).dropna().groupby("bin").sum()
        g["station"], g["ws"] = s, wsname
        rows.append(g.reset_index())
B = pd.concat(rows)
B.to_csv(f"{OUT}/WS1_bins.csv", index=False)
for wsname in ["refWS", "ownWS"]:
    b = B[B.ws == wsname]
    tot_n = b.groupby("station").n.transform("sum")
    b = b.assign(dE=(b.p_obs - b.p_exp) / tot_n, dT=(b.t3_obs - b.t3_exp) / b.groupby("station").t3_exp.transform("sum"),
                 uplift_kW=(b.p_obs - b.p_exp) / b.n, freq=b.n / tot_n)
    t = b[b.station == T13].set_index("bin")
    o = b[b.station != T13].groupby("bin")[["dE", "dT", "uplift_kW"]].median()
    net = pd.DataFrame({"freq": t.freq, "n_T13": t.n,
                        "uplift_kW_T13": t.uplift_kW, "uplift_kW_fleet_med": o.uplift_kW.reindex(t.index),
                        "uplift_kW_net": t.uplift_kW - o.uplift_kW.reindex(t.index).fillna(0),
                        "dE_net": t.dE - o.dE.reindex(t.index).fillna(0), "dT3_net": t.dT - o.dT.reindex(t.index).fillna(0)})
    net["share_E"] = net.dE_net / net.dE_net.sum()
    net["share_T3"] = net.dT3_net / net.dT3_net.sum()
    net["p_exp_mean_kW"] = (t.p_exp / t.n)
    net["uplift_pct"] = 100 * net.uplift_kW_net / net.p_exp_mean_kW
    net.to_csv(f"{OUT}/WS1_T13_net_{wsname}.csv")
    log(f"\n-- {wsname}: per 1 m/s bin (T13 minus fleet median)")
    log(net[["freq", "n_T13", "p_exp_mean_kW", "uplift_kW_net", "uplift_pct", "share_E", "share_T3"]].round(3).to_string())
    for lo, hi in [(0, 10), (10, 18), (4, 10), (10, 13), (13, 18)]:
        sel = (net.index >= lo) & (net.index < hi)
        log(f"   {wsname} {lo}-{hi} m/s: share of energy gain {net.share_E[sel].sum():.3f}; share of T^3 gain {net.share_T3[sel].sum():.3f}; "
            f"mean net uplift {net.uplift_kW_net[sel].mean():.1f} kW; time share {net.freq[sel].sum():.3f}")

# fleet power curve vs refWS (pre window, unmodified turbines) to locate rated wind speed
pc = []
for s, (pre, post) in frames.items():
    if s == T13:
        continue
    pc.append(pre.groupby(np.floor(pre["rws"].clip(0, 25)))["p"].mean().rename(s))
pc = pd.concat(pc, axis=1).median(axis=1)
pc.to_csv(f"{OUT}/WS2_fleet_power_curve_refWS.csv", header=["p_kW"])
log("\nFleet median power curve vs reference wind speed (pre window):")
log(pc.round(0).to_string())

# ---------------- 2. robustness: cells on reference wind speed instead of reference power
log("\n=== 2. Cells = reference wind speed 0.5 m/s x yaw 8 sectors ===")
fw = {s: frame(s, "rws") for s in STATIONS if s in P.columns}
tab = pd.DataFrame({s: eff(*fw[s]) for s in fw}).T
oth = [s for s in tab.index if s != T13]
med = tab.loc[oth, ["E", "R_pp", "D3"]].median()
corr = tab.loc[T13, ["E", "R_pp", "D3"]] - med
rank = {k: int((tab[k] >= tab.loc[T13, k]).sum()) for k in ["E", "R_pp", "D3"]}
log(tab.round(4).to_string())
pre, post = fw[T13]
pg, qg = dict(tuple(pre.groupby("week"))), dict(tuple(post.groupby("week")))
pw, qw = list(pg), list(qg)
bs = []
for _ in range(NBOOT):
    e = eff(pd.concat([pg[w] for w in rng.choice(pw, len(pw))]), pd.concat([qg[w] for w in rng.choice(qw, len(qw))]))
    bs.append({k: e[k] - med[k] for k in ["E", "R_pp", "D3"]})
bs = pd.DataFrame(bs)
bs["C"] = bs["D3"] / bs["E"]
res = pd.DataFrame({"estimate": list(corr) + [corr["D3"] / corr["E"]],
                    "lo": bs.quantile(0.025).to_numpy(), "hi": bs.quantile(0.975).to_numpy(),
                    "rank": [rank["E"], rank["R_pp"], rank["D3"], np.nan]}, index=["E", "R_pp", "D3", "C"])
res.to_csv(f"{OUT}/WS3_T13_corrected_wscells.csv")
log("\nT13 corrected, ws cells (95% week bootstrap):")
log(res.round(4).to_string())
log("compare paper default (power cells): E 0.0481 [0.0329,0.0625]; R 1.67; D3 0.0230; C 0.477")
log("\nDONE")
