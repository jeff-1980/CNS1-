"""AeroUp paper, revision analyses (descriptive; no new pass/fail gates). Answers a pre-submission review:
 R1 joint farm-wide week-block bootstrap: every draw resamples calendar weeks for all turbines at once and
    re-estimates all 21 turbine effects, the placebo median drift, the corrected E, R, D_m and C, and T13's rank.
 R2 placebo in time: pseudo-retrofit dates inside the pre-retrofit period (no turbine changed), all 21 turbines,
    same estimator and drift correction -> null distribution of corrected effects (time x space); plus a +4%
    injection into T13's own data at each pseudo date (single-turbine recovery error on the target itself).
 R3 per-turbine (not only median) recovery error of the frozen K1 injection on the real window.
 R4 mechanism decomposition: energy = mean torque x speed split for T13, and counterfactual C for (a) pure scaling,
    (b) T13's measured gain-by-power-bin profile at fixed rotor speed, (c) the same with the region-2 K*omega^2 speed
    response, compared with measured C.
 R5 concurrent-change screen for T13: table inventory, event/alarm tables if present, and monthly control
    signatures (rated rotor speed, part-load pitch, power at rated, power set-point, anemometer ratio) T13 minus
    fleet median, with step size and rank at the retrofit date.
Same data, filter, T = P/omega, cells (reference power 50 kW x yaw 8 sectors) and windows as paper_aeroup.py.
Usage: python paper_aeroup_rev.py <ROOT> <OUTDIR> [NBOOT=1000]
"""
import os
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]
NBOOT = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
os.makedirs(OUT, exist_ok=True)
LOGF = f"{OUT}/log_rev.txt"
if os.path.exists(LOGF):
    os.remove(LOGF)
RATED = 2300.0
T13 = 2304522
STATIONS = list(range(2304510, 2304531))
NAME = {s: f"T{s - 2304509:02d}" for s in STATIONS}
PRE = ("2020-01-01", "2021-09-22 23:59")
POST = ("2021-09-30", "2022-07-13 23:59")
RETRO = pd.Timestamp("2021-09-26", tz="UTC")
MS = [3, 10 / 3, 4, 6, 10]
MK = {m: ("D" + (f"{m:g}" if m != 10 / 3 else "10_3")) for m in MS}
rng = np.random.default_rng(20261106)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOGF, "a") as fh:
        fh.write(s + "\n")


# ------------------------------------------------------------------ inventory of tables (for R5)
inv = {}
with zipfile.ZipFile(f"{ROOT}/2021.zip") as z:
    for n in z.namelist():
        base = n.split("/")[-1]
        key = base.split("_")[0] if base.startswith("tbl") else base
        inv.setdefault(key, []).append(n)
log("tables in 2021.zip:", {k: len(v) for k, v in inv.items()})
for p in sorted(os.listdir(ROOT)):
    log("ROOT file:", p)


def cols_of(table, year=2021):
    with zipfile.ZipFile(f"{ROOT}/{year}.zip") as z:
        n = sorted(n for n in z.namelist() if n.split("/")[-1].startswith(table + "_"))[0]
        with z.open(n) as f:
            return list(pd.read_csv(f, nrows=0).columns)


def read_table(table, cols, years=(2020, 2021, 2022)):
    out = []
    for y in years:
        with zipfile.ZipFile(f"{ROOT}/{y}.zip") as z:
            for n in sorted(n for n in z.namelist() if n.split("/")[-1].startswith(table + "_")):
                with z.open(n) as f:
                    head = pd.read_csv(f, nrows=0).columns
                use = [c for c in ["TimeStamp", "StationId"] + cols if c in head]
                with z.open(n) as f:
                    out.append(pd.read_csv(f, usecols=use, low_memory=False))
    df = pd.concat(out, ignore_index=True)
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], utc=True, format="mixed")
    return df.drop_duplicates(["TimeStamp", "StationId"])


ct = cols_of("tblSCTurbine")
PITCH = [c for c in ct if "pitc" in c.lower() and c.endswith("_mean")][:1]
WS = "wtc_AcWindSp_mean" if "wtc_AcWindSp_mean" in ct else None
log("tblSCTurbine columns:", ct)
grid = read_table("tblSCTurGrid", ["wtc_ActPower_mean"])
tur = read_table("tblSCTurbine", ["wtc_MainSRpm_mean", "wtc_YawPos_mean", "wtc_PowerRef_endvalue"] + PITCH + ([WS] if WS else []))
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
del sd

# ------------------------------------------------------------------ R5a. monthly control signatures BEFORE the normal-operation filter is applied
log("\n=== R5. concurrent-change screen for T13 ===")
d5 = df[(df["ShutdownDuration"].fillna(1) == 0) & (df["wtc_ActPower_mean"] > 0)].copy()
d5["month"] = d5["TimeStamp"].dt.tz_convert(None).dt.to_period("M")
d5["rpm"] = d5["wtc_MainSRpm_mean"]
sig = {}
at_rated = d5["wtc_ActPower_mean"] >= 0.95 * RATED
sig["rpm_p99_at_rated"] = d5[at_rated].groupby(["month", "StationId"])["rpm"].quantile(0.99)
sig["rpm_median_at_rated"] = d5[at_rated].groupby(["month", "StationId"])["rpm"].median()
sig["power_mean_at_rated"] = d5[at_rated].groupby(["month", "StationId"])["wtc_ActPower_mean"].mean()
sig["setpoint_share_below_2290"] = (d5["wtc_PowerRef_endvalue"].fillna(RATED) < 2290).groupby([d5["month"], d5["StationId"]]).mean()
if PITCH:
    part = d5["wtc_ActPower_mean"] < 0.5 * RATED
    sig["pitch_median_part_load"] = d5[part].groupby(["month", "StationId"])[PITCH[0]].median()
if WS:
    sig["ws_mean"] = d5.groupby(["month", "StationId"])[WS].mean()
del d5
rows = []
for k, s in sig.items():
    w = s.unstack("StationId")
    if k == "ws_mean":  # anemometer ratio to fleet median (sensor change screen)
        w = w.div(w.median(axis=1), axis=0)
    w = w.loc[pd.Period("2020-10", "M"):pd.Period("2022-06", "M")]
    dev = w.sub(w.drop(columns=[T13], errors="ignore").median(axis=1), axis=0)
    pre_m = dev.loc[pd.Period("2021-01", "M"):pd.Period("2021-08", "M")].mean()
    post_m = dev.loc[pd.Period("2021-10", "M"):pd.Period("2022-05", "M")].mean()
    step = post_m - pre_m
    rows.append({"signature": k, "T13_step": step.get(T13, np.nan),
                 "rank_abs_step_of_21": int((step.abs() >= abs(step.get(T13, np.nan))).sum()),
                 "fleet_abs_step_median": step.drop(T13, errors="ignore").abs().median()})
    dev[T13].rename(k).to_frame().to_csv(f"{OUT}/R5_T13_monthly_{k}.csv")
R5 = pd.DataFrame(rows)
R5.to_csv(f"{OUT}/R5_control_signature_steps.csv", index=False)
log(R5.round(4).to_string(index=False))
log("(step = mean T13-minus-fleet deviation Oct21-May22 minus Jan-Aug21; rank 1 = largest absolute step among 21)")
# event-type tables if present
for key in inv:
    if any(w in key.lower() for w in ("alarm", "event", "log", "status")) and key.startswith("tbl"):
        try:
            c = cols_of(key)
            log(f"\n{key} columns: {c}")
            ev = read_table(key, [x for x in c if x not in ("TimeStamp", "StationId")][:6], years=(2021, 2022))
            ev = ev[ev["StationId"].isin(STATIONS)]
            win = ev[(ev["TimeStamp"] >= RETRO - pd.Timedelta(days=60)) & (ev["TimeStamp"] <= RETRO + pd.Timedelta(days=60))]
            cnt = win.groupby("StationId").size()
            log(f"{key}: records within +-60 d of retrofit, T13 {cnt.get(T13, 0)}; fleet median {cnt.drop(T13, errors='ignore').median()}")
            code_col = [x for x in ev.columns if x not in ("TimeStamp", "StationId")]
            if code_col:
                t13 = win[win["StationId"] == T13]
                log(f"T13 most frequent {code_col[0]} values in +-60 d:", t13[code_col[0]].value_counts().head(15).to_dict())
        except Exception as ex:  # descriptive screen only
            log(f"{key}: could not summarise ({ex})")

# ------------------------------------------------------------------ normal-operation filter and pivots
ok = (df["ShutdownDuration"].fillna(1) == 0) & (df["wtc_ActPower_mean"] > 0) & (df["wtc_PowerRef_endvalue"].fillna(RATED) >= 2290)
df = df[ok].copy()
df["omega"] = df["wtc_MainSRpm_mean"] * 2 * np.pi / 60
df = df[df["omega"] > 0.3]
df["T"] = df["wtc_ActPower_mean"] * 1e3 / df["omega"] / 1e6
log("\nnormal-operation rows:", len(df))
P = df.pivot_table(index="TimeStamp", columns="StationId", values="wtc_ActPower_mean")
Y = df.pivot_table(index="TimeStamp", columns="StationId", values="wtc_YawPos_mean")
TQ = df.pivot_table(index="TimeStamp", columns="StationId", values="T")
W = df.pivot_table(index="TimeStamp", columns="StationId", values="omega")
del df


def frame_for(target):
    refs = [s for s in P.columns if s not in (target, T13)]
    n = P[refs].notna().sum(axis=1)
    yaw = np.deg2rad(Y[refs])
    ang = np.rad2deg(np.arctan2(np.sin(yaw).mean(axis=1), np.cos(yaw).mean(axis=1))) % 360
    f = pd.DataFrame({"p": P[target], "t": TQ[target], "w": W[target], "rp": P[refs].mean(axis=1), "sec": ang // 45, "n": n})
    f = f[(f["n"] >= 10) & f["p"].notna()].dropna(subset=["rp", "sec", "t", "w"])
    f["cell"] = ((f["rp"] // 50).astype(int) * 8 + f["sec"].astype(int)).astype(int)
    f["rated"] = (f["p"] >= 0.95 * RATED).astype(float)
    f["wid"] = ((f.index - pd.Timestamp("2019-12-30", tz="UTC")).days // 7).astype(int)
    return f


VCOLS = ["p", "rated", "w", "t1"] + [MK[m] for m in MS]


def vals(f):
    v = np.column_stack([f["p"], f["rated"], f["w"], f["t"]] + [f["t"].to_numpy() ** m for m in MS])
    return v


def eff_w(c_pre, v_pre, wt_pre, c_post, v_post, wt_post, ncell):
    """weighted reference-cell estimator; weights = bootstrap multiplicities (1 = plain estimate)"""
    N = np.bincount(c_pre, weights=wt_pre, minlength=ncell)
    S = np.column_stack([np.bincount(c_pre, weights=wt_pre * v_pre[:, j], minlength=ncell) for j in range(v_pre.shape[1])])
    ok = N[c_post] > 0
    cm = S[c_post[ok]] / N[c_post[ok], None]
    wq = wt_post[ok]
    e = (cm * wq[:, None]).sum(0) / wq.sum()
    o = (v_post[ok] * wq[:, None]).sum(0) / wq.sum()
    r = {"E": o[0] / e[0] - 1, "R_pp": 100 * (o[1] - e[1]), "W": o[2] / e[2] - 1, "T1": o[3] / e[3] - 1}
    for i, m in enumerate(MS):
        r[MK[m]] = (o[4 + i] / e[4 + i]) ** (1 / m) - 1
    return r


KEYS = ["E", "R_pp", "W", "T1"] + [MK[m] for m in MS]


class Prep:
    def __init__(self, f, a, b):
        pre, post = f.loc[a[0]:a[1]], f.loc[b[0]:b[1]]
        self.ok = len(pre) > 1000 and len(post) > 1000
        self.cp, self.vp, self.wkp = pre["cell"].to_numpy(), vals(pre), pre["wid"].to_numpy()
        self.cq, self.vq, self.wkq = post["cell"].to_numpy(), vals(post), post["wid"].to_numpy()
        self.post = post
        self.nc = int(max(self.cp.max(initial=0), self.cq.max(initial=0))) + 1

    def est(self, cnt_pre=None, cnt_post=None, vq=None):
        wp = np.ones(len(self.cp)) if cnt_pre is None else cnt_pre[self.wkp]
        wq = np.ones(len(self.cq)) if cnt_post is None else cnt_post[self.wkq]
        return eff_w(self.cp, self.vp, wp, self.cq, self.vq if vq is None else vq, wq, self.nc)


def corrected(effs):
    t = pd.DataFrame(effs).T
    oth = [s for s in t.index if s != T13]
    c = t.loc[T13, KEYS] - t.loc[oth, KEYS].median()
    for m in MS:
        c["C_" + MK[m]] = c[MK[m]] / c["E"]
    rk = {k: int((t[k] >= t.loc[T13, k]).sum()) for k in ["E", "R_pp", "D3"]}
    return c, rk, t


frames = {s: frame_for(s) for s in STATIONS if s in P.columns}
prep = {s: Prep(frames[s], PRE, POST) for s in frames}
prep = {s: p for s, p in prep.items() if p.ok}
c0, rk0, t0 = corrected({s: p.est() for s, p in prep.items()})
log("\npoint estimate (reproduces paper): ", c0[["E", "R_pp", "D3", "C_D3", "W", "T1"]].round(4).to_dict(), "ranks", rk0)

# ------------------------------------------------------------------ R1. joint farm-wide bootstrap
log(f"\n=== R1. joint week-block bootstrap of the whole farm ({NBOOT} draws) ===")
weeks_pre = np.array(sorted(set().union(*[set(p.wkp.tolist()) for p in prep.values()])))
weeks_post = np.array(sorted(set().union(*[set(p.wkq.tolist()) for p in prep.values()])))
NW = int(max(weeks_pre.max(), weeks_post.max())) + 1
bs = []
for i in range(NBOOT):
    dp = np.bincount(rng.choice(weeks_pre, len(weeks_pre)), minlength=NW).astype(float)
    dq = np.bincount(rng.choice(weeks_post, len(weeks_post)), minlength=NW).astype(float)
    c, rk, _ = corrected({s: p.est(dp, dq) for s, p in prep.items()})
    row = c.to_dict()
    row.update({f"rank_{k}": v for k, v in rk.items()})
    bs.append(row)
bs = pd.DataFrame(bs)
bs.to_csv(f"{OUT}/R1_joint_bootstrap.csv", index=False)
summ = pd.DataFrame({"estimate": c0, "lo": bs[c0.index].quantile(0.025), "hi": bs[c0.index].quantile(0.975)})
summ.to_csv(f"{OUT}/R1_joint_bootstrap_summary.csv")
log(summ.round(4).to_string())
for k in ["E", "R_pp", "D3"]:
    log(f"share of draws with T13 rank 1 on {k}: {(bs['rank_' + k] == 1).mean():.3f}; corrected {k} > 0: {(bs[k] > 0).mean():.3f}")

# ------------------------------------------------------------------ R2. placebo in time (pseudo-retrofit dates before 2021-09-22)
log("\n=== R2. pseudo-retrofit dates inside the pre-retrofit period ===")
FAKE = ["2020-07-01", "2020-09-01", "2020-11-01", "2021-01-01", "2021-03-01"]
rows, inj = [], []
for d in FAKE:
    dd = pd.Timestamp(d, tz="UTC")
    a = (PRE[0], str((dd - pd.Timedelta(days=8)).date()) + " 23:59")
    b = (d, str(min(dd + pd.Timedelta(days=290), pd.Timestamp("2021-09-22", tz="UTC")).date()) + " 23:59")
    pr = {s: Prep(frames[s], a, b) for s in frames}
    pr = {s: p for s, p in pr.items() if p.ok}
    eff = {s: p.est() for s, p in pr.items()}
    t = pd.DataFrame(eff).T
    for s in t.index:  # every turbine as pseudo-treated, corrected by the median of the other 20
        o = [x for x in t.index if x != s]
        c = t.loc[s, KEYS] - t.loc[o, KEYS].median()
        rows.append({"date": d, "station": s, "turbine": NAME[s], **c.to_dict(),
                     "rank_E": int((t["E"] >= t.loc[s, "E"]).sum()), "rank_R": int((t["R_pp"] >= t.loc[s, "R_pp"]).sum())})
    if T13 in pr:  # +4% injected into T13's own post data at the pseudo date
        p = pr[T13]
        q = p.post.copy()
        p2 = np.minimum(q["p"] * 1.04, RATED)
        q["t"] = q["t"] * p2 / q["p"]
        q["p"] = p2
        q["rated"] = (q["p"] >= 0.95 * RATED).astype(float)
        est = p.est(vq=vals(q))
        o = [x for x in t.index if x != T13]
        med = t.loc[o, KEYS].median()
        tru = {"E": q["p"].mean() / p.post["p"].mean() - 1, "R_pp": 100 * (q["rated"].mean() - p.post["rated"].mean()),
               "D3": ((q["t"] ** 3).mean() / (p.post["t"] ** 3).mean()) ** (1 / 3) - 1}
        inj.append({"date": d, **{k + "_true": tru[k] for k in tru},
                    **{k + "_rec_minus_true": est[k] - med[k] - tru[k] for k in tru},
                    "C_true": tru["D3"] / tru["E"], "C_rec": (est["D3"] - med["D3"]) / (est["E"] - med["E"])})
R2 = pd.DataFrame(rows)
R2.to_csv(f"{OUT}/R2_pseudo_dates_all_turbines.csv", index=False)
INJ = pd.DataFrame(inj)
INJ.to_csv(f"{OUT}/R2_T13_injection_at_pseudo_dates.csv", index=False)
log("T13 corrected effects at pseudo dates (true effect 0):")
log(R2[R2.station == T13][["date", "E", "R_pp", "D3", "W", "rank_E", "rank_R"]].round(4).to_string(index=False))
nul = R2[R2.station != T13]
for k in ["E", "R_pp", "D3"]:
    allnull = R2[k]
    log(f"{k}: null distribution over {len(allnull)} turbine-date pairs: sd {allnull.std():.4f}, 95th pct {allnull.quantile(0.95):.4f}, "
        f"max {allnull.max():.4f}; real T13 {c0[k]:.4f}; share of null >= real {(allnull >= c0[k]).mean():.4f}")
log("T13-only null (5 dates):", R2[R2.station == T13][["E", "R_pp", "D3"]].abs().max().round(4).to_dict())
log("\n+4% injected into T13 at pseudo dates (recovered minus true):")
log(INJ.round(4).to_string(index=False))

# ------------------------------------------------------------------ R4. mechanism decomposition
log("\n=== R4. mechanism: torque x speed split and counterfactual load cost ratios ===")
log(f"T13 corrected: E {c0['E']:.4f}; mean torque T1 {c0['T1']:.4f}; rotor speed W {c0['W']:.4f}; "
    f"share of energy gain carried by speed W/E {c0['W'] / c0['E']:.3f}; by torque T1/E {c0['T1'] / c0['E']:.3f}")
p13 = prep[T13]
pre = frames[T13].loc[PRE[0]:PRE[1]]
post = p13.post
# measured relative uplift by reference-power bin (T13 minus fleet median), from the same estimator
bins = np.arange(0, 2400, 200)


def bin_uplift(s):
    p = prep[s]
    N = np.bincount(p.cp, minlength=p.nc)
    S = np.bincount(p.cp, weights=p.vp[:, 0], minlength=p.nc)
    ok = N[p.cq] > 0
    ex = S[p.cq[ok]] / N[p.cq[ok]]
    rp = p.post["rp"].to_numpy()[ok]
    b = np.clip((rp // 200) * 200, 0, 2200)
    g = pd.DataFrame({"b": b, "o": p.vq[ok, 0], "e": ex}).groupby("b").sum()
    return g.o / g.e - 1


U = pd.DataFrame({s: bin_uplift(s) for s in prep}).T
u_net = (U.loc[T13] - U.drop(T13).median()).reindex(bins).fillna(0)
log("T13 net relative uplift by reference-power bin:", u_net.round(4).to_dict())
pre_b = np.clip((pre["rp"].to_numpy() // 200) * 200, 0, 2200)
u = u_net.reindex(pre_b).to_numpy()
p0, t0_, w0 = pre["p"].to_numpy(), pre["t"].to_numpy(), pre["w"].to_numpy()
w_rated = np.nanquantile(w0[p0 >= 0.95 * RATED], 0.5)
reg2 = (p0 < 0.9 * RATED) & (w0 < 0.98 * w_rated)


def cf(p1, w1):
    t1 = p1 * 1e3 / w1 / 1e6
    r = {"E": p1.mean() / p0.mean() - 1}
    for m in MS:
        r[MK[m]] = ((t1 ** m).mean() / (t0_ ** m).mean()) ** (1 / m) - 1
    return r


k = 1 + c0["E"]
cases = {
    "a_pure_scaling_fixed_speed": cf(np.minimum(p0 * k, RATED), w0),
    "b_measured_bin_profile_fixed_speed": cf(np.minimum(p0 * (1 + u), RATED), w0),
    "c_bin_profile_plus_Kw2_speed": cf(np.minimum(p0 * (1 + u), RATED), np.where(reg2, w0 * (1 + u) ** (1 / 3), w0)),
}
R4 = pd.DataFrame(cases).T
for m in MS:
    R4["C_" + MK[m]] = R4[MK[m]] / R4["E"]
meas = {"E": c0["E"], **{MK[m]: c0[MK[m]] for m in MS}, **{"C_" + MK[m]: c0["C_" + MK[m]] for m in MS}}
R4.loc["measured"] = pd.Series(meas)
R4.to_csv(f"{OUT}/R4_mechanism_counterfactuals.csv")
log(R4[["E", "D3", "C_D3", "C_D10_3", "C_D10"]].round(4).to_string())
log(f"(region 2 = P < 0.9 rated and omega < 0.98 x median rated speed {w_rated:.3f} rad/s; share of pre records {reg2.mean():.3f})")

# ------------------------------------------------------------------ R3. per-turbine K1 recovery error is computed from paper run CSV (D_known_truth.csv) in make_rev_tables
log("\nDONE")
