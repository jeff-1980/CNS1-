"""Q1 scan pre-pool baseline for Q-20261030-1: did the AeroUp blade upgrade at Hill of
Towie raise drivetrain load exposure?

Data: Hill of Towie open dataset v2.1.0 (Zenodo 10.5281/zenodo.22662930, CC BY 4.0), yearly
zips 2020-2022 (10-min tables tblSCTurGrid, tblSCTurbine, tblSCTurTemp; TimeStamp marks the
END of the 10-min period), ShutdownDuration.zip (TimeStamp_StartFormat = START of period).
21 Siemens SWT-2.3-VS-82, rated 2300 kW. StationId 2304510 = T01 ... 2304530 = T21.

Design (revised 2026-11-01 BEFORE any run, after reading the install-date file): the 2022
rollout has long interim periods (first works 2022-07/09, completion up to 2023-05) and no
not-yet-treated controls after 2023-05, so the staggered TWFE in the first draft is not
identified cleanly. The baseline therefore uses the clean single-turbine window that RES
also used: T13 upgraded 2021-09-23..29, every other turbine untouched until 2022-07-14.
  pre  = 2020-01-01 .. 2021-09-22,  post = 2021-09-30 .. 2022-07-13.
Reference-turbine method: for each 10-min stamp, cell = (reference mean power, 50 kW bins)
x (reference circular-mean yaw, 8 sectors), reference = all other turbines operating
normally (>= 10 needed). Cell means of the target turbine in pre are applied to post;
effect = observed - expected. Normal operation: ShutdownDuration == 0, ActPower > 0, and no
curtailment (PowerRef_endvalue >= 2290 when present).

Outcomes for the target turbine:
  E  energy ratio  sum(P_obs) / sum(P_exp) - 1             (known-truth anchor: RES T13 +4.3 %)
  R  time at rated: share of stamps with P >= 0.95 * 2300  (percentage points)
  D3, D10_3  torque damage-equivalent proxy ratio, T = P / omega_main, m = 3 and 10/3
  H  gearbox HS bearing (generator side) temperature residual, cell = own power bin x
     ambient 5 K bin (informational)
CI: weekly block bootstrap (500 draws) over pre and post weeks.

Pre-registered thresholds (fixed before running):
  S  method sanity: T13 E between +2 % and +7 %; otherwise stop, method invalid.
  P1 T13 R increase >= 1.0 pp with 95 % CI excluding 0.
  P2 placebos: (a) each of the other 20 turbines at the real date: median |R effect| < half of
     T13's; (b) T13 with fake split 2020-12-01 inside pre: |R effect| < half of T13's.
Keep in pool only if S, P1 and P2 all pass. D and H are reported, not gated.
"""
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = sys.argv[1]
NBOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 500
RATED = 2300.0
T13 = 2304522
STATIONS = list(range(2304510, 2304531))
PRE = ("2020-01-01", "2021-09-22 23:59")
POST = ("2021-09-30", "2022-07-13 23:59")
FAKE_SPLIT = "2020-12-01"
rng = np.random.default_rng(20261101)


def read_table(table, cols, years=(2020, 2021, 2022)):
    out = []
    for y in years:
        with zipfile.ZipFile(f"{ROOT}/{y}.zip") as z:
            names = sorted(n for n in z.namelist() if n.split("/")[-1].startswith(table + "_"))
            if not names:
                raise SystemExit(f"{table} not in {y}.zip")
            for n in names:
                with z.open(n) as f:
                    head = pd.read_csv(f, nrows=0).columns
                use = [c for c in ["TimeStamp", "StationId"] + cols if c in head]
                missing = set(cols) - set(use)
                if missing:
                    print(f"WARNING {n}: missing {sorted(missing)}")
                with z.open(n) as f:
                    out.append(pd.read_csv(f, usecols=use, low_memory=False))
    df = pd.concat(out, ignore_index=True)
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], utc=True, format="mixed")
    return df.drop_duplicates(["TimeStamp", "StationId"])


grid = read_table("tblSCTurGrid", ["wtc_ActPower_mean"])
tur = read_table("tblSCTurbine", ["wtc_MainSRpm_mean", "wtc_YawPos_mean", "wtc_PowerRef_endvalue"])
tmp = read_table("tblSCTurTemp", ["wtc_HSGenTmp_mean", "wtc_AmbieTmp_mean"])
df = grid.merge(tur, on=["TimeStamp", "StationId"], how="inner").merge(
    tmp, on=["TimeStamp", "StationId"], how="left")

with zipfile.ZipFile(f"{ROOT}/Hill_of_Towie_ShutdownDuration.zip") as z:
    name = [n for n in z.namelist() if n.endswith(".csv")][0]
    with z.open(name) as f:
        sd = pd.read_csv(f)
sd["TimeStamp"] = pd.to_datetime(sd["TimeStamp_StartFormat"], utc=True, format="mixed") + pd.Timedelta(minutes=10)
sd["StationId"] = sd["TurbineName"].str[1:].astype(int) + 2304509
sd = sd.drop_duplicates(["TimeStamp", "StationId"])
df = df.merge(sd[["TimeStamp", "StationId", "ShutdownDuration"]], on=["TimeStamp", "StationId"], how="left")
print("rows merged:", len(df), " share with shutdown info:", round(df["ShutdownDuration"].notna().mean(), 3))

ok = (df["ShutdownDuration"].fillna(1) == 0) & (df["wtc_ActPower_mean"] > 0)
if "wtc_PowerRef_endvalue" in df:
    ok &= df["wtc_PowerRef_endvalue"].fillna(RATED) >= 2290
df = df[ok].copy()
df["omega"] = df["wtc_MainSRpm_mean"] * 2 * np.pi / 60
df = df[df["omega"] > 0.3]
df["T"] = df["wtc_ActPower_mean"] * 1e3 / df["omega"] / 1e6  # MNm, electrical torque proxy
print("normal-operation rows:", len(df), " per turbine median:", int(df.groupby("StationId").size().median()))

P = df.pivot_table(index="TimeStamp", columns="StationId", values="wtc_ActPower_mean")
Y = df.pivot_table(index="TimeStamp", columns="StationId", values="wtc_YawPos_mean")
TQ = df.pivot_table(index="TimeStamp", columns="StationId", values="T")
HS = df.pivot_table(index="TimeStamp", columns="StationId", values="wtc_HSGenTmp_mean")
AMB = df.pivot_table(index="TimeStamp", columns="StationId", values="wtc_AmbieTmp_mean")


def frame_for(target, exclude=()):
    refs = [s for s in P.columns if s != target and s not in exclude]
    rp = P[refs]
    n = rp.notna().sum(axis=1)
    yaw = np.deg2rad(Y[refs])
    ang = np.rad2deg(np.arctan2(np.sin(yaw).mean(axis=1), np.cos(yaw).mean(axis=1))) % 360
    f = pd.DataFrame({
        "p": P[target], "t": TQ[target], "hs": HS.get(target), "amb": AMB.get(target),
        "rp": rp.mean(axis=1), "sec": ang // 45, "n": n})
    f = f[(f["n"] >= 10) & f["p"].notna()].dropna(subset=["rp", "sec", "t"])
    f["cell"] = (f["rp"] // 50).astype(int).astype(str) + "_" + f["sec"].astype(int).astype(str)
    f["rated"] = (f["p"] >= 0.95 * RATED).astype(float)
    f["week"] = f.index.to_period("W").astype(str)
    return f


def effects(pre, post):
    cm = pd.DataFrame({"p": pre["p"], "rated": pre["rated"], "t3": pre["t"] ** 3,
                       "t103": pre["t"] ** (10 / 3)}).groupby(pre["cell"]).mean()
    po = post[post["cell"].isin(cm.index)]
    exp = cm.loc[po["cell"]].to_numpy()
    obs = np.column_stack([po["p"], po["rated"], po["t"] ** 3, po["t"] ** (10 / 3)])
    e, o = exp.mean(axis=0), obs.mean(axis=0)
    return {"E": o[0] / e[0] - 1, "R_pp": 100 * (o[1] - e[1]),
            "D3": (o[2] / e[2]) ** (1 / 3) - 1, "D10_3": (o[3] / e[3]) ** (3 / 10) - 1,
            "coverage": len(po) / max(len(post), 1)}


def boot(pre, post, n):
    pg, qg = dict(tuple(pre.groupby("week"))), dict(tuple(post.groupby("week")))
    pw, qw = list(pg), list(qg)
    res = []
    for _ in range(n):
        a = pd.concat([pg[w] for w in rng.choice(pw, len(pw))])
        b = pd.concat([qg[w] for w in rng.choice(qw, len(qw))])
        res.append(effects(a, b))
    return pd.DataFrame(res)


def split(f, pre_win, post_win):
    return f.loc[pre_win[0]:pre_win[1]], f.loc[post_win[0]:post_win[1]]


def hs_residual(f, pre_win, post_win):
    g = f.dropna(subset=["hs", "amb"]).copy()
    g["c2"] = (g["p"] // 100).astype(int).astype(str) + "_" + (g["amb"] // 5).astype(int).astype(str)
    a, b = split(g, pre_win, post_win)
    m = a.groupby("c2")["hs"].mean()
    b = b[b["c2"].isin(m.index)]
    return float((b["hs"] - m.loc[b["c2"]].to_numpy()).mean())


f13 = frame_for(T13)
pre, post = split(f13, PRE, POST)
main = effects(pre, post)
ci = boot(pre, post, NBOOT).quantile([0.025, 0.975])
print(f"\nT13 main (pre {len(pre)} stamps, post {len(post)} stamps)")
for k in ["E", "R_pp", "D3", "D10_3"]:
    print(f"  {k}: {main[k]:+.4f}  95% CI [{ci.loc[0.025, k]:+.4f}, {ci.loc[0.975, k]:+.4f}]")
print(f"  coverage of post stamps by pre cells: {main['coverage']:.3f}")
print(f"  H (HS bearing temp residual, K): {hs_residual(f13, PRE, POST):+.3f}")

plc = []
for s in STATIONS:
    if s == T13 or s not in P.columns:
        continue
    a, b = split(frame_for(s, exclude=(T13,)), PRE, POST)
    if len(a) > 1000 and len(b) > 1000:
        e = effects(a, b)
        e["station"] = s
        plc.append(e)
plc = pd.DataFrame(plc)
print("\nPlacebo (a) other turbines, real date:")
print(plc[["station", "E", "R_pp", "D3"]].round(4).to_string(index=False))
med_abs = plc["R_pp"].abs().median()
print(f"  median |R_pp| = {med_abs:.3f}")

fa, fb = split(f13, (PRE[0], "2020-11-30 23:59"), (FAKE_SPLIT, PRE[1]))
fake = effects(fa, fb)
print(f"\nPlacebo (b) T13 fake split {FAKE_SPLIT}: E {fake['E']:+.4f} R_pp {fake['R_pp']:+.3f} D3 {fake['D3']:+.4f}")

S = 0.02 <= main["E"] <= 0.07
P1 = main["R_pp"] >= 1.0 and ci.loc[0.025, "R_pp"] > 0
P2 = med_abs < 0.5 * abs(main["R_pp"]) and abs(fake["R_pp"]) < 0.5 * abs(main["R_pp"])
print(f"\nS (sanity) {'PASS' if S else 'FAIL'}; P1 {'PASS' if P1 else 'FAIL'}; P2 {'PASS' if P2 else 'FAIL'}")
print("VERDICT:", "KEEP" if (S and P1 and P2) else ("METHOD INVALID" if not S else "DROP"))
