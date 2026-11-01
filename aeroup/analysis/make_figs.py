"""Figures for the AeroUp paper from paper_aeroup.py CSV outputs. Usage: python make_figs.py <OUTDIR> <FIGDIR>"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D, F = sys.argv[1], sys.argv[2]
os.makedirs(F, exist_ok=True)
T13 = 2304522
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 300})
NAME = lambda s: f"T{int(s) - 2304509:02d}"


def save(fig, n):
    fig.tight_layout()
    fig.savefig(f"{F}/{n}.png")
    fig.savefig(f"{F}/{n}.pdf")
    plt.close(fig)


# ---- Fig 2: per-turbine effects at the T13 date (placebo distribution vs T13)
A = pd.read_csv(f"{D}/A_per_turbine_T13window.csv", index_col="station")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, (k, lab, sc) in zip(axs, [("E", "Energy ratio change (%)", 100), ("R_pp", "Time at rated power (pp)", 1),
                                   ("D3", "Torque DEL proxy, m = 3 (%)", 100)]):
    v = (A[k] * sc).sort_values()
    c = [BLUE if s == T13 else GREY for s in v.index]
    ax.barh(range(len(v)), v.to_numpy(), color=c, height=0.7)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(v.drop(T13).median(), color=ORANGE, lw=1, ls="--")
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels([NAME(s) for s in v.index], fontsize=6)
    ax.set_xlabel(lab)
    ax.grid(axis="y", visible=False)
fig.text(0.5, 0.995, "Blue: T13 (retrofitted).  Grey: 20 unmodified turbines.  Dashed: their median (null drift).", ha="center", va="top", fontsize=7, color=MUTED)
save(fig, "fig2_per_turbine")

# ---- Fig 3: monthly event study
M = pd.read_csv(f"{D}/F_monthly_event_study.csv")
M["month"] = pd.PeriodIndex(M["month"], freq="M").to_timestamp()
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True)
for ax, (k, lab, sc) in zip(axs, [("R_pp", "Time at rated (pp)", 1), ("E", "Energy ratio change (%)", 100)]):
    o = M[M.station != T13].groupby("month")[k]
    q1, q3, md = o.quantile(0.25) * sc, o.quantile(0.75) * sc, o.median() * sc
    ax.fill_between(q1.index, q1, q3, color=GREY, alpha=0.35, lw=0, label="20 unmodified turbines (IQR)")
    ax.plot(md.index, md, color=GREY, lw=1.2, label="median")
    t = M[M.station == T13].set_index("month")[k] * sc
    ax.plot(t.index, t, color=BLUE, lw=2, marker="o", ms=3, label="T13")
    ax.axvspan(pd.Timestamp("2021-09-23"), pd.Timestamp("2021-09-29"), color=ORANGE, alpha=0.4, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel(lab)
axs[0].text(pd.Timestamp("2021-10-05"), axs[0].get_ylim()[1] * 0.85, "T13 retrofit", color=INK, fontsize=7)
axs[0].legend(loc="upper left", fontsize=6, ncol=3)
save(fig, "figS4_event_study_raw")

# ---- Fig 4: where the energy and load gains occur (T13 minus median of others, per ref-power bin)
B = pd.read_csv(f"{D}/B_power_bin_decomposition.csv")
B["dp"] = (B.p_obs - B.p_exp) / B.groupby("station").n.transform("sum")
B["dt3"] = (B.t3_obs - B.t3_exp) / B.groupby("station").t3_exp.transform("sum")
B["ep"] = B.p_exp / B.groupby("station").p_exp.transform("sum")
b13 = B[B.station == T13].set_index("bin")
bo = B[B.station != T13].groupby("bin")[["dp", "dt3"]].median()
net_p = (b13.dp - bo.dp.reindex(b13.index).fillna(0))
net_t = (b13.dt3 - bo.dt3.reindex(b13.index).fillna(0))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
x = b13.index.to_numpy() + 100
axs[0].bar(x, 100 * net_p / net_p.sum(), width=170, color=BLUE)
axs[0].set_ylabel("Share of energy gain (%)")
axs[1].bar(x, 100 * net_t / net_t.sum(), width=170, color=ORANGE)
axs[1].set_ylabel("Share of torque-cubed gain (%)")
for ax in axs:
    ax.set_xlabel("Farm reference power (kW, 200 kW bins)")
    ax.axhline(0, color=MUTED, lw=0.8)
save(fig, "fig4_power_bins")

# ---- Fig 5: load cost ratio vs exponent: observed vs pure-scaling benchmark
S = pd.read_csv(f"{D}/A_T13_corrected_summary.csv").set_index("quantity")
SC = pd.read_csv(f"{D}/D_pure_scaling_benchmark.csv")
ms = [("D3", 3), ("D10_3", 10 / 3), ("D4", 4), ("D6", 6), ("D8", 8), ("D10", 10)]
fig, ax = plt.subplots(figsize=(3.5, 2.8))
obs = [S.loc["C_" + k] for k, _ in ms]
mm = [m for _, m in ms]
ax.errorbar(mm, [o.estimate for o in obs], yerr=[[o.estimate - o.lo for o in obs], [o.hi - o.estimate for o in obs]],
            color=BLUE, marker="o", ms=5, lw=1.5, capsize=2, label="T13 measured (drift-corrected)")
e13 = S.loc["E", "estimate"]
kk = SC.loc[(SC.data == "T13_pre"), "k"].to_numpy()
k0 = kk[np.argmin(np.abs(kk - (1 + e13)))]
bench = SC[(SC.data == "T13_pre") & (SC.k == k0)].iloc[0]
ax.plot(mm, [bench["C_" + k] for k, _ in ms], color=GREY, marker="s", ms=4, lw=1.2, ls="--",
        label=f"pure power scaling ×{k0:.2f}, capped")
ax.set_xlabel("S-N exponent m")
ax.set_ylabel("Load cost ratio C = ΔDEL / ΔE")
ax.legend(fontsize=6, loc="upper right")
save(fig, "fig5_cost_ratio")

# ---- Fig 6: replication and null check (pooled R, E, D3)
G = pd.read_csv(f"{D}/G_replication_pooled.csv").set_index("set")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.0))
labs = {"K3_rollout_ref_T13": "2022–23 rollout\n(ref T13)", "K3b_null_ref_T01": "null 2020→21\n(ref T01)",
        "K3b_null_ref_T21": "null 2020→21\n(ref T21)"}
for ax, (k, lab, sc) in zip(axs, [("R_pp", "Time at rated (pp)", 1), ("E", "Energy (%)", 100), ("D3", "Torque DEL m=3 (%)", 100)]):
    for i, s in enumerate(G.index):
        c = BLUE if s.startswith("K3_") else GREY
        ax.errorbar(G.loc[s, k] * sc, i, xerr=[[(G.loc[s, k] - G.loc[s, k + "_lo"]) * sc], [(G.loc[s, k + "_hi"] - G.loc[s, k]) * sc]],
                    color=c, marker="o", ms=5, capsize=2)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(G)))
    ax.set_yticklabels([labs.get(s, s) for s in G.index] if ax is axs[0] else [], fontsize=6)
    ax.set_xlabel(lab)
    ax.set_ylim(-0.6, len(G) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
save(fig, "fig6_replication")

# ---- Fig S1: torque spectrum T13
C = pd.read_csv(f"{D}/C_torque_spectrum.csv")
c13 = C[C.station == T13]
mid = (c13.t_lo + c13.t_hi) / 2
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
axs[0].plot(mid, c13.exp * 100, color=GREY, lw=1.5, label="expected (pre-retrofit)")
axs[0].plot(mid, c13.obs * 100, color=BLUE, lw=1.5, label="observed (post)")
axs[0].set_xlabel("Main-shaft torque proxy (MNm)")
axs[0].set_ylabel("Share of 10-min records (%)")
axs[0].legend(fontsize=6)
oth = C[C.station != T13].assign(d=lambda d: d.obs - d.exp).groupby("bin").d.median().to_numpy()
w = mid.to_numpy()[1] - mid.to_numpy()[0]
axs[1].bar(mid, (c13.obs - c13.exp).to_numpy() * 100 - oth * 100, width=w * 0.85, color=BLUE)
axs[1].axhline(0, color=MUTED, lw=0.8)
axs[1].set_xlabel("Main-shaft torque proxy (MNm)")
axs[1].set_ylabel("Obs. − exp., corrected (pp)")
save(fig, "figS1_torque_spectrum")

# ---- Fig S2: sensitivity grid
E = pd.read_csv(f"{D}/E_sensitivity.csv").set_index("config")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8))
for ax, (k, lab, sc) in zip(axs, [("E", "Energy (%)", 100), ("R_pp", "Time at rated (pp)", 1), ("C_D3", "C (m = 3)", 1)]):
    v = E[k] * sc
    ax.scatter(v, range(len(v)), color=[BLUE if i == "default" else GREY for i in v.index], s=14, zorder=3)
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels(v.index if ax is axs[0] else [], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel(lab)
    ax.grid(axis="y", visible=False)
save(fig, "figS2_sensitivity")

# ---- Fig S3: known-truth recovery
K = pd.read_csv(f"{D}/D_known_truth.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
for ax, (k, lab, sc) in zip(axs, [("E", "Energy (%)", 100), ("R_pp", "Time at rated (pp)", 1), ("D3", "Torque DEL m=3 (%)", 100)]):
    t, r = K[k + "_true"] * sc, K[k + "_rec"] * sc
    ax.scatter(t, r, color=BLUE, s=14)
    lo, hi = min(t.min(), r.min()), max(t.max(), r.max())
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel(f"Injected truth: {lab}")
    ax.set_ylabel("Recovered")
save(fig, "figS3_known_truth")
# ---- Fig 3: same-calendar-month year-on-year difference-in-differences (T13 minus fleet median)
M = pd.read_csv(f"{D}/F_monthly_event_study.csv")
M["p"] = pd.PeriodIndex(M["month"], freq="M")
rows = []
for k in ["E", "R_pp", "D3"]:
    v = M.pivot(index="p", columns="station", values=k)
    for p in v.index:
        q = p - 12
        if q in v.index:
            d = v.loc[p] - v.loc[q]
            rows.append({"p": p, "k": k, "did": d[T13] - d.drop(T13).median(),
                         "lo": (d.drop(T13) - d.drop(T13).median()).quantile(0.1), "hi": (d.drop(T13) - d.drop(T13).median()).quantile(0.9),
                         "rank": int((d >= d[T13]).sum())})
Yd = pd.DataFrame(rows)
Yd.to_csv(f"{D}/F2_yoy_same_month.csv", index=False)
fig, axs = plt.subplots(3, 1, figsize=(7.2, 5.0), sharex=True)
for ax, (k, lab, sc) in zip(axs, [("E", "Energy (%)", 100), ("R_pp", "Time at rated (pp)", 1), ("D3", "Torque DEL m=3 (%)", 100)]):
    g = Yd[Yd.k == k]
    x = g.p.dt.to_timestamp()
    ax.fill_between(x, g.lo * sc, g.hi * sc, color=GREY, alpha=0.35, lw=0, label="unmodified turbines, 10–90%")
    post = (g.p >= pd.Period("2021-10", "M")).to_numpy()
    trans = (g.p == pd.Period("2021-09", "M")).to_numpy()
    pre = ~post & ~trans
    ax.bar(x, g.did * sc, width=20, color=np.where(post, BLUE, np.where(trans, "#ffffff", GREY)),
           edgecolor=np.where(trans, ORANGE, "none"), hatch=None)
    ax.axvline(pd.Timestamp("2021-09-26"), color=ORANGE, lw=1.2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel(lab)
    mp, mq = g.did[pre].mean() * sc, g.did[post].mean() * sc
    ax.set_title(f"pre-retrofit months mean {mp:+.2f}   |   post-retrofit months mean {mq:+.2f}   (Sep 2021 = retrofit month, open bar)",
                 fontsize=6.5, color=MUTED, loc="left")
axs[0].legend(loc="lower left", fontsize=6)
axs[-1].set_xlabel("Month (difference to the same month one year earlier)")
save(fig, "fig3_event_study")
print("figures written to", F)
