"""Graphical abstract (Elsevier: landscape, >= 1328 x 531 px). Usage: python make_graphical_abstract.py <RUN_OUT> <FIGDIR> [R1_joint_bootstrap_summary.csv]"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R, F = sys.argv[1], sys.argv[2]
A = pd.read_csv(f"{R}/A_T13_corrected_summary.csv").set_index("quantity")
B = pd.read_csv(f"{R}/D_pure_scaling_benchmark.csv")
B = B[(B.data == "T13_pre") & (B.k == 1.05)].iloc[0]
BLUE, GREY, ORANGE, INK, MUTED = "#2a78d6", "#9a9993", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED})
fig = plt.figure(figsize=(13.28, 5.31), dpi=150)
ax0 = fig.add_axes([0, 0, 1, 1]); ax0.axis("off"); ax0.set_xlim(0, 100); ax0.set_ylim(0, 40)


def box(x, y, w, h, title, body, c):
    ax0.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=c, lw=2))
    ax0.text(x + w / 2, y + h - 2.5, title, ha="center", va="top", fontsize=13, fontweight="bold", color=INK)
    ax0.text(x + w / 2, y + h - 8, body, ha="center", va="top", fontsize=11, color=INK, linespacing=1.5)


box(2, 6, 27, 29, "Natural experiment", "Hill of Towie, 21 turbines\n1 retrofitted with vortex\ngenerators (Sep 2021),\n20 unmodified for 9 months\n\nopen 10-min SCADA\nreference-cell estimator\nplacebo turbines and\npseudo-retrofit dates", GREY)
box(35, 6, 25, 29, "Retrofit effect", "", BLUE)
for i, (lab, q, unit, sc) in enumerate([("Energy", "E", "%", 100), ("Time at rated", "R_pp", " pp", 1), ("Torque exposure (m=3)", "D3", "%", 100)]):
    y = 26 - 6.5 * i
    ax0.text(47.5, y, lab, ha="center", fontsize=11, color=MUTED)
    ax0.text(47.5, y - 3.2, f"+{A.loc[q,'estimate']*sc:.1f}{unit}", ha="center", fontsize=17, fontweight="bold", color=BLUE)
ax0.text(47.5, 7.5, "rank 1 of 21 turbines", ha="center", fontsize=10, color=MUTED)
for x0, x1 in [(29.8, 34.2), (60.8, 65.2)]:
    ax0.add_patch(FancyArrowPatch((x0, 20.5), (x1, 20.5), arrowstyle="-|>", mutation_scale=22, lw=2, color=MUTED))
ax = fig.add_axes([0.715, 0.2, 0.26, 0.62])
ms = [3, 10 / 3, 4, 6, 10]
keys = ["D3", "D10_3", "D4", "D6", "D10"]
C = [A.loc[k, "estimate"] / A.loc["E", "estimate"] for k in keys]
import os
J = pd.read_csv(sys.argv[3], index_col=0) if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else None
src = J if J is not None else A
lo = [src.loc[f"C_{k}", "lo"] if f"C_{k}" in src.index else np.nan for k in keys]
hi = [src.loc[f"C_{k}", "hi"] if f"C_{k}" in src.index else np.nan for k in keys]
bench = [B[f"C_{k}"] for k in keys]
ax.plot(ms, bench, "-", color=GREY, lw=2, label="pure power scaling")
ax.fill_between(ms, lo, hi, color=BLUE, alpha=0.2, lw=0)
ax.plot(ms, C, "-o", color=BLUE, lw=2, label="measured")
ax.set_xlabel("S-N exponent m"); ax.set_ylabel("Exposure ratio C = D_m / E")
ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=10, loc="upper right")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.set_title("Exposure ratio 0.48 at m = 3, below\npure power scaling (part-load gain, speed)", fontsize=11.5, color=INK)
fig.savefig(f"{F}/graphical_abstract.png", dpi=150)
fig.savefig(f"{F}/graphical_abstract.pdf")
