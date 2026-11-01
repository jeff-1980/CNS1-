"""Fig. 6: T13's real drift-corrected effects against the pseudo-retrofit null (105 turbine-date pairs).
Usage: python make_fig_null.py <R2 csv> <FIGDIR>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = pd.read_csv(sys.argv[1]); F = sys.argv[2]
REAL = {"E": 0.0481, "R_pp": 1.6656, "D3": 0.0230}
BLUE, GREY, ORANGE, MUTED = "#2a78d6", "#9a9993", "#eb6834", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "savefig.dpi": 300})
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
low = D.turbine.isin(["T15", "T16", "T17"])
for ax, (k, lab, sc), l in zip(axs, [("E", "Energy (%)", 100), ("R_pp", "Time at rated (pp)", 1), ("D3", "Torque exposure m=3 (%)", 100)], "abc"):
    x = D[k] * sc
    bins = np.linspace(min(x.min(), 0) - 0.1 * abs(x.min()), max(x.max(), REAL[k] * sc) * 1.05, 30)
    ax.hist([x[~low & (D.turbine != "T13")], x[low], x[D.turbine == "T13"]], bins=bins, stacked=True,
            color=[GREY, "#d4d3cf", BLUE], label=["other turbines", "T15–T17", "T13 at pseudo dates"])
    ax.axvline(REAL[k] * sc, color=BLUE, lw=1.8)
    ax.axvline(x.quantile(0.95), color=MUTED, lw=0.8, ls="--")
    ax.text(REAL[k] * sc, ax.get_ylim()[1] * 0.95, " T13 real", color=BLUE, fontsize=7, va="top")
    ax.set_xlabel(lab); ax.set_ylabel("turbine-date pairs" if l == "a" else "")
    ax.text(-0.2, 1.03, l, transform=ax.transAxes, fontsize=10, fontweight="bold")
axs[0].legend(fontsize=6, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(f"{F}/fig6_null.png"); fig.savefig(f"{F}/fig6_null.pdf")
