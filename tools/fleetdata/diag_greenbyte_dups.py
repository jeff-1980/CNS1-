#!/usr/bin/env python3
"""Why do some Greenbyte yearly exports have ~40x the expected rows? Read one Turbine_Data member and report rows per
timestamp, the spacing of distinct timestamps, and for core channels how often repeated timestamps carry conflicting
values. Read-only.  Usage: diag_greenbyte_dups.py <zip> [member-substring] > report.txt"""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fleetdata.common import WIND_CORE_GREENBYTE, canon_raw, greenbyte_header  # noqa: E402

z = sys.argv[1]
sub = sys.argv[2] if len(sys.argv) > 2 else "Turbine_Data"
with zipfile.ZipFile(z) as zf:
    n = sorted(m for m in zf.namelist() if sub in m and os.path.basename(m).startswith("Turbine_Data"))[0]
    with zf.open(n) as f:
        lines = [f.readline().decode("utf-8", "replace") for _ in range(40)]
    i, cols = greenbyte_header(lines)
    import re
    core = [c for c in cols if any(re.search(rx, canon_raw(c), re.I) for _, rx in WIND_CORE_GREENBYTE[:6])]
    with zf.open(n) as f:
        df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"), skiprows=i + 1, header=None,
                         names=cols, usecols=["Date and time"] + core, na_values=["NaN"], low_memory=False)
print("member", n, "columns", len(cols), "rows", len(df))
t = pd.to_datetime(df["Date and time"], errors="coerce")
print("unparseable times", int(t.isna().sum()))
vc = t.value_counts()
print("distinct timestamps", len(vc), "rows per timestamp:", vc.describe().round(2).to_dict())
print("rows-per-timestamp histogram (top 10):", vc.value_counts().head(10).to_dict())
u = pd.Series(sorted(vc.index))
print("spacing of distinct timestamps:", u.diff().value_counts().head(5).to_dict())
print("share of timestamps on the 10-min grid:", float((u.dt.minute % 10 == 0).mean() * (u.dt.second == 0).mean()))
print("first rows:\n", df.head(12).to_string()[:3000])
dup = df[t.duplicated(keep=False)]
print("rows in repeated timestamps", len(dup), " exact duplicate rows", int(df.duplicated().sum()))
g = dup.assign(_t=t[dup.index]).groupby("_t")
for c in core:
    nun = g[c].nunique(dropna=True)
    print(f"{c}: timestamps with >1 distinct non-missing value {int((nun > 1).sum())} of {len(nun)}; "
          f"share of rows non-missing {df[c].notna().mean():.3f}")
    bad = nun[nun > 1].index[:2]
    for b in bad:
        print("   example", b, list(dup.loc[t[dup.index] == b, c].dropna().unique()[:6]))
