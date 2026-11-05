"""Synthetic round-trip test: build a 3-unit fleet, reload it, check layout, daily counts and the h2kit export."""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

from .core import Writer, open_fleet


def main(out=None):
    out = out or tempfile.mkdtemp(prefix="fleetdata_selftest_")
    rng = np.random.default_rng(1)
    w = Writer(out, dataset="synthetic", licence="none", interval="10min", overwrite=True)
    w.add_channel("power_kw", "P", core=True)
    w.add_channel("wind_ms", "WS", core=True)
    units = []
    ev = []
    for i in range(3):
        u = f"U{i}"
        t = pd.date_range("2024-01-01 00:10", "2024-03-01", freq="10min", tz="UTC")
        t = t[rng.random(len(t)) > 0.05]  # 5% missing
        ws = rng.gamma(2, 3, len(t))
        w.add_ts(u, pd.DataFrame({"time": t, "power_kw": np.clip(ws ** 3, 0, 2000), "wind_ms": ws}))
        w.add_ts(u, pd.DataFrame({"time": t[:10], "power_kw": 0.0, "wind_ms": 0.0}))  # duplicate chunk: must be dropped
        units.append(dict(unit_id=u, dataset="synthetic", site="S", rated_kw=2000, meta={"i": i}))
        ev.append(dict(unit_id=u, start="2024-02-10 05:00", end="2024-02-10 09:00", kind="fault", code="F1",
                       text="gearbox", stopping=True, source="synthetic"))
        ev.append(dict(unit_id=u, start="2024-01-20 10:00", kind="maintenance", text="service", source="synthetic"))

    def exposure(u, ts):
        day = ts["time"].dt.floor("D")
        return pd.DataFrame({"energy_kwh": ts["power_kw"].groupby(day).sum() / 6.0})

    f = w.finish(units, pd.DataFrame(ev), daily_exposure=exposure)
    probs = f.validate()
    d = f.daily()
    checks = {
        "no validation problems": not probs,
        "3 units": len(f.unit_ids) == 3,
        "duplicates dropped": f.ts("U0")["time"].is_unique,
        "6 events": len(f.events()) == 6,
        "3 faults": len(f.events(kind="fault")) == 3,
        "fault counted on day": int(d.loc[d["date"] == pd.Timestamp("2024-02-10", tz="UTC"), "n_fault"].sum()) == 3,
        "coverage ~0.95": 0.9 < d["coverage"].median() < 1.0,
        "energy column": "energy_kwh" in d,
        "panel": len(f.panel(["power_kw"], freq="W")) > 0,
    }
    try:
        cfg = f.to_h2kit(os.path.join(out, "h2kit"))
        evcsv = pd.read_csv(os.path.join(out, "h2kit", "events.csv"))
        av = json.load(open(os.path.join(out, "h2kit", "avoid.json")))
        checks["h2kit export"] = os.path.exists(cfg) and len(evcsv) == 3 and av.get("U0") == ["2024-01-20"]
    except ImportError:
        checks["h2kit export (needs pyyaml)"] = False
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if probs:
        print("problems:", probs)
    ok = all(checks.values())
    print("SELFTEST", "PASS" if ok else "FAIL", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
