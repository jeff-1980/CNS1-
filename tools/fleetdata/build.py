#!/usr/bin/env python3
"""Build unified fleet datasets from the raw public files on jeff's PC (CPU only, raw files are read-only).

  python build.py kelmarsh    [--out /mnt/d/audit_data/_unified] [--channels core|means|all] [--overwrite]
  python build.py penmanshiel
  python build.py hill_of_towie
  python build.py care_a | care_b | care_c
  python build.py ult_revco
  python build.py --selftest
  python build.py --check <built dir>       # validate + print a summary

Each build writes <out>/<name>/ and a build log <out>/<name>.log.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fleetdata import open_fleet  # noqa: E402

DATASETS = ["kelmarsh", "penmanshiel", "hill_of_towie", "care_a", "care_b", "care_c", "ult_revco"]


def summary(f):
    d = f.daily()
    lines = [repr(f), f"time convention: {f.meta.get('time_convention')}",
             f"channels: {len(f.channels)} ({int(f.channels['core'].sum()) if len(f.channels) else 0} core)",
             f"events by kind: {f.meta.get('events_by_kind')}"]
    if len(d):
        lines.append(f"days: {len(d)}, from {d['date'].min()} to {d['date'].max()}")
        if "coverage" in d:
            lines.append(f"median daily coverage {d['coverage'].median():.3f}")
    for n in f.meta.get("notes", []):
        lines.append(f"note: {n}")
    probs = f.validate()
    lines.append("VALIDATE " + ("OK" if not probs else "PROBLEMS: " + "; ".join(probs)))
    return "\n".join(lines), not probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", choices=DATASETS)
    ap.add_argument("--out", default="/mnt/d/audit_data/_unified")
    ap.add_argument("--channels", default="means", choices=["core", "means", "all"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--units", default=None, help="comma list of turbine numbers (wind) for a quick partial build")
    ap.add_argument("--src", default=None, help="override raw data location")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", default=None)
    ap.add_argument("--headers", action="store_true", help="wind Greenbyte: list raw columns only, then exit")
    a = ap.parse_args()
    if a.selftest:
        from fleetdata.selftest import main as st
        sys.exit(st())
    if a.check:
        s, ok = summary(open_fleet(a.check))
        print(s)
        sys.exit(0 if ok else 1)
    if not a.dataset:
        ap.error("dataset required")
    if a.headers:
        from fleetdata.adapters import greenbyte
        os.makedirs(a.out, exist_ok=True)
        p = os.path.join(a.out, f"{a.dataset}_headers.txt")
        print(greenbyte.list_headers(a.dataset, a.src, p)[:3000])
        print("written", p)
        sys.exit(0)
    out = os.path.join(a.out, a.dataset + ("" if not a.units else "_partial"))
    os.makedirs(a.out, exist_ok=True)
    logf = open(out + ".log", "a", encoding="utf-8")

    def log(*x):
        msg = time.strftime("%H:%M:%S ") + " ".join(str(i) for i in x)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    units = [int(u) for u in a.units.split(",")] if a.units else None
    t0 = time.time()
    log(f"build {a.dataset} -> {out} channels={a.channels}")
    if a.dataset in ("kelmarsh", "penmanshiel"):
        from fleetdata.adapters import greenbyte
        f = greenbyte.build(a.dataset, out, zips_dir=a.src, channels=a.channels, overwrite=a.overwrite, units=units,
                            log=log)
    elif a.dataset == "hill_of_towie":
        from fleetdata.adapters import hill_of_towie
        f = hill_of_towie.build(out, root=a.src, channels=a.channels, overwrite=a.overwrite, units=units, log=log)
    elif a.dataset.startswith("care_"):
        from fleetdata.adapters import care
        f = care.build(a.dataset[-1].upper(), out, root=a.src, overwrite=a.overwrite, log=log)
    elif a.dataset == "ult_revco":
        from fleetdata.adapters import ult
        f = ult.build(out, root=a.src, overwrite=a.overwrite, log=log)
    s, ok = summary(f)
    log(s)
    log(f"done in {(time.time() - t0) / 60:.1f} min")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
