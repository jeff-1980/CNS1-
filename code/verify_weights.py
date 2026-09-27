"""Verify the trained weights against the SHA-256 values in results/registry/training_instances.csv.

Usage: python code/verify_weights.py   (expects the weight archive unpacked into results/, e.g. results/ckpt_2x2/)
Rows whose weights were not kept (earlier screening runs, weights_sha256 = "missing") are counted separately.
"""
import csv, hashlib, os, pathlib, sys
ROOT = pathlib.Path(os.environ.get("CNS_ROOT", pathlib.Path(__file__).resolve().parents[1]))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
ok = bad = absent = notkept = 0; bad_rows = []
for r in csv.DictReader(open(ROOT / "results/registry/training_instances.csv", newline="")):
    paths, hashes = r["weights_path"], r["weights_sha256"]
    if not paths or hashes in ("", "missing"): notkept += 1; continue
    pairs = [(p.split("=", 1)[-1].strip(), h.split("=", 1)[-1].strip()) for p, h in zip(paths.split("|"), hashes.split("|"))]
    for p, h in pairs:
        f = (ROOT / p.replace("code/../", "")).resolve()
        if not f.exists(): absent += 1; continue
        if sha(f) == h: ok += 1
        else: bad += 1; bad_rows.append(r["instance_id"])
print(f"weights verified {ok}, mismatched {bad}, not found {absent}; registry rows without kept weights {notkept}")
if bad_rows: print("mismatch:", bad_rows[:10])
sys.exit(1 if bad else 0)
