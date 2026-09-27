"""Check the raw data against the recorded SHA-256 hashes and build the CWRU 16-file subset.

Usage: python code/prepare_data.py            (paths from the environment variables in README)
Hashes: results/registry/raw_hashes.csv (dataset, file name, size, SHA-256).
"""
import csv, hashlib, os, pathlib, sys
ROOT = pathlib.Path(os.environ.get("CNS_ROOT", pathlib.Path(__file__).resolve().parents[1]))
DIRS = {"CWRU40": os.environ.get("CWRU40_DIR", ROOT / "data/cwru_12k_de"), "PU": os.environ.get("PU_DIR", ROOT / "data/paderborn"),
        "JNU": os.environ.get("JNU_DIR", ROOT / "data/jnu")}
CWRU16 = pathlib.Path(os.environ.get("CWRU16_DIR", ROOT / "data/cwru_16"))

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

rows = [r for r in csv.DictReader(open(ROOT / "results/registry/raw_hashes.csv", newline="")) if r["kind"] in ("data", "subset_link")]
bad = 0
for ds in ("CWRU40", "JNU", "PU"):
    want = [r for r in rows if r["kind"] == "data" and r["dataset"] == ds]
    base = pathlib.Path(DIRS[ds]); index = {p.name: p for p in base.rglob("*") if p.is_file()} if base.exists() else {}
    miss = [r for r in want if pathlib.Path(r["path"]).name not in index]
    wrong = [r for r in want if pathlib.Path(r["path"]).name in index and sha(index[pathlib.Path(r["path"]).name]) != r["sha256"]]
    bad += len(miss) + len(wrong)
    print(f"{ds:7s} expected {len(want):5d}  missing {len(miss):5d}  hash mismatch {len(wrong):3d}  ({base})")
CWRU16.mkdir(parents=True, exist_ok=True)
for r in rows:
    if r["kind"] == "subset_link" and r["dataset"] == "CWRU16":
        n = pathlib.Path(r["path"]).name; src = pathlib.Path(DIRS["CWRU40"]) / n; dst = CWRU16 / n
        if src.exists() and not dst.exists(): dst.symlink_to(src.resolve())
print("CWRU16 subset:", len(list(CWRU16.glob("*.mat"))), "files in", CWRU16)
sys.exit(1 if bad else 0)
