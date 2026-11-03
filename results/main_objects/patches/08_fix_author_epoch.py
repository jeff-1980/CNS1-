"""[audit] Object 08: correct author_epoch from the saved checkpoint.
The recorder callback ran before ModelCheckpoint (PL 1.7 moves checkpoint callbacks to the end), so the
detected best epoch lagged one validation and a best at the final validation was missed.
Truth = ckpt['epoch'] + 1 (one validation per epoch). Original value kept as author_epoch_recorder_lagged."""
import glob, json, math, os, torch
n = bad = 0
rows = []
for f in sorted(glob.glob(os.path.expanduser("~/audit_ext/main_records/08/*/seed*.json"))):
    r = json.load(open(f))
    g = r["group"].split("|")[-1]; s = r["seed"]
    ck = torch.load(os.path.expanduser(f"~/audit_ext/work08/main/{g}_s{s}/best_model/model.ckpt"), map_location="cpu")
    e = int(ck["epoch"]) + 1
    vl = {int(k): v["val"]["loss"] for k, v in r["epochs"].items()}
    argmin = min(vl, key=lambda k: (vl[k], k))
    r.setdefault("author_epoch_recorder_lagged", r["author_epoch"])
    r["author_epoch"] = e
    r["author_epoch_source"] = "best_model/model.ckpt['epoch'] + 1"
    rec = r["epochs"][str(e)]["test"]["acc"]; a = r["author_reported_acc"]
    r["author_check"] = dict(recorded=rec, reported=a, abs_diff=abs(rec - a), matches=math.isclose(rec, a, abs_tol=1e-6))
    json.dump(r, open(f, "w"), indent=1)
    n += 1; bad += not r["author_check"]["matches"]
    rows.append((g, s, e, argmin, r["author_epoch_recorder_lagged"], r["author_check"]["matches"]))
    assert e == argmin, (g, s, e, argmin)
print(f"{n} records corrected; author_check mismatches after correction: {bad}; ckpt epoch == argmin val_loss in all")
for x in rows: print(*x)
