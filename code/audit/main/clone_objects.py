"""正式阶段 §2/§3：把 attempt_order.csv 里的合格对象按检索时记录的 commit 取到本地，并给出每个仓库的代码概况，
供逐个写协议钩子（逐 epoch 记录报告划分准确率）用。只读取代码，不运行任何对象代码。

每个仓库：git init + fetch --depth 1 <commit>（取不到固定 commit 时记为失败，不退回默认分支）。
概况：大小、训练入口候选、epoch 循环与保存 / 选模相关行（torch.save / save_weights / ModelCheckpoint / best / val_acc 等），
写 objects_inventory.json，可提交回仓库。克隆前检查剩余空间（WSL 下看 /mnt/c 的真实剩余），低于 --min-free-gb 就停。
在 code/ 下运行：python -m audit.main.clone_objects --order ../results/main_frame/2026-10-19/attempt_order.csv --dest ~/audit_ext/main_objects --out ../results/main_frame/2026-10-19"""
import argparse, csv, datetime as dt, json, os, re, shutil, subprocess

PATTERNS = {
    "epoch_loop": re.compile(r"for\s+\w*epoch\w*\s+in|\.fit\(|epochs?\s*=", re.I),
    "save": re.compile(r"torch\.save|save_weights|\.save\(|ModelCheckpoint|save_model|state_dict\(\)", re.I),
    "select": re.compile(r"best_?(acc|loss|model|val|epoch)|val_?acc|max_acc|min_loss|early_?stop", re.I),
    "seed": re.compile(r"manual_seed|random\.seed|np\.random\.seed|set_seed|tf\.random\.set_seed", re.I),
}


def free_gb(path):
    # WSL 里 df 报的是 vhdx 上限，不是真实剩余；有 /mnt/c 时以 Windows C 盘真实剩余为准
    return shutil.disk_usage("/mnt/c" if os.path.isdir("/mnt/c") else path).free / 2**30


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def scan(root):
    hits = {k: [] for k in PATTERNS}; train = []; size = 0
    for d, _, fs in os.walk(root):
        if "/.git" in d: continue
        for f in fs:
            p = os.path.join(d, f); size += os.path.getsize(p); rel = os.path.relpath(p, root)
            if not f.endswith((".py", ".ipynb")): continue
            if re.search(r"train|main|run", f, re.I): train.append(rel)
            try: lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
            except OSError: continue
            for i, ln in enumerate(lines):
                for k, rx in PATTERNS.items():
                    if rx.search(ln) and len(hits[k]) < 40: hits[k].append(f"{rel}:{i + 1}: {ln.strip()[:160]}")
    return dict(size_mb=round(size / 2**20, 1), entry_candidates=sorted(train)[:30], **hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", required=True); ap.add_argument("--dest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--only", default="", help="逗号分隔的 attempt_order，只重取这些，结果并入已有 objects_inventory.json")
    a = ap.parse_args()
    dest = os.path.expanduser(a.dest); os.makedirs(dest, exist_ok=True)
    rows = list(csv.DictReader(open(a.order)))
    only = {int(x) for x in a.only.split(",") if x}
    inv_path = os.path.join(a.out, "objects_inventory.json")
    keep = [o for o in json.load(open(inv_path))["objects"] if int(o["attempt_order"]) not in only] if only else []
    if only: rows = [r for r in rows if int(r["attempt_order"]) in only]
    inv = []
    for r in rows:
        free = free_gb(dest)
        if free < a.min_free_gb: inv.append(dict(r, status=f"stopped_low_disk:{free:.1f}GB")); break
        d = os.path.join(dest, f"{int(r['attempt_order']):02d}_" + r["full_name"].replace("/", "__"))
        if not os.path.isdir(os.path.join(d, ".git")):
            os.makedirs(d, exist_ok=True); run(["git", "init", "-q"], d)
            run(["git", "remote", "add", "origin", f"https://github.com/{r['full_name']}.git"], d)
        p = run(["git", "fetch", "-q", "--depth", "1", "origin", r["commit"]], d)
        if p.returncode != 0: inv.append(dict(r, status="fetch_failed", error=p.stderr.strip()[:300])); continue
        run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], d)
        head = run(["git", "rev-parse", "HEAD"], d).stdout.strip()
        inv.append(dict(r, status="ok" if head == r["commit"] else "commit_mismatch", path=d, head=head, **scan(d)))
        print(f"{r['attempt_order']:>2} {r['full_name']}: {inv[-1]['status']} {inv[-1].get('size_mb')} MB", flush=True)
    inv = sorted(keep + inv, key=lambda o: int(o["attempt_order"]))
    meta = dict(built_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), n=len(inv),
                status={s: sum(i["status"] == s for i in inv) for s in sorted({i["status"] for i in inv})},
                free_gb_after=round(free_gb(dest), 1), registration="https://osf.io/2tqv6")
    json.dump(dict(meta=meta, objects=inv), open(inv_path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
