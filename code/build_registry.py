"""门 A 第 2 项：逐训练实例注册表。每行 = 一次训练得到的一个权重。
epochs 从日志逐 run 读取（该 run 最后一个 'ep N val' 行），不从计数推断。"""
import json, csv, hashlib, os, re, collections, pathlib
R = pathlib.Path("results"); L = pathlib.Path("logs")
def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
CODE_NOW = sha("code/prior_screen.py")
# 结果文件 → (日志, 数据集, 文件子集/协议, 角色, 权重目录)
SRC = {
 "cwru_prior_screen.json": ("cwru_run.log",        "CWRU", "40file", "analysis", None),
 "cwru_ep100_probe.json":  ("cwru_ep100_probe.log","CWRU", "40file", "analysis", None),
 "cwru_16file_probe.json": ("cwru_16file_probe.log","CWRU","16file", "analysis", None),
 "cwru40_ckpt_probe.json": ("cwru_common_testset.log","CWRU","40file","analysis","ckpt_cwru40"),
 "cwru16_ckpt_probe.json": ("cwru_common_testset.log","CWRU","16file","analysis","ckpt_cwru16"),
 "jnu_prior_screen.json":  ("jnu_run.log",  "JNU", "modeA_rpm600", "analysis", None),
 "jnu_fine_snr.json":      ("jnu_fine.log", "JNU", "modeA_rpm600|arm=orig", "analysis", "ckpt_jnu_fine"),
 "jnu_arm_balanced.json":  ("jnu_arms.log", "JNU", "modeA_rpm600|arm=balanced", "analysis", None),
 "jnu_arm_subsample.json": ("jnu_arms.log", "JNU", "modeA_rpm600|arm=subsample", "analysis", None),
 "pu_fine_snr.json":       ("pu_fine.log",  "PU",  "N15_M07_F10|time60/20/20|K001-3,KA04/15/16,KI04/14/17", "analysis", None),
 "jnu_smoke.json": (None,"JNU","modeA_rpm600","smoke",None), "jnu_fine_smoke.json": (None,"JNU","modeA_rpm600","smoke",None),
 "jnu_arm_smoke_balanced.json": (None,"JNU","arm=balanced","smoke",None), "jnu_arm_smoke_subsample.json": (None,"JNU","arm=subsample","smoke",None),
 "pu_smoke.json": (None,"PU","-","smoke",None), "pu_probe.json": (None,"PU","individual-split (失败协议)","smoke",None),
 "pu_probe2.json": (None,"PU","time-split","smoke",None),
}
def log_runs(logname):
    """返回 [(model, seed, last_epoch)]，按日志顺序；共享日志按出现顺序切分。"""
    out, cur, last = [], None, None
    for line in open(L / logname):
        m = re.match(r"--- (\S+) seed(\d+) ---", line)
        if m:
            if cur: out.append((*cur, last))
            cur, last = (m.group(1), int(m.group(2))), None
        e = re.search(r"ep ?(\d+) val", line)
        if e: last = int(e.group(1))
    if cur: out.append((*cur, last))
    return out
log_cache = {}
rows = []
for fn, (lg, ds, subset, role, ck) in SRC.items():
    recs = json.load(open(R / fn))
    runs = None
    if lg:
        runs = log_cache.setdefault(lg, log_runs(lg))
    for i, r in enumerate(recs):
        ep = None
        if runs:
            if lg == "cwru_common_testset.log":   # 两个结果文件共用一份日志：40file 在前、16file 在后
                off = 0 if fn.startswith("cwru40") else 6
                cand = runs[off:off+6]
            elif lg == "jnu_arms.log":
                off = 0 if "balanced" in fn else 25
                cand = runs[off:off+25]
            else:
                cand = runs
            hit = [e for (m, s, e) in cand if m == r["model"] and s == r["seed"]]
            ep = hit[0] if len(hit) == 1 else (hit if hit else None)
        w = None; wh = "missing"
        if ck:
            p = R / ck / f"{r['model']}_s{r['seed']}.pt"
            if p.exists(): w, wh = str(p), sha(p)
        snr = r.get("snr_grid") or sorted(map(float, r["observed"].keys()), reverse=True)
        rows.append(dict(
            instance_id=f"{fn[:-5]}::{r['model']}::s{r['seed']}", source_json=fn, role=role,
            dataset=ds, subset_protocol=subset, split_seed=42 if ds == "CWRU" else "",
            model=r["model"], seed=r["seed"], seed_type="torch+numpy global (train_one); eval noise RNG 未单独设种",
            epochs=ep if ep is not None else "unknown", n_train=r.get("n_train", ""),
            best_val_acc=r.get("best_val_acc", ""), noise_impl="add_awgn_tensor (per-window power)",
            snr_grid="|".join(f"{s:g}" for s in snr), collapse_rule="conc>=0.90 per SNR",
            weights_path=w or "", weights_sha256=wh,
            code_hash_at_run="not captured", code_hash_prior_screen_now=CODE_NOW[:16]))
out = R / "registry/training_instances.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
an = [r for r in rows if r["role"] == "analysis"]
print("rows", len(rows), "| analysis", len(an), "| smoke", len(rows) - len(an))
print("by source:", {k: v for k, v in collections.Counter(r["source_json"] for r in an).items()})
print("epochs:", collections.Counter((r["source_json"], str(r["epochs"])) for r in an))
print("weights:", collections.Counter(r["weights_sha256"] != "missing" for r in an))
print("n_train empty:", sum(1 for r in an if r["n_train"] == ""))
