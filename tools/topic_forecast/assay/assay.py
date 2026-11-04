#!/usr/bin/env python3
"""Layer-3 'gold assay' v0: score (a) candidate-direction cards and (b) radar topic pairs.

score_card: deterministic H1–H5-style gates on the 7-tuple fields a candidate has BEFORE any kill test
            (data access, hardware, lever, mech relevance, collision, crowding). Replayed on the ledger
            to check that it ranks the 3 survivors above the killed/dropped directions.
score_pair: multiplies the radar emergence probability by topic-level priors learned from the 2322
            NC/SA cards (share of pure-computational papers on topic a, share with a sister-journal
            lever on a or b), a field filter for the partner topic b, and a crowding penalty.
Outputs: assay/replay_ledger.csv, assay/assayed_top50.csv, assay/assay_summary.json
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

W_HW = {"no": 1.0, "yes": 0.0}
W_PUBLIC = {"yes": 1.0, "apply": 0.6, "unclear": 0.5, "no": 0.1}
W_COLL = {"low": 1.0, "mid": 0.6, "high": 0.15}
W_CROWD = {"low": 1.0, "mid": 0.7, "high": 0.3}
W_MECH = {"core": 1.0, "adjacent": 0.8, "tagged_only": 0.3}
LEVERS = {"meta_research", "large_scale_operational_data", "model_plus_real_event", "pure_method_theory", "new_dataset"}


def score_card(c):
    """Pre-test assay score in [0, 1.2]; also returns the gate breakdown."""
    d = c["data"]
    g = {}
    g["H1_hardware"] = W_HW.get(c.get("requires_hardware", "no"), 0.5)
    g["H1_data"] = W_PUBLIC.get(d.get("public", "unclear"), 0.5)
    g["H3_collision"] = W_COLL.get(c.get("collision", "mid"), 0.6)
    g["crowding"] = W_CROWD.get(c.get("crowding", "mid"), 0.7)
    g["mech"] = W_MECH.get(c.get("mech_relevance", "adjacent"), 0.8)
    lever = c.get("lever", "none")
    ceiling = 1.0 if lever in LEVERS else 0.3
    if d.get("real_world") == "yes" and d.get("events") == "yes" and (d.get("longitudinal_years") or 0) >= 3:
        ceiling *= 1.2  # the one pattern that has produced our survivors
    g["H4_ceiling"] = ceiling
    g["pure_comp"] = 1.0 if c.get("pure_computational") == "yes" else 0.5
    s = 1.0
    for v in g.values():
        s *= v
    return s, g


def replay(ledger_path, out_csv, failed_ideas=None):
    """failed_ideas: optional scan_daily/failed_ideas.jsonl (protocol §A); rows with full 7-tuple fields are appended."""
    rows = [json.loads(l) for l in open(ledger_path, encoding="utf-8") if l.strip()]
    if failed_ideas and os.path.exists(failed_ideas):
        for l in open(failed_ideas, encoding="utf-8"):
            r = json.loads(l) if l.strip() else None
            if r and "data" in r and "lever" in r:
                r.setdefault("id", r.get("seed_id") or r.get("idea", "")[:40]); r.setdefault("object", r.get("idea", ""))
                r.setdefault("outcome", "dropped"); r.setdefault("death_code", r.get("reason_code", ""))
                rows.append(r)
    scored = []
    for r in rows:
        s, g = score_card(r)
        scored.append((s, r, g))
    scored.sort(key=lambda x: -x[0])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "score", "outcome", "death_code", "object"] + list(scored[0][2].keys()))
        for i, (s, r, g) in enumerate(scored, 1):
            w.writerow([i, r["id"], f"{s:.3f}", r["outcome"], r["death_code"], r["object"]] + [f"{v:.2f}" for v in g.values()])
    alive = [s for s, r, g in scored if r["outcome"] == "alive"]
    dead = [s for s, r, g in scored if r["outcome"] != "alive"]
    # AUC alive vs not
    auc = sum((a > b) + 0.5 * (a == b) for a in alive for b in dead) / (len(alive) * len(dead))
    ranks = {r["id"]: i for i, (s, r, g) in enumerate(scored, 1)}
    return {"n": len(rows), "n_alive": len(alive), "auc_alive_vs_rest": round(auc, 3),
            "survivor_ranks": {k: v for k, v in ranks.items() if k.startswith("S")},
            "top5": [r["id"] for s, r, g in scored[:5]]}


# ---------- radar pair assay ----------
# OpenAlex field ids: 12 arts, 14 business, 20 economics, 29 nursing, 32 psychology, 33 social sciences,
# 34 veterinary, 35 dentistry, 36 health professions -> not engineering; 27 medicine, 28 neuroscience, 30 pharma -> crowded/medical
FIELD_W = {"12": 0.0, "14": 0.0, "20": 0.0, "29": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0,
           "27": 0.3, "28": 0.3, "30": 0.3, "13": 0.5, "24": 0.3, "11": 0.5}


def topic_priors(cards_path, abstracts_path, prior_pc=0.11, prior_lever=0.11, k=5):
    """Per-topic shares from the NC/SA cards (smoothed towards the corpus-wide shares)."""
    topics = {}
    for l in open(abstracts_path, encoding="utf-8"):
        w = json.loads(l)
        topics[w["id"]] = [t["id"] for t in w.get("topics", [])][:3]
    n, pc, lv, rw, core = {}, {}, {}, {}, {}
    for l in open(cards_path, encoding="utf-8"):
        c = json.loads(l)
        for t in topics.get(c["id"], []):
            n[t] = n.get(t, 0) + 1
            pc[t] = pc.get(t, 0) + (c["pure_computational"] == "yes")
            lv[t] = lv.get(t, 0) + (c["lever"] != "none")
            rw[t] = rw.get(t, 0) + (c["data"]["real_world"] == "yes")
            core[t] = core.get(t, 0) + (c["mech_relevance"] == "core")
    out = {}
    for t in n:
        out[t] = {"n_cards": n[t],
                  "pure_comp": (pc[t] + k * prior_pc) / (n[t] + k),
                  "lever": (lv[t] + k * prior_lever) / (n[t] + k),
                  "real_world": (rw[t] + k * 0.10) / (n[t] + k),
                  "core": (core[t] + k * 0.31) / (n[t] + k)}
    return out, {"pure_comp": prior_pc, "lever": prior_lever, "real_world": 0.10, "core": 0.31}


def score_pairs(forecast_csv, topics_all, priors, default, out_csv, top=50):
    rows = list(csv.DictReader(open(forecast_csv, encoding="utf-8")))
    scored = []
    for r in rows:
        a, b = r["a"], r["b"]
        fb = topics_all.get(b, {}).get("field", "")
        field_w = FIELD_W.get(fb, 1.0)
        pa, pb = priors.get(a, default), priors.get(b, default)
        feas = 0.5 * (pa["pure_comp"] + pb["pure_comp"])          # can we do it without a lab?
        ceiling = max(pa["lever"], pb["lever"])                    # has a sister-journal lever appeared on either topic?
        realw = max(pa["real_world"], pb["real_world"])            # does real-world data exist on either side?
        nb = float(r["n_b"])
        crowd = min(1.0, nb / 20000.0)                             # huge generic partner topics are crowded
        mech = pa["core"] if a in priors else 0.31
        p = float(r["p_emerge"])
        s = p * field_w * feas * (0.5 + ceiling) * (0.5 + realw) * (1 - 0.7 * crowd) * (0.5 + mech)
        scored.append((s, r, dict(field_w=field_w, feas=feas, ceiling=ceiling, realw=realw, crowd=crowd, mech=mech)))
    scored.sort(key=lambda x: -x[0])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "assay", "p_emerge", "a_name", "b_name", "b_subfield", "cum_ab", "n_a", "n_b",
                    "field_w", "feas", "ceiling", "realw", "crowd", "mech"])
        for i, (s, r, g) in enumerate(scored[:top], 1):
            w.writerow([i, f"{s:.4f}", f"{float(r['p_emerge']):.3f}", r["a_name"], r["b_name"], r["b_subfield"],
                        r["cum_ab"], int(float(r["n_a"])), int(float(r["n_b"]))] + [f"{v:.2f}" for v in g.values()])
    return [(f"{s:.4f}", r["a_name"], r["b_name"]) for s, r, g in scored[:20]]


def main():
    ext = len(sys.argv) > 1 and sys.argv[1] in ("ext", "v1")   # ext -> results/ext/ forecast; v1 -> results/v1/ns (NS-label radar, non-AI partners, mech-core sources only)
    v1 = len(sys.argv) > 1 and sys.argv[1] == "v1"
    suf = "_v1" if v1 else ("_ext" if ext else "")
    rep = replay(os.path.join(HERE, "ledger.jsonl"), os.path.join(HERE, f"replay_ledger{suf}.csv"),
                 failed_ideas="/mnt/project-files/scan_daily/failed_ideas.jsonl")
    print("replay:", json.dumps(rep, ensure_ascii=False))
    topics_all = json.load(open(os.path.join(ROOT, "data/openalex/topics_all.json"), encoding="utf-8"))
    priors, default = topic_priors(os.path.join(ROOT, "cards/ncsa_cards.jsonl"),
                                   os.path.join(ROOT, "data/k3_ncsa_abstracts.jsonl"))
    fc = os.path.join(ROOT, "results/v1/ns/forecast_T2025_top2000_nonAI.csv" if v1 else
                      ("results/ext/forecast_T2025_top2000.csv" if ext else "results/forecast_T2025_top2000.csv"))
    if v1:   # keep only pairs whose source topic is in the mechanical core set
        mech_ids = set(json.load(open(os.path.join(ROOT, "data/openalex/mech_topics.json"), encoding="utf-8")))
        rows = [r for r in csv.DictReader(open(fc, encoding="utf-8")) if r["a"] in mech_ids]
        fc = os.path.join(HERE, "_v1_mech_input.csv")
        with open(fc, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("v1 mech-core pairs:", len(rows))
    top = score_pairs(fc, topics_all, priors, default,
                      os.path.join(HERE, f"assayed_top50{suf}.csv"))
    for t in top:
        print(*t, sep=" | ")
    # Evaluation-type declaration (protocol addition 2026-11-03 §B):
    #   replay      = relative_only  (ledger fields and weights were written from the same lessons; not independent)
    #   radar K1/K3 = real_gt        (OpenAlex counts; labels are what actually emerged by T+3 / in NC-SA)
    #   assayed pairs = no ground truth yet (prospective); only a kill test on data can grade them
    json.dump({"evaluation_type": {"replay": "relative_only", "radar_K1_K3": "real_gt", "assayed_pairs": "prospective_no_gt"},
               "replay": rep, "n_topics_with_priors": len(priors), "assayed_top20": top},
              open(os.path.join(HERE, f"assay_summary{suf}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
