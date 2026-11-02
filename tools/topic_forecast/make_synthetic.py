#!/usr/bin/env python3
"""Write a small synthetic dataset in the fetch_openalex.py layout, for testing model.py offline.

Pairs connect with probability rising in latent similarity and node popularity, so a working
pipeline should beat chance clearly. Usage: python make_synthetic.py OUTDIR
"""
import json
import os
import sys

import numpy as np

out = sys.argv[1]
rng = np.random.default_rng(0)
N, NM = 400, 60
years = list(range(2008, 2026))
ids = [f"T{10000 + i}" for i in range(N)]
topics = {t: {"name": f"topic {t}", "subfield": ("2210" if i < NM else str(1700 + i % 30)),
              "subfield_name": "Mechanical Engineering" if i < NM else "Other",
              "field": "22" if i < NM else str(10 + i % 15), "domain": "3"} for i, t in enumerate(ids)}
os.makedirs(out, exist_ok=True)
json.dump(topics, open(os.path.join(out, "topics_all.json"), "w"))
json.dump(ids[:NM], open(os.path.join(out, "mech_topics.json"), "w"))
emb = rng.normal(size=(N, 4))
base = rng.lognormal(4, 1, N)
trend = rng.normal(0.05, 0.15, N)
for k, y in enumerate(years):
    size = base * np.exp(trend * k)
    os.makedirs(os.path.join(out, "totals"), exist_ok=True)
    os.makedirs(os.path.join(out, "ns"), exist_ok=True)
    json.dump({t: int(rng.poisson(size[i])) for i, t in enumerate(ids)}, open(os.path.join(out, "totals", f"{y}.json"), "w"))
    json.dump({t: int(rng.poisson(size[i] * 0.01)) for i, t in enumerate(ids)}, open(os.path.join(out, "ns", f"{y}.json"), "w"))
    d = os.path.join(out, "cooc", str(y)); os.makedirs(d, exist_ok=True)
    drift = emb + 0.05 * k * np.sin(np.arange(N))[:, None]
    for i in range(NM):
        sim = drift[i] @ drift.T / 4
        lam = np.exp(sim * 1.5 - 4) * np.sqrt(size[i] * size) / 30
        c = rng.poisson(lam)
        json.dump({ids[j]: int(c[j]) for j in range(N) if c[j] > 0}, open(os.path.join(d, f"{ids[i]}.json"), "w"))
with open(os.path.join(out, "k3_ncsa_works.jsonl"), "w") as f:
    for w in range(300):
        ts = list(rng.choice(ids[:NM], 1)) + list(rng.choice(ids, 2, replace=False))
        f.write(json.dumps({"id": f"W{w}", "doi": None, "year": 2024, "source": "S1", "topics": ts}) + "\n")
print("synthetic data written to", out)
