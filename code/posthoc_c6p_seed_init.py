"""【探索性 / 事后，不改变任何预注册判定】C6-P 诊断：吸引子对噪声抽样是否稳定；同种子三臂（初始化相同）是否落到同一类（与打乱种子的置换零分布比较）。"""
import os, json, collections, itertools, numpy as np
R = os.path.join(os.environ.get("CNS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
rng = np.random.default_rng(0)
for d in ("d1", "d2"):
    rows = json.load(open(f"{R}/c6p_{d}.json")); names = rows[0]["classes"]
    agree5 = np.mean([len(set(r["observed"]["-8.0"]["tops"])) == 1 for r in rows])
    print(f"== {d} 5 次噪声抽样主导类别完全一致的实例：{agree5:.2f}")
    for m in sorted({r["model"] for r in rows}, key=lambda x: [r["model"] for r in rows].index(x)):
        top = {(r["arm"], r["seed"]): r["observed"]["-8.0"]["top_mode"] for r in rows if r["model"] == m}
        # within-seed agreement across the 3 arms (same init) vs. shuffled seeds
        def stat(t): return np.mean([len({t[(a, s)] for a in ("S_a", "S_b", "S_ALL")}) == 1 for s in range(5)])
        obs = stat(top); null = []
        for _ in range(5000):
            t = {}
            for a in ("S_a", "S_b", "S_ALL"):
                p = rng.permutation(5)
                for s in range(5): t[(a, s)] = top[(a, p[s])]
            null.append(stat(t))
        p = np.mean(np.array(null) >= obs - 1e-9)
        per_seed = [" ".join(names[top[(a, s)]][:4] for a in ("S_a", "S_b", "S_ALL")) for s in range(5)]
        print(f"  {m:9s} 同种子三臂完全相同 {obs:.1f}（打乱种子期望 {np.mean(null):.2f}，p={p:.3f}）  逐种子[S_a S_b S_ALL]: {per_seed}")
    # per-class clean recall (S_ALL) to see what is learned
    for m in sorted({r["model"] for r in rows}):
        pass
