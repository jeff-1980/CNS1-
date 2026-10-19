"""正式阶段 §6：从论文A 抽样框（OpenAlex 重拉版）随机抽 80 篇做静态编码。
按 DOI 字母序排列框内带 DOI 的论文，numpy default_rng(20261017).choice(n, 80, replace=False) 无放回抽样。
没有 DOI 的论文无法按 DOI 排序，不进入抽样（篇数写在 meta 里）。
在 code/ 下运行：python -m audit.main.static_sample --frame DIR（写 static_sample_80.json/.csv）"""
import argparse, csv, json, os
import numpy as np

SEED, N = 20261017, 80


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frame", required=True); a = ap.parse_args()
    works = json.load(open(os.path.join(a.frame, "openalex_frame.json")))["works"]
    pool = sorted((w for w in works if w["doi"]), key=lambda w: w["doi"])
    idx = np.random.default_rng(SEED).choice(len(pool), N, replace=False)
    rows = [dict(draw_order=i + 1, pool_index=int(k), **pool[k]) for i, k in enumerate(idx)]
    meta = dict(seed=SEED, n=N, method="default_rng(seed).choice(len(pool), n, replace=False) over DOI-sorted pool",
                n_frame=len(works), n_pool=len(pool), n_no_doi_excluded=len(works) - len(pool), numpy=np.__version__,
                registration="https://osf.io/2tqv6")
    json.dump(dict(meta=meta, works=rows), open(os.path.join(a.frame, "static_sample_80.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(a.frame, "static_sample_80.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
