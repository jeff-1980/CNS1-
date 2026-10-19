"""正式阶段 §2 抽样框来源 1 与 §6 静态编码抽样框：按论文A 记录的 OpenAlex 检索式重拉逐篇清单。

论文A 当年只保存了计数与分组统计（N=1347），没有逐篇清单（2026-10-19 核查）。这里按其记录的检索式重拉：
  title_and_abstract.search:"bearing fault diagnosis" AND (noise OR noisy OR SNR)
  from_publication_date:2019-01-01, to_publication_date:2025-12-31, type:article
OpenAlex 持续更新，篇数与 1347 的差异如实记录（偏离记录）。只用标准库；用 page 普通分页（不带 API key 时
cursor 翻页返回 401，普通分页上限 10000 条，远大于本框），礼貌池带 mailto。
在 code/ 下运行：python -m audit.main.openalex_frame --out DIR [--mailto ADDR]"""
import argparse, csv, datetime as dt, json, os, time, urllib.parse, urllib.request

SEARCH = '"bearing fault diagnosis" AND (noise OR noisy OR SNR)'
FILTER = ("title_and_abstract.search:" + SEARCH +
          ",from_publication_date:2019-01-01,to_publication_date:2025-12-31,type:article")
FIELDS = "id,doi,title,publication_year,primary_location,open_access,cited_by_count,type"


def fetch(params, mailto):
    if mailto: params = dict(params, mailto=mailto)
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r: return json.load(r)
        except Exception as e:
            print(f"retry {i + 1}: {e}"); time.sleep(2 ** (i + 1))
    raise RuntimeError(url)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--mailto", default="")
    a = ap.parse_args()
    t0 = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows, page, count = [], 1, None
    while True:
        r = fetch(dict(filter=FILTER, select=FIELDS, per_page=200, page=page, sort="publication_date:asc"), a.mailto)
        count = r["meta"]["count"]
        if count > 10000: raise SystemExit(f"{count} > 10000: basic paging cannot reach all works")
        for w in r["results"]:
            loc = (w.get("primary_location") or {}).get("source") or {}
            rows.append(dict(openalex_id=w["id"], doi=(w.get("doi") or "").replace("https://doi.org/", "").lower(),
                             title=w.get("title"), year=w.get("publication_year"), venue=loc.get("display_name"),
                             is_oa=(w.get("open_access") or {}).get("is_oa"), oa_url=(w.get("open_access") or {}).get("oa_url"),
                             cited_by=w.get("cited_by_count")))
        if not r["results"] or len(rows) >= count: break
        page += 1; time.sleep(0.5)
    ids = [x["openalex_id"] for x in rows]
    if len(set(ids)) != len(ids) or len(ids) != count:   # 同日期并列时页边界可能不稳，出问题就重跑
        raise SystemExit(f"paging inconsistent: {len(ids)} rows, {len(set(ids))} unique, meta count {count}")
    meta = dict(filter=FILTER, paging="page, per_page=200, sort=publication_date:asc", fetched_at_utc=t0, meta_count=count, n_rows=len(rows),
                n_with_doi=sum(bool(x["doi"]) for x in rows), paperA_recorded_n=1347,
                registration="https://osf.io/2tqv6")
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(meta=meta, works=rows), open(os.path.join(a.out, "openalex_frame.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, "openalex_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["openalex_id"]); w.writeheader(); w.writerows(rows)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
