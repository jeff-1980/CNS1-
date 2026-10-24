"""正式阶段 §2 扩大检索（偏离，jeff 2026-10-20 裁定）：原抽样框试到第 8 个对象只有 1 个可运行，按此比例 32 个对象
远不到 §2 的 15 个下限，于是在原框之外补一轮检索。纳入 / 排除标准、原因码、选择方式都不变；原框 32 个对象照常先试完，
扩大检索得到的新合格对象按同一种子排序，接在第 33 号之后。

两路来源，结果都写进 --out（建议 results/main_frame/2026-10-20_expansion），分开提交，先存档再筛选：
  github  对 QUERIES 里每个查询式做 GitHub 仓库检索（同 github_search 的字段），超过 1000 条时按创建日期递归二分；
          合并去重后去掉 --seen（原检索 search_raw.json）里已有的仓库，写 search_raw.json/.csv。
          之后用原脚本的 screen 步骤抽 DOI：python -m audit.main.github_search screen --out DIR
  epmc    Europe PMC 全文检索 EPMC_QUERY（同行评审、全文在 EPMC 的轴承论文里提到 github 的），取 fullTextXML 抽仓库链接，
          写 epmcsearch_code_links.json/.csv（字段与 epmc_code_links 相同，供 candidate_dossier --link-files 读）。
然后：python -m audit.main.candidate_dossier --frame DIR --link-files epmcsearch_code_links.json
        --exclude ../results/main_frame/2026-10-19/candidate_dossier.json
需要已登录的 gh。单连接；GitHub search 每次间隔 2.5 秒，Europe PMC 每次间隔 1 秒。
在 code/ 下运行：python -m audit.main.expand_search {github,epmc} --out DIR [--seen FILE]"""
import argparse, csv, datetime as dt, json, os, re, time, urllib.parse
from audit.main.github_search import QUERY as ORIGINAL_QUERY, gh, run_query
from audit.main.epmc_code_links import BASE, code_links, get

QUERIES = [f'"{t}" in:name,description,readme' for t in (
    "bearing fault", "rolling bearing", "rolling element bearing", "bearing diagnosis", "Case Western Reserve",
    "rotating machinery", "machinery fault diagnosis", "intelligent fault diagnosis")] + [
    "CWRU in:name,description,readme", "Paderborn bearing in:name,description,readme"]
EPMC_QUERY = ('("bearing fault" OR "rolling bearing" OR "bearing diagnosis" OR "rotating machinery") AND "github" '
              'AND PUB_YEAR:[2019 TO 2025] AND IN_EPMC:y AND NOT SRC:PPR')
NOW = lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sliced(q, lo, hi, log):
    """created:lo..hi 内取全；超过 1000 条就把日期区间对半分。"""
    sq = f"{q} created:{lo.isoformat()}..{hi.isoformat()}"
    # 先只取计数再决定是否二分：run_query 会一直翻页，超过 1000 条时第 11 页起 GitHub 返回 422
    n = gh("search/repositories", dict(q=sq, per_page=1))["total_count"]; time.sleep(2.5)
    if n > 1000 and lo < hi:
        mid = lo + (hi - lo) // 2
        return sliced(q, lo, mid, log) + sliced(q, mid + dt.timedelta(days=1), hi, log)
    if n > 1000: raise SystemExit(f"{sq}: {n} > 1000 within a single day")
    n, items, inc = (run_query(sq) if n else (0, [], False)); time.sleep(2.5)
    log.append(dict(query=sq, total_count=n, returned=len(items), incomplete=inc))
    return items


def github(out, seen_file):
    t0 = NOW(); seen = {r["full_name"].lower() for r in json.load(open(seen_file))["repos"]}
    per_query, rows, got = [], [], set()
    for q in QUERIES:
        log = []; items = sliced(q, dt.date(2008, 1, 1), dt.date.today(), log)
        new = 0
        for it in items:
            fn = it["full_name"]
            if fn.lower() in got or fn.lower() in seen: continue
            got.add(fn.lower()); new += 1
            br = gh(f"repos/{fn}/branches/{it['default_branch']}")
            rows.append(dict(full_name=fn, html_url=it["html_url"], default_branch=it["default_branch"],
                             head_commit=(br or {}).get("commit", {}).get("sha"), stars=it["stargazers_count"],
                             created_at=it["created_at"], pushed_at=it["pushed_at"], language=it.get("language"),
                             description=it.get("description"), homepage=it.get("homepage"), fork=it["fork"],
                             archived=it.get("archived"), query=q))
        per_query.append(dict(query=q, total_count=sum(s["total_count"] for s in log), returned=len(items),
                              new_unique=new, incomplete=any(s["incomplete"] for s in log), slices=log))
        print(f"{q}: {per_query[-1]['total_count']} hits, {new} new", flush=True)
    meta = dict(step="expansion_github", queries=QUERIES, original_query=ORIGINAL_QUERY, seen_file=seen_file,
                n_seen=len(seen), searched_at_utc=t0, finished_at_utc=NOW(), per_query=per_query, n_unique=len(rows),
                total_count=None, incomplete_results=any(p["incomplete"] for p in per_query), slices=None,
                registration="https://osf.io/2tqv6", deviation="results/DEVIATIONS_AUDIT.md 2026-10-20 扩大检索")
    os.makedirs(out, exist_ok=True)
    json.dump(dict(meta=meta, repos=rows), open(os.path.join(out, "search_raw.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(out, "search_raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["full_name"]); w.writeheader(); w.writerows(rows)
    print(json.dumps({k: v for k, v in meta.items() if k != "per_query"}, indent=1))


def epmc(out):
    t0 = NOW(); res, cursor = [], "*"
    while True:
        r = get(f"{BASE}/search?" + urllib.parse.urlencode(dict(query=EPMC_QUERY, format="json", resultType="lite",
                                                                pageSize=1000, cursorMark=cursor)))
        res += r["resultList"]["result"]; count = r["hitCount"]
        if not r["resultList"]["result"] or r.get("nextCursorMark") in (None, cursor): break
        cursor = r["nextCursorMark"]; time.sleep(1)
    print(f"Europe PMC hits {count}, rows {len(res)}", flush=True)
    rows = []
    for x in res:
        row = dict(openalex_id=None, doi=(x.get("doi") or "").lower(), year=x.get("pubYear"), pmcid=x.get("pmcid"),
                   title=x.get("title"), fulltext="no_pmcid", links=[])
        if x.get("pmcid"):
            try:
                xml = get(f"{BASE}/{x['pmcid']}/fullTextXML", as_json=False)
                if xml:
                    row["fulltext"] = "xml"
                    row["links"] = code_links(re.sub(r"<[^>]+>", " ", re.sub(r'<ext-link[^>]*href="([^"]+)"', r" \1 ", xml)))
                else:
                    row["fulltext"] = "xml_404"
            except Exception as e:
                row["fulltext"] = f"fetch_error:{type(e).__name__}"
            time.sleep(1)
        rows.append(row)
    st = {}
    for r in rows: k = r["fulltext"].split(":")[0]; st[k] = st.get(k, 0) + 1
    meta = dict(step="expansion_epmc", query=EPMC_QUERY, started_at_utc=t0, finished_at_utc=NOW(), hit_count=count,
                n_works=len(rows), n_no_doi=sum(not r["doi"] for r in rows), fulltext_status=st,
                n_link=sum(bool(r["links"]) for r in rows), registration="https://osf.io/2tqv6")
    os.makedirs(out, exist_ok=True)
    # 无 DOI 的论文仍保留（候选资料包里 paper_dois 为空，人工筛选时按 E2 判定）
    json.dump(dict(meta=meta, works=rows), open(os.path.join(out, "epmcsearch_code_links.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(out, "epmcsearch_code_links.csv"), "w", newline="") as f:
        c = csv.writer(f); c.writerow(["doi", "year", "pmcid", "fulltext", "links", "title"])
        for r in rows: c.writerow([r["doi"], r["year"], r["pmcid"], r["fulltext"], " ".join(r["links"]), r["title"]])
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["github", "epmc"]); ap.add_argument("--out", required=True)
    ap.add_argument("--seen", default="../results/main_frame/2026-10-19/search_raw.json")
    a = ap.parse_args()
    github(a.out, a.seen) if a.step == "github" else epmc(a.out)
