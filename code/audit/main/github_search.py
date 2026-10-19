"""正式阶段 §2 抽样框来源 2：GitHub 仓库检索（预注册 results/PREREG_AUDIT_main.md，OSF 2tqv6）。

两步，分开提交，保证“完整结果列表先存档、再筛选”：
  search  执行冻结的查询式 "bearing fault diagnosis" in:name,description,readme，记录每个仓库的
          全名、默认分支及其当前 commit、星数、创建/推送时间与检索时间，写 search_raw.json/.csv。
          GitHub 每个查询最多返回 1000 条；total_count 超过 1000 时按仓库创建年份切分同一查询
          （created:YYYY-01-01..YYYY-12-31），切分方式写进输出的 meta。
  screen  只读 search_raw.json：取每个仓库的 README（及仓库描述与主页），抽出 DOI，
          保留至少有一个 DOI 的仓库，写 screen_doi.json/.csv。DOI 是否为同行评审论文留到人工筛选。
需要已登录的 gh CLI。在 code/ 下运行：
  python -m audit.main.github_search search --out DIR
  python -m audit.main.github_search screen --out DIR"""
import argparse, base64, csv, datetime as dt, json, os, re, subprocess, sys, time

QUERY = '"bearing fault diagnosis" in:name,description,readme'
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>\]\)\},;]+)", re.I)


def gh(path, params=None, retries=4):
    cmd = ["gh", "api", "-X", "GET", path, "-H", "Accept: application/vnd.github+json"]
    for k, v in (params or {}).items(): cmd += ["-f", f"{k}={v}"]
    for i in range(retries):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0: return json.loads(p.stdout)
        if "Not Found" in p.stderr or "404" in p.stderr: return None
        wait = 60 if ("rate limit" in p.stderr.lower() or "403" in p.stderr) else 2 ** (i + 1)
        print(f"gh error ({p.stderr.strip()[:120]}); retry in {wait}s", file=sys.stderr); time.sleep(wait)
    raise RuntimeError(f"gh api {path} failed: {p.stderr.strip()[:300]}")


def run_query(q):
    items, page = [], 1
    first = gh("search/repositories", dict(q=q, per_page=100, page=1, sort="updated", order="desc"))
    total = first["total_count"]; items += first["items"]
    while len(items) < min(total, 1000) and first["items"]:
        page += 1; time.sleep(2.5)   # search API：认证后 30 次/分钟
        r = gh("search/repositories", dict(q=q, per_page=100, page=page, sort="updated", order="desc"))
        if not r["items"]: break
        items += r["items"]
    return total, items, bool(first.get("incomplete_results"))


def search(out):
    t0 = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    total, items, incomplete = run_query(QUERY)
    slices = None
    if total > 1000:
        slices, items = [], []
        for y in range(2008, dt.date.today().year + 1):
            q = f"{QUERY} created:{y}-01-01..{y}-12-31"
            n, it, inc = run_query(q); slices.append(dict(query=q, total_count=n, returned=len(it), incomplete=inc))
            if n > 1000: raise SystemExit(f"{y}: {n} > 1000 results; split further before continuing")
            items += it; time.sleep(2.5)
    seen, rows = set(), []
    for it in items:
        fn = it["full_name"]
        if fn in seen: continue
        seen.add(fn)
        br = gh(f"repos/{fn}/branches/{it['default_branch']}")
        rows.append(dict(full_name=fn, html_url=it["html_url"], default_branch=it["default_branch"],
                         head_commit=(br or {}).get("commit", {}).get("sha"), stars=it["stargazers_count"],
                         created_at=it["created_at"], pushed_at=it["pushed_at"], language=it.get("language"),
                         description=it.get("description"), homepage=it.get("homepage"), fork=it["fork"],
                         archived=it.get("archived")))
    meta = dict(query=QUERY, searched_at_utc=t0, finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                total_count=total, incomplete_results=incomplete, slices=slices, n_unique=len(rows),
                registration="https://osf.io/2tqv6")
    os.makedirs(out, exist_ok=True)
    json.dump(dict(meta=meta, repos=rows), open(os.path.join(out, "search_raw.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(out, "search_raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["full_name"]); w.writeheader(); w.writerows(rows)
    print(json.dumps(meta, indent=1))


def screen(out):
    raw = json.load(open(os.path.join(out, "search_raw.json")))
    keep = []
    for r in raw["repos"]:
        rd = gh(f"repos/{r['full_name']}/readme")
        text = base64.b64decode(rd["content"]).decode("utf-8", "replace") if rd and rd.get("content") else ""
        text += "\n" + (r.get("description") or "") + "\n" + (r.get("homepage") or "")
        dois = sorted({d.rstrip(".").lower() for d in DOI_RE.findall(text)})
        if dois: keep.append(dict(r, dois=dois))
    json.dump(dict(meta=dict(raw["meta"], screened_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                             n_screened=len(raw["repos"]), n_with_doi=len(keep)), repos=keep),
              open(os.path.join(out, "screen_doi.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(out, "screen_doi.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["full_name", "head_commit", "stars", "dois"])
        for r in keep: w.writerow([r["full_name"], r["head_commit"], r["stars"], " ".join(r["dois"])])
    print(f"screened {len(raw['repos'])}, with DOI {len(keep)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["search", "screen"]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (search if a.step == "search" else screen)(a.out)
