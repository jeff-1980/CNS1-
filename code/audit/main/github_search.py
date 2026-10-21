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


def gh(path, params=None, retries=8):
    cmd = ["gh", "api", "-X", "GET", path, "-H", "Accept: application/vnd.github+json"]
    for k, v in (params or {}).items(): cmd += ["-f", f"{k}={v}"]
    i = limited = 0
    while i < retries:
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0: return json.loads(p.stdout)
        if "Not Found" in p.stderr or "404" in p.stderr: return None
        if ("rate limit" in p.stderr.lower() or "403" in p.stderr) and limited < 90:   # 限流等待不算重试次数，最多约 90 分钟
            limited += 1; wait = 60
        else:
            i += 1; wait = min(2 ** i, 120)   # 网络 / TLS 超时：最多 8 次，等待封顶 2 分钟
        print(f"gh error ({p.stderr.strip()[:120]}); retry in {wait}s", file=sys.stderr); time.sleep(wait)
    raise RuntimeError(f"gh api {path} failed: {p.stderr.strip()[:300]}")


def run_query(q):
    items, page = [], 1
    first = gh("search/repositories", dict(q=q, per_page=100, page=1, sort="updated", order="desc"))
    total = first["total_count"]; items += first["items"]
    # GitHub 只给前 1000 条（10 页 × 100）；实际返回常少于 total_count，所以按页数而不是按条数停
    while page < 10 and len(items) < total and first["items"]:
        page += 1; time.sleep(2.5)   # search API：认证后 30 次/分钟
        r = gh("search/repositories", dict(q=q, per_page=100, page=page, sort="updated", order="desc"))
        if not r["items"]: break
        items += r["items"]
    return total, items, bool(first.get("incomplete_results")) or len(items) < min(total, 1000)


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
    part = os.path.join(out, "screen_doi.partial.json")   # 断点：中途失败后重跑从这里接着筛
    done = json.load(open(part)) if os.path.exists(part) else {}
    for n, r in enumerate(raw["repos"]):
        if r["full_name"] in done: continue
        rd = gh(f"repos/{r['full_name']}/readme")
        text = base64.b64decode(rd["content"]).decode("utf-8", "replace") if rd and rd.get("content") else ""
        text += "\n" + (r.get("description") or "") + "\n" + (r.get("homepage") or "")
        done[r["full_name"]] = sorted({d.rstrip(".").lower() for d in DOI_RE.findall(text)})
        if n % 50 == 0: json.dump(done, open(part, "w")); print(f"screened {len(done)}/{len(raw['repos'])}", flush=True)
    keep = [dict(r, dois=done[r["full_name"]]) for r in raw["repos"] if done[r["full_name"]]]
    json.dump(dict(meta=dict(raw["meta"], screened_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                             n_screened=len(raw["repos"]), n_with_doi=len(keep)), repos=keep),
              open(os.path.join(out, "screen_doi.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(out, "screen_doi.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["full_name", "head_commit", "stars", "dois"])
        for r in keep: w.writerow([r["full_name"], r["head_commit"], r["stars"], " ".join(r["dois"])])
    if os.path.exists(part): os.remove(part)
    print(f"screened {len(raw['repos'])}, with DOI {len(keep)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["search", "screen"]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (search if a.step == "search" else screen)(a.out)
