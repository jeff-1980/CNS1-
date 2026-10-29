"""正式阶段 §2 抽样框来源 1 的自动初筛：在 OpenAlex 框（openalex_frame.json）里找给出公开代码链接的论文。

两路：
  (a) DOI 对照：框内论文的 DOI 出现在 GitHub 检索筛出的仓库（screen_doi.json）里；
  (b) 开放全文：对 is_oa 的论文取 oa_url（PDF 或 HTML），抽出 github / gitee / gitlab / codeocean / zenodo 链接。
非开放获取的论文只能经 (a) 进入，这一限制写进偏离记录。只读不存全文；单连接、每次请求间隔 1 秒。
命中只是候选，是否“论文自己的代码”留到人工筛选。PDF 需要 pypdf（缺则记为 no_pdf_parser）。
在 code/ 下运行：python -m audit.main.openalex_code_links --frame DIR [--limit N]（输出写回 DIR）"""
import argparse, csv, datetime as dt, io, json, os, re, time, urllib.request

LINK_RE = re.compile(r"https?://(?:www\.)?(?:github\.com|gitee\.com|gitlab\.com|codeocean\.com|zenodo\.org)/[^\s\"'<>()\[\]{},;]+", re.I)
UA = "Mozilla/5.0 (audit of checkpoint selection; https://osf.io/2tqv6)"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.headers.get("Content-Type", ""), r.read(30_000_000)


def text_of(ctype, body):
    if "pdf" in ctype.lower() or body[:5] == b"%PDF-":
        try:
            import pypdf
        except ImportError:
            return None, "no_pdf_parser"
        try:
            return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(body)).pages), "pdf"
        except Exception as e:
            return None, f"pdf_error:{type(e).__name__}"
    return body.decode("utf-8", "replace"), "html"


def clean(links):
    out = set()
    for u in links:
        u = u.rstrip(".").rstrip("/")
        if re.search(r"github\.com/(?:[^/]+\.)?(?:features|about|login|site|topics|sponsors)\b", u, re.I): continue
        out.add(u)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frame", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    works = json.load(open(os.path.join(a.frame, "openalex_frame.json")))["works"]
    repos = json.load(open(os.path.join(a.frame, "screen_doi.json")))["repos"]
    by_doi = {}
    for r in repos:
        for d in r["dois"]: by_doi.setdefault(d, []).append(r["full_name"])
    t0 = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows, n_fetch = [], 0
    for w in works:
        row = dict(openalex_id=w["openalex_id"], doi=w["doi"], year=w["year"], is_oa=w["is_oa"],
                   gh_repos=by_doi.get(w["doi"], []), fulltext="not_oa", links=[])
        if w["is_oa"] and w["oa_url"] and (not a.limit or n_fetch < a.limit):
            n_fetch += 1
            try:
                ctype, body = get(w["oa_url"])
                txt, row["fulltext"] = text_of(ctype, body)
                if txt: row["links"] = clean(LINK_RE.findall(txt))
            except Exception as e:
                row["fulltext"] = f"fetch_error:{getattr(e, 'code', type(e).__name__)}"
            time.sleep(1)
            if n_fetch % 50 == 0: print(f"{n_fetch} fetched", flush=True)
        elif w["is_oa"]:
            row["fulltext"] = "no_oa_url" if not w["oa_url"] else "skipped_limit"
        row["candidate"] = bool(row["gh_repos"] or row["links"])
        rows.append(row)
    st = {}
    for r in rows: k = r["fulltext"].split(":")[0]; st[k] = st.get(k, 0) + 1
    meta = dict(started_at_utc=t0, finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                n_works=len(rows), n_oa=sum(bool(r["is_oa"]) for r in rows), fulltext_status=st,
                n_doi_match=sum(bool(r["gh_repos"]) for r in rows), n_link=sum(bool(r["links"]) for r in rows),
                n_candidate=sum(r["candidate"] for r in rows), limit=a.limit, registration="https://osf.io/2tqv6")
    json.dump(dict(meta=meta, works=rows), open(os.path.join(a.frame, "openalex_code_links.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(a.frame, "openalex_code_links.csv"), "w", newline="") as f:
        c = csv.writer(f); c.writerow(["openalex_id", "doi", "year", "is_oa", "fulltext", "candidate", "gh_repos", "links"])
        for r in rows: c.writerow([r["openalex_id"], r["doi"], r["year"], r["is_oa"], r["fulltext"], r["candidate"],
                                   " ".join(r["gh_repos"]), " ".join(r["links"])])
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
