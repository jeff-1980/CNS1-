"""正式阶段 §2 抽样框来源 1 的第二轮初筛：经 Europe PMC 取正文全文 XML 找代码链接。

第一轮（openalex_code_links）直接读出版商 oa_url，568 篇开放获取里 331 篇被拦（MDPI 403 占一半），取到的 HTML
多是只有摘要的落地页。这一轮把框内全部带 DOI 的论文按 DOI 批量查 Europe PMC（每次 50 个 DOI），对有 PMCID 且
inEPMC 的论文取 fullTextXML，只在正文里抽链接，避开网页导航里的无关链接。只读不存全文；单连接、请求间隔 1 秒。
在 code/ 下运行：python -m audit.main.epmc_code_links --frame DIR（读 openalex_frame.json，写 epmc_code_links.json/.csv）"""
import argparse, csv, datetime as dt, json, os, re, time, urllib.parse, urllib.request

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
LINK_RE = re.compile(r"https?://(?:www\.)?(?:github\.com|gitee\.com|gitlab\.com|codeocean\.com|zenodo\.org)/[^\s\"'<>()\[\]{},;]+", re.I)
REPO_RE = re.compile(r"^https?://(?:www\.)?(?:github|gitee|gitlab)\.com/[^/]+/[^/#?]+", re.I)


def get(url, as_json=True):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "checkpoint-audit (osf.io/2tqv6)"}),
                                        timeout=60) as r:
                b = r.read(); return json.loads(b) if as_json else b.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            err = e
        except Exception as e:
            err = e
        time.sleep(2 ** (i + 1))
    raise RuntimeError(f"{url}: {err}")


def code_links(text):
    """只留仓库级链接（owner/repo）与 Zenodo / Code Ocean 记录页。"""
    out = set()
    for u in LINK_RE.findall(text):
        u = u.rstrip(".").rstrip("/")
        m = REPO_RE.match(u)
        if m: out.add(m.group(0).removesuffix(".git").lower()); continue
        if re.match(r"^https?://(?:www\.)?(zenodo\.org/(records?|doi)/|codeocean\.com/capsule/)", u, re.I): out.add(u)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frame", required=True); a = ap.parse_args()
    works = json.load(open(os.path.join(a.frame, "openalex_frame.json")))["works"]
    dois = sorted({w["doi"] for w in works if w["doi"]})
    t0 = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    hits = {}
    for i in range(0, len(dois), 50):
        q = " OR ".join(f'DOI:"{d}"' for d in dois[i:i + 50])
        r = get(f"{BASE}/search?" + urllib.parse.urlencode(dict(query=q, format="json", resultType="lite", pageSize=100)))
        for x in r["resultList"]["result"]:
            if x.get("doi"): hits[x["doi"].lower()] = dict(pmcid=x.get("pmcid"), in_epmc=x.get("inEPMC") == "Y",
                                                           is_oa=x.get("isOpenAccess") == "Y")
        time.sleep(1)
    print(f"Europe PMC records {len(hits)}; with PMCID {sum(bool(h['pmcid']) for h in hits.values())}", flush=True)
    rows = []
    for w in works:
        h = hits.get(w["doi"], {})
        row = dict(openalex_id=w["openalex_id"], doi=w["doi"], year=w["year"], pmcid=h.get("pmcid"),
                   fulltext="no_epmc_record" if not h else "no_pmc_fulltext", links=[])
        if h.get("pmcid") and h.get("in_epmc"):
            try:
                xml = get(f"{BASE}/{h['pmcid']}/fullTextXML", as_json=False)
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
    meta = dict(started_at_utc=t0, finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                n_works=len(rows), n_epmc_records=len(hits), fulltext_status=st,
                n_link=sum(bool(r["links"]) for r in rows), registration="https://osf.io/2tqv6")
    json.dump(dict(meta=meta, works=rows), open(os.path.join(a.frame, "epmc_code_links.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(a.frame, "epmc_code_links.csv"), "w", newline="") as f:
        c = csv.writer(f); c.writerow(["openalex_id", "doi", "year", "pmcid", "fulltext", "links"])
        for r in rows: c.writerow([r["openalex_id"], r["doi"], r["year"], r["pmcid"], r["fulltext"], " ".join(r["links"])])
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
