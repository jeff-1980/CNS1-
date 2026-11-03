"""正式阶段 §2 合格性筛选的资料包：为每个候选仓库取 README 摘录、文件树与依赖文件，供人工按纳入 / 排除标准判定。

候选 = GitHub 检索里带 DOI 的仓库（screen_doi.json）∪ OpenAlex 框内论文全文给出、且不在前者里的仓库链接
（openalex_code_links.json、epmc_code_links.json；只收 owner/repo 级 GitHub 链接）。
每个仓库固定在检索时记录的 head_commit（链接来源的仓库取当前默认分支 head，并记下 commit）。
只读，需要已登录的 gh。在 code/ 下运行：python -m audit.main.candidate_dossier --frame DIR（写 candidate_dossier.json）"""
import argparse, base64, datetime as dt, json, os, re
from audit.main.github_search import gh

DEP_FILES = re.compile(r"(^|/)(requirements[^/]*\.txt|environment[^/]*\.ya?ml|setup\.py|pyproject\.toml|Pipfile)$", re.I)
README_CHARS, TREE_MAX = 6000, 1500


def repo_of(url):
    m = re.match(r"^https?://(?:www\.)?github\.com/([^/]+)/([^/#?]+)", url, re.I)
    return f"{m.group(1)}/{m.group(2).removesuffix('.git')}" if m else None


def text(path):
    r = gh(path)
    return base64.b64decode(r["content"]).decode("utf-8", "replace") if r and r.get("content") else ""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frame", required=True)
    ap.add_argument("--link-files", default="openalex_code_links.json,epmc_code_links.json", help="DIR 下给出论文代码链接的文件")
    ap.add_argument("--exclude", default="", help="已筛过的 candidate_dossier.json，其中的仓库不再收（扩大检索用）")
    a = ap.parse_args()
    F = lambda n: json.load(open(os.path.join(a.frame, n)))
    cands = {r["full_name"].lower(): dict(full_name=r["full_name"], source="github_search", commit=r["head_commit"],
                                          paper_dois=r["dois"], stars=r["stars"], language=r["language"])
             for r in F("screen_doi.json")["repos"]}
    for name in a.link_files.split(","):
        for w in F(name)["works"]:
            for u in w["links"]:
                fn = repo_of(u)
                if fn and fn.lower() not in cands:
                    cands[fn.lower()] = dict(full_name=fn, source=f"openalex_fulltext:{name.split('_')[0]}", commit=None,
                                             paper_dois=[w["doi"]], stars=None, language=None)
                elif fn and w["doi"] not in cands[fn.lower()]["paper_dois"]:
                    cands[fn.lower()].setdefault("frame_dois", []).append(w["doi"])
    seen = {r["full_name"].lower() for r in json.load(open(a.exclude))["repos"]} if a.exclude else set()
    excluded = sorted(k for k in cands if k in seen)
    cands = {k: v for k, v in cands.items() if k not in seen}
    out = []
    for i, c in enumerate(sorted(cands.values(), key=lambda c: c["full_name"].lower())):
        meta = gh(f"repos/{c['full_name']}")
        if not meta: out.append(dict(c, status="not_found")); continue
        if not c["commit"]:
            br = gh(f"repos/{c['full_name']}/branches/{meta['default_branch']}") or {}
            c["commit"] = br.get("commit", {}).get("sha")
            c["stars"], c["language"] = meta["stargazers_count"], meta.get("language")
        tree = gh(f"repos/{c['full_name']}/git/trees/{c['commit']}", dict(recursive="1")) or {}
        paths = [t["path"] for t in tree.get("tree", []) if t["type"] == "blob"]
        deps = {p: text(f"repos/{c['full_name']}/contents/{p}?ref={c['commit']}")[:3000]
                for p in [p for p in paths if DEP_FILES.search(p)][:5]}
        readme = gh(f"repos/{c['full_name']}/readme", dict(ref=c["commit"]))
        rd = base64.b64decode(readme["content"]).decode("utf-8", "replace") if readme and readme.get("content") else ""
        out.append(dict(c, status="ok", description=meta.get("description"), license=(meta.get("license") or {}).get("spdx_id"),
                        archived=meta.get("archived"), n_files=len(paths), tree_truncated=bool(tree.get("truncated")),
                        n_py=sum(p.endswith(".py") for p in paths), n_ipynb=sum(p.endswith(".ipynb") for p in paths),
                        train_files=[p for p in paths if re.search(r"train", p.split("/")[-1], re.I)][:30],
                        paths=paths[:TREE_MAX], dep_files=deps, readme_excerpt=rd[:README_CHARS], readme_chars=len(rd)))
        if (i + 1) % 20 == 0: print(f"{i + 1}/{len(cands)}", flush=True)
    meta = dict(built_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), n_candidates=len(out),
                by_source={s: sum(o["source"] == s for o in out) for s in sorted({o["source"] for o in out})},
                link_files=a.link_files, excluded_as_already_screened=len(excluded), exclude_file=a.exclude or None,
                registration="https://osf.io/2tqv6")
    json.dump(dict(meta=meta, repos=out), open(os.path.join(a.frame, "candidate_dossier.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
