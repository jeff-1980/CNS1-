#!/usr/bin/env python3
"""Fetch titles + abstracts for the K3 NC/SA mechanical works (for structured-card decomposition).

Reads  <data>/k3_ncsa_works.jsonl  (ids from fetch_openalex.py)
Writes <data>/k3_ncsa_abstracts.jsonl  one line per work:
       {id, doi, year, source, title, abstract, cited_by_count, topics:[{id,name,subfield}], concepts? no}
About 50 list requests (50 ids per filter). Resumable. Needs OPENALEX_API_KEY.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.openalex.org"
KEY = os.environ.get("OPENALEX_API_KEY", "")


def get(params):
    if KEY:
        params["api_key"] = KEY
    url = f"{API}/works?{urllib.parse.urlencode(params, safe=':|,*-')}"
    for i in range(6):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa
            print("retry", i, repr(e)[:120], flush=True)
            time.sleep(2 ** i)
    raise RuntimeError("failed: " + url.replace(KEY, "KEY"))


def inv_to_text(inv):
    if not inv:
        return ""
    pos = []
    for w, ps in inv.items():
        for p in ps:
            pos.append((p, w))
    return " ".join(w for _, w in sorted(pos))


def main():
    data = sys.argv[1]
    src = os.path.join(data, "k3_ncsa_works.jsonl")
    dst = os.path.join(data, "k3_ncsa_abstracts.jsonl")
    ids = [json.loads(l)["id"] for l in open(src, encoding="utf-8")]
    done = set()
    if os.path.exists(dst):
        done = {json.loads(l)["id"] for l in open(dst, encoding="utf-8")}
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} works, {len(done)} done, {len(todo)} to fetch", flush=True)
    with open(dst, "a", encoding="utf-8") as out:
        for k in range(0, len(todo), 50):
            batch = todo[k:k + 50]
            d = get({"filter": "openalex_id:" + "|".join(batch), "per_page": 50,
                     "select": "id,doi,title,publication_year,abstract_inverted_index,cited_by_count,"
                               "primary_location,topics"})
            for w in d.get("results", []):
                out.write(json.dumps({
                    "id": w["id"].rsplit("/", 1)[-1], "doi": w.get("doi"), "year": w.get("publication_year"),
                    "source": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "title": w.get("title"), "abstract": inv_to_text(w.get("abstract_inverted_index")),
                    "cited_by_count": w.get("cited_by_count"),
                    "topics": [{"id": t["id"].rsplit("/", 1)[-1], "name": t["display_name"],
                                "subfield": (t.get("subfield") or {}).get("display_name")}
                               for t in (w.get("topics") or [])]}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  {min(k + 50, len(todo))}/{len(todo)}", flush=True)
            time.sleep(0.2)
    n = sum(1 for _ in open(dst, encoding="utf-8"))
    empty = sum(1 for l in open(dst, encoding="utf-8") if not json.loads(l)["abstract"])
    print(f"DONE {n} records, {empty} without abstract", flush=True)


if __name__ == "__main__":
    main()
