#!/usr/bin/env python3
"""Fetch OpenAlex aggregate counts for the mechanical topic-pair forecaster (v0).

Everything is pulled with group_by count queries; no per-work bulk download.
Outputs (under --out):
  topics_all.json          all OpenAlex topics: id -> {name, subfield, field, domain}
  mech_topics.json         list of topic ids in the mechanical subfields
  totals/<Y>.json          topic -> number of articles in year Y
  ns/<Y>.json              topic -> number of articles in Nature/Science-family journals in year Y
  cooc/<Y>/<T>.json        for mech topic T: partner topic -> number of articles in year Y tagged with both
  k3_ncsa_works.jsonl      NC + SA mechanical articles 2023-2026 (id, doi, year, topic ids)
  fetch_log.txt            request log incl. OpenAlex meta.cost_usd when present

Resumable: every finished output file is skipped on rerun. Stops with exit code 3 on HTTP 429/409
(daily budget used up) so it can be resumed the next day.

Needs env var OPENALEX_API_KEY (free key from openalex.org/settings/api; ~10k list queries/day).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.openalex.org"
# Mechanical-related subfields (OpenAlex/ASJC ids). Names are printed at start for checking.
MECH_SUBFIELDS = ["2210", "2211", "2209", "2213"]
# Optional extension (--ext): co-occurrence is fetched for these topics too, in this order
# (energy and transport first, then electrical, control, civil/structural, computational mechanics).
EXT_SUBFIELDS = {
    "2100": "General Energy", "2102": "Energy Engineering and Power Technology", "2103": "Fuel Technology",
    "2104": "Nuclear Energy and Engineering", "2105": "Renewable Energy, Sustainability and the Environment",
    "2202": "Aerospace Engineering", "2203": "Automotive Engineering", "2212": "Ocean Engineering",
    "3313": "Transportation",
    "2208": "Electrical and Electronic Engineering", "2207": "Control and Systems Engineering",
    "2205": "Civil and Structural Engineering", "2206": "Computational Mechanics",
}
# Nature/Science family journals used as the "high-profile" venue set (resolved by ISSN at run time).
NS_ISSNS = {
    "Nature": "0028-0836", "Science": "0036-8075", "Nature Communications": "2041-1723",
    "Science Advances": "2375-2548", "Nature Materials": "1476-1122",
    "Nature Machine Intelligence": "2522-5839", "Science Robotics": "2470-9476",
    "Nature Energy": "2058-7546", "Nature Electronics": "2520-1131",
    "Nature Sustainability": "2398-9629", "Nature Computational Science": "2662-8457",
    "Communications Engineering": "2731-3395",
}
KEY = os.environ.get("OPENALEX_API_KEY", "")
LOG = None
N_REQ = 0


class BudgetExhausted(Exception):
    pass


def short(oid):
    return oid.rsplit("/", 1)[-1] if oid else oid


def get(path, params, tries=6):
    global N_REQ
    p = dict(params)
    if KEY:
        p["api_key"] = KEY
    url = f"{API}{path}?{urllib.parse.urlencode(p, safe=':|,*-')}"
    shown = url.replace(KEY, "KEY") if KEY else url
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            N_REQ += 1
            cost = (data.get("meta") or {}).get("cost_usd")
            LOG.write(f"{time.strftime('%H:%M:%S')}\t{cost}\t{shown}\n")
            time.sleep(0.12)
            return data
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            LOG.write(f"HTTP {e.code}\t{shown}\t{body}\n")
            LOG.flush()
            if e.code in (409, 429) and ("budget" in body.lower() or "credit" in body.lower() or i >= 3):
                raise BudgetExhausted(body)
            if e.code in (400, 404):
                raise
            time.sleep(2 ** i)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            LOG.write(f"NETERR {e}\t{shown}\n")
            time.sleep(2 ** i)
    raise RuntimeError(f"failed after {tries} tries: {shown}")


def list_all(path, params, key="results"):
    """Cursor-paginate a list endpoint (results or group_by buckets)."""
    out, cursor = [], "*"
    while cursor:
        d = get(path, {**params, "per_page": 100, "cursor": cursor})
        items = d.get(key) or []
        out.extend(items)
        nxt = (d.get("meta") or {}).get("next_cursor")
        if not items or nxt == cursor:
            break
        cursor = nxt
    return out


GROUP_CURSOR_OK = True


def group_counts(filt):
    global GROUP_CURSOR_OK
    params = {"filter": filt, "group_by": "topics.id"}
    buckets = None
    if GROUP_CURSOR_OK:
        try:
            buckets = list_all("/works", params, key="group_by")
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            GROUP_CURSOR_OK = False
            print("NOTE: group_by cursor paging refused; falling back to single page of 200", flush=True)
    if buckets is None:
        buckets = get("/works", {**params, "per_page": 200}).get("group_by") or []
        if len(buckets) >= 200:
            LOG.write(f"TRUNCATED group_by at 200 buckets: {filt}\n")
    return {short(b["key"]): b["count"] for b in buckets if b.get("key") and b["key"] != "unknown"}


def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--y0", type=int, default=2008)
    ap.add_argument("--y1", type=int, default=2025)
    ap.add_argument("--probe", action="store_true", help="~8 cheap requests: check subfields, topic counts, group_by paging")
    ap.add_argument("--max-requests", type=int, default=9500, help="stop politely after this many requests")
    ap.add_argument("--ext", action="store_true", help="also fetch co-occurrence for energy/transport topics")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    LOG = open(os.path.join(a.out, "fetch_log.txt"), "a", encoding="utf-8", buffering=1)
    if not KEY:
        print("WARNING: OPENALEX_API_KEY not set; free budget is ~10 list queries/day", flush=True)
    years = list(range(a.y0, a.y1 + 1))
    art = "type:article"

    def budget_ok():
        if N_REQ >= a.max_requests:
            print(f"reached --max-requests {a.max_requests}; rerun later to resume", flush=True)
            sys.exit(3)

    if a.probe:
        try:
            for sf in MECH_SUBFIELDS:
                d = get(f"/subfields/{sf}", {})
                print("subfield", sf, d.get("display_name"), "| field:", (d.get("field") or {}).get("display_name"), flush=True)
            d = get("/topics", {"filter": "subfield.id:" + "|".join(MECH_SUBFIELDS), "per_page": 1})
            print("mechanical topics:", d["meta"].get("count"), "| cost_usd:", d["meta"].get("cost_usd"), flush=True)
            d = get("/topics", {"per_page": 1})
            print("all topics:", d["meta"].get("count"), flush=True)
            t = "T10220"
            d = get("/works", {"filter": f"topics.id:{t},publication_year:2020,{art}", "group_by": "topics.id",
                               "per_page": 100, "cursor": "*"})
            print("group_by page1 buckets:", len(d.get("group_by") or []), "| next_cursor:",
                  bool((d.get("meta") or {}).get("next_cursor")), "| meta:", {k: v for k, v in (d.get("meta") or {}).items()
                                                                                  if k != "next_cursor"}, flush=True)
            print("first buckets:", (d.get("group_by") or [])[:3], flush=True)
        except Exception as e:  # noqa
            print("PROBE ERROR:", repr(e)[:400], flush=True)
        print("requests used:", N_REQ, flush=True)
        return
    try:
        for sf in MECH_SUBFIELDS:
            d = get(f"/subfields/{sf}", {})
            print("subfield", sf, d.get("display_name"), "| field:", (d.get("field") or {}).get("display_name"), flush=True)

        f_all = os.path.join(a.out, "topics_all.json")
        if not os.path.exists(f_all):
            tops = list_all("/topics", {"select": "id,display_name,subfield,field,domain"})
            save(f_all, {short(t["id"]): {"name": t["display_name"],
                                          "subfield": short((t.get("subfield") or {}).get("id")),
                                          "subfield_name": (t.get("subfield") or {}).get("display_name"),
                                          "field": short((t.get("field") or {}).get("id")),
                                          "domain": short((t.get("domain") or {}).get("id"))} for t in tops})
        topics_all = json.load(open(f_all, encoding="utf-8"))
        mech = sorted(t for t, v in topics_all.items() if v["subfield"] in MECH_SUBFIELDS)
        save(os.path.join(a.out, "mech_topics.json"), mech)
        print(f"topics: {len(topics_all)} total, {len(mech)} mechanical", flush=True)
        ext = []
        if a.ext:
            order = {sf: i for i, sf in enumerate(EXT_SUBFIELDS)}
            ext = sorted((t for t, v in topics_all.items() if v["subfield"] in EXT_SUBFIELDS),
                         key=lambda t: (order[topics_all[t]["subfield"]], t))
            save(os.path.join(a.out, "ext_topics.json"), ext)
            print(f"ext topics: {len(ext)} in {len(EXT_SUBFIELDS)} subfields", flush=True)

        f_ns = os.path.join(a.out, "ns_sources.json")
        if not os.path.exists(f_ns):
            d = get("/sources", {"filter": "issn:" + "|".join(NS_ISSNS.values()), "per_page": 100,
                                 "select": "id,display_name,issn_l"})
            save(f_ns, {short(s["id"]): s["display_name"] for s in d["results"]})
        ns_src = json.load(open(f_ns, encoding="utf-8"))
        print("NS-family sources resolved:", len(ns_src), sorted(ns_src.values()), flush=True)
        ns_filter = "primary_location.source.id:" + "|".join(ns_src)
        ncsa = [k for k, v in ns_src.items() if v in ("Nature Communications", "Science Advances")]

        for y in years:
            f = os.path.join(a.out, "totals", f"{y}.json")
            if not os.path.exists(f):
                save(f, group_counts(f"publication_year:{y},{art}"))
            f = os.path.join(a.out, "ns", f"{y}.json")
            if not os.path.exists(f):
                save(f, group_counts(f"publication_year:{y},{art},{ns_filter}"))
            budget_ok()
        print("totals + NS counts done", flush=True)

        f_k3 = os.path.join(a.out, "k3_ncsa_works.jsonl")
        if not os.path.exists(f_k3):
            works = list_all("/works", {
                "filter": f"publication_year:2023-2026,{art},primary_location.source.id:{'|'.join(ncsa)},"
                          f"topics.subfield.id:{'|'.join(MECH_SUBFIELDS)}",
                "select": "id,doi,publication_year,topics,primary_location"})
            with open(f_k3 + ".tmp", "w", encoding="utf-8") as fh:
                for w in works:
                    fh.write(json.dumps({"id": short(w["id"]), "doi": w.get("doi"), "year": w.get("publication_year"),
                                         "source": short(((w.get("primary_location") or {}).get("source") or {}).get("id")),
                                         "topics": [short(t["id"]) for t in (w.get("topics") or [])]}) + "\n")
            os.replace(f_k3 + ".tmp", f_k3)
            print("K3 NC/SA mechanical works:", len(works), flush=True)

        focus = mech + ext  # mechanical first, then the extension
        todo = [(y, t) for t in focus for y in years
                if not os.path.exists(os.path.join(a.out, "cooc", str(y), f"{t}.json"))]
        print(f"co-occurrence queries left: {len(todo)} of {len(years) * len(focus)}", flush=True)
        for i, (y, t) in enumerate(todo):
            save(os.path.join(a.out, "cooc", str(y), f"{t}.json"),
                 group_counts(f"topics.id:{t},publication_year:{y},{art}"))
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} requests_this_run={N_REQ}", flush=True)
            budget_ok()
        print("ALL DONE requests_this_run=", N_REQ, flush=True)
    except BudgetExhausted as e:
        print("OpenAlex budget exhausted for today; rerun tomorrow to resume.", str(e)[:200], flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
