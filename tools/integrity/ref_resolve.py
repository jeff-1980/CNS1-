#!/usr/bin/env python3
"""ref_resolve.py - resolve every DOI in a reference list and compare it with
what the manuscript says (ScientistOne audit I3 "reference verification").

For each DOI found in the input file (Markdown reference list, .bib, or the
manuscript itself) the script fetches Crossref metadata (OpenAlex as fallback)
and compares year, first-author surname and title words against the text of
the reference entry that carries the DOI. Entries without a DOI are listed as
UNRESOLVED so a human (or nature-ref-verifier) can look them up.

Network: api.crossref.org and api.openalex.org are blocked from the cloud
container; run this on a machine with normal internet (jeff's PC).

Usage:
    python3 ref_resolve.py REFS.md [-o report.md] [--mailto you@example.org] [--sleep 1.0]
    python3 ref_resolve.py REFS.md --offline       # only list DOIs / entries, no network
    python3 ref_resolve.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>\]\)|,;]+)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d\d\b")
STOP = set("a an the of on in for and or to with from by at as is are its into via over under using based".split())


def split_entries(text: str):
    """Yield (label, entry_text). Works for numbered Markdown lists, table rows and .bib."""
    if "@" in text and re.search(r"^@\w+\{", text, re.M):
        for m in re.finditer(r"^@\w+\{([^,]+),(.*?)^\}", text, re.M | re.S):
            yield m.group(1).strip(), m.group(2)
        return
    cur_label, cur = None, []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:\[?(\d+)\]?\.?|\|\s*(\d+)\s*\|)\s*(.*)", line)
        if m and (m.group(1) or m.group(2)):
            if cur:
                yield cur_label, " ".join(cur)
            cur_label, cur = (m.group(1) or m.group(2)), [m.group(3)]
        elif cur and line.strip():
            cur.append(line.strip())
    if cur:
        yield cur_label, " ".join(cur)


def clean_doi(d: str) -> str:
    return d.rstrip(".").rstrip("|").strip()


def fetch_json(url: str, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "cns1-ref-resolve/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def crossref(doi: str, mailto: str | None):
    q = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    if mailto:
        q += f"?mailto={urllib.parse.quote(mailto)}"
    m = fetch_json(q)["message"]
    year = None
    for k in ("published-print", "published-online", "issued", "created"):
        parts = m.get(k, {}).get("date-parts", [[None]])
        if parts and parts[0] and parts[0][0]:
            year = parts[0][0]
            break
    auth = m.get("author") or []
    return {
        "source": "crossref",
        "title": (m.get("title") or [""])[0],
        "year": year,
        "first_author": (auth[0].get("family") if auth else "") or "",
        "container": (m.get("container-title") or [""])[0],
        "volume": m.get("volume", ""),
        "page": m.get("page", "") or m.get("article-number", ""),
    }


def openalex(doi: str, mailto: str | None):
    q = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}"
    if mailto:
        q += f"?mailto={urllib.parse.quote(mailto)}"
    w = fetch_json(q)
    auths = w.get("authorships") or []
    fa = auths[0]["author"]["display_name"].split()[-1] if auths else ""
    return {
        "source": "openalex",
        "title": w.get("title") or "",
        "year": w.get("publication_year"),
        "first_author": fa,
        "container": ((w.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
        "volume": (w.get("biblio") or {}).get("volume", ""),
        "page": (w.get("biblio") or {}).get("first_page", ""),
    }


def tokens(s: str):
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP and len(t) > 2}


def compare(entry: str, meta: dict):
    """Return list of problems (empty = consistent)."""
    probs = []
    years = {int(y) for y in re.findall(r"\b((?:19|20)\d\d)\b", entry)}
    if meta["year"] and years and meta["year"] not in years:
        probs.append(f"year: entry {sorted(years)} vs {meta['source']} {meta['year']}")
    fa = meta["first_author"].lower()
    if fa and fa not in entry.lower():
        probs.append(f"first author '{meta['first_author']}' not in entry")
    tt, et = tokens(meta["title"]), tokens(entry)
    if tt:
        overlap = len(tt & et) / len(tt)
        if overlap < 0.6:
            probs.append(f"title overlap {overlap:.0%}: '{meta['title'][:80]}'")
    return probs


def run(path: Path, out: Path | None, mailto: str | None, offline: bool, sleep: float):
    text = path.read_text(errors="replace")
    rows = []
    for label, entry in split_entries(text):
        dois = [clean_doi(d) for d in DOI_RE.findall(entry)]
        if not dois:
            rows.append((label, "UNRESOLVED", "-", "no DOI in entry; check by hand / nature-ref-verifier", entry[:90]))
            continue
        doi = dois[0]
        if offline:
            rows.append((label, "DOI", doi, "offline", entry[:90]))
            continue
        meta, err = None, ""
        for fn in (crossref, openalex):
            try:
                meta = fn(doi, mailto)
                break
            except Exception as e:  # noqa: BLE001
                err = f"{fn.__name__}: {e}"
            time.sleep(sleep)
        time.sleep(sleep)
        if meta is None:
            rows.append((label, "NOT RESOLVED", doi, err, entry[:90]))
            continue
        probs = compare(entry, meta)
        status = "OK" if not probs else "CHECK"
        rows.append((label, status, doi, "; ".join(probs) or f"{meta['source']}: {meta['first_author']} {meta['year']}, {meta['container']}", entry[:90]))
    n = {s: sum(1 for r in rows if r[1] == s) for s in ("OK", "CHECK", "NOT RESOLVED", "UNRESOLVED", "DOI")}
    lines = [f"# Reference resolution: {path.name}", "", f"Entries: {len(rows)}; " + "; ".join(f"{k} {v}" for k, v in n.items() if v), "",
             "CHECK = metadata from the registry disagrees with the entry (year / first author / title words). "
             "NOT RESOLVED = DOI not found in Crossref or OpenAlex (possible fabricated or mistyped DOI). "
             "UNRESOLVED = entry has no DOI.", "",
             "| # | status | DOI | detail | entry |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    rep = "\n".join(lines) + "\n"
    if out:
        out.write_text(rep)
        print(f"wrote {out}")
    else:
        print(rep)
    return rows


def selftest():
    md = ("1. Chekroud AM et al. Illusory generalizability of clinical prediction models. Science 2024;383:164-167. "
          "doi:10.1126/science.adg8538\n2. Smith J. A paper with no doi. J Foo 2020.\n")
    rows = list(split_entries(md))
    assert len(rows) == 2 and rows[0][0] == "1", rows
    meta = {"source": "x", "title": "Illusory generalizability of clinical prediction models", "year": 2024,
            "first_author": "Chekroud", "container": "Science", "volume": "383", "page": "164"}
    assert compare(rows[0][1], meta) == [], compare(rows[0][1], meta)
    bad = dict(meta, year=2023, first_author="Jones", title="Something else entirely about cats")
    assert len(compare(rows[0][1], bad)) == 3
    bib = "@article{chek2024,\n title={Illusory generalizability},\n doi={10.1126/science.adg8538},\n year={2024}\n}\n"
    assert list(split_entries(bib))[0][0] == "chek2024"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs", nargs="?", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--mailto")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.refs:
        ap.error("refs file required")
    run(a.refs, a.out, a.mailto, a.offline, a.sleep)


if __name__ == "__main__":
    main()
