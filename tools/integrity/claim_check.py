#!/usr/bin/env python3
"""claim_check.py - trace every number in a manuscript back to a result file.

Deterministic, no model call. Borrowed idea: ScientistOne's "Ground" step and
ARIS's evidence_check.py (number must appear in the cited artifact). This
script works the other way round, AI-first: it does not need a hand-written
claims list. It pulls every numeric token out of the manuscript, drops the
ones that are obviously not results (DOIs, years, citation numbers, figure /
table / section numbers, equation labels), and then searches every result
file under the given directories for a value that rounds to the same number
at the manuscript's displayed precision (also trying x100 for percentages and
percentage points, and x1000 / /1000 for unit changes).

Output: a Markdown report listing, per number, where it was found (file and
column / line) or NOT FOUND. A NOT FOUND is a lead, not a verdict: the number
may come from a figure, from the literature, or from arithmetic on two
results. Those still need a sentence of provenance in the methods or a line
in the claims table; the point is that nothing reaches a submission unseen.

Usage:
    python3 claim_check.py MANUSCRIPT.md -r results_dir [-r more_dir] [-o report.md]
                           [--min-digits 2] [--context 40]
    python3 claim_check.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

NUM_RE = re.compile(r"(?<![\w./-])([+\-−]?\d+(?:[.,]\d+)*)(\s?%|\s?pp)?(?![\w/])")
DOI_RE = re.compile(r"10\.\d{4,9}/\S+")
YEAR_RE = re.compile(r"^(19|20)\d\d$")
# contexts whose following number is a label, not a result
LABEL_BEFORE = re.compile(
    r"\b(?:Fig|Figure|Table|Tab|Section|Sec|Eq|Equation|Ref|Appendix|Supplementary|Supp|Note|S|T|WT|turbine|ref|step|phase|part|chapter|p|pp|page|vol|no|issue|arXiv)\.?\s*$",
    re.IGNORECASE,
)
CITATION_BEFORE = re.compile(r"[\[(]\s*$")  # "[12]" / "(3)" integers are citations or labels
MANUSCRIPT_SKIP_LINE = re.compile(r"^\s*(\d+\.\s|\[\d+\]|\|?\s*#\s*\|)")  # reference list / tables of refs


def norm_number(tok: str) -> float | None:
    t = tok.replace("−", "-").replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def displayed_decimals(tok: str) -> int:
    t = tok.replace(",", "")
    return len(t.split(".")[1]) if "." in t else 0


def extract_manuscript_numbers(text: str, min_digits: int, context: int):
    """Yield (token, value, decimals, suffix, context_snippet)."""
    seen = set()
    in_refs = False
    for line in text.splitlines():
        if re.match(r"^#+\s*(References|Bibliography|参考文献)", line, re.IGNORECASE):
            in_refs = True
        elif re.match(r"^#+\s", line):
            in_refs = False
        if in_refs or MANUSCRIPT_SKIP_LINE.match(line) and "|" not in line:
            continue
        clean = DOI_RE.sub(" ", line)
        if clean.lstrip().startswith("#"):
            clean = re.sub(r"^\s*#+\s*[\d.]+", " ", clean)
        for m in NUM_RE.finditer(clean):
            tok, suffix = m.group(1), (m.group(2) or "").strip()
            before = clean[max(0, m.start() - 14):m.start()]
            if LABEL_BEFORE.search(before):
                continue
            if CITATION_BEFORE.search(before) and "." not in tok and suffix == "":
                continue
            digits = re.sub(r"\D", "", tok)
            if YEAR_RE.match(digits) and "." not in tok and suffix == "":
                continue
            if len(digits) < min_digits and "." not in tok and suffix == "":
                continue
            val = norm_number(tok)
            if val is None:
                continue
            key = (tok, suffix)
            if key in seen:
                continue
            seen.add(key)
            lo, hi = max(0, m.start() - context), min(len(clean), m.end() + context)
            yield tok, val, displayed_decimals(tok), suffix, clean[lo:hi].strip()


def load_result_values(dirs, exts=(".csv", ".json", ".md", ".txt", ".log", ".tsv"), exclude=(), raw_threshold=400):
    """Return list of (value, file, where) for every numeric cell in result files."""
    out = []
    exclude = {Path(e).resolve() for e in exclude}
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if not p.is_file() or p.suffix.lower() not in exts or p.stat().st_size > 50_000_000:
                continue
            if p.resolve() in exclude:
                continue
            try:
                txt = p.read_text(errors="replace")
            except OSError:
                continue
            if p.suffix.lower() in (".csv", ".tsv"):
                delim = "\t" if p.suffix.lower() == ".tsv" else ","
                rows = list(csv.reader(txt.splitlines(), delimiter=delim))
                if not rows:
                    continue
                header = rows[0]
                for i, row in enumerate(rows[1:], start=2):
                    label = row[0] if row else ""
                    for j, cell in enumerate(row):
                        v = norm_number(cell.strip())
                        if v is not None and not math.isnan(v):
                            col = header[j] if j < len(header) else f"col{j}"
                            out.append((v, str(p), f"row {i} ({label}) col {col}"))
            elif p.suffix.lower() == ".json":
                try:
                    obj = json.loads(txt)
                except json.JSONDecodeError:
                    continue

                def walk(o, path):
                    if isinstance(o, dict):
                        for k, v in o.items():
                            walk(v, f"{path}.{k}")
                    elif isinstance(o, list):
                        for k, v in enumerate(o):
                            walk(v, f"{path}[{k}]")
                    elif isinstance(o, (int, float)) and not isinstance(o, bool):
                        out.append((float(o), str(p), path.lstrip(".")))

                walk(obj, "")
            else:
                for i, line in enumerate(txt.splitlines(), start=1):
                    for m in NUM_RE.finditer(DOI_RE.sub(" ", line)):
                        v = norm_number(m.group(1))
                        if v is not None:
                            out.append((v, str(p), f"line {i}"))
    # tier: a file with many numeric cells (bootstrap draws, per-record dumps)
    # will contain almost any 2-digit number by chance; matches there are weak
    counts = {}
    for _, f, _ in out:
        counts[f] = counts.get(f, 0) + 1
    return [(v, f, w, "summary" if counts[f] <= raw_threshold else "raw") for v, f, w in out]


def candidates(val: float, suffix: str):
    """Scale factors to try: as-is, percent<->fraction, per-mille, thousands."""
    facs = [1.0]
    if suffix in ("%", "pp") or abs(val) >= 1:
        facs.append(0.01)
    facs += [100.0, 0.001, 1000.0]
    return [(val * f, f) for f in facs]


def match(val, decimals, suffix, pool):
    """Find pool entries that round to val at the displayed precision."""
    hits = []
    for target, f in candidates(val, suffix):
        # tolerance: half a unit of the last displayed digit, scaled
        dec = decimals + (2 if f == 0.01 else 0) + (3 if f == 0.001 else 0)
        tol = 0.5 * 10 ** (-dec) * 1.0000001
        for v, fname, where, tier in pool:
            if abs(v - target) <= tol:
                hits.append((f, fname, where, v, tier))
    # summary files first, then exact scale, then dedupe by file
    hits.sort(key=lambda h: (h[4] != "summary", h[0] != 1.0, h[1]))
    seen, uniq = set(), []
    for h in hits:
        if h[1] not in seen:
            seen.add(h[1])
            uniq.append(h)
    return uniq


def run(manuscript: Path, dirs, out: Path | None, min_digits: int, context: int, max_hits: int):
    text = manuscript.read_text(errors="replace")
    pool = load_result_values(dirs, exclude=[manuscript])
    rows, n_found, n_weak = [], 0, 0
    for tok, val, dec, suffix, ctx in extract_manuscript_numbers(text, min_digits, context):
        hits = match(val, dec, suffix, pool)
        if hits and hits[0][4] == "summary":
            n_found += 1
        elif hits:
            n_weak += 1
        rows.append((tok + (suffix and " " + suffix), ctx, hits))
    lines = [f"# Claim check: {manuscript.name}", "",
             f"Result dirs: {', '.join(str(d) for d in dirs)}  ",
             f"Numeric cells indexed: {len(pool)}  ",
             f"Manuscript numbers checked: {len(rows)}; found in summary files: {n_found}; "
             f"WEAK (only in raw/bootstrap files, likely chance): {n_weak}; "
             f"NOT FOUND: {len(rows) - n_found - n_weak}", "",
             "Matching rule: result value rounds to the manuscript value at its displayed precision; "
             "scale factors tried: x1, x0.01 (percent/pp), x100, x0.001, x1000. "
             "NOT FOUND and WEAK = need a provenance note (figure, literature, derived arithmetic, raw count) or a correction.", "",
             "## NOT FOUND", "", "| number | context |", "|---|---|"]
    for tok, ctx, hits in rows:
        if not hits:
            lines.append(f"| {tok} | …{ctx.replace('|', '/')}… |")
    lines += ["", "## WEAK (only in raw / bootstrap files)", "", "| number | context | first match |", "|---|---|---|"]
    for tok, ctx, hits in rows:
        if hits and hits[0][4] != "summary":
            f, fname, where, v, _ = hits[0]
            lines.append(f"| {tok} | …{ctx.replace('|', '/')}… | {Path(fname).name}: {where} = {v:g} |")
    lines += ["", "## Found in summary files", "", "| number | context | source (first match) |", "|---|---|---|"]
    for tok, ctx, hits in rows:
        if hits and hits[0][4] == "summary":
            f, fname, where, v, _ = hits[0]
            scale = "" if f == 1.0 else f" (x{f:g})"
            extra = f" +{len(hits) - 1} more files" if len(hits) > 1 else ""
            lines.append(f"| {tok} | …{ctx.replace('|', '/')}… | {Path(fname).name}: {where} = {v:g}{scale}{extra} |")
    report = "\n".join(lines) + "\n"
    if out:
        out.write_text(report)
        print(f"wrote {out}")
    else:
        print(report)
    return rows


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "r.csv").write_text("quantity,estimate,lo,hi\nE,0.048095,0.0329,0.0625\nC,0.4772,0.388,0.539\n")
        (d / "m.md").write_text(
            "We find E = 4.8% [3.3, 6.2] and C ≈ 0.48 (Fig. 3, Table 2, ref [12], 2023). "
            "The missing value 7.77 is not in any file. DOI 10.1016/j.renene.2023.04.104.\n"
            "## References\n1. Foo 2020, vol 12, 3.14\n")
        rows = run(d / "m.md", [d], None, 2, 20, 3)
        got = {tok: bool(h) for tok, _, h in rows}
        assert got.get("4.8 %") is True, got
        assert got.get("3.3") is True and got.get("6.2") is True, got
        assert got.get("0.48") is True, got
        assert got.get("7.77") is False, got
        assert "3.14" not in got and "12" not in got, got
        print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manuscript", nargs="?", type=Path)
    ap.add_argument("-r", "--results", action="append", type=Path, default=[])
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--min-digits", type=int, default=2, help="ignore bare integers shorter than this")
    ap.add_argument("--context", type=int, default=40)
    ap.add_argument("--max-hits", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.manuscript or not a.results:
        ap.error("manuscript and at least one -r results dir are required")
    run(a.manuscript, a.results, a.out, a.min_digits, a.context, a.max_hits)


if __name__ == "__main__":
    main()
