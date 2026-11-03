#!/usr/bin/env python3
"""script_audit.py - three-vote integrity audit of analysis scripts.

Borrowed from ScientistOne (audit I2 specification violation, I4 method-code
alignment) and ARIS (experiment-integrity checklist, reviewer-independence:
the reviewer gets raw files and a role, never the author's summary).

Two phases:

  build    Write one self-contained prompt packet (Markdown) that embeds the
           raw text of every script, plus the preregistration / method text if
           given, followed by the checklist and a strict JSON answer format.
           The packet is what each of the 3 reviewers reads. Nothing in it is
           written by the executor except file names.

  collect  Read the 3 reviewers' JSON answers, majority-vote each checklist
           item, and write AUDIT.md + AUDIT.json. An item is FAIL if >= 2
           reviewers say FAIL; WARN if >= 2 say WARN or FAIL but no FAIL
           majority; else PASS. Disagreements are listed, not hidden.

Reviewer backend (jeff 2026-11-03): three fresh **Claude Sonnet** subagents
spawned from a Claude Code session, each given ONLY the packet path and told to
write its answer to votes/vote_N.json. DeepSeek via `--backend deepseek`
(needs DEEPSEEK_API_KEY) is optional and kept for machines without Claude Code.

Usage:
    python3 script_audit.py build  -s a.py -s b.py [-p PREREG.md] [-m METHODS.md] -o audit_dir
    # then spawn 3 Sonnet subagents: "Read audit_dir/PACKET.md and follow it; write audit_dir/votes/vote_<k>.json"
    python3 script_audit.py collect -o audit_dir
    python3 script_audit.py run -s a.py ... -o audit_dir --backend deepseek   # optional API route
    python3 script_audit.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

CHECKLIST = [
    ("A_test_contamination", "Test / held-out data contamination",
     "Is any information from the evaluation set (labels, statistics, file identity, normalisation constants, "
     "early-stopping signal) used during training, model selection, threshold choice or feature design? "
     "Trace every variable that touches the held-out set."),
    ("B_split_unit", "Split by the right unit",
     "Are train and test split by the physical unit the claim is about (bearing, turbine, machine, drive, "
     "subject) rather than by sample / window / file, so that no unit appears on both sides? For bearing data "
     "(PU, CWRU, XJTU-SY) the split must be bearing-disjoint."),
    ("C_selection_rule", "Model / epoch / checkpoint selection rule",
     "Which epoch, seed, hyper-parameter or checkpoint is reported, and is the rule fixed BEFORE seeing test "
     "results? Flag 'best test epoch' reporting, seed cherry-picking and unreported sweeps."),
    ("D_metric_integrity", "Metric integrity",
     "Is any score normalised by the model's own output statistics, clipped, or computed on a filtered subset "
     "that removes hard cases? Is 'ground truth' loaded from the dataset, or derived from a model / the method "
     "itself? Is the evaluation type (real_gt / synthetic_proxy / simulation_only) clear from the code?"),
    ("E_prereg_alignment", "Alignment with preregistration / method description",
     "Does the code do what the preregistration or methods text says (same data, filter, estimator, kill-test "
     "thresholds, number of seeds)? List every deviation, even ones that look harmless."),
    ("F_silent_failures", "Silent failures and dead paths",
     "Are there try/except blocks that swallow errors, default values that hide missing data, functions defined "
     "but never called, or hard-coded numbers that should come from data?"),
]

ANSWER_SCHEMA = {
    "reviewer": "<model name you are>",
    "items": {k: {"status": "PASS|WARN|FAIL", "evidence": "file:line ...", "note": "one or two sentences"} for k, _, _ in CHECKLIST},
    "overall": "PASS|WARN|FAIL",
    "most_damaging_finding": "one sentence, or 'none'",
}


def build(scripts, prereg, methods, out: Path, max_chars: int):
    out.mkdir(parents=True, exist_ok=True)
    (out / "votes").mkdir(exist_ok=True)
    parts = ["# Script integrity audit packet", "",
             "You are an independent integrity auditor. Assume the evaluation is compromised somewhere and look for "
             "where. Read every file below yourself. Nothing here is a summary: the files are reproduced verbatim. "
             "Do not trust comments or docstrings; trust only what the code does. Cite file:line for every finding.", ""]
    if prereg:
        parts += ["## Preregistration / frozen plan (what the code is supposed to do)", "", "```", Path(prereg).read_text(errors="replace")[:max_chars], "```", ""]
    if methods:
        parts += ["## Methods text as written in the manuscript", "", "```", Path(methods).read_text(errors="replace")[:max_chars], "```", ""]
    parts += ["## Scripts under audit", ""]
    for s in scripts:
        txt = Path(s).read_text(errors="replace")
        if len(txt) > max_chars:
            txt = txt[:max_chars] + f"\n# [truncated at {max_chars} chars]\n"
        numbered = "\n".join(f"{i:4d}: {l}" for i, l in enumerate(txt.splitlines(), 1))
        parts += [f"### FILE {Path(s).name}", "", "```", numbered, "```", ""]
    parts += ["## Checklist", ""]
    for k, title, q in CHECKLIST:
        parts += [f"### {k}: {title}", "", q, ""]
    parts += ["## Answer format", "",
              "Reply with ONLY one JSON object (no prose before or after) of exactly this shape:", "",
              "```json", json.dumps(ANSWER_SCHEMA, indent=2, ensure_ascii=False), "```", "",
              "Rules: FAIL = a concrete defect that would change or invalidate a reported number; WARN = a practice "
              "that needs a disclosed caveat; PASS = checked and clean. Never answer PASS for an item you could not "
              "verify from the files; use WARN and say what is missing."]
    packet = out / "PACKET.md"
    packet.write_text("\n".join(parts) + "\n")
    manifest = {"scripts": [str(Path(s).resolve()) for s in scripts], "prereg": prereg, "methods": methods}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {packet} ({packet.stat().st_size} bytes). Now obtain 3 votes into {out / 'votes'}/vote_<k>.json")
    return packet


def parse_vote(p: Path):
    txt = p.read_text(errors="replace")
    start, end = txt.find("{"), txt.rfind("}")
    return json.loads(txt[start:end + 1])


def collect(out: Path):
    votes = sorted((out / "votes").glob("vote_*.json"))
    if len(votes) < 2:
        sys.exit(f"need at least 2 votes in {out / 'votes'}, found {len(votes)}")
    parsed = []
    for v in votes:
        try:
            parsed.append((v.name, parse_vote(v)))
        except Exception as e:  # noqa: BLE001
            print(f"skip {v.name}: {e}", file=sys.stderr)
    n = len(parsed)
    need = n // 2 + 1
    result, lines = {}, [f"# Script integrity audit ({n} votes, majority = {need})", ""]
    for k, title, _ in CHECKLIST:
        st = [p["items"].get(k, {}).get("status", "WARN").upper() for _, p in parsed]
        c = Counter(st)
        if c["FAIL"] >= need:
            verdict = "FAIL"
        elif c["FAIL"] + c["WARN"] >= need:
            verdict = "WARN"
        else:
            verdict = "PASS"
        result[k] = {"verdict": verdict, "votes": st}
        lines += [f"## {k}: {title} — **{verdict}** (votes: {', '.join(st)})", ""]
        for name, p in parsed:
            it = p["items"].get(k, {})
            lines.append(f"- {name} [{it.get('status', '?')}] {it.get('evidence', '')} — {it.get('note', '')}")
        lines.append("")
    overall = "FAIL" if any(r["verdict"] == "FAIL" for r in result.values()) else (
        "WARN" if any(r["verdict"] == "WARN" for r in result.values()) else "PASS")
    lines += ["## Most damaging finding per reviewer", ""]
    for name, p in parsed:
        lines.append(f"- {name} ({p.get('reviewer', '?')}): {p.get('most_damaging_finding', '')}")
    lines = [lines[0], "", f"**Overall: {overall}**", ""] + lines[1:]
    (out / "AUDIT.md").write_text("\n".join(lines) + "\n")
    (out / "AUDIT.json").write_text(json.dumps({"overall": overall, "items": result,
                                                "reviewers": [p.get("reviewer") for _, p in parsed]}, indent=2))
    print(f"overall {overall}; wrote {out / 'AUDIT.md'}")
    return overall


def call_openai_compatible(base: str, key: str, model: str, prompt: str, temperature: float):
    body = json.dumps({"model": model, "temperature": temperature,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def run_api(out: Path, backend: str, votes: int):
    packet = (out / "PACKET.md").read_text()
    if backend == "deepseek":
        base, key, model = "https://api.deepseek.com", os.environ.get("DEEPSEEK_API_KEY"), "deepseek-chat"
    else:
        sys.exit("only --backend deepseek is implemented for the API route; use Sonnet subagents otherwise")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")
    for k in range(1, votes + 1):
        ans = call_openai_compatible(base, key, model, packet, temperature=0.7)
        (out / "votes" / f"vote_{k}.json").write_text(ans)
        print(f"vote {k} saved")


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "a.py").write_text("import numpy as np\nX_test = load()\nthr = X_test.mean()\n")
        out = d / "audit"
        build([d / "a.py"], None, None, out, 20000)
        assert "FILE a.py" in (out / "PACKET.md").read_text()
        v = {"reviewer": "t", "items": {k: {"status": "PASS", "evidence": "", "note": ""} for k, _, _ in CHECKLIST},
             "overall": "PASS", "most_damaging_finding": "none"}
        v1 = json.loads(json.dumps(v)); v1["items"]["A_test_contamination"]["status"] = "FAIL"
        v2 = json.loads(json.dumps(v)); v2["items"]["A_test_contamination"]["status"] = "FAIL"; v2["items"]["C_selection_rule"]["status"] = "WARN"
        for i, vv in enumerate((v, v1, v2), 1):
            (out / "votes" / f"vote_{i}.json").write_text("```json\n" + json.dumps(vv) + "\n```")
        assert collect(out) == "FAIL"
        res = json.loads((out / "AUDIT.json").read_text())
        assert res["items"]["A_test_contamination"]["verdict"] == "FAIL"
        assert res["items"]["C_selection_rule"]["verdict"] == "PASS"
        print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", nargs="?", choices=["build", "collect", "run"])
    ap.add_argument("-s", "--script", action="append", default=[])
    ap.add_argument("-p", "--prereg")
    ap.add_argument("-m", "--methods")
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--max-chars", type=int, default=60000)
    ap.add_argument("--backend", default="subagent")
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.phase or not a.out:
        ap.error("phase and -o are required")
    if a.phase == "build":
        build(a.script, a.prereg, a.methods, a.out, a.max_chars)
    elif a.phase == "collect":
        collect(a.out)
    else:
        build(a.script, a.prereg, a.methods, a.out, a.max_chars)
        run_api(a.out, a.backend, a.votes)
        collect(a.out)


if __name__ == "__main__":
    main()
