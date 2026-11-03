# tools/integrity — pre-submission integrity checks

Added 2026-11-03 after reviewing Google ScientistOne/ScientistTwo (arXiv 2605.26340, 2609.19644)
and the ARIS skill pack (wanshuiyin/Auto-claude-code-research-in-sleep). Three scripts, all
standard library only.

| script | what it does | when |
|---|---|---|
| `claim_check.py` | pulls every number out of a manuscript and looks for it in the result files; reports NOT FOUND / WEAK (only in raw or bootstrap dumps) / found in summary files | before every submission or resubmission |
| `ref_resolve.py` | resolves every DOI in the reference list via Crossref (OpenAlex fallback) and flags year / first-author / title mismatches and DOIs that do not resolve | before every submission; run on a machine with normal internet (the cloud container cannot reach Crossref/OpenAlex) |
| `script_audit.py` | builds a verbatim packet of the frozen analysis scripts + preregistration + methods text with a six-item checklist; three fresh reviewers answer in JSON; majority vote → AUDIT.md | after scripts are frozen, before kill tests are declared PASS, and again before submission |

## Reviewer rule (jeff 2026-11-03)

Reviews are done by **three fresh Claude Sonnet subagents** started from a Claude Code session.
Each gets only the packet path and the instruction to follow it; the executor never passes a
summary, an interpretation or a list of what changed (ARIS reviewer-independence protocol).
DeepSeek through `script_audit.py run --backend deepseek` is an optional route for machines
without Claude Code and needs `DEEPSEEK_API_KEY`.

## Typical sequence

```bash
python3 tools/integrity/claim_check.py paper/manuscript.md -r results/ -r frozen/ -o integrity/claim_check.md
python3 tools/integrity/ref_resolve.py paper/refs.md -o integrity/ref_resolve.md --mailto <your email>
python3 tools/integrity/script_audit.py build -s frozen/a.py -s frozen/b.py -p frozen/PLAN.md -m paper/methods.md -o integrity/audit
# spawn 3 Sonnet subagents: "Read integrity/audit/PACKET.md and follow it; write votes/vote_<k>.json"
python3 tools/integrity/script_audit.py collect -o integrity/audit
```

Each script has `--selftest`. `claim_check.py` finds numbers, it does not judge them: a NOT FOUND
needs a provenance sentence, and a match only in a 50,000-cell bootstrap dump is reported as WEAK
because almost any two-digit number occurs there by chance.
