#!/usr/bin/env bash
# Monthly incremental refresh of the topic-forecast radar (WSL). Written 2026-11-04, NOT yet run.
#
# What it does
#   1. Re-fetches the latest COMPLETE publication year (OpenAlex keeps indexing late works) for totals/, ns/
#      and cooc/ of all 485 source topics (137 mechanical + 348 ext). If a newly completed year has appeared
#      (e.g. 2026 in January 2027) it is fetched as well. Partial current-year data is never added: model.py
#      treats the last year in totals/ as complete.
#      Superseded files are MOVED to <data>/_superseded/<date>/ (never deleted).
#      Cost per refreshed year: ~90 list requests for totals+ns + ~2,100 for cooc + ~650 for cooc_ns  ->  ~$0.3 (budget $1/day free).
#   2. Re-runs the primary radar model.py --ext --target ns (results/refresh/<date>/ns) and the control
#      model.py --ext (old co-occurrence target, results/refresh/<date>/control) (fixed origins: val T=2018, test T=2021, K3 at T=2020, final forecast at the last
#      year) into results/refresh/<date>/, and copies the refreshed aggregate data into data/openalex/.
#   3. Appends one row to results/refresh_log.csv: date, last data year, requests used, test AUC / P@100 of the
#      model and the best baseline, K1, K3 shares.
#   Rolling origin: model.py currently has fixed split years. If it later gains a --test-T option, this script
#   also scores every origin T = 2021 .. last_year-3 and logs one row per origin (detected via --help).
#
# Usage:  bash refresh_monthly.sh [--year YYYY] [--no-fetch]
#   --year     year to refresh (default: last complete year = current year - 1)
#   --no-fetch only rerun the model on the data already present
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA=${TF_DATA:-/mnt/d/audit_data/topic_forecast}
REPO_DATA="$HERE/data/openalex"
VENV=$HOME/tfvenv
TODAY=$(date +%F)
YEAR=$(( $(date +%Y) - 1 ))
FETCH=1
while [ $# -gt 0 ]; do
  case "$1" in
    --year) YEAR="$2"; shift 2 ;;
    --no-fetch) FETCH=0; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
if [ -z "${OPENALEX_API_KEY:-}" ]; then
  line="$(grep -h '^export OPENALEX_API_KEY=' "$HOME/.profile" "$HOME/.bashrc" 2>/dev/null | tail -1 || true)"
  [ -n "$line" ] && eval "$line"
fi
LOGF="$DATA/fetch_log.txt"
n_before=$(grep -c "" "$LOGF" 2>/dev/null || echo 0)

if [ "$FETCH" = 1 ]; then
  last_have=$(ls "$DATA/totals" | sed 's/\.json$//' | sort -n | tail -1)
  years="$YEAR"
  # a newly completed year beyond what we have (only if it is not the current, partial year)
  for y in $(seq $((last_have + 1)) "$YEAR"); do years="$years $y"; done
  years=$(echo $years | tr ' ' '\n' | sort -nu | tr '\n' ' ')
  for y in $years; do
    if [ "$y" -ge "$(date +%Y)" ]; then echo "skip partial year $y"; continue; fi
    bak="$DATA/_superseded/$TODAY/$y"
    for sub in totals ns; do
      if [ -f "$DATA/$sub/$y.json" ]; then mkdir -p "$bak/$sub"; mv "$DATA/$sub/$y.json" "$bak/$sub/"; fi
    done
    for sub in cooc cooc_ns; do if [ -d "$DATA/$sub/$y" ]; then mkdir -p "$bak/$sub"; mv "$DATA/$sub/$y" "$bak/$sub/"; fi; done
    echo "$(date '+%F %T') fetching year $y"
    # resumable: rerun after exit 3 (budget) until done
    for i in $(seq 1 48); do
      set +e; "$VENV/bin/python" "$HERE/fetch_openalex.py" --out "$DATA" --ext --ns-cooc --y0 "$y" --y1 "$y" --max-requests 9500; rc=$?; set -e
      [ "$rc" -eq 0 ] && break
      [ "$rc" -ne 3 ] && { echo "fetch failed rc=$rc"; exit "$rc"; }
      echo "$(date '+%F %T') budget used, retry in 30 min"; sleep 1800
    done
    [ "$rc" -eq 0 ] || { echo "fetch for $y still incomplete"; exit 3; }
  done
fi
n_req=$(( $(grep -c "" "$LOGF") - n_before ))

OUT="$HERE/results/refresh/$TODAY"
mkdir -p "$OUT"
"$VENV/bin/python" "$HERE/model.py" --data "$DATA" --out "$OUT/ns" --ext --target ns > "$OUT/ns.log" 2>&1
"$VENV/bin/python" "$HERE/model.py" --data "$DATA" --out "$OUT/control" --ext > "$OUT/control.log" 2>&1
ORIGINS=""
if "$VENV/bin/python" "$HERE/model.py" --help 2>/dev/null | grep -q -- "--test-T"; then
  last=$(ls "$DATA/totals" | sed 's/\.json$//' | sort -n | tail -1)
  for T in $(seq 2021 $((last - 3))); do
    "$VENV/bin/python" "$HERE/model.py" --data "$DATA" --out "$OUT/T$T" --ext --test-T "$T" > "$OUT/T$T.log" 2>&1
    ORIGINS="$ORIGINS $T"
  done
fi

# copy refreshed aggregates into the repo (small: ~50 MB)
mkdir -p "$REPO_DATA"
cp "$DATA"/topics_all.json "$DATA"/mech_topics.json "$DATA"/ns_sources.json "$DATA"/ext_topics.json "$REPO_DATA"/
for sub in totals ns cooc cooc_ns; do rm -rf "$REPO_DATA/$sub.new"; cp -r "$DATA/$sub" "$REPO_DATA/$sub.new"; \
  rm -rf "$REPO_DATA/$sub"; mv "$REPO_DATA/$sub.new" "$REPO_DATA/$sub"; done
find "$REPO_DATA" -name "*.tmp" -delete

"$VENV/bin/python" - "$HERE/results/refresh_log.csv" "$OUT" "$DATA" "$n_req" "$TODAY" $ORIGINS <<'EOF'
import csv, json, os, sys
log, out, data, n_req, today, origins = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:]
last_year = max(int(f[:-5]) for f in os.listdir(os.path.join(data, "totals")) if f.endswith(".json"))
rows = []
for d, origin in [(os.path.join(out, "ns"), "fixed_T2021_ns_primary"), (os.path.join(out, "control"), "fixed_T2021_cooc_control")] + [(os.path.join(out, f"T{t}"), f"T{t}") for t in origins]:
    m = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
    s = json.load(open(os.path.join(d, "summary.json")))
    test = [r for r in m if r["split"].startswith("test")]
    mod = next(r for r in test if r["scorer"] == "model")
    base = [r for r in test if r["scorer"] != "model"]
    best = max(base, key=lambda r: float(r["AUC"]))
    k3 = s.get("K3") or {}
    rows.append({"date": today, "origin": origin, "data_last_year": last_year, "requests": n_req,
                 "auc_model": round(float(mod["AUC"]), 4), "p100_model": mod["P@100"],
                 "auc_best_baseline": round(float(best["AUC"]), 4), "best_baseline": best["scorer"],
                 "p100_best_baseline": max(float(r["P@100"]) for r in base), "K1_pass": s.get("K1_pass"),
                 "K3_top10_model": k3.get("share_top10pct_model"), "K3_top10_PA": k3.get("share_top10pct_PA"),
                 "results_dir": os.path.relpath(d, os.path.dirname(log))})
new = not os.path.exists(log)
with open(log, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    if new:
        w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
EOF
echo "REFRESH DONE $(date '+%F %T')  requests=$n_req  results=$OUT"
