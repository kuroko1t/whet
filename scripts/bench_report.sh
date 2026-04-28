#!/usr/bin/env bash
# Aggregate every benchmarks/results/results-*.jsonl into a Markdown leaderboard.
# Multi-run aware: groups by (model, task) and reports pass_count/n_runs plus
# averages over runs. Latest JSONL timestamp wins per (model, task) group.
# Usage: scripts/bench_report.sh [> leaderboard.md]
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/benchmarks/results"

if ! ls "$RESULTS_DIR"/results-*.jsonl >/dev/null 2>&1; then
    echo "# Whet Bench Leaderboard"
    echo
    echo "_No results yet. Run \`scripts/run_bench.sh -m <model>\`._"
    exit 0
fi

python3 - "$RESULTS_DIR" <<'PY'
import json, glob, os, sys
from collections import defaultdict

results_dir = sys.argv[1]
runs = []
for p in sorted(glob.glob(os.path.join(results_dir, "results-*.jsonl"))):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

# Each run has (model, task, run_index, ts). For each (model, task) keep only
# the runs from the most recent _batch_ — runs minted within 5 minutes of the
# latest ts. A single `-n N` invocation completes its runs within seconds, so
# this isolates the latest invocation while ignoring earlier (possibly
# different-version) batches that share the same 24-hour window.
import datetime as _dt

BATCH_WINDOW_S = 300  # 5 minutes

def parse_ts(t):
    return _dt.datetime.fromisoformat(t.replace("Z", "+00:00"))

latest_ts = {}
for r in runs:
    key = (r["model"], r["task"])
    ts = parse_ts(r["ts"])
    if key not in latest_ts or ts > latest_ts[key]:
        latest_ts[key] = ts

groups = defaultdict(list)
for r in runs:
    key = (r["model"], r["task"])
    ts = parse_ts(r["ts"])
    if (latest_ts[key] - ts).total_seconds() <= BATCH_WINDOW_S:
        groups[key].append(r)

models = sorted({k[0] for k in groups})
tasks = sorted({k[1] for k in groups})

def pareto_score(pass_rate, total_tokens, total_seconds, total_runs):
    """Higher is better. Combines correctness, token efficiency, and wall-clock speed.

    Score = pass_rate × tok_factor × time_factor where:
      tok_factor  = 1 / (1 + avg_tok / 10000)    # 5K tok → 0.67, 50K tok → 0.17
      time_factor = 1 / (1 + avg_sec / 120)      # 60s → 0.67, 300s → 0.29
    """
    if total_runs == 0:
        return 0.0
    avg_tok = total_tokens / total_runs
    avg_sec = total_seconds / total_runs
    tok_factor = 1.0 / (1.0 + avg_tok / 10000.0)
    time_factor = 1.0 / (1.0 + avg_sec / 120.0)
    return pass_rate * tok_factor * time_factor

print("# Whet Bench Leaderboard")
print()
print(f"_Aggregated from {len(runs)} run record(s); {len(groups)} (model, task) groups using the latest batch per pair._")
print()

# Per-model summary.
print("## Model summary")
print()
print("| Model | Pass rate | Tasks fully passed | Total tokens | Avg time (s) | Pareto score |")
print("|---|---|---|---|---|---|")
agg = defaultdict(lambda: dict(pass_=0, tries=0, fully=0, tasks=0, tok=0, dur=0.0))
for (m, t), grp in groups.items():
    a = agg[m]
    n = len(grp)
    p = sum(1 for r in grp if r["pass"])
    a["pass_"] += p
    a["tries"] += n
    a["tasks"] += 1
    if p == n:
        a["fully"] += 1
    a["tok"] += sum(r["total_tokens"] for r in grp)
    a["dur"] += sum(r["duration_s"] for r in grp)

ranked = []
for m in models:
    a = agg[m]
    rate = a["pass_"] / a["tries"] if a["tries"] else 0
    score = pareto_score(rate, a["tok"], a["dur"], a["tries"])
    ranked.append((m, a, rate, score))

ranked.sort(key=lambda x: x[3], reverse=True)
for m, a, rate, score in ranked:
    avg = a["dur"] / a["tries"] if a["tries"] else 0
    print(f"| `{m}` | {rate*100:.0f}% ({a['pass_']}/{a['tries']}) | {a['fully']}/{a['tasks']} | {a['tok']:,} | {avg:.1f} | **{score:.3f}** |")

# Per-task detail.
print()
print("## Per-task detail")
print()
for t in tasks:
    print(f"### {t}")
    print()
    print("| Model | Result | Avg tokens | Avg time (s) | Avg iters | Tool fail | Re-prompts | Text→tool |")
    print("|---|---|---|---|---|---|---|---|")
    for m in models:
        grp = groups.get((m, t), [])
        if not grp:
            print(f"| `{m}` | — | — | — | — | — | — | — |")
            continue
        n = len(grp)
        p = sum(1 for r in grp if r["pass"])
        if p == n:
            mark = f"PASS ({p}/{n})"
        elif p == 0:
            mark = f"**FAIL ({p}/{n})**"
        else:
            mark = f"_partial ({p}/{n})_"
        avg_tok = sum(r["total_tokens"] for r in grp) / n
        avg_dur = sum(r["duration_s"] for r in grp) / n
        avg_it = sum(r["llm_calls"] for r in grp) / n
        tf = sum(r["tool_calls_failed"] for r in grp)
        rp = sum(r["reprompts"] for r in grp)
        t2 = sum(r["text_to_tool_fallbacks"] for r in grp)
        print(f"| `{m}` | {mark} | {avg_tok:,.0f} | {avg_dur:.1f} | {avg_it:.1f} | {tf} | {rp} | {t2} |")
    print()
PY
