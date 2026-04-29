#!/usr/bin/env bash
# Run the Whet benchmark suite for one or more models.
#
# Usage:
#   scripts/run_bench.sh -m <model[,model2,...]> [-t <task[,task2,...]>] [-n N] [-w <whet_bin>]
#
#   -n N : run each (model, task) N times to expose stochastic variance.
#          Default: 1. Recommended: 3 for headline comparisons.
#
# Each benchmark is expected to contain:
#   prompt.txt   — instruction passed to whet via -p
#   verify.sh    — exits 0 on pass, non-zero on fail (cwd is the workspace copy)
#   workspace/   — initial state copied for each run
#
# Results are appended to benchmarks/results/results-<timestamp>.jsonl
# and a Markdown leaderboard is written to benchmarks/results/leaderboard.md.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$REPO_ROOT/benchmarks"
RESULTS_DIR="$BENCH_DIR/results"
# Prefer the freshly built local binary so benchmarks see the current source.
if [ -z "${WHET_BIN:-}" ] && [ -x "$REPO_ROOT/target/release/whet" ]; then
    WHET_BIN="$REPO_ROOT/target/release/whet"
else
    WHET_BIN="${WHET_BIN:-whet}"
fi
MODELS=""
TASKS=""
RUNS=1

while getopts "m:t:n:w:h" opt; do
    case "$opt" in
        m) MODELS="$OPTARG" ;;
        t) TASKS="$OPTARG" ;;
        n) RUNS="$OPTARG" ;;
        w) WHET_BIN="$OPTARG" ;;
        h|*)
            sed -n '2,20p' "$0"
            exit 0
            ;;
    esac
done

case "$RUNS" in
    ''|*[!0-9]*) echo "ERROR: -n must be a positive integer" >&2; exit 1 ;;
esac
if [ "$RUNS" -lt 1 ]; then
    echo "ERROR: -n must be >= 1" >&2
    exit 1
fi

if [ -z "$MODELS" ]; then
    echo "ERROR: -m <model[,model2,...]> is required" >&2
    exit 1
fi

if [ ! -x "$WHET_BIN" ] && ! command -v "$WHET_BIN" >/dev/null 2>&1; then
    echo "ERROR: whet binary '$WHET_BIN' not found" >&2
    exit 1
fi
echo "Using whet: $WHET_BIN"

# Default: every benchmarks/<dir> that contains prompt.txt + verify.sh + workspace/
if [ -z "$TASKS" ]; then
    TASKS=$(find "$BENCH_DIR" -mindepth 1 -maxdepth 1 -type d \
        -exec test -f '{}/prompt.txt' \; -print \
        | xargs -I{} basename {} \
        | sort \
        | paste -sd, -)
fi

mkdir -p "$RESULTS_DIR"
TS=$(date -u +"%Y%m%dT%H%M%SZ")
JSONL="$RESULTS_DIR/results-$TS.jsonl"
echo "Results: $JSONL"

IFS=',' read -ra MODEL_ARR <<<"$MODELS"
IFS=',' read -ra TASK_ARR <<<"$TASKS"

run_one() {
    local model="$1" task="$2" run_index="$3"
    local task_dir="$BENCH_DIR/$task"
    if [ ! -f "$task_dir/prompt.txt" ] || [ ! -x "$task_dir/verify.sh" ] || [ ! -d "$task_dir/workspace" ]; then
        echo "  SKIP $task (missing prompt.txt / verify.sh / workspace/)"
        return
    fi

    # Two separate dirs: `run_dir` is the agent's workspace (only the task files);
    # `logs_dir` is a sibling that holds .stats.log, .stdout.log, .verify.log so a
    # recursive `grep .` inside verify.sh doesn't pick up Whet's tool-call traces
    # as task content.
    local run_dir logs_dir
    run_dir=$(mktemp -d -t "whet-bench-${task}-r${run_index}-XXXXXX")
    logs_dir="${run_dir}.logs"
    mkdir -p "$logs_dir"
    cp -r "$task_dir/workspace/." "$run_dir/"

    local stats_log="$logs_dir/stats.log"
    local stdout_log="$logs_dir/stdout.log"
    local verify_log="$logs_dir/verify.log"
    local events_jsonl="$logs_dir/events.jsonl"
    local prompt
    prompt=$(cat "$task_dir/prompt.txt")

    local t0 t1 dur whet_rc verify_rc
    t0=$(date +%s.%N)
    (
        cd "$run_dir" \
            && WHET_STATS_JSONL="$events_jsonl" \
               "$WHET_BIN" -y -m "$model" -p "$prompt" >"$stdout_log" 2>"$stats_log"
    )
    whet_rc=$?
    t1=$(date +%s.%N)
    dur=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')

    (
        cd "$run_dir" && "$task_dir/verify.sh" >"$verify_log" 2>&1
    )
    verify_rc=$?

    # Parse Whet session stats from stderr (best-effort; missing fields → 0).
    local llm_calls prompt_tokens completion_tokens tool_ok tool_failed t2t reprompts
    llm_calls=$(awk -F: '/LLM calls:/ {gsub(/ /, "", $2); print $2; exit}' "$stats_log")
    prompt_tokens=$(awk -F: '/Prompt tokens:/ {gsub(/ /, "", $2); print $2; exit}' "$stats_log")
    completion_tokens=$(awk -F: '/Completion tokens:/ {gsub(/ /, "", $2); print $2; exit}' "$stats_log")
    tool_ok=$(awk '/Tool calls:/ {match($0, /\(([0-9]+) ok/, a); print a[1]; exit}' "$stats_log")
    tool_failed=$(awk '/Tool calls:/ {match($0, /([0-9]+) failed/, a); print a[1]; exit}' "$stats_log")
    t2t=$(awk -F: '/Text->tool fallbacks:/ {gsub(/ /, "", $2); print $2; exit}' "$stats_log")
    reprompts=$(awk -F: '/Re-prompts:/ {gsub(/ /, "", $2); print $2; exit}' "$stats_log")

    : "${llm_calls:=0}" "${prompt_tokens:=0}" "${completion_tokens:=0}"
    : "${tool_ok:=0}" "${tool_failed:=0}" "${t2t:=0}" "${reprompts:=0}"

    local pass="false"
    if [ "$verify_rc" -eq 0 ]; then pass="true"; fi

    # Emit one JSON line. Use python for safe escaping (avoids quoting issues).
    python3 - "$model" "$task" "$run_index" "$pass" "$whet_rc" "$verify_rc" "$dur" \
        "$llm_calls" "$prompt_tokens" "$completion_tokens" \
        "$tool_ok" "$tool_failed" "$t2t" "$reprompts" "$run_dir" "$logs_dir" <<'PY' >>"$JSONL"
import json, sys, datetime
m, task, run_idx, p, wrc, vrc, dur, lc, pt, ct, tok, tf, t2, rp, rd, ld = sys.argv[1:]
print(json.dumps({
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "model": m,
    "task": task,
    "run_index": int(run_idx),
    "pass": p == "true",
    "whet_exit": int(wrc),
    "verify_exit": int(vrc),
    "duration_s": float(dur),
    "llm_calls": int(lc),
    "prompt_tokens": int(pt),
    "completion_tokens": int(ct),
    "total_tokens": int(pt) + int(ct),
    "tool_calls_ok": int(tok),
    "tool_calls_failed": int(tf),
    "text_to_tool_fallbacks": int(t2),
    "reprompts": int(rp),
    "run_dir": rd,
    "logs_dir": ld,
}))
PY

    local mark="FAIL"
    [ "$pass" = "true" ] && mark="PASS"
    printf "  %-22s %-18s [run %d/%d] %s  %ss  %s tok\n" \
        "$model" "$task" "$run_index" "$RUNS" "$mark" "$dur" "$((prompt_tokens + completion_tokens))"
}

export JSONL BENCH_DIR

# Make sure none of the candidate models are loaded before we start.
# This prevents WSL OOM when stale models from a previous run are still
# resident in RAM/VRAM when ollama tries to load the new one on top.
for m in "${MODEL_ARR[@]}"; do
    ollama stop "$m" >/dev/null 2>&1 || true
done

for model in "${MODEL_ARR[@]}"; do
    # Unload every other candidate before the batch so only one model is
    # ever resident at a time.
    for other in "${MODEL_ARR[@]}"; do
        [ "$other" = "$model" ] && continue
        ollama stop "$other" >/dev/null 2>&1 || true
    done

    echo "==> Model: $model  (n=$RUNS)"
    for task in "${TASK_ARR[@]}"; do
        i=1
        while [ "$i" -le "$RUNS" ]; do
            run_one "$model" "$task" "$i"
            i=$((i + 1))
        done
    done
done

# Final cleanup: leave nothing loaded so the user's GPU/RAM is free.
for m in "${MODEL_ARR[@]}"; do
    ollama stop "$m" >/dev/null 2>&1 || true
done

echo
echo "Done. JSONL: $JSONL"

# Regenerate Markdown leaderboard from all JSONL files.
"$REPO_ROOT/scripts/bench_report.sh" >"$RESULTS_DIR/leaderboard.md"
echo "Leaderboard: $RESULTS_DIR/leaderboard.md"
