"""Analyze every JSONL + stats.log under benchmarks/results and /tmp.

Used by the blog post research. Outputs:
  - failure-modes.csv   : per-run failure classification
  - tool-usage.csv      : per (model, task) tool call counts
  - quant-sweep.md      : qwen3.6 quantization comparison summary
  - timeline.md         : Whet improvement timeline (before/after metrics)

Run from the repo root:
    python3 scripts/analyze_bench.py
"""
import json
import re
import os
import glob
import csv
from collections import defaultdict, Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "benchmarks" / "results"
OUT_DIR = REPO / "benchmarks" / "results" / "analysis"
OUT_DIR.mkdir(exist_ok=True)


def classify_failure(rec):
    """Return a short tag describing the run's outcome."""
    if rec["pass"]:
        return "pass"

    run_dir = Path(rec.get("run_dir", ""))
    logs_dir = Path(str(run_dir) + ".logs")
    verify_log = logs_dir / "verify.log" if logs_dir.is_dir() else run_dir / ".verify.log"
    stats_log = logs_dir / "stats.log" if logs_dir.is_dir() else run_dir / ".stats.log"
    stdout_log = logs_dir / "stdout.log" if logs_dir.is_dir() else run_dir / ".stdout.log"

    verify_text = verify_log.read_text(errors="replace") if verify_log.exists() else ""
    stats_text = stats_log.read_text(errors="replace") if stats_log.exists() else ""
    stdout_text = stdout_log.read_text(errors="replace") if stdout_log.exists() else ""

    # Ollama runtime failures (unloaded mid-run, OOM, etc.)
    if "model runner has unexpectedly stopped" in stdout_text:
        return "ollama_runner_crash"
    if "Internal Server Error" in stdout_text and rec["whet_exit"] != 0:
        return "ollama_500"

    # Verifier-detected tampering
    if "was modified" in verify_text and ("test" in verify_text or "init_db" in verify_text):
        return "test_tampering_caught"
    if "package.json was modified" in verify_text:
        return "npm_install_side_effect"
    if "tsconfig.json was modified" in verify_text:
        return "tsconfig_modified"

    # Domain-specific verifier hints
    if "SQL injection still works" in verify_text:
        return "missed_security_fix"
    if "duplicated guard" in verify_text or "duplicated rounding" in verify_text:
        return "incomplete_dedup"
    if "tests still pass against a mutated" in verify_text:
        return "weak_tests"  # tests didn't catch our mutation

    # Test-count gates
    if "expected >=5 collected tests" in verify_text:
        return "missing_new_test"

    # Behavioural assertions
    if "farewell was not added" in verify_text:
        return "no_edit_attempt"
    if "function should be renamed" in verify_text:
        return "incomplete_rename"
    if "missing endpoint" in verify_text:
        return "incomplete_enumeration"
    if "compute' still referenced" in verify_text or "compute' still" in verify_text:
        return "incomplete_rename"
    if "recieve' still present" in verify_text:
        return "incomplete_typo_fix"

    # apply_diff anchor failures (tool-level)
    if "Could not locate hunk anchored" in stats_text:
        return "apply_diff_anchor_miss"

    # Generic patterns
    n_iter = rec.get("llm_calls", 0)
    n_tools_total = rec.get("tool_calls_ok", 0) + rec.get("tool_calls_failed", 0)
    n_tools_failed = rec.get("tool_calls_failed", 0)
    completion_tok = rec.get("completion_tokens", 0)

    # Lots of tool failures suggests the edit tool kept rejecting (whitespace mismatches)
    if n_tools_failed >= 3:
        return "edit_tool_thrash"

    # Hit max iterations without succeeding
    if n_iter >= 9:
        return "max_iterations"

    # Read but did not edit (early give-up)
    has_edits = bool(re.search(r"\[tool: (edit_file|apply_diff|write_file)\]", stats_text))
    if not has_edits and n_iter <= 4:
        return "early_giveup"

    # Edited but the verifier still failed for an unclassified reason
    if has_edits:
        return "wrong_solution"

    return "other_fail"


def parse_tool_calls(stats_text):
    """Return list of tool names invoked, in order."""
    return re.findall(r"\[tool: (\w+)\]", stats_text)


def main():
    runs = []
    for jp in sorted(RESULTS_DIR.glob("results-*.jsonl")):
        for line in jp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["_jsonl"] = jp.name
            runs.append(rec)

    print(f"Loaded {len(runs)} runs across {len(list(RESULTS_DIR.glob('results-*.jsonl')))} JSONL files")

    # 1. Failure classification CSV
    out = OUT_DIR / "failure-modes.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ts", "model", "task", "run_index", "pass",
            "duration_s", "total_tokens", "llm_calls",
            "tool_ok", "tool_failed", "category",
        ])
        cats = Counter()
        for r in runs:
            cat = classify_failure(r)
            cats[cat] += 1
            w.writerow([
                r["ts"], r["model"], r["task"], r["run_index"], r["pass"],
                f"{r['duration_s']:.1f}", r["total_tokens"], r["llm_calls"],
                r.get("tool_calls_ok", 0), r.get("tool_calls_failed", 0),
                cat,
            ])
    print(f"\n== failure-modes.csv → {out} ==")
    for cat, n in cats.most_common():
        print(f"  {cat:<28} {n}")

    # 2. Tool usage histogram per (model, task), latest batch only
    # Use the same 5-min latest-batch logic as bench_report.sh
    import datetime as dt
    def parse_ts(t):
        return dt.datetime.fromisoformat(t.replace("Z", "+00:00"))

    latest_ts = {}
    for r in runs:
        key = (r["model"], r["task"])
        ts = parse_ts(r["ts"])
        if key not in latest_ts or ts > latest_ts[key]:
            latest_ts[key] = ts

    tool_usage = defaultdict(lambda: defaultdict(int))   # tool_usage[(model, task)][tool] = count
    for r in runs:
        key = (r["model"], r["task"])
        if (latest_ts[key] - parse_ts(r["ts"])).total_seconds() > 300:
            continue
        run_dir = Path(r.get("run_dir", ""))
        logs_dir = Path(str(run_dir) + ".logs")
        stats = (logs_dir / "stats.log") if logs_dir.is_dir() else (run_dir / ".stats.log")
        if not stats.exists():
            continue
        for tool in parse_tool_calls(stats.read_text(errors="replace")):
            tool_usage[key][tool] += 1

    out = OUT_DIR / "tool-usage.csv"
    with out.open("w", newline="") as f:
        all_tools = sorted({t for d in tool_usage.values() for t in d})
        w = csv.writer(f)
        w.writerow(["model", "task"] + all_tools)
        for (m, t), counts in sorted(tool_usage.items()):
            w.writerow([m, t] + [counts.get(tool, 0) for tool in all_tools])
    print(f"\n== tool-usage.csv → {out} ({len(tool_usage)} groups) ==")

    # 3. Quant sweep summary for qwen3.6-family models
    quant_models = {
        "qwen3.6:35b-a3b-q4_K_M": "Q4_K_M (23GB, KV ?)",
        "qwen3.6-q3":              "UD-Q3_K_M (15GB)",
    }
    summary = defaultdict(lambda: dict(pass_=0, n=0, dur=0.0, tok=0))
    for r in runs:
        if r["model"] not in quant_models:
            continue
        key = (r["model"], r["task"])
        if (latest_ts[key] - parse_ts(r["ts"])).total_seconds() > 300:
            continue
        s = summary[r["model"], r["task"]]
        s["n"] += 1
        if r["pass"]:
            s["pass_"] += 1
        s["dur"] += r["duration_s"]
        s["tok"] += r["total_tokens"]

    out = OUT_DIR / "quant-sweep.md"
    with out.open("w") as f:
        f.write("# Qwen3.6-35B-A3B quantization sweep (latest batch per model+task)\n\n")
        f.write("| Model | Task | Pass | Avg time (s) | Avg tokens |\n")
        f.write("|---|---|---|---|---|\n")
        for (m, t), s in sorted(summary.items()):
            avg_t = s["dur"] / s["n"] if s["n"] else 0
            avg_k = s["tok"] / s["n"] if s["n"] else 0
            f.write(f"| `{m}` | {t} | {s['pass_']}/{s['n']} | {avg_t:.1f} | {avg_k:,.0f} |\n")
    print(f"\n== quant-sweep.md → {out} ==")

    # 4. Per-model summary on the latest batch
    print("\n== per-model headline (latest batch) ==")
    by_model = defaultdict(lambda: dict(pass_=0, n=0, fully=0, tasks=0, dur=0.0, tok=0))
    by_model_task = defaultdict(list)
    for r in runs:
        key = (r["model"], r["task"])
        if (latest_ts[key] - parse_ts(r["ts"])).total_seconds() > 300:
            continue
        by_model_task[(r["model"], r["task"])].append(r)
    for (m, t), grp in by_model_task.items():
        s = by_model[m]
        s["tasks"] += 1
        n = len(grp)
        p = sum(1 for r in grp if r["pass"])
        s["n"] += n
        s["pass_"] += p
        if p == n:
            s["fully"] += 1
        s["dur"] += sum(r["duration_s"] for r in grp)
        s["tok"] += sum(r["total_tokens"] for r in grp)
    for m, s in sorted(by_model.items()):
        rate = s["pass_"] / s["n"] * 100 if s["n"] else 0
        avg_t = s["dur"] / s["n"] if s["n"] else 0
        print(f"  {m:<32} pass {s['pass_']}/{s['n']} ({rate:.0f}%)  fully {s['fully']}/{s['tasks']}  avg {avg_t:.1f}s  total {s['tok']:,} tok")


if __name__ == "__main__":
    main()
