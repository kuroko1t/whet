"""Generate a per-config quant sweep table by mapping JSONL files to their
ollama / KV cache configuration. Used by the blog post."""
import json
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "benchmarks" / "results"

# JSONL → (config_label, batch_role)
# Compiled from session history.
JSONL_CONFIG = {
    # First baseline: q4_K_M with KV f16, original verify infra (some polluted)
    "results-20260427T185607Z.jsonl": ("q4_K_M + KV f16", "original"),
    # Tasks 8 + verify-fix + apply_diff iterations
    "results-20260428T072504Z.jsonl": ("q4_K_M + KV f16", "task8 baseline"),
    "results-20260428T093818Z.jsonl": ("q4_K_M + KV f16", "post apply_diff fix"),
    "results-20260428T101613Z.jsonl": ("q4_K_M + KV f16", "post verify-infra fix"),
    # KV q8_0 enabled
    "results-20260428T110509Z.jsonl": ("q4_K_M + KV q8_0", "KV q8_0 on q4_K_M"),
    # UD-Q3_K_M
    "results-20260428T113620Z.jsonl": ("UD-Q3_K_M + KV q8_0", "primary winner"),
    "results-20260428T121801Z.jsonl": ("UD-Q3_K_M + KV f16", "f16 experiment"),
    # Smoke
    "results-20260428T132234Z.jsonl": ("UD-Q3_K_M + KV q8_0", "smoke (1 run)"),
    # 5 new tasks (KV q8_0 in effect)
    "results-20260428T154351Z.jsonl": ("UD-Q3_K_M + KV q8_0", "5 new tasks"),
    # task13 re-bench post node_modules fix (KV q8_0)
    "results-20260428T163323Z.jsonl": ("UD-Q3_K_M + KV q8_0", "task13 post node_modules fix"),
}


def main():
    runs = []
    for jp in sorted(RESULTS_DIR.glob("results-*.jsonl")):
        cfg = JSONL_CONFIG.get(jp.name, ("?", "?"))
        for line in jp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_config"] = cfg[0]
            rec["_batch_note"] = cfg[1]
            rec["_jsonl"] = jp.name
            runs.append(rec)

    print(f"Loaded {len(runs)} runs.")

    # qwen3.6 family only
    qwen36 = [r for r in runs if r["model"] in
              ("qwen3.6:35b-a3b-q4_K_M", "qwen3.6-q3")]

    # Aggregate per (config, task)
    summary = defaultdict(lambda: dict(p=0, n=0, dur=0.0, tok=0, jsonl=set()))
    for r in qwen36:
        key = (r["_config"], r["task"])
        s = summary[key]
        s["n"] += 1
        if r["pass"]: s["p"] += 1
        s["dur"] += r["duration_s"]
        s["tok"] += r["total_tokens"]
        s["jsonl"].add(r["_jsonl"])

    out = REPO / "benchmarks" / "results" / "analysis" / "quant-sweep-clean.md"
    with out.open("w") as f:
        f.write("# Qwen3.6-35B-A3B quantization × KV-cache sweep\n\n")
        f.write("Each row aggregates every run for that (config, task) across all JSONL batches.\n\n")
        # Order: configs, then tasks
        configs = ["q4_K_M + KV f16", "q4_K_M + KV q8_0", "UD-Q3_K_M + KV q8_0", "UD-Q3_K_M + KV f16"]
        tasks = sorted({r["task"] for r in qwen36})
        f.write("| Task | " + " | ".join(configs) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(configs)) + "|\n")
        for t in tasks:
            row = [t]
            for c in configs:
                s = summary.get((c, t))
                if not s or s["n"] == 0:
                    row.append("—")
                    continue
                avg = s["dur"] / s["n"]
                pct = s["p"] / s["n"] * 100
                cell = f"**{s['p']}/{s['n']}** {avg:.0f}s ({s['tok']//s['n']:,} tok)"
                row.append(cell)
            f.write("| " + " | ".join(row) + " |\n")
    print(f"\nQuant sweep → {out}")
    out.read_text() and print("\n" + out.read_text())


if __name__ == "__main__":
    main()
