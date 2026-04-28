# Whet Benchmarks

Lightweight, reproducible coding-agent benchmarks used to compare LLMs on the
exact tasks Whet is designed to handle.

## Task layout

Every benchmark directory follows the same shape:

```
benchmarks/<task_name>/
  prompt.txt   # instruction passed to whet via -p (UTF-8, no trailing prose)
  verify.sh   # exits 0 on pass, non-zero on fail; cwd is the workspace copy
  workspace/  # initial files; copied to a tempdir before each run
```

Add new tasks by creating a directory with these three entries.

## Running

```sh
# Single run, all tasks, one model
scripts/run_bench.sh -m qwen3:14b

# Three runs per (model, task) — recommended for headline comparisons
scripts/run_bench.sh -m qwen3:14b,devstral:24b -n 3

# Subset of tasks
scripts/run_bench.sh -m qwen3:14b -t task6_debug,task7_dedupe

# Override the whet binary
scripts/run_bench.sh -w ./target/release/whet -m qwen3:14b
```

Each invocation appends a JSONL file under `benchmarks/results/` and rewrites
`benchmarks/results/leaderboard.md` from every JSONL it finds. Within each
`(model, task)` pair the leaderboard uses the most recent batch (runs minted
within a 24-hour window of the latest timestamp).

## Reproducibility

LLM outputs are stochastic. To minimise variance, set deterministic options in
your `~/.whet/config.toml` before running the suite:

```toml
[llm.options]
temperature = 0.0
seed = 42
num_ctx = 32768       # ensure the model can hold the whole agent loop
```

Even with these, MoE expert routing and tool-call ordering can still vary —
that's why `-n 3` is recommended. The leaderboard reports `pass_count/n_runs`
per task so partial reliability is visible.

## Anti-tampering

Tasks whose verifiers run a unit-test file inside the workspace (currently only
`task6_debug`) pin that file's SHA-256 in `verify.sh`. If the agent edits the
test file to make it pass, the SHA check fires and the task is marked FAIL.

When adding a similar task, compute the canonical SHA once with
`sha256sum workspace/<test_file>` and embed it in `verify.sh`.

## Output schema

Each JSONL line:

```json
{
  "ts": "2026-04-27T17:00:00Z",
  "model": "qwen3:14b",
  "task": "task6_debug",
  "run_index": 1,
  "pass": true,
  "whet_exit": 0,
  "verify_exit": 0,
  "duration_s": 12.3,
  "llm_calls": 3,
  "prompt_tokens": 1234,
  "completion_tokens": 567,
  "total_tokens": 1801,
  "tool_calls_ok": 4,
  "tool_calls_failed": 1,
  "text_to_tool_fallbacks": 0,
  "reprompts": 0,
  "run_dir": "/tmp/whet-bench-...",
  "logs_dir": "/tmp/whet-bench-....logs"
}
```

`run_dir` holds the agent's workspace copy. `logs_dir` (a sibling) holds
`stats.log` (Whet stderr — tool calls + session stats), `stdout.log` (Whet
stdout) and `verify.log` (the verifier's output). Logs are kept outside the
workspace so a recursive `grep` inside `verify.sh` doesn't pick up Whet's
tool-call traces as if they were task content. Both dirs are preserved on
disk; failures can be inspected manually.

## Leaderboard scoring

`leaderboard.md` ranks models by a Pareto score that combines correctness,
token efficiency, and wall-clock speed:

```
score = pass_rate × tok_factor × time_factor
tok_factor  = 1 / (1 + avg_tokens  / 10000)   # 5K tok → 0.67, 50K tok → 0.17
time_factor = 1 / (1 + avg_seconds / 120)     # 60s   → 0.67, 300s   → 0.29
```

A model that solves the same tasks with half the tokens *or* twice the speed
beats one that just edges it on raw pass rate.
