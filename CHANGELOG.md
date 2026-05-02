# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-02

This release is a substantial evolution since `0.1.0`: subagents, persistent memory, Claude-Code-style UX polish, plan-first conversation behaviour, a reproducible benchmark suite, and — as the headline change — a deliberate refocus to **local LLMs only**.

### Added

- **Subagents** (#4, #5): `/agent <task>` slash command spawns a sequential child agent with isolated memory and read-tracking, returning a single summary to the parent. The model itself can also call `subagent` autonomously when it judges the subtask large enough. Depth cap = 1; `SubagentGuard` RAII restores parent state even on child panic.
- **Persistent cross-session memory** (#9): SQLite-backed `remember` tool + auto-load injects per-project memories into the system prompt at startup.
- **`/doctor` diagnostics** (#3): five checks (ollama reachable, model present, config parses, `~/.whet` writable, MCP binaries on `$PATH`) with PASS / WARN / FAIL per row plus an overall verdict.
- **Compact tool-call display** (#3): `Read(path)`, `Edit(path)`, … replaces the legacy `[tool: read_file] {"path":"x"}` form. 16 unit tests cover every built-in tool, MCP fallback, and char-boundary-safe truncation.
- **Spinner / "thinking…" indicator** (#3): braille frames at 80 ms/frame while waiting for the first streamed token. TTY-aware — no-ops when stderr isn't a terminal.
- **Diff preview after edits** (#3): every successful `edit_file` / `apply_diff` prints a colored unified-style preview (red `-` / green `+`).
- **Plan-first on open-ended questions** (#13): when the user asks a strategic question ("what should we do?", "次は何やる？", "A or B?"), Whet replies with 2–3 options + tradeoffs + a recommendation and waits for the user to pick — no edits until then. Concrete directives still act immediately. Detection is a tight English/Japanese heuristic; a regression guard rejects bare prose disjunctions to avoid false positives on directive prompts.
- **Adaptive iteration cap + per-call token cap** (#2): bounds runaway generations on borderline tasks. Default `num_predict = 8192` per LLM call (overridable via config).
- **Reproducible benchmark suite**: 13 tasks (single-file edits, multi-file refactors, planning chains, security fixes, TDD-style test generation, TypeScript, large investigation, long-conversation recall) with tamper-proof verifiers (SHA-pinned sources, mutation testing, false-positive rejection).
  - **task14_callsites** (#6): large investigation across 16 files — exercises subagent / focused-context patterns.
  - **task15_long_recall** (#12): 15 fact files force compaction mid-task; verifies that compaction preserves enough information for the writeup turn.
- **Bench analysis scripts**: `analyze_bench.py` (failure modes per task), `analyze_quant_sweep.py` (per-quantization comparison).
- **JSONL stats sink**: `WHET_STATS_JSONL=path/to/file.jsonl` writes one structured record per run for downstream analysis.

### Changed

- **Token-based compaction trigger** (#10): replaces the old message-count threshold. Default ratio is `0.85 × num_ctx`, falling back to an absolute 5000-token threshold when `num_ctx` isn't known. The default is purely beneficial — fits-context runs are unaffected; overflow runs are saved.
- **System prompt CORE RULE 2 split**: `ACT, DON'T ASK` → `ACT ON CONCRETE DIRECTIVES, DISCUSS OPEN-ENDED QUESTIONS`. The harness's act-don't-ask re-prompt is now gated by user-input classification so that an open-ended response (option list + clarifying question) is no longer chastised.
- **`apply_diff`** now reports per-hunk outcomes with atomic-per-file rollback — a partial failure rolls the file back rather than leaving it in a half-applied state.
- **`edit_file` whitespace-normalised fallback**: when the exact `old_text` doesn't match, retries with whitespace normalisation before failing.
- **Re-prompt on premature exit**: when the model exits after only read-type tool calls (no edits / shell), the harness re-prompts to push it into action.

### Fixed

- **Subagent hardening** (#7): panic safety on child loops, clearer error semantics on tool failures, child tool list properly scoped (no shell escape via the parent's allowlist).
- **Compaction fallback** (#11): if the summariser can't reduce memory below the threshold, fall through to a deterministic truncation rather than looping.
- **Single-shot conversation persistence**: `-c` / `--continue` now actually resumes single-shot turns (the prior version skipped the save step in non-interactive mode).
- **Config test isolation**: `Config::load()` tests now accept an explicit home dir to avoid cross-test interference.
- **Clippy 1.95**: collapsed `if-else` into a match guard, replaced `is_none / return None` with `?` where applicable.

### Removed (BREAKING)

- **Anthropic and Gemini provider modules** (#14): `src/llm/anthropic.rs` and `src/llm/gemini.rs` are gone (1243 lines deleted). Configs with `provider = "anthropic"` or `provider = "gemini"` now fall back to the Ollama default. Whet is local-only by design — no API-key env vars are read anywhere in the codebase. The remaining providers are `"ollama"` (default) and `"openai_compat"`, the latter covering llama.cpp, LM Studio, vLLM, and any other local OpenAI-compatible inference server.

## [0.1.0] - 2026-02-19

### Added

- Interactive terminal coding agent with REPL interface
- **LLM providers**: Ollama, Anthropic Claude, Google Gemini, OpenAI-compatible APIs
- **11 built-in tools**: `read_file`, `write_file`, `edit_file`, `apply_diff`, `list_dir`, `grep`, `repo_map`, `shell`, `git`, `web_fetch`, `web_search`
- **MCP support**: Extend with external tools via Model Context Protocol servers
- **Permission system**: 3 modes (`default`, `accept_edits`, `yolo`) with interactive approval
- **Git safety tiers**: Always-allowed, approval-required, and always-blocked commands
- **Session management**: `--resume` (session picker), `--resume <id>` (direct), `--continue` (latest)
- **Context compression**: Automatic and manual (`/compact`) conversation summarization
- **Project instructions**: `WHET.md` support for project-specific context
- **Skills system**: Custom prompt templates loaded from `~/.whet/skills/`
- **Slash commands**: `/model`, `/mode`, `/plan`, `/test`, `/init`, `/compact`, `/skills`, `/clear`, `/help`
- **Plan mode**: Read-only analysis mode for code exploration
- **Auto test-fix loop**: `/test` command with up to 5 fix iterations
- **Streaming support**: Real-time token streaming for all providers
- **Security**: Path sandboxing, shell command safety checks, write protection
- **Single-shot mode**: `whet "fix the bug"` or `whet -p "explain main.rs"`

[0.2.0]: https://github.com/kuroko1t/whet/releases/tag/v0.2.0
[0.1.0]: https://github.com/kuroko1t/whet/releases/tag/v0.1.0
