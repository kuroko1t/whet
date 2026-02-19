# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/kuroko1t/whet/releases/tag/v0.1.0
