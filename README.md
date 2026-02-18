<p align="center">
  <img src="assets/banner.svg" alt="HermitClaw" width="800">
</p>

<p align="center">
  <a href="https://github.com/kuroko1t/hermitclaw/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/kuroko1t/hermitclaw/ci.yml?branch=main&style=for-the-badge&label=CI" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.75+-orange?style=for-the-badge&logo=rust" alt="Rust"></a>
</p>

<p align="center"><strong>The hermit needs no network.</strong></p>

<p align="center">
A terminal-based AI coding agent written in Rust.<br>
Run it fully offline with local LLMs, or connect to cloud providers when you need more power.
</p>

---

## Highlights

- **Local-first** — works with [Ollama](https://ollama.com/), llama.cpp, LM Studio, or any OpenAI-compatible server
- **Cloud-ready** — supports Anthropic Claude and Google Gemini when you need stronger models
- **Secure by default** — permission gates, path safety, git safety, no telemetry
- **Single binary** — no runtime dependencies, fast startup, ships as one executable
- **Extensible** — 11 built-in tools, custom skills, MCP server integration

## Quick Start

```bash
# 1. Install Ollama and pull a model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b

# 2. Build and install hermitclaw
git clone https://github.com/kuroko1t/hermitclaw.git
cd hermitclaw
cargo install --path .

# 3. Start coding
hermitclaw chat
```

No API keys. No sign-up. No internet required.

## Demo

```
$ hermitclaw chat

hermitclaw v0.1.0
The hermit needs no network.

Model: qwen2.5:7b
Permission: default
Type Ctrl+D to exit.

You> Find all TODO comments and fix them

  [tool: grep] {"pattern": "TODO", "path": "."}

I found 3 TODO comments. Let me fix them one by one.

  [tool: read_file] {"path": "src/main.rs"}
  [tool: edit_file] {"path": "src/main.rs", ...}

  Tool 'edit_file' wants to execute:
    path: src/main.rs
    old_text: // TODO: add retry logic
    new_text: for attempt in 1..=3 { ... }
  Allow? [y/N/a(lways)] y

Done! Fixed all 3 TODO items.
```

## LLM Providers

| Provider | Network | API Key | Config `provider` |
|---|---|---|---|
| [Ollama](https://ollama.com/) | Local only | Not required | `"ollama"` |
| OpenAI-compatible (llama.cpp, LM Studio, vLLM, LocalAI) | Local or remote | Optional | `"openai_compat"` |
| [Anthropic Claude](https://www.anthropic.com/) | Cloud | Required (`ANTHROPIC_API_KEY`) | `"anthropic"` |
| [Google Gemini](https://ai.google.dev/) | Cloud | Required (`GEMINI_API_KEY`) | `"gemini"` |

## Built-in Tools

| Tool | Category | Description |
|---|---|---|
| `read_file` | File | Read file contents |
| `write_file` | File | Create or overwrite a file |
| `edit_file` | File | Replace an exact text match in a file |
| `apply_diff` | File | Apply a unified diff patch (multi-hunk supported) |
| `list_dir` | File | List directory contents (recursive option) |
| `grep` | Code | Search for regex patterns recursively |
| `repo_map` | Code | Show project structure with function/class/type definitions |
| `shell` | System | Execute a shell command |
| `git` | System | Safe git commands only (`status`, `diff`, `log`, `add`, `commit`, `branch`, `show`, `stash`) |
| `web_fetch` | Web | Fetch and extract text from a URL |
| `web_search` | Web | Search the web via DuckDuckGo |

> Web tools are disabled by default. Enable with `web_enabled = true` in config.

## Interactive Commands

| Command | Description |
|---|---|
| `/model <name>` | Switch LLM model at runtime |
| `/mode <mode>` | Change permission mode (`default`, `accept_edits`, `yolo`) |
| `/plan` | Toggle plan mode — read-only analysis using safe tools only |
| `/test [cmd]` | Auto test-fix loop: run tests, let AI fix failures, repeat (max 5 rounds) |
| `/skills` | List loaded skill files |
| `/clear` | Clear conversation and start fresh |
| `/help` | Show all commands |
| `Ctrl+D` | Exit |

## Skills

Customize the system prompt with reusable prompt templates. Place `.md` files in `~/.hermitclaw/skills/`:

```bash
mkdir -p ~/.hermitclaw/skills
echo "Always write tests for new code." > ~/.hermitclaw/skills/testing.md
echo "Use Japanese for comments." > ~/.hermitclaw/skills/japanese.md
```

Skills are automatically loaded at startup and injected into the system prompt. Use `/skills` to see loaded skills.

## Permission System

hermitclaw asks before doing anything risky. Choose your comfort level:

| Mode | File reads | File writes | Shell / Git |
|---|---|---|---|
| `default` | Auto | **Ask** | **Ask** |
| `accept_edits` | Auto | Auto | **Ask** |
| `yolo` | Auto | Auto | Auto |

```bash
# Set in chat
/mode accept_edits

# Or in config.toml
[agent]
permission_mode = "accept_edits"
```

## Security

hermitclaw is **secure by default**:

- **Local inference** — with Ollama, code and prompts never leave your machine
- **Path safety** — sensitive files are blocked (`/etc/shadow`, `~/.ssh`, `~/.aws`, `~/.gnupg`, etc.) with both logical normalization and symlink resolution
- **Git safety** — destructive commands (`push`, `reset`, `clean`, `checkout`, `rebase`, `merge`) are blocked
- **Permission gates** — file writes and shell commands require explicit approval (unless you opt out)
- **Context compression** — automatic conversation summarization prevents unbounded memory growth
- **No telemetry** — zero tracking, zero analytics, zero phone-home

## Architecture

```
hermitclaw (single binary)

  Terminal (REPL) <--> Agent Loop
                        |
                        +-- LLM Provider
                        |     Ollama / OpenAI / Anthropic / Gemini
                        |
                        +-- Tool Executor
                        |     11 built-in + MCP + Skills
                        |
                        +-- Security Layer
                        |     Path safety, Permissions, Git safety
                        |
                        +-- SQLite Memory
                              Chat history, Resume
```

## Configuration

Config file: `~/.hermitclaw/config.toml`

```toml
[llm]
provider = "ollama"             # "ollama", "openai_compat", "anthropic", "gemini"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"
# api_key = "sk-..."            # for OpenAI-compatible / cloud providers
# streaming = true              # token-by-token streaming (default: true)

[agent]
max_iterations = 10
# permission_mode = "default"   # default | accept_edits | yolo
# web_enabled = false           # enable web_fetch / web_search
# context_compression = true    # auto-summarize long conversations
# skills_dir = "~/.hermitclaw/skills"

[memory]
database_path = "~/.hermitclaw/memory.db"
```

<details>
<summary>Anthropic Claude</summary>

```toml
[llm]
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"
# api_key = "sk-ant-..."        # or set ANTHROPIC_API_KEY env var
```
</details>

<details>
<summary>Google Gemini</summary>

```toml
[llm]
provider = "gemini"
model = "gemini-2.0-flash"
# api_key = "..."               # or set GEMINI_API_KEY env var
```
</details>

<details>
<summary>OpenAI-compatible (llama.cpp, LM Studio, vLLM, etc.)</summary>

```toml
[llm]
provider = "openai_compat"
model = "your-model-name"
base_url = "http://localhost:8080"
api_key = "sk-..."              # optional, depends on server
```
</details>

### MCP (Model Context Protocol)

Extend hermitclaw with external tools via MCP servers:

```toml
[[mcp.servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

MCP tools are auto-discovered and registered as `mcp_{server}_{tool}`.

## CLI

```bash
hermitclaw chat                        # start interactive chat
hermitclaw chat -m llama3.2:3b         # use a specific model
hermitclaw chat --continue             # resume last conversation
hermitclaw chat -p "explain main.rs"   # single-shot mode (non-interactive)
hermitclaw chat -p "fix the bug" -y    # single-shot + skip all permission prompts
hermitclaw tools                       # list available tools
hermitclaw config                      # show current configuration
```

### Requirements

- Rust 1.75+
- [Ollama](https://ollama.com/) (or any supported LLM provider)

## Development

```bash
cargo test                     # run tests
cargo fmt -- --check           # check formatting
cargo clippy --all-targets     # lint
```

### CI

All pull requests are checked by GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

| Job | Description |
|---|---|
| **Format** | `cargo fmt -- --check` |
| **Clippy** | `cargo clippy --all-targets` with `-Dwarnings` |
| **Test** | `cargo test` on Ubuntu and macOS |
| **Build** | `cargo build --release` |

All checks must pass before merging.

## Contributing

Contributions are welcome! Please follow these guidelines:

- **Issues** — Use [GitHub Issues](https://github.com/kuroko1t/hermitclaw/issues) for bug reports and feature requests. For bugs, include: Rust version, OS, LLM provider/model, steps to reproduce, and error output.
- **Pull Requests** — Fork the repo, create a feature branch from `main`, ensure `cargo fmt` / `cargo clippy` / `cargo test` all pass, and write a clear PR description.
- **Commits** — Use imperative mood with category prefix: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

## License

[MIT](LICENSE)
