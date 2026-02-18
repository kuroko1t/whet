<p align="center">
  <img src="assets/banner.svg" alt="HermitClaw" width="800">
</p>

<p align="center">
  <a href="https://github.com/kuroko1t/hermitclaw/actions/workflows/ci.yml"><img src="https://github.com/kuroko1t/hermitclaw/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.75+-orange.svg" alt="Rust"></a>
</p>

---

hermitclaw is an AI coding agent that can run **entirely on your machine**. Your code, your prompts, your data — nothing leaves your computer unless you choose to use a cloud provider.

Use local LLMs via [Ollama](https://ollama.com/) for **zero API costs** and **zero network dependency**, or connect to cloud providers (Anthropic Claude, Google Gemini, OpenAI-compatible) when you need more power. Either way, hermitclaw provides **built-in security guardrails**.

## Quick Start

**1. Install Ollama and pull a model**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
```

**2. Install hermitclaw**

```bash
cargo install --path .
```

**3. Start coding**

```bash
hermitclaw chat
```

That's it. No API keys. No sign-up. No internet.

## Demo

```
$ hermitclaw chat

 hermitclaw v0.1.0 (qwen2.5:7b)
 Type /help for commands, Ctrl+D to exit

You> Find all TODO comments in this project and fix them

 hermitclaw is thinking...

[tool] grep { pattern: "TODO", path: "." }
Found 3 matches:
  src/main.rs:42:  // TODO: add retry logic
  src/config.rs:18:  // TODO: validate port range
  src/lib.rs:7:  // TODO: implement Display

[tool] read_file { path: "src/main.rs" }
[tool] edit_file { path: "src/main.rs", old_text: "// TODO: add retry logic", new_text: "..." }
 Allow edit to src/main.rs? [y/n] y
...

 Done! Fixed 3 TODO items across 3 files.
```

## LLM Providers

hermitclaw supports 4 LLM providers:

| Provider | Network | API Key | Config `provider` |
|---|---|---|---|
| [Ollama](https://ollama.com/) | Local only | Not required | `"ollama"` |
| OpenAI-compatible (llama.cpp, LM Studio, vLLM, LocalAI) | Local or remote | Optional | `"openai_compat"` |
| [Anthropic Claude](https://www.anthropic.com/) | Cloud | Required (`ANTHROPIC_API_KEY`) | `"anthropic"` |
| [Google Gemini](https://ai.google.dev/) | Cloud | Required (`GEMINI_API_KEY`) | `"gemini"` |

## Tools

hermitclaw comes with **11 built-in tools** that cover the full coding workflow:

### File Operations

| Tool | Description |
|---|---|
| `read_file` | Read file contents |
| `write_file` | Create or overwrite a file |
| `edit_file` | Replace an exact text match in a file |
| `apply_diff` | Apply a unified diff patch (multi-hunk supported) |
| `list_dir` | List directory contents (recursive option) |

### Code Intelligence

| Tool | Description |
|---|---|
| `grep` | Search for regex patterns recursively (skips `.git`, `target`, `node_modules`) |
| `repo_map` | Show project structure with function/class/type definitions |

### System

| Tool | Description |
|---|---|
| `shell` | Execute a shell command |
| `git` | Safe git commands only (`status`, `diff`, `log`, `add`, `commit`, `branch`, `show`, `stash`) |

### Web (opt-in)

| Tool | Description |
|---|---|
| `web_fetch` | Fetch and extract text from a URL |
| `web_search` | Search the web via DuckDuckGo |

> Web tools are disabled by default. Enable with `web_enabled = true` in config.

## Interactive Commands

Inside the chat, use slash commands for quick actions:

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
- **Path safety** — sensitive files are blocked (`/etc/shadow`, `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.docker/config.json`, `~/.kube/config`, etc.) with both logical normalization and symlink resolution
- **Git safety** — destructive commands (`push`, `reset`, `clean`, `checkout`, `rebase`, `merge`) are blocked
- **Permission gates** — file writes and shell commands require explicit approval (unless you opt out)
- **Context compression** — automatic conversation summarization prevents unbounded memory growth
- **No telemetry** — zero tracking, zero analytics, zero phone-home

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    hermitclaw                         │
│                  (single binary)                      │
│                                                      │
│  ┌──────────┐    ┌─────────────────────────────┐    │
│  │ Terminal  │◄──►│        Agent Loop            │    │
│  │  (REPL)  │    │                              │    │
│  └──────────┘    │  User input                  │    │
│                  │    → LLM inference            │    │
│                  │      → Tool execution         │    │
│                  │        → Response             │    │
│                  └──────┬───────────┬────────────┘    │
│                         │           │                │
│         ┌───────────────┤           │                │
│         ▼               ▼           ▼                │
│  ┌────────────┐  ┌──────────┐  ┌─────────┐         │
│  │    LLM     │  │ Security │  │ SQLite  │         │
│  │  Provider  │  │  Layer   │  │ Memory  │         │
│  │            │  │          │  │         │         │
│  │ · Ollama   │  │ · Path   │  │ · Chat  │         │
│  │ · OpenAI-  │  │   safety │  │   history│         │
│  │   compat   │  │ · Perms  │  │ · Resume│         │
│  │ · Anthropic│  │ · Context│  │         │         │
│  │ · Gemini   │  │   compress│  │         │         │
│  └────────────┘  └──────────┘  └─────────┘         │
│         ▲                                            │
│  ┌────────────┐  ┌────────────────────────────┐     │
│  │    MCP     │  │     11 Built-in Tools      │     │
│  │  Servers   │  │  + Skills + MCP tools      │     │
│  │ (optional) │  └────────────────────────────┘     │
│  └────────────┘                                      │
└──────────────────────────────────────────────────────┘
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

### Provider Examples

**Ollama (local, default):**

```toml
[llm]
provider = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"
```

**Anthropic Claude:**

```toml
[llm]
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"
# api_key = "sk-ant-..."        # or set ANTHROPIC_API_KEY env var
```

**Google Gemini:**

```toml
[llm]
provider = "gemini"
model = "gemini-2.0-flash"
# api_key = "..."               # or set GEMINI_API_KEY env var
```

**OpenAI-compatible (llama.cpp, LM Studio, vLLM, etc.):**

```toml
[llm]
provider = "openai_compat"
model = "your-model-name"
base_url = "http://localhost:8080"
api_key = "sk-..."              # optional, depends on server
```

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

## Building from Source

```bash
git clone https://github.com/kuroko1t/hermitclaw.git
cd hermitclaw
cargo build --release
./target/release/hermitclaw chat
```

### Requirements

- Rust 1.75+
- [Ollama](https://ollama.com/) (or any supported LLM provider)

## Development

### Running Tests

```bash
cargo test
```

### Linting

```bash
cargo fmt -- --check
cargo clippy --all-targets
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

### Issues

- Use [GitHub Issues](https://github.com/kuroko1t/hermitclaw/issues) for bug reports and feature requests
- For bugs, include: Rust version, OS, LLM provider/model, steps to reproduce, and error output
- Check existing issues before opening a new one

### Pull Requests

1. Fork the repository and create a feature branch from `main`
2. Make your changes, ensuring:
   - `cargo fmt` passes (no formatting issues)
   - `cargo clippy --all-targets` passes with zero warnings
   - `cargo test` passes (all tests green)
   - New functionality includes tests
3. Write a clear PR description explaining **what** and **why**
4. One logical change per PR — avoid mixing unrelated changes

### Commit Messages

- Use imperative mood: "add feature" not "added feature"
- Keep the first line under 72 characters
- Prefix with category when appropriate: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

## License

[MIT](LICENSE)
