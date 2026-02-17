# hermitclaw

> The hermit needs no network.

Fully offline, secure-by-default AI agent. The only Claw-family agent that works with **zero internet connection**.

## Features

| Feature | OpenClaw | PicoClaw | ZeroClaw | **hermitclaw** |
|---|---|---|---|---|
| Local LLM inference | - | - | - | Ollama |
| Network required | Yes | Yes | Yes | **No** |
| Path safety | - | - | Partial | **Built-in** |
| API cost | $$$ | $$$ | $$$ | **$0** |
| Language | Python | Go | Rust | **Rust** |
| Single binary | - | Yes | Yes | **Yes** |
| OpenAI-compat API | - | - | - | **Yes** |
| MCP support | - | - | - | **Yes** |
| Streaming | - | - | - | **Yes** |

## Quick Start

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
```

### 2. Install hermitclaw

```bash
cargo install --path .
```

### 3. Run

```bash
hermitclaw chat
```

## Usage

```bash
# Interactive chat (default model: qwen2.5:7b)
hermitclaw chat

# Use a specific model
hermitclaw chat -m llama3.2:3b

# Continue last conversation
hermitclaw chat --continue

# List available tools
hermitclaw tools

# Show configuration
hermitclaw config
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              hermitclaw                      │
│            (single binary)                   │
│                                              │
│  ┌─────────┐    ┌────────────────────┐      │
│  │ Terminal │◄──►│    Agent Loop      │      │
│  │   (TUI)  │    │                    │      │
│  └─────────┘    │ 1. User input      │      │
│                  │ 2. LLM inference   │      │
│                  │ 3. Tool execution  │      │
│                  │ 4. Response        │      │
│                  └────────┬───────────┘      │
│                           │                  │
│            ┌──────────────┼──────────┐      │
│            ▼              ▼          ▼      │
│  ┌──────────────┐ ┌──────────┐ ┌────────┐  │
│  │ LLM Provider │ │ Security │ │ SQLite │  │
│  │ Ollama/      │ │  (path)  │ │(memory)│  │
│  │ OpenAI-compat│ └──────────┘ └────────┘  │
│  └──────────────┘                           │
│            ▲                                │
│            │                                │
│  ┌──────────────┐                           │
│  │ MCP Servers  │ (optional external tools) │
│  └──────────────┘                           │
└─────────────────────────────────────────────┘
         Network connection: NONE
```

## Security Model

hermitclaw is secure by default:

- **Local inference**: Your code and prompts never leave your machine — all LLM inference runs locally via Ollama
- **Path restrictions**: File tools block access to sensitive paths (`/etc/shadow`, `~/.ssh`, `~/.aws`, etc.)
- **Git safety**: Only safe git commands allowed (status, diff, log, add, commit, branch, show, stash). Destructive commands (push, reset, clean) are blocked.
- **No telemetry**: Zero data leaves your machine. Ever.

## Configuration

Config file: `~/.hermitclaw/config.toml`

```toml
[llm]
provider = "ollama"          # "ollama" or "openai_compat"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"
# api_key = "sk-..."         # Optional: for OpenAI-compatible servers
# streaming = true           # Enable streaming responses

[agent]
max_iterations = 10

[memory]
database_path = "~/.hermitclaw/memory.db"

# MCP servers
# [[mcp.servers]]
# name = "filesystem"
# command = "npx"
# args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

### OpenAI-Compatible Providers

hermitclaw supports any OpenAI-compatible API server (llama.cpp, LM Studio, vLLM, LocalAI, etc.):

```toml
[llm]
provider = "openai_compat"
model = "your-model-name"
base_url = "http://localhost:8080"
streaming = true
```

### MCP (Model Context Protocol)

hermitclaw can connect to external MCP servers to extend its tool capabilities:

```toml
[[mcp.servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

MCP tools are automatically discovered and registered with the prefix `mcp_{server}_{tool}`.

## Built-in Tools

| Tool | Description | Permissions |
|---|---|---|
| `read_file` | Read file contents | filesystem_read |
| `list_dir` | List directory contents | filesystem_read |
| `write_file` | Write content to a file | filesystem_read + write |
| `shell` | Execute a shell command | subprocess |
| `grep` | Search for patterns in files recursively | filesystem_read |
| `edit_file` | Edit a file by replacing exact text match | filesystem_read + write |
| `git` | Execute safe git commands | subprocess (restricted) |
| `repo_map` | Show project structure with symbol definitions | filesystem_read |

## License

MIT
