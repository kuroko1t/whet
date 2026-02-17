# hermitclaw

> The hermit needs no network.

Fully offline, secure-by-default AI agent. The only Claw-family agent that works with **zero internet connection**.

## Features

| Feature | OpenClaw | PicoClaw | ZeroClaw | **hermitclaw** |
|---|---|---|---|---|
| Local LLM inference | - | - | - | Ollama |
| Network required | Yes | Yes | Yes | **No** |
| Sandboxed tools | - | - | Partial | **Namespace isolation** |
| API cost | $$$ | $$$ | $$$ | **$0** |
| Language | Python | Go | Rust | **Rust** |
| Single binary | - | Yes | Yes | **Yes** |

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

# Disable sandbox (development mode)
hermitclaw chat --no-sandbox

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
│  │   Ollama     │ │ Sandbox  │ │ SQLite │  │
│  │ (local LLM)  │ │(unshare) │ │(memory)│  │
│  └──────────────┘ └──────────┘ └────────┘  │
└─────────────────────────────────────────────┘
         Network connection: NONE
```

## Security Model

hermitclaw is secure by default:

- **Network isolation**: Tool execution runs in Linux namespace with `--net` flag, creating an empty network namespace with no connectivity
- **Path restrictions**: File tools block access to sensitive paths (`/etc/shadow`, `~/.ssh`, `~/.aws`, etc.)
- **Execution timeout**: Sandboxed commands are killed after 30 seconds
- **No telemetry**: Zero data leaves your machine. Ever.
- **Local inference**: Your prompts never touch a cloud API

## Configuration

Config file: `~/.hermitclaw/config.toml`

```toml
[llm]
provider = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"

[agent]
max_iterations = 10
sandbox = true

[memory]
database_path = "~/.hermitclaw/memory.db"
```

## Built-in Tools

| Tool | Description | Permissions |
|---|---|---|
| `read_file` | Read file contents | filesystem_read |
| `list_dir` | List directory contents | filesystem_read |
| `write_file` | Write content to a file | filesystem_read + write |
| `shell` | Execute a shell command | subprocess (sandboxed) |

## License

MIT
