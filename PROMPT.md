# hermitclaw - Implementation Prompt

## Overview

hermitclaw is a fully offline, secure-by-default AI agent written in Rust.
It is the ONLY Claw-family AI agent that works with ZERO internet connection.
The name "Hermit" represents isolation (offline) and protection (shell) — like a hermit crab.

Key differentiators vs OpenClaw/PicoClaw/ZeroClaw:
- **Embedded local LLM inference** via Ollama (no cloud API needed)
- **Sandboxed tool execution** via Linux namespaces (every tool runs isolated)
- **Encrypted local memory** via SQLite + encryption
- **Single binary** deployment
- **Zero network by default**

## Your Role

You are a senior Rust engineer implementing hermitclaw.
Each time you receive this prompt, follow these steps:

1. **Assess current state**: Check which files exist, run `cargo build`, run `cargo test`
2. **Determine current phase**: Find the first incomplete phase below
3. **Work on that phase**: Implement it fully, ensure its completion criteria pass
4. **Commit your work**: `git add` changed files and `git commit` with a descriptive message
5. **Move to next phase** if time permits

IMPORTANT RULES:
- Always run `cargo build` after making changes to verify compilation
- Always run `cargo test` after implementing tests
- Fix any compilation errors before moving to the next phase
- Do NOT skip phases. Complete them in order.
- When ALL phases are complete and all tests pass, output: <promise>HERMITCLAW COMPLETE</promise>

## Tech Stack

- Language: Rust (edition 2021)
- LLM Backend: Ollama (already installed at localhost:11434)
- HTTP Client: reqwest (blocking feature)
- CLI: clap (derive feature)
- Serialization: serde, serde_json
- Terminal UI: rustyline for readline, colored for terminal colors
- Database: rusqlite with bundled feature
- TOML config: toml crate
- Sandbox: std::process::Command with `unshare` for Linux namespace isolation
- Async: Do NOT use async. Use blocking/synchronous code throughout for simplicity.

---

## Phase 1: Project Scaffold

**Goal**: Basic Rust project structure that compiles.

Create the following structure:
```
hermitclaw/
├── Cargo.toml
├── src/
│   ├── main.rs          # CLI entry point (clap)
│   ├── lib.rs           # Re-exports modules
│   ├── llm/
│   │   ├── mod.rs       # LLM trait + Ollama implementation
│   │   └── ollama.rs    # Ollama HTTP client
│   ├── agent/
│   │   ├── mod.rs       # Agent loop logic
│   │   └── prompt.rs    # System prompt templates
│   ├── tools/
│   │   ├── mod.rs       # Tool trait + registry
│   │   ├── read_file.rs
│   │   ├── list_dir.rs
│   │   ├── write_file.rs
│   │   └── shell.rs     # Sandboxed shell command execution
│   ├── sandbox/
│   │   ├── mod.rs       # Sandbox trait + namespace implementation
│   │   └── namespace.rs # Linux namespace sandbox via unshare
│   ├── memory/
│   │   ├── mod.rs       # Conversation memory
│   │   └── store.rs     # SQLite storage
│   └── config.rs        # TOML configuration
├── config.example.toml  # Example configuration file
└── tests/
    └── integration_test.rs
```

### Cargo.toml dependencies:
```toml
[package]
name = "hermitclaw"
version = "0.1.0"
edition = "2021"
description = "Fully offline, secure-by-default AI agent. The hermit needs no network."

[dependencies]
clap = { version = "4", features = ["derive"] }
reqwest = { version = "0.12", features = ["blocking", "json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rustyline = "15"
colored = "3"
rusqlite = { version = "0.32", features = ["bundled"] }
toml = "0.8"
chrono = { version = "0.4", features = ["serde"] }
dirs = "6"
uuid = { version = "1", features = ["v4"] }
```

### main.rs CLI structure:
```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "hermitclaw")]
#[command(about = "Fully offline, secure-by-default AI agent. The hermit needs no network.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive chat
    Chat {
        /// Ollama model to use
        #[arg(short, long, default_value = "qwen2.5:7b")]
        model: String,
        /// Disable sandbox for tool execution
        #[arg(long)]
        no_sandbox: bool,
    },
    /// List available tools
    Tools,
    /// Show configuration
    Config,
}
```

**Completion criteria**:
- [ ] All files in the structure above exist with appropriate module declarations
- [ ] `cargo build` succeeds with zero errors
- [ ] Running `cargo run -- --help` shows the CLI help text

---

## Phase 2: Ollama LLM Integration

**Goal**: Connect to local Ollama and perform text generation with tool calling.

### LLM Trait (src/llm/mod.rs):
```rust
pub trait LlmProvider {
    fn chat(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<LlmResponse, LlmError>;
}

pub struct Message {
    pub role: Role,        // System, User, Assistant, Tool
    pub content: String,
    pub tool_call_id: Option<String>,
}

pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

pub struct LlmResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
}
```

### Ollama Client (src/llm/ollama.rs):
- Use reqwest::blocking::Client to POST to `http://localhost:11434/api/chat`
- Ollama's chat API supports tool calling natively. Use the `tools` field in the request.
- Request format:
```json
{
  "model": "qwen2.5:7b",
  "messages": [...],
  "tools": [...],
  "stream": false
}
```
- Parse the response to extract either content or tool_calls
- Handle connection errors gracefully (show "Is Ollama running?" message)

### Test:
Write a test in `tests/integration_test.rs` that:
- Creates an OllamaClient
- Sends a simple "Hello" message (skip if Ollama not running - use `#[ignore]`)

**Completion criteria**:
- [ ] `cargo build` succeeds
- [ ] OllamaClient implements LlmProvider trait
- [ ] Ollama API request/response serialization works correctly
- [ ] Error handling for connection failures is implemented

---

## Phase 3: Tool System

**Goal**: Define tools that the agent can call, with JSON Schema definitions for LLM tool calling.

### Tool Trait (src/tools/mod.rs):
```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError>;
    fn permissions(&self) -> ToolPermissions;
}

pub struct ToolPermissions {
    pub filesystem_read: bool,
    pub filesystem_write: bool,
    pub network: bool,
    pub subprocess: bool,
}

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}
```

### Implement these tools:

1. **read_file** - Read file contents
   - Parameters: `{ "path": string }`
   - Permissions: filesystem_read only
   - Returns file contents or error

2. **list_dir** - List directory contents
   - Parameters: `{ "path": string, "recursive": bool (optional, default false) }`
   - Permissions: filesystem_read only
   - Returns list of files/directories

3. **write_file** - Write content to a file
   - Parameters: `{ "path": string, "content": string }`
   - Permissions: filesystem_read + filesystem_write
   - Returns success message

4. **shell** - Execute a shell command (sandboxed)
   - Parameters: `{ "command": string, "working_dir": string (optional) }`
   - Permissions: subprocess
   - Returns stdout/stderr

### ToolRegistry:
- `register(tool: Box<dyn Tool>)` - Add a tool
- `get(name: &str) -> Option<&dyn Tool>` - Get tool by name
- `list() -> Vec<&dyn Tool>` - List all tools
- `definitions() -> Vec<ToolDefinition>` - Generate ToolDefinition list for LLM

**Completion criteria**:
- [ ] `cargo build` succeeds
- [ ] All 4 tools implemented
- [ ] ToolRegistry can register, list, and retrieve tools
- [ ] `definitions()` returns valid JSON Schema for each tool
- [ ] Unit tests for read_file and list_dir pass (`cargo test`)

---

## Phase 4: Agent Loop

**Goal**: Implement the ReAct-style agent loop that connects LLM to tools.

### Agent (src/agent/mod.rs):
```rust
pub struct Agent {
    llm: Box<dyn LlmProvider>,
    tools: ToolRegistry,
    memory: Vec<Message>,
    config: AgentConfig,
}

pub struct AgentConfig {
    pub model: String,
    pub max_iterations: usize,  // Max tool-calling rounds per user message (default: 10)
    pub sandbox_enabled: bool,
}
```

### Agent Loop Logic:
```
fn process_message(&mut self, user_input: &str) -> String:
    1. Add user message to memory
    2. Loop (max_iterations):
       a. Call llm.chat(memory, tool_definitions)
       b. If response has content and no tool_calls -> return content
       c. If response has tool_calls:
          - For each tool_call:
            - Find tool in registry
            - Execute tool (with sandbox if enabled)
            - Add assistant message (with tool_call) to memory
            - Add tool result message to memory
          - Continue loop (let LLM see tool results)
    3. Return final response
```

### System Prompt (src/agent/prompt.rs):
Create a system prompt that:
- Identifies the agent as "hermitclaw"
- Explains it runs fully offline
- Instructs to use tools when needed
- Tells to be concise and helpful

### Interactive Chat Loop (update main.rs):
For the `Chat` command:
1. Initialize Agent with OllamaClient and all tools
2. Use rustyline for input with history
3. Display assistant responses with colored output
4. Handle Ctrl+C and Ctrl+D gracefully
5. Print tool calls being executed (e.g., "[tool: read_file] reading ./src/main.rs")

**Completion criteria**:
- [ ] `cargo build` succeeds
- [ ] Agent loop handles multi-turn tool calling
- [ ] `cargo run -- chat` starts an interactive session
- [ ] Tool call results are fed back to the LLM
- [ ] Ctrl+C exits cleanly

---

## Phase 5: Sandbox

**Goal**: Execute tool commands in an isolated Linux namespace.

### Sandbox Trait (src/sandbox/mod.rs):
```rust
pub trait Sandbox {
    fn execute(&self, command: &str, permissions: &ToolPermissions, working_dir: Option<&str>) -> Result<SandboxResult, SandboxError>;
}

pub struct SandboxResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub enum SandboxError {
    PermissionDenied(String),
    ExecutionFailed(String),
    Timeout,
}
```

### Namespace Sandbox (src/sandbox/namespace.rs):
Use `unshare` command to create isolated execution:

```rust
fn execute(&self, command: &str, permissions: &ToolPermissions, working_dir: Option<&str>) -> Result<SandboxResult, SandboxError> {
    let mut cmd = std::process::Command::new("unshare");

    // Network isolation: --net creates a new (empty) network namespace
    if !permissions.network {
        cmd.arg("--net");
    }

    // Mount namespace for filesystem isolation
    cmd.arg("--mount");

    // Run the actual command
    cmd.args(["--", "sh", "-c", command]);

    if let Some(dir) = working_dir {
        cmd.current_dir(dir);
    }

    // Set timeout
    // Use std::process with wait_timeout pattern or spawned thread
    // Kill after 30 seconds

    // Capture output
    let output = cmd.output()?;
    Ok(SandboxResult { ... })
}
```

### Integration:
- The `shell` tool should use Sandbox for execution when sandbox is enabled
- Other tools (read_file, write_file, list_dir) restrict paths:
  - Deny access to paths outside the current working directory
  - Deny access to dotfiles and hidden directories (except explicitly allowed)
  - Deny access to sensitive paths: /etc/shadow, /etc/passwd, ~/.ssh, etc.
- Add `--no-sandbox` flag to CLI for development/testing

### NoOp Sandbox:
Also implement a `NoOpSandbox` that just runs commands directly (for when sandbox is disabled).

**Completion criteria**:
- [ ] `cargo build` succeeds
- [ ] Shell tool runs commands inside namespace when sandbox enabled
- [ ] Network is blocked by default in sandboxed execution
- [ ] Path restrictions prevent access to sensitive files
- [ ] `--no-sandbox` flag disables sandboxing
- [ ] Unit test: sandboxed command cannot access network

---

## Phase 6: Memory & Configuration

**Goal**: Persist conversations and support configuration files.

### SQLite Memory Store (src/memory/store.rs):
```sql
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_call_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
```

- Store conversations in `~/.hermitclaw/memory.db`
- Auto-create directory and database on first run
- Load previous conversation on startup (optional `--continue` flag)
- `hermitclaw chat --new` starts a fresh conversation

### Configuration (src/config.rs):
```toml
# ~/.hermitclaw/config.toml
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

- Load config from `~/.hermitclaw/config.toml` if it exists
- Fall back to defaults if no config file
- CLI arguments override config file values
- Create `config.example.toml` with all options documented

**Completion criteria**:
- [ ] `cargo build` succeeds
- [ ] Conversations are saved to SQLite
- [ ] `--continue` resumes last conversation
- [ ] Config file is loaded from `~/.hermitclaw/config.toml`
- [ ] CLI args override config values
- [ ] `cargo test` passes for memory store operations

---

## Phase 7: Polish & README

**Goal**: Error handling, UX improvements, and documentation.

### Error Handling:
- Wrap all errors in a custom `HermitclawError` enum
- Provide helpful error messages:
  - "Ollama is not running. Start it with: ollama serve"
  - "Model 'xxx' not found. Pull it with: ollama pull xxx"
  - "Sandbox requires Linux. Use --no-sandbox on other platforms."

### UX Improvements:
- Show a startup banner with version info
- Show which model is loaded
- Show "[thinking...]" while waiting for LLM response
- Color tool calls in cyan, errors in red, assistant text in white
- Show token count / response time after each response

### CLI subcommands:
- `hermitclaw tools` - List all available tools with descriptions
- `hermitclaw config` - Show current configuration

### README.md:
Create a README.md with:
- Project name and one-line description: "hermitclaw — The hermit needs no network."
- Feature comparison table (vs OpenClaw, PicoClaw, ZeroClaw)
- Quick start (3 steps: install Ollama, install hermitclaw, run)
- Architecture diagram (text-based)
- Security model explanation
- License: MIT

**Completion criteria**:
- [ ] `cargo build` succeeds (zero warnings with `#![warn(clippy::all)]`)
- [ ] `cargo test` all tests pass
- [ ] `cargo run -- --help` shows clean help
- [ ] `cargo run -- tools` lists tools
- [ ] `cargo run -- config` shows config
- [ ] README.md exists with all sections above
- [ ] All error paths show helpful messages

---

## Final Checklist

Before outputting the completion promise, verify ALL of these:

1. `cargo build --release` succeeds with zero errors
2. `cargo test` - all tests pass
3. `cargo run -- --help` works
4. `cargo run -- tools` lists 4 tools
5. `cargo run -- config` shows configuration
6. README.md exists and is complete
7. config.example.toml exists
8. All 7 phases are complete

If ALL checks pass: <promise>HERMITCLAW COMPLETE</promise>
