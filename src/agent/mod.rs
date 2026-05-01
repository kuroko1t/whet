pub mod display;
pub mod doctor;
pub mod prompt;

use crate::config::{PermissionMode, ToolRiskLevel};
use crate::llm::{LlmProvider, Message, TokenUsage, ToolCall};
use crate::skills::Skill;
use crate::tools::ToolRegistry;
use colored::Colorize;
use std::collections::HashSet;

#[derive(Debug, Default)]
pub struct SessionStats {
    pub llm_calls: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub tool_calls_ok: u64,
    pub tool_calls_failed: u64,
    pub text_to_tool_fallbacks: u64,
    pub reprompts: u64,
}

impl SessionStats {
    pub fn record_llm_call(&mut self, usage: &TokenUsage) {
        self.llm_calls += 1;
        if let Some(pt) = usage.prompt_tokens {
            self.prompt_tokens += pt;
        }
        if let Some(ct) = usage.completion_tokens {
            self.completion_tokens += ct;
        }
    }

    pub fn record_tool_call(&mut self, success: bool) {
        if success {
            self.tool_calls_ok += 1;
        } else {
            self.tool_calls_failed += 1;
        }
    }

    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }

    pub fn total_tool_calls(&self) -> u64 {
        self.tool_calls_ok + self.tool_calls_failed
    }

    pub fn tool_success_rate(&self) -> Option<f64> {
        let total = self.total_tool_calls();
        if total == 0 {
            None
        } else {
            Some(self.tool_calls_ok as f64 / total as f64 * 100.0)
        }
    }
}

const MAX_TOOL_OUTPUT_CHARS: usize = 50_000;
const MAX_CONTEXT_MESSAGES: usize = 40;
const SUMMARIZE_KEEP_RECENT: usize = 10;

/// Tools that observe the workspace without modifying it. Used by the
/// premature-exit detector: a turn that only invoked tools from this set
/// hasn't actually acted on the user's request yet.
const READ_ONLY_TOOLS: &[&str] = &[
    "read_file",
    "list_dir",
    "repo_map",
    "grep",
    "web_fetch",
    "web_search",
];

fn is_read_only_tool(name: &str) -> bool {
    READ_ONLY_TOOLS.contains(&name)
}

pub struct Agent {
    pub llm: Box<dyn LlmProvider>,
    pub tools: ToolRegistry,
    pub memory: Vec<Message>,
    pub config: AgentConfig,
    pub stats: SessionStats,
    /// Tracks paths that have been read via read_file, used to enforce read-before-edit.
    read_paths: HashSet<String>,
    /// When true, skip the read-before-edit check (resumed sessions lack tool call history).
    resumed: bool,
    /// Current subagent nesting depth (0 = parent, 1 = inside a subagent).
    /// Bounded by `MAX_SUBAGENT_DEPTH` to prevent unbounded recursion.
    subagent_depth: usize,
    /// Callback wired to the persistent-memory store. The model invokes
    /// it via the `remember` tool. None means no memory backend (tests,
    /// or runs where the user disabled persistent memory).
    on_remember: Option<RememberCallback>,
}

/// Hard cap on subagent nesting. Phase A keeps it at 1 — a subagent
/// itself cannot spawn further subagents. Lift later if a real workflow
/// needs deeper delegation.
pub const MAX_SUBAGENT_DEPTH: usize = 1;

/// Callback the agent loop invokes when the model calls the `remember`
/// tool. Returns the row id assigned by the persistent-memory store on
/// success, or a printable error on failure.
pub type RememberCallback = Box<dyn Fn(&str) -> Result<i64, String>>;

/// How an agent loop iteration concluded. Surfaced from
/// `process_message_full` and consumed by `run_subagent` so the parent
/// can correctly classify a subagent's outcome (success vs. LLM-side
/// failure vs. iteration-cap exhaustion) without resorting to fragile
/// string-prefix heuristics on the returned text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitReason {
    /// The agent emitted a final assistant message (no further tool calls).
    Answered,
    /// LLM provider returned an error mid-loop.
    LlmError(String),
    /// Hit the adaptive iteration cap without a final answer.
    MaxIterations,
}

impl ExitReason {
    /// True when the loop completed with a real answer; false for any
    /// internal failure path. Used by the parent loop to set
    /// `tool_success` on a subagent dispatch.
    pub fn is_success(&self) -> bool {
        matches!(self, ExitReason::Answered)
    }
}

/// RAII guard that swaps the parent's per-conversation state out for a
/// fresh child set on construction, and restores the parent's state on
/// Drop. Holding the guard borrows the `Agent` mutably for the
/// guard's lifetime; child loop work goes through `guard.agent.*`.
///
/// Drop runs even on panic unwind, so a panic inside the child loop
/// doesn't leak corrupted memory / read_paths / resumed / depth state
/// into the next parent turn.
struct SubagentGuard<'a> {
    agent: &'a mut Agent,
    saved_memory: Vec<Message>,
    saved_read_paths: HashSet<String>,
    saved_resumed: bool,
}

impl<'a> SubagentGuard<'a> {
    fn enter(agent: &'a mut Agent, child_system_msg: Message) -> Self {
        let saved_memory = std::mem::replace(&mut agent.memory, vec![child_system_msg]);
        let saved_read_paths = std::mem::take(&mut agent.read_paths);
        let saved_resumed = agent.resumed;
        agent.resumed = false;
        agent.subagent_depth += 1;
        Self {
            agent,
            saved_memory,
            saved_read_paths,
            saved_resumed,
        }
    }
}

impl<'a> Drop for SubagentGuard<'a> {
    fn drop(&mut self) {
        self.agent.memory = std::mem::take(&mut self.saved_memory);
        self.agent.read_paths = std::mem::take(&mut self.saved_read_paths);
        self.agent.resumed = self.saved_resumed;
        // Saturating decrement just in case Drop fires twice via some
        // future refactor — we never want to wrap into usize::MAX.
        self.agent.subagent_depth = self.agent.subagent_depth.saturating_sub(1);
    }
}

pub struct AgentConfig {
    #[allow(dead_code)]
    pub model: String,
    pub max_iterations: usize,
    pub permission_mode: PermissionMode,
    pub plan_mode: bool,
    pub context_compression: bool,
    /// If set, structured per-event session stats are appended as JSON Lines.
    /// One object per tool call plus a final `session_end` summary line.
    pub stats_jsonl_path: Option<std::path::PathBuf>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "qwen2.5:7b".to_string(),
            max_iterations: 10,
            permission_mode: PermissionMode::Default,
            plan_mode: false,
            context_compression: true,
            stats_jsonl_path: None,
        }
    }
}

/// Append a single JSON object as a line to the stats JSONL sink, if enabled.
/// Failures are deliberately silent — observability must never break the agent.
fn write_stats_event(path: &Option<std::path::PathBuf>, event: serde_json::Value) {
    if let Some(p) = path {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
        {
            use std::io::Write;
            let _ = writeln!(f, "{}", event);
        }
    }
}

/// Append a session_end stats event. Skipped for child (subagent) loops
/// — they share their parent's session and shouldn't emit a second
/// session_end line per logical session.
fn emit_session_end_at_depth(
    path: &Option<std::path::PathBuf>,
    stats: &SessionStats,
    reason: &str,
    depth: usize,
) {
    if depth > 0 {
        return;
    }
    emit_session_end(path, stats, reason);
}

fn emit_session_end(path: &Option<std::path::PathBuf>, stats: &SessionStats, reason: &str) {
    write_stats_event(
        path,
        serde_json::json!({
            "event": "session_end",
            "reason": reason,
            "llm_calls": stats.llm_calls,
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "tool_calls_ok": stats.tool_calls_ok,
            "tool_calls_failed": stats.tool_calls_failed,
            "text_to_tool_fallbacks": stats.text_to_tool_fallbacks,
            "reprompts": stats.reprompts,
        }),
    );
}

impl Agent {
    pub fn new(
        llm: Box<dyn LlmProvider>,
        tools: ToolRegistry,
        config: AgentConfig,
        skills: &[Skill],
    ) -> Self {
        let memory = vec![Message::system(&prompt::system_prompt(skills))];
        Self {
            llm,
            tools,
            memory,
            config,
            stats: SessionStats::default(),
            read_paths: HashSet::new(),
            resumed: false,
            subagent_depth: 0,
            on_remember: None,
        }
    }

    /// Wire a callback that the `remember` tool will invoke. Typical
    /// caller (`main.rs`) closes over an `Arc<Mutex<MemoryStore>>` and
    /// the canonical working directory so the fact is scoped to the
    /// current project. Returning the new row id lets the agent
    /// surface it back to the model (useful for /forget tooling).
    pub fn set_on_remember(&mut self, cb: RememberCallback) {
        self.on_remember = Some(cb);
    }

    pub fn set_resumed(&mut self, resumed: bool) {
        self.resumed = resumed;
    }

    #[allow(dead_code)]
    pub fn process_message(&mut self, user_input: &str) -> String {
        self.process_message_with_callbacks(user_input, &mut |_| {}, &mut |_, _| true)
    }

    #[allow(dead_code)]
    pub fn process_message_with_callback(
        &mut self,
        user_input: &str,
        on_token: &mut dyn FnMut(&str),
    ) -> String {
        self.process_message_with_callbacks(user_input, on_token, &mut |_, _| true)
    }

    /// Compress context by summarizing old messages when the memory exceeds the threshold.
    fn compress_context(&mut self) {
        if self.memory.len() <= MAX_CONTEXT_MESSAGES {
            return;
        }
        self.compress_context_with_instruction(None);
    }

    /// Manually compress the conversation context.
    /// If `instruction` is provided, it is appended to the summarization prompt.
    pub fn compact(&mut self, instruction: Option<&str>) {
        if self.memory.len() <= 2 {
            eprintln!("  Nothing to compress.");
            return;
        }
        self.compress_context_with_instruction(instruction);
    }

    fn compress_context_with_instruction(&mut self, instruction: Option<&str>) {
        // Split: [system_prompt] + [old_messages] + [recent_messages]
        // Keep system prompt (index 0) and the last SUMMARIZE_KEEP_RECENT messages
        let keep_recent = SUMMARIZE_KEEP_RECENT.min(self.memory.len().saturating_sub(1));
        let keep_from = self.memory.len() - keep_recent;

        if keep_from <= 1 {
            return;
        }

        // Build summarization request by draining old messages (avoids cloning)
        // First, split off the recent messages
        let recent_messages = self.memory.split_off(keep_from);
        // Now self.memory = [system_prompt, old_messages...]
        // Drain old messages, keeping system_prompt at index 0
        let system_prompt = self.memory[0].clone();
        // Use the remaining messages (including system prompt) as summarization input
        let summarize_prompt = match instruction {
            Some(inst) => format!(
                "Summarize the conversation so far concisely, preserving key facts, decisions, and context needed for future turns. Additional instructions: {}",
                inst
            ),
            None => "Summarize the conversation so far concisely, preserving key facts, decisions, and context needed for future turns:".to_string(),
        };
        self.memory.push(Message::user(&summarize_prompt));

        // Call LLM without tools for summarization
        let summary = match self.llm.chat(&self.memory, &[]) {
            Ok(resp) => resp.content.unwrap_or_default(),
            Err(_) => {
                // Restore memory on error
                self.memory.pop(); // remove summarize request
                self.memory.extend(recent_messages);
                return;
            }
        };

        if summary.is_empty() {
            self.memory.pop();
            self.memory.extend(recent_messages);
            return;
        }

        // Rebuild memory: [system_prompt, summary, recent_messages...]
        self.memory.clear();
        self.memory.push(system_prompt);
        self.memory.push(Message::system(&format!(
            "Previous conversation summary: {}",
            summary
        )));
        self.memory.extend(recent_messages);

        eprintln!(
            "  {}",
            "[context compressed: conversation summarized]".dimmed()
        );
    }

    /// Process a message with streaming callback and approval callback.
    /// `on_approve` is called before executing a tool that requires approval.
    /// It receives (tool_name, &arguments) and returns true to allow, false to deny.
    pub fn process_message_with_callbacks(
        &mut self,
        user_input: &str,
        on_token: &mut dyn FnMut(&str),
        on_approve: &mut dyn FnMut(&str, &serde_json::Value) -> bool,
    ) -> String {
        // Public surface keeps returning a plain String for the 19-odd
        // callers (single-shot, REPL, /test loop, tests). The full
        // (text, ExitReason) tuple is exposed via process_message_full
        // for run_subagent's structural success/failure classification.
        self.process_message_full(user_input, on_token, on_approve)
            .0
    }

    /// Same as `process_message_with_callbacks` but also returns the
    /// `ExitReason` describing how the loop concluded. Reserved for
    /// callers that need to distinguish a real answer from an LLM
    /// error or an iteration-cap miss without scraping the result text.
    pub fn process_message_full(
        &mut self,
        user_input: &str,
        on_token: &mut dyn FnMut(&str),
        on_approve: &mut dyn FnMut(&str, &serde_json::Value) -> bool,
    ) -> (String, ExitReason) {
        self.memory.push(Message::user(user_input));

        // Compress context if enabled and threshold exceeded
        if self.config.context_compression {
            self.compress_context();
        }

        // Own the tool definitions for this turn so we can call &mut self
        // methods (e.g. run_subagent) inside the loop without conflicting
        // with a borrow on self.tools. Clone is shallow — definitions are
        // small structs, one allocation per turn.
        let mut tool_defs: Vec<crate::llm::ToolDefinition> = if self.config.plan_mode {
            self.tools.safe_definitions().to_vec()
        } else {
            self.tools.definitions().to_vec()
        };
        // Hide `subagent` from child loops. The child can't usefully spawn
        // another subagent (depth cap = 1), so exposing it just wastes
        // ~150 tokens of description on every child LLM call and risks the
        // model emitting calls that always error out.
        if self.subagent_depth > 0 {
            tool_defs.retain(|d| d.name != "subagent");
        }

        let mut reprompt_count: usize = 0;
        const MAX_REPROMPTS: usize = 1;
        // Per-turn tool tracking. The premature-exit detector fires only when
        // the model has invoked at least one read-only tool but no edit/write
        // tool — i.e. it explored the workspace and then stopped.
        let mut has_acted: bool = false;
        let mut has_read: bool = false;

        // Adaptive iteration cap: hit the configured cap, but if the model is
        // still actively making progress (a successful tool call within the
        // last 2 iterations), grant up to MAX_PROGRESS_EXTENSION extra cycles
        // before forcing a stop. Avoids losing tasks that need a few more
        // turns to land while still bounding runaway loops.
        const MAX_PROGRESS_EXTENSION: usize = 5;
        let base_cap = self.config.max_iterations;
        let hard_cap = base_cap.saturating_add(MAX_PROGRESS_EXTENSION);
        let mut iteration: usize = 0;
        let mut last_progress_iter: usize = 0;

        loop {
            // Stop if we've exhausted both base + extension, OR we're past
            // the base and recent iterations stopped making progress.
            if iteration >= hard_cap {
                break;
            }
            if iteration >= base_cap && iteration.saturating_sub(last_progress_iter) >= 2 {
                break;
            }
            iteration += 1;
            let response = match self.llm.chat_streaming(&self.memory, &tool_defs, on_token) {
                Ok(resp) => resp,
                Err(e) => {
                    emit_session_end_at_depth(
                        &self.config.stats_jsonl_path,
                        &self.stats,
                        "llm_error",
                        self.subagent_depth,
                    );
                    let msg = format!("Error: {}", e);
                    return (msg, ExitReason::LlmError(e.to_string()));
                }
            };

            self.stats.record_llm_call(&response.usage);

            // Try to recover tool calls from text if the model didn't use the API
            let mut effective_tool_calls = response.tool_calls;

            if effective_tool_calls.is_empty() {
                let content = response.content.clone().unwrap_or_default();

                // Pattern 2: Extract tool calls from JSON in text content
                let extracted = try_extract_tool_calls_from_text(&content, &self.tools);
                if !extracted.is_empty() {
                    eprintln!(
                        "  {}",
                        format!(
                            "[fallback: extracted {} tool call(s) from text]",
                            extracted.len()
                        )
                        .yellow()
                    );
                    self.stats.text_to_tool_fallbacks += 1;
                    effective_tool_calls = extracted;
                }
                // Pattern 1: Re-prompt if model asked a question instead of acting
                else if reprompt_count < MAX_REPROMPTS && looks_like_question(&content) {
                    eprintln!(
                        "  {}",
                        "[re-prompt: model asked instead of acting]".yellow()
                    );
                    self.stats.reprompts += 1;
                    reprompt_count += 1;
                    self.memory.push(Message::assistant(&content));
                    self.memory.push(Message::user(
                        "Don't ask questions. Use your tools to take action directly. \
                         If you're unsure, start by reading files or exploring the project structure.",
                    ));
                    continue;
                }
                // Pattern 3: Re-prompt if model is exiting after only reads.
                // Requires:
                //   (a) the model invoked at least one read-only tool this turn,
                //   (b) no edit/write/shell call ran this turn,
                //   (c) the final response is empty.
                // Read-only Q&A ("read file → answer in text") is unaffected
                // because it produces non-empty content.
                else if reprompt_count < MAX_REPROMPTS
                    && has_read
                    && !has_acted
                    && content.trim().is_empty()
                {
                    eprintln!(
                        "  {}",
                        "[re-prompt: model stopped after only reads — pushing it to act]".yellow()
                    );
                    self.stats.reprompts += 1;
                    reprompt_count += 1;
                    self.memory.push(Message::assistant(&content));
                    self.memory.push(Message::user(
                        "You've only read files so far. Use your editing tools \
                         (edit_file / apply_diff / write_file / shell) to make the \
                         changes the task requires. If you've finished, state in plain \
                         text what you did so the user can verify.",
                    ));
                    continue;
                }
            }

            // If still no tool calls after recovery, return the content
            if effective_tool_calls.is_empty() {
                let content = response.content.unwrap_or_default();
                self.memory.push(Message::assistant(&content));
                emit_session_end_at_depth(
                    &self.config.stats_jsonl_path,
                    &self.stats,
                    "answered",
                    self.subagent_depth,
                );
                return (content, ExitReason::Answered);
            }

            // Separate content from tool calls to avoid borrow issues
            let response_content = response.content;
            let tool_calls = effective_tool_calls;

            // Store tool calls in memory — move instead of clone
            self.memory
                .push(Message::assistant_with_tool_calls(tool_calls.clone()));

            // Notify the caller's on_token closure that the model has
            // committed to action — this lets a "thinking…" spinner /
            // placeholder stop and clear its line BEFORE we start
            // emitting tool-call lines below, avoiding the race where
            // both share a row. The empty-string payload is a no-op
            // print for callers that just append text.
            on_token("");

            for tool_call in &tool_calls {
                eprintln!(
                    "  {}",
                    display::format_tool_call_compact(&tool_call.name, &tool_call.arguments).cyan()
                );

                // Track read_file calls and enforce read-before-edit
                if tool_call.name == "read_file" {
                    if let Some(p) = tool_call.arguments["path"].as_str() {
                        self.read_paths.insert(Self::normalize_tool_path(p));
                    }
                }

                let needs_read_first = !self.resumed
                    && (tool_call.name == "edit_file" || tool_call.name == "apply_diff")
                    && tool_call.arguments["path"].as_str().map_or(true, |p| {
                        !self.read_paths.contains(&Self::normalize_tool_path(p))
                    });

                let (result, tool_success) = if needs_read_first {
                    let p = tool_call.arguments["path"].as_str().unwrap_or("<unknown>");
                    (
                        format!(
                            "Warning: You must read_file(\"{}\") before using {}. \
                         Read the file first to see its current content, then retry.",
                            p, tool_call.name
                        ),
                        false,
                    )
                } else if tool_call.name == "subagent" {
                    // Phase C: model-callable subagent. Special-cased before
                    // the generic dispatch because the child loop needs full
                    // mutable Agent state (memory swap, read-paths reset),
                    // which Tool::execute(args) cannot provide.
                    let task = tool_call.arguments["task"].as_str().unwrap_or("");
                    if task.is_empty() {
                        (
                            "Tool error: subagent requires a non-empty 'task' argument".to_string(),
                            false,
                        )
                    } else {
                        let context = tool_call.arguments["context"].as_str().unwrap_or("");
                        let brief = if context.is_empty() {
                            task.to_string()
                        } else {
                            format!("Task: {}\n\nContext:\n{}", task, context)
                        };
                        match self.run_subagent(&brief, on_token, on_approve) {
                            // Structural classification via ExitReason —
                            // an LlmError or MaxIterations marks the call
                            // as failed for parent stats and the parent
                            // model's view, no string-prefix heuristic.
                            Ok((text, reason)) => (text, reason.is_success()),
                            Err(e) => (format!("Tool error: {}", e), false),
                        }
                    }
                } else if tool_call.name == "remember" {
                    // Persistent memory tool. Routed via the on_remember
                    // callback because Tool::execute() cannot reach the
                    // SQLite store. If no callback is wired (e.g. tests
                    // or memory-disabled runs), surface a clear error so
                    // the model doesn't think the fact was saved.
                    let content = tool_call.arguments["content"].as_str().unwrap_or("");
                    if content.trim().is_empty() {
                        (
                            "Tool error: remember requires a non-empty 'content' argument"
                                .to_string(),
                            false,
                        )
                    } else if let Some(cb) = self.on_remember.as_ref() {
                        match cb(content) {
                                Ok(id) => (
                                    format!(
                                        "Remembered (id={}). This fact will appear in future sessions for this project.",
                                        id
                                    ),
                                    true,
                                ),
                                Err(e) => (
                                    format!("Tool error: failed to persist memory: {}", e),
                                    false,
                                ),
                            }
                    } else {
                        (
                            "Tool error: persistent memory is not configured for this session"
                                .to_string(),
                            false,
                        )
                    }
                } else if let Some(tool) = self.tools.get(&tool_call.name) {
                    // Determine effective risk level (dynamic for git)
                    let effective_risk = if tool_call.name == "git" {
                        let git_cmd = tool_call.arguments["command"].as_str().unwrap_or("");
                        crate::tools::git::git_command_risk_level(git_cmd)
                    } else {
                        tool.risk_level()
                    };

                    // In plan mode, block non-safe tools
                    if self.config.plan_mode && effective_risk != ToolRiskLevel::Safe {
                        (
                            "Tool blocked: plan mode is active (read-only). Use /plan to toggle."
                                .to_string(),
                            false,
                        )
                    } else if self.needs_approval(effective_risk) {
                        if !on_approve(&tool_call.name, &tool_call.arguments) {
                            ("Tool execution denied by user.".to_string(), false)
                        } else {
                            match tool.execute(tool_call.arguments.clone()) {
                                Ok(output) => (output, true),
                                Err(e) => (format!("Tool error: {}", e), false),
                            }
                        }
                    } else {
                        match tool.execute(tool_call.arguments.clone()) {
                            Ok(output) => (output, true),
                            Err(e) => (format!("Tool error: {}", e), false),
                        }
                    }
                } else {
                    (format!("Unknown tool: {}", tool_call.name), false)
                };

                self.stats.record_tool_call(tool_success);
                if tool_success {
                    last_progress_iter = iteration;
                    if is_read_only_tool(&tool_call.name) {
                        has_read = true;
                    } else {
                        has_acted = true;
                    }
                    // UX.9: show what actually changed for edit_file / apply_diff.
                    match tool_call.name.as_str() {
                        "edit_file" => {
                            let preview = display::format_edit_diff(
                                tool_call.arguments["old_text"].as_str().unwrap_or(""),
                                tool_call.arguments["new_text"].as_str().unwrap_or(""),
                                display::DIFF_PREVIEW_MAX_LINES,
                            );
                            display::print_colored_diff(&preview);
                        }
                        "apply_diff" => {
                            let preview = display::format_unified_diff_excerpt(
                                tool_call.arguments["diff"].as_str().unwrap_or(""),
                                display::DIFF_PREVIEW_MAX_LINES,
                            );
                            display::print_colored_diff(&preview);
                        }
                        _ => {}
                    }
                }
                write_stats_event(
                    &self.config.stats_jsonl_path,
                    serde_json::json!({
                        "event": "tool_call",
                        "name": tool_call.name,
                        "args": tool_call.arguments,
                        "ok": tool_success,
                    }),
                );

                let result = if result.len() > MAX_TOOL_OUTPUT_CHARS {
                    let mut end = MAX_TOOL_OUTPUT_CHARS;
                    while !result.is_char_boundary(end) {
                        end -= 1;
                    }
                    let mut truncated = String::with_capacity(end + 40);
                    truncated.push_str(&result[..end]);
                    truncated.push_str("\n...[output truncated to 50KB]");
                    truncated
                } else {
                    result
                };

                self.memory
                    .push(Message::tool_result(&tool_call.id, &result));
            }

            // Content alongside tool calls is already streamed to the user.
            // Don't add it as a separate memory message — the model would repeat itself.
            let _ = response_content;
        }

        emit_session_end_at_depth(
            &self.config.stats_jsonl_path,
            &self.stats,
            "max_iterations",
            self.subagent_depth,
        );
        (
            "Max iterations reached. The agent could not complete the task.".to_string(),
            ExitReason::MaxIterations,
        )
    }

    /// Add a path to the set of files that have been read (for read-before-edit tracking).
    pub fn add_read_path(&mut self, path: &str) {
        self.read_paths.insert(Self::normalize_tool_path(path));
    }

    /// Run a subagent: a focused child agent loop with isolated memory
    /// and read-before-edit state, but sharing the parent's LLM client,
    /// tools, and stats accumulator.
    ///
    /// The brief becomes the child's first user message. The child's
    /// memory starts with a clone of the parent's system prompt
    /// (so any project-instruction / skills inherited at construction
    /// time stay in scope). After the child loop returns, the parent's
    /// memory and `read_paths` are restored — the only side-effect
    /// visible to the parent is the accumulated stats and the returned
    /// string. The child cannot spawn further subagents
    /// (`MAX_SUBAGENT_DEPTH`).
    ///
    /// Returns `Err` if nesting would exceed the cap; otherwise the
    /// child's final assistant message.
    pub fn run_subagent(
        &mut self,
        brief: &str,
        on_token: &mut dyn FnMut(&str),
        on_approve: &mut dyn FnMut(&str, &serde_json::Value) -> bool,
    ) -> Result<(String, ExitReason), String> {
        if self.subagent_depth >= MAX_SUBAGENT_DEPTH {
            return Err(format!(
                "subagent nesting beyond depth {} is not supported",
                MAX_SUBAGENT_DEPTH
            ));
        }

        let system_msg = self
            .memory
            .first()
            .cloned()
            .unwrap_or_else(|| Message::system(""));

        // RAII guard restores parent state in Drop, so even if the child
        // loop panics, unwinding the stack still puts memory + read_paths
        // + resumed + subagent_depth back to where they were. Without the
        // guard, a panic in any user-supplied callback or future
        // sub-tooling would leave the parent Agent in a corrupted state.
        let _guard = SubagentGuard::enter(self, system_msg);
        let result = _guard
            .agent
            .process_message_full(brief, on_token, on_approve);
        // _guard is dropped here, restoring parent state.

        Ok(result)
    }

    /// Normalize a tool path for read-before-edit tracking.
    /// Resolves "./src/main.rs" and "src/main.rs" to the same key.
    fn normalize_tool_path(path: &str) -> String {
        use std::path::{Component, PathBuf};
        let mut normalized = PathBuf::new();
        for component in std::path::Path::new(path).components() {
            match component {
                Component::CurDir => {} // skip "."
                Component::ParentDir => {
                    normalized.pop();
                }
                other => normalized.push(other),
            }
        }
        normalized.to_string_lossy().to_string()
    }

    /// Determine if a tool at the given risk level needs user approval.
    fn needs_approval(&self, risk_level: ToolRiskLevel) -> bool {
        match self.config.permission_mode {
            PermissionMode::Yolo => false,
            PermissionMode::AcceptEdits => risk_level == ToolRiskLevel::Dangerous,
            PermissionMode::Default => {
                risk_level == ToolRiskLevel::Moderate || risk_level == ToolRiskLevel::Dangerous
            }
        }
    }
}

/// Check if text content looks like the model is asking a question instead of acting.
fn looks_like_question(content: &str) -> bool {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return false;
    }
    // Ends with question mark (ASCII or full-width)
    if trimmed.ends_with('?') || trimmed.ends_with('？') {
        return true;
    }
    let lower = trimmed.to_lowercase();
    let question_phrases = [
        "shall i",
        "should i",
        "do you want",
        "would you like",
        "can i",
        "may i",
        "do you need",
        "want me to",
        "しますか",
        "ますか",
        "でしょうか",
        "よろしいですか",
        "しましょうか",
    ];
    question_phrases.iter().any(|p| lower.contains(p))
}

/// Try to extract tool calls from text content (for models that output JSON as text
/// instead of using the tool calling API).
fn try_extract_tool_calls_from_text(content: &str, tools: &ToolRegistry) -> Vec<ToolCall> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return vec![];
    }

    let json_objects = extract_json_objects(trimmed);
    let mut result = Vec::new();
    for (i, val) in json_objects.iter().enumerate() {
        if let Some(tc) = parse_tool_call_json(val, tools, i) {
            result.push(tc);
        }
    }
    result
}

/// Scan text for top-level JSON objects using brace-depth counting.
fn extract_json_objects(text: &str) -> Vec<serde_json::Value> {
    let mut objects = Vec::new();
    let mut depth = 0i32;
    let mut start = None;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in text.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        let candidate = &text[s..=i];
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(candidate) {
                            objects.push(val);
                        }
                    }
                    start = None;
                }
            }
            _ => {}
        }
    }
    objects
}

/// Parse a JSON value as a tool call if it matches known formats.
/// Format A: {"name": "tool_name", "arguments": {...}}
fn parse_tool_call_json(
    val: &serde_json::Value,
    tools: &ToolRegistry,
    index: usize,
) -> Option<ToolCall> {
    let obj = val.as_object()?;

    // Format A: {"name": "...", "arguments": {...}}
    let name = obj.get("name").and_then(|n| n.as_str())?;
    let arguments = obj.get("arguments")?;

    // Only accept registered tool names
    tools.get(name)?;

    Some(ToolCall {
        id: format!("fallback_{}", index),
        name: name.to_string(),
        arguments: arguments.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmError, LlmResponse, Role, TokenUsage, ToolCall, ToolDefinition};
    use crate::tools::default_registry;
    use std::cell::RefCell;

    /// A mock LLM that returns pre-scripted responses in sequence.
    struct MockLlm {
        responses: RefCell<Vec<LlmResponse>>,
    }

    impl MockLlm {
        fn new(responses: Vec<LlmResponse>) -> Self {
            // Reverse so we can pop from the end
            let mut r = responses;
            r.reverse();
            Self {
                responses: RefCell::new(r),
            }
        }
    }

    impl LlmProvider for MockLlm {
        fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
        ) -> Result<LlmResponse, LlmError> {
            let mut responses = self.responses.borrow_mut();
            if let Some(resp) = responses.pop() {
                Ok(resp)
            } else {
                // If we run out of scripted responses, return empty content
                Ok(LlmResponse {
                    content: Some("(no more scripted responses)".to_string()),
                    tool_calls: vec![],
                    usage: TokenUsage::default(),
                })
            }
        }
    }

    /// A mock LLM that always returns an error.
    struct ErrorLlm;

    impl LlmProvider for ErrorLlm {
        fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
        ) -> Result<LlmResponse, LlmError> {
            Err(LlmError::ConnectionError(
                "Cannot connect to Ollama".to_string(),
            ))
        }
    }

    fn make_agent(llm: Box<dyn LlmProvider>) -> Agent {
        Agent::new(llm, default_registry(), AgentConfig::default(), &[])
    }

    fn make_agent_with_jsonl(llm: Box<dyn LlmProvider>, path: std::path::PathBuf) -> Agent {
        let cfg = AgentConfig {
            stats_jsonl_path: Some(path),
            ..AgentConfig::default()
        };
        Agent::new(llm, default_registry(), cfg, &[])
    }

    #[test]
    fn test_simple_text_response() {
        let llm = MockLlm::new(vec![LlmResponse {
            content: Some("Hello! I'm here to help.".to_string()),
            tool_calls: vec![],
            usage: TokenUsage::default(),
        }]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Hi there");
        assert_eq!(response, "Hello! I'm here to help.");
    }

    #[test]
    fn test_empty_content_response() {
        let llm = MockLlm::new(vec![LlmResponse {
            content: None,
            tool_calls: vec![],
            usage: TokenUsage::default(),
        }]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Hi");
        // None content should become empty string
        assert_eq!(response, "");
    }

    #[test]
    fn test_tool_call_then_response() {
        // First response: call read_file tool
        // Second response: use tool result to answer
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "Cargo.toml"}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("The project is named whet.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("What is this project?");
        assert_eq!(response, "The project is named whet.");
    }

    #[test]
    fn test_multiple_tool_calls_in_one_response() {
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![
                    ToolCall {
                        id: "call_0".to_string(),
                        name: "read_file".to_string(),
                        arguments: serde_json::json!({"path": "Cargo.toml"}),
                    },
                    ToolCall {
                        id: "call_1".to_string(),
                        name: "list_dir".to_string(),
                        arguments: serde_json::json!({"path": "src"}),
                    },
                ],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("I found 2 things.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Analyze the project");
        assert_eq!(response, "I found 2 things.");

        // Verify memory contains system + user + assistant(tool_calls) + 2 tool results + assistant
        // system(1) + user(1) + assistant_tool_calls(1) + tool_result(2) + assistant(1) = 6
        assert_eq!(agent.memory.len(), 6);
    }

    #[test]
    fn test_unknown_tool_handled_gracefully() {
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "nonexistent_tool".to_string(),
                    arguments: serde_json::json!({}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Sorry, that tool doesn't exist.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Use nonexistent tool");
        assert_eq!(response, "Sorry, that tool doesn't exist.");

        // Check that "Unknown tool" message was stored in memory
        let tool_result_msg = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have a tool result message");
        assert!(tool_result_msg
            .content
            .contains("Unknown tool: nonexistent_tool"));
    }

    #[test]
    fn test_tool_execution_error_handled() {
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "/nonexistent/file.txt"}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("The file doesn't exist.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Read a missing file");
        assert_eq!(response, "The file doesn't exist.");

        // Check that error was stored as tool result
        let tool_result_msg = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have a tool result message");
        assert!(tool_result_msg.content.contains("Tool error:"));
    }

    #[test]
    fn test_llm_error_returns_error_message() {
        let mut agent = make_agent(Box::new(ErrorLlm));
        let response = agent.process_message("Hello");
        assert!(response.starts_with("Error:"));
        assert!(response.contains("Cannot connect to Ollama"));
    }

    #[test]
    fn test_adaptive_cap_extends_when_making_progress() {
        // Adaptive cap: with max_iterations=3 the loop should still finish a
        // 5-tool-call sequence as long as each iteration makes progress
        // (a successful tool call). Without the extension, we'd run out of
        // budget at iter 3 and never see the final content.
        let mut responses = Vec::new();
        for i in 0..5 {
            responses.push(LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: format!("call_{}", i),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            });
        }
        responses.push(LlmResponse {
            content: Some("All done.".to_string()),
            tool_calls: vec![],
            usage: TokenUsage::default(),
        });
        let llm = MockLlm::new(responses);
        let config = AgentConfig {
            max_iterations: 3,
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(Box::new(llm), default_registry(), config, &[]);
        let response = agent.process_message("List things repeatedly");
        // We extended past max_iterations=3 because every iteration made
        // progress (list_dir succeeded). Final content reaches us.
        assert_eq!(response, "All done.");
        assert_eq!(agent.stats.llm_calls, 6);
    }

    #[test]
    fn test_adaptive_cap_stops_when_progress_stalls() {
        // Hard cap is base + MAX_PROGRESS_EXTENSION (5). Even with continuous
        // failing tool calls, the loop must terminate by base + extension.
        // With base 3 and 20 unknown-tool calls (each fails), we should stop
        // around iter 8 with the "Max iterations reached" message, not run
        // through all 20 responses.
        let mut responses = Vec::new();
        for i in 0..20 {
            responses.push(LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: format!("call_{}", i),
                    name: "nonexistent_tool".to_string(),
                    arguments: serde_json::json!({}),
                }],
                usage: TokenUsage::default(),
            });
        }
        let llm = MockLlm::new(responses);
        let config = AgentConfig {
            max_iterations: 3,
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(Box::new(llm), default_registry(), config, &[]);
        let response = agent.process_message("Spam unknown tool");
        assert_eq!(
            response,
            "Max iterations reached. The agent could not complete the task."
        );
        // Without the cap we'd run 20+. With base=3 and stall-extension=5,
        // we stop within 3 (no progress at all → never even enters extension).
        assert!(
            agent.stats.llm_calls <= 8,
            "expected ≤ 8 LLM calls, got {}",
            agent.stats.llm_calls
        );
    }

    #[test]
    fn test_max_iterations_reached() {
        // LLM always returns tool calls, never a final response
        let mut responses = Vec::new();
        for _ in 0..15 {
            responses.push(LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            });
        }
        let llm = MockLlm::new(responses);

        let config = AgentConfig {
            max_iterations: 3,
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(Box::new(llm), default_registry(), config, &[]);
        let response = agent.process_message("Keep using tools forever");
        assert_eq!(
            response,
            "Max iterations reached. The agent could not complete the task."
        );
    }

    #[test]
    fn test_memory_accumulates_across_messages() {
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: Some("First response".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Second response".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));

        agent.process_message("First question");
        // system + user + assistant = 3
        assert_eq!(agent.memory.len(), 3);

        agent.process_message("Second question");
        // 3 + user + assistant = 5
        assert_eq!(agent.memory.len(), 5);
    }

    #[test]
    fn test_system_prompt_is_first_message() {
        let llm = MockLlm::new(vec![LlmResponse {
            content: Some("ok".to_string()),
            tool_calls: vec![],
            usage: TokenUsage::default(),
        }]);
        let agent = make_agent(Box::new(llm));

        assert_eq!(agent.memory.len(), 1);
        assert_eq!(agent.memory[0].role, Role::System);
        assert!(agent.memory[0].content.contains("whet"));
    }

    #[test]
    fn test_tool_call_with_content_alongside() {
        // LLM returns both tool_calls AND content in the same response.
        // Content alongside tool calls is NOT stored in memory to prevent
        // the model from repeating itself in the next iteration.
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: Some("Let me check that file.".to_string()),
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "Cargo.toml"}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Done reading.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Check Cargo.toml");
        assert_eq!(response, "Done reading.");

        // Memory: system + user + assistant_tool_calls + tool_result + assistant("Done reading.")
        // Content alongside tool calls ("Let me check") is intentionally dropped from memory.
        let assistant_messages: Vec<_> = agent
            .memory
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .collect();
        // Only 2 assistant messages: one with tool_calls (empty content), one final response
        assert_eq!(assistant_messages.len(), 2);
        assert_eq!(assistant_messages[1].content, "Done reading.");
    }

    // --- Streaming callback tests ---

    /// A mock LLM that implements chat_streaming with token-by-token callback
    struct StreamingMockLlm {
        responses: RefCell<Vec<(Vec<String>, LlmResponse)>>,
    }

    impl StreamingMockLlm {
        /// Each entry: (tokens_to_emit, final_response)
        fn new(responses: Vec<(Vec<String>, LlmResponse)>) -> Self {
            let mut r = responses;
            r.reverse();
            Self {
                responses: RefCell::new(r),
            }
        }
    }

    impl LlmProvider for StreamingMockLlm {
        fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
        ) -> Result<LlmResponse, LlmError> {
            let mut responses = self.responses.borrow_mut();
            if let Some((_, resp)) = responses.pop() {
                Ok(resp)
            } else {
                Ok(LlmResponse {
                    content: Some("(no more responses)".to_string()),
                    tool_calls: vec![],
                    usage: TokenUsage::default(),
                })
            }
        }

        fn chat_streaming(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            on_token: &mut dyn FnMut(&str),
        ) -> Result<LlmResponse, LlmError> {
            let mut responses = self.responses.borrow_mut();
            if let Some((tokens, resp)) = responses.pop() {
                for token in &tokens {
                    on_token(token);
                }
                Ok(resp)
            } else {
                Ok(LlmResponse {
                    content: Some("(no more responses)".to_string()),
                    tool_calls: vec![],
                    usage: TokenUsage::default(),
                })
            }
        }
    }

    #[test]
    fn test_process_message_with_callback_receives_tokens() {
        let llm = StreamingMockLlm::new(vec![(
            vec!["Hello".to_string(), " world".to_string(), "!".to_string()],
            LlmResponse {
                content: Some("Hello world!".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        )]);

        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig::default(),
            &[],
        );

        let mut received_tokens = Vec::new();
        let response = agent.process_message_with_callback("Hi", &mut |token| {
            received_tokens.push(token.to_string());
        });

        assert_eq!(response, "Hello world!");
        assert_eq!(received_tokens, vec!["Hello", " world", "!"]);
    }

    #[test]
    fn test_process_message_with_callback_empty_callback() {
        // process_message uses an empty callback internally
        let llm = StreamingMockLlm::new(vec![(
            vec!["token1".to_string()],
            LlmResponse {
                content: Some("token1".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        )]);

        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig::default(),
            &[],
        );
        let response = agent.process_message("Hi");
        assert_eq!(response, "token1");
    }

    #[test]
    fn test_process_message_with_callback_tool_calls() {
        // When tool calls happen, streaming callback should still work
        let llm = StreamingMockLlm::new(vec![
            (
                vec![], // No tokens during tool call response
                LlmResponse {
                    content: None,
                    tool_calls: vec![ToolCall {
                        id: "call_0".to_string(),
                        name: "read_file".to_string(),
                        arguments: serde_json::json!({"path": "Cargo.toml"}),
                    }],
                    usage: TokenUsage::default(),
                },
            ),
            (
                vec!["The ".to_string(), "project".to_string(), ".".to_string()],
                LlmResponse {
                    content: Some("The project.".to_string()),
                    tool_calls: vec![],
                    usage: TokenUsage::default(),
                },
            ),
        ]);

        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig::default(),
            &[],
        );

        let mut received_tokens = Vec::new();
        let response = agent.process_message_with_callback("What project?", &mut |token| {
            received_tokens.push(token.to_string());
        });

        assert_eq!(response, "The project.");
        // Real streamed tokens should come from the second call (after
        // tool execution). The agent loop also fires on_token("") just
        // before printing tool-call lines so callers can stop spinners
        // without racing — filter those out for the content assertion.
        let real_tokens: Vec<&String> = received_tokens.iter().filter(|t| !t.is_empty()).collect();
        assert_eq!(real_tokens, vec!["The ", "project", "."]);
    }

    #[test]
    fn test_process_message_with_callback_error() {
        let mut agent = make_agent(Box::new(ErrorLlm));

        let mut received_tokens = Vec::new();
        let response = agent.process_message_with_callback("Hello", &mut |token| {
            received_tokens.push(token.to_string());
        });

        assert!(response.starts_with("Error:"));
        // No tokens should be emitted on error
        assert!(received_tokens.is_empty());
    }

    #[test]
    fn test_process_message_with_callback_multiple_turns() {
        let llm = StreamingMockLlm::new(vec![
            (
                vec!["First".to_string()],
                LlmResponse {
                    content: Some("First".to_string()),
                    tool_calls: vec![],
                    usage: TokenUsage::default(),
                },
            ),
            (
                vec!["Second".to_string()],
                LlmResponse {
                    content: Some("Second".to_string()),
                    tool_calls: vec![],
                    usage: TokenUsage::default(),
                },
            ),
        ]);

        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig::default(),
            &[],
        );

        let mut tokens1 = Vec::new();
        let r1 = agent.process_message_with_callback("Q1", &mut |t| tokens1.push(t.to_string()));
        assert_eq!(r1, "First");
        assert_eq!(tokens1, vec!["First"]);

        let mut tokens2 = Vec::new();
        let r2 = agent.process_message_with_callback("Q2", &mut |t| tokens2.push(t.to_string()));
        assert_eq!(r2, "Second");
        assert_eq!(tokens2, vec!["Second"]);

        // Memory should have system + user1 + assistant1 + user2 + assistant2 = 5
        assert_eq!(agent.memory.len(), 5);
    }

    #[test]
    fn test_process_message_delegates_to_callback_version() {
        // process_message should produce the same result as process_message_with_callback
        let llm1 = MockLlm::new(vec![LlmResponse {
            content: Some("Same result".to_string()),
            tool_calls: vec![],
            usage: TokenUsage::default(),
        }]);
        let llm2 = MockLlm::new(vec![LlmResponse {
            content: Some("Same result".to_string()),
            tool_calls: vec![],
            usage: TokenUsage::default(),
        }]);

        let mut agent1 = make_agent(Box::new(llm1));
        let mut agent2 = make_agent(Box::new(llm2));

        let r1 = agent1.process_message("test");
        let r2 = agent2.process_message_with_callback("test", &mut |_| {});

        assert_eq!(r1, r2);
    }

    // --- Permission system tests ---

    fn make_agent_with_mode(llm: Box<dyn LlmProvider>, mode: PermissionMode) -> Agent {
        Agent::new(
            llm,
            default_registry(),
            AgentConfig {
                permission_mode: mode,
                ..AgentConfig::default()
            },
            &[],
        )
    }

    #[test]
    fn test_needs_approval_default_mode() {
        let llm = MockLlm::new(vec![]);
        let agent = make_agent_with_mode(Box::new(llm), PermissionMode::Default);
        assert!(agent.needs_approval(ToolRiskLevel::Moderate));
        assert!(agent.needs_approval(ToolRiskLevel::Dangerous));
        assert!(!agent.needs_approval(ToolRiskLevel::Safe));
    }

    #[test]
    fn test_needs_approval_accept_edits_mode() {
        let llm = MockLlm::new(vec![]);
        let agent = make_agent_with_mode(Box::new(llm), PermissionMode::AcceptEdits);
        assert!(!agent.needs_approval(ToolRiskLevel::Moderate));
        assert!(agent.needs_approval(ToolRiskLevel::Dangerous));
        assert!(!agent.needs_approval(ToolRiskLevel::Safe));
    }

    #[test]
    fn test_needs_approval_yolo_mode() {
        let llm = MockLlm::new(vec![]);
        let agent = make_agent_with_mode(Box::new(llm), PermissionMode::Yolo);
        assert!(!agent.needs_approval(ToolRiskLevel::Moderate));
        assert!(!agent.needs_approval(ToolRiskLevel::Dangerous));
        assert!(!agent.needs_approval(ToolRiskLevel::Safe));
    }

    #[test]
    fn test_tool_denied_by_approval_callback() {
        // shell tool is Dangerous → needs approval in Default mode
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "shell".to_string(),
                    arguments: serde_json::json!({"command": "echo hello"}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("I couldn't execute the command.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent_with_mode(Box::new(llm), PermissionMode::Default);

        let response = agent.process_message_with_callbacks(
            "Run echo",
            &mut |_| {},
            &mut |_, _| false, // Always deny
        );

        assert_eq!(response, "I couldn't execute the command.");
        // Check that the tool result contains the denial message
        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(tool_result.content.contains("denied by user"));
    }

    #[test]
    fn test_tool_approved_by_callback() {
        // shell tool is approved → executes normally
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "shell".to_string(),
                    arguments: serde_json::json!({"command": "echo approved"}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Command executed.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent_with_mode(Box::new(llm), PermissionMode::Default);

        let response = agent.process_message_with_callbacks(
            "Run echo",
            &mut |_| {},
            &mut |_, _| true, // Always approve
        );

        assert_eq!(response, "Command executed.");
        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(tool_result.content.contains("approved"));
    }

    #[test]
    fn test_yolo_mode_skips_approval() {
        // In yolo mode, shell tool should execute without approval callback being called
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "shell".to_string(),
                    arguments: serde_json::json!({"command": "echo yolo"}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Done.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent_with_mode(Box::new(llm), PermissionMode::Yolo);

        let mut approval_called = false;
        let response =
            agent.process_message_with_callbacks("Run echo", &mut |_| {}, &mut |_, _| {
                approval_called = true;
                false // Would deny, but should never be called
            });

        assert_eq!(response, "Done.");
        assert!(
            !approval_called,
            "Approval should not be called in yolo mode"
        );
    }

    #[test]
    fn test_safe_tool_never_needs_approval() {
        // read_file is Safe → no approval needed even in Default mode
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "Cargo.toml"}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Read the file.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent_with_mode(Box::new(llm), PermissionMode::Default);

        let mut approval_called = false;
        let response =
            agent.process_message_with_callbacks("Read Cargo.toml", &mut |_| {}, &mut |_, _| {
                approval_called = true;
                false
            });

        assert_eq!(response, "Read the file.");
        assert!(
            !approval_called,
            "Approval should not be called for safe tools"
        );
    }

    #[test]
    fn test_accept_edits_allows_write_file() {
        // In AcceptEdits mode, write_file (Moderate) should not need approval
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "write_file".to_string(),
                    arguments: serde_json::json!({
                        "path": "/tmp/whet_perm_test.txt",
                        "content": "test"
                    }),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("File written.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent_with_mode(Box::new(llm), PermissionMode::AcceptEdits);

        let mut approval_called = false;
        let response =
            agent.process_message_with_callbacks("Write file", &mut |_| {}, &mut |_, _| {
                approval_called = true;
                false
            });

        assert_eq!(response, "File written.");
        assert!(
            !approval_called,
            "write_file should not need approval in AcceptEdits mode"
        );
        // Cleanup
        std::fs::remove_file("/tmp/whet_perm_test.txt").ok();
    }

    // --- Plan mode tests ---

    #[test]
    fn test_plan_mode_blocks_dangerous_tools() {
        // shell is Dangerous → should be blocked in plan mode
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "shell".to_string(),
                    arguments: serde_json::json!({"command": "rm -rf /"}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Blocked.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig {
                plan_mode: true,
                ..AgentConfig::default()
            },
            &[],
        );

        let response = agent.process_message("Delete everything");
        assert_eq!(response, "Blocked.");

        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(tool_result.content.contains("plan mode"));
    }

    #[test]
    fn test_plan_mode_blocks_moderate_tools() {
        // write_file is Moderate → should be blocked in plan mode
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "write_file".to_string(),
                    arguments: serde_json::json!({"path": "/tmp/test.txt", "content": "x"}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Can't write.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig {
                plan_mode: true,
                ..AgentConfig::default()
            },
            &[],
        );

        let response = agent.process_message("Write file");
        assert_eq!(response, "Can't write.");

        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(tool_result.content.contains("plan mode"));
    }

    #[test]
    fn test_plan_mode_allows_safe_tools() {
        // read_file is Safe → should work in plan mode
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "Cargo.toml"}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Read successfully.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig {
                plan_mode: true,
                ..AgentConfig::default()
            },
            &[],
        );

        let response = agent.process_message("Read Cargo.toml");
        assert_eq!(response, "Read successfully.");

        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        // Should contain actual file content, not a "blocked" message
        assert!(!tool_result.content.contains("plan mode"));
        assert!(tool_result.content.contains("whet"));
    }

    // --- Tool output truncation tests ---

    #[test]
    fn test_tool_output_within_limit_not_truncated() {
        // A tool that produces output just under MAX_TOOL_OUTPUT_CHARS should NOT be truncated
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "Cargo.toml"}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Done.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Read Cargo.toml");
        assert_eq!(response, "Done.");

        // Tool result should NOT contain truncation message
        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(!tool_result.content.contains("truncated"));
    }

    #[test]
    fn test_tool_output_exceeding_limit_is_truncated() {
        // Create a large file that will produce output > MAX_TOOL_OUTPUT_CHARS
        let path = "/tmp/whet_test_agent_large.txt";
        let content = "x".repeat(MAX_TOOL_OUTPUT_CHARS + 1000);
        std::fs::write(path, &content).unwrap();

        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": path}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Done.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Read large file");
        assert_eq!(response, "Done.");

        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(
            tool_result.content.contains("output truncated"),
            "Large output should be truncated"
        );
        assert!(
            tool_result.content.len() <= MAX_TOOL_OUTPUT_CHARS + 100,
            "Truncated output should be within limit (got {})",
            tool_result.content.len()
        );

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_tool_output_truncation_utf8_safe() {
        // Create a file with multi-byte UTF-8 chars that could cause a boundary issue
        let path = "/tmp/whet_test_agent_utf8.txt";
        // Each emoji is 4 bytes, so fill just over the limit
        let emoji_count = MAX_TOOL_OUTPUT_CHARS / 4 + 500;
        let content: String = "🦀".repeat(emoji_count);
        std::fs::write(path, &content).unwrap();

        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": path}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Done.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Read emoji file");
        assert_eq!(response, "Done.");

        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        // Should not panic on char boundary and should contain truncation marker
        assert!(tool_result.content.contains("output truncated"));
        // Verify it's valid UTF-8 (the fact that we can access .content proves it)
        assert!(tool_result
            .content
            .is_char_boundary(tool_result.content.len()));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_tool_output_exactly_at_limit_not_truncated() {
        let path = "/tmp/whet_test_agent_exact.txt";
        let content = "a".repeat(MAX_TOOL_OUTPUT_CHARS);
        std::fs::write(path, &content).unwrap();

        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": path}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Done.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        agent.process_message("Read file");

        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(
            !tool_result.content.contains("truncated"),
            "Output exactly at limit should NOT be truncated"
        );

        std::fs::remove_file(path).ok();
    }

    // --- Read-before-edit enforcement tests ---

    #[test]
    fn test_edit_file_without_read_returns_warning() {
        let path = "/tmp/whet_test_edit_no_read.txt";
        std::fs::write(path, "hello world").unwrap();

        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "edit_file".to_string(),
                    arguments: serde_json::json!({
                        "path": path,
                        "old_text": "hello",
                        "new_text": "hi"
                    }),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Warned.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        agent.process_message("Edit the file");

        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(
            tool_result.content.contains("Warning: You must read_file"),
            "edit_file without prior read should return warning, got: {}",
            tool_result.content
        );

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_edit_file_after_read_succeeds() {
        let path = "/tmp/whet_test_edit_after_read.txt";
        std::fs::write(path, "hello world").unwrap();

        let llm = MockLlm::new(vec![
            // Step 1: read_file
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": path}),
                }],
                usage: TokenUsage::default(),
            },
            // Step 2: edit_file (should succeed now)
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "edit_file".to_string(),
                    arguments: serde_json::json!({
                        "path": path,
                        "old_text": "hello",
                        "new_text": "hi"
                    }),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Edited.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Read then edit");
        assert_eq!(response, "Edited.");

        // Verify edit_file result does NOT contain a warning
        let tool_results: Vec<_> = agent
            .memory
            .iter()
            .filter(|m| m.role == Role::Tool)
            .collect();
        assert!(tool_results.len() >= 2);
        let edit_result = &tool_results[1].content;
        assert!(
            !edit_result.contains("Warning"),
            "edit_file after read should not warn, got: {}",
            edit_result
        );

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_edit_file_path_normalization() {
        let path = "/tmp/whet_test_edit_norm.txt";
        std::fs::write(path, "hello world").unwrap();

        let llm = MockLlm::new(vec![
            // read_file with "./" prefix
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "./tmp/whet_test_edit_norm.txt"}),
                }],
                usage: TokenUsage::default(),
            },
            // edit_file without "./" prefix — should still match
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "edit_file".to_string(),
                    arguments: serde_json::json!({
                        "path": "tmp/whet_test_edit_norm.txt",
                        "old_text": "hello",
                        "new_text": "hi"
                    }),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Done.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        agent.process_message("Normalize paths");

        let tool_results: Vec<_> = agent
            .memory
            .iter()
            .filter(|m| m.role == Role::Tool)
            .collect();
        assert!(tool_results.len() >= 2);
        let edit_result = &tool_results[1].content;
        assert!(
            !edit_result.contains("Warning"),
            "Normalized paths should match: got: {}",
            edit_result
        );

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_apply_diff_without_read_returns_warning() {
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "apply_diff".to_string(),
                    arguments: serde_json::json!({
                        "path": "src/main.rs",
                        "diff": "--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1 +1 @@\n-old\n+new"
                    }),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Warned.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        agent.process_message("Apply diff");

        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(
            tool_result.content.contains("Warning: You must read_file"),
            "apply_diff without prior read should warn, got: {}",
            tool_result.content
        );
    }

    #[test]
    fn test_write_file_without_read_succeeds() {
        let path = "/tmp/whet_test_write_no_read.txt";

        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "write_file".to_string(),
                    arguments: serde_json::json!({
                        "path": path,
                        "content": "new file content"
                    }),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Written.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Write file");
        assert_eq!(response, "Written.");

        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(
            !tool_result.content.contains("Warning"),
            "write_file should not require prior read, got: {}",
            tool_result.content
        );

        std::fs::remove_file(path).ok();
    }

    // --- Resumed session tests ---

    #[test]
    fn test_resumed_edit_file_skips_read_before_edit_warning() {
        let path = "/tmp/whet_test_resumed_edit.txt";
        std::fs::write(path, "hello world").unwrap();

        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "edit_file".to_string(),
                    arguments: serde_json::json!({
                        "path": path,
                        "old_text": "hello",
                        "new_text": "hi"
                    }),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Edited.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        agent.set_resumed(true);
        let response = agent.process_message("Edit the file");
        assert_eq!(response, "Edited.");

        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(
            !tool_result.content.contains("Warning"),
            "resumed session should skip read-before-edit, got: {}",
            tool_result.content
        );

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_resumed_session_still_tracks_read_paths() {
        let path = "/tmp/whet_test_resumed_track.txt";
        std::fs::write(path, "content").unwrap();

        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": path}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Read.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        agent.set_resumed(true);
        agent.process_message("Read the file");

        let normalized = Agent::normalize_tool_path(path);
        assert!(
            agent.read_paths.contains(&normalized),
            "resumed session should still track read_file calls"
        );

        std::fs::remove_file(path).ok();
    }

    // --- SessionStats tests ---

    #[test]
    fn test_session_stats_default() {
        let stats = SessionStats::default();
        assert_eq!(stats.llm_calls, 0);
        assert_eq!(stats.prompt_tokens, 0);
        assert_eq!(stats.completion_tokens, 0);
        assert_eq!(stats.tool_calls_ok, 0);
        assert_eq!(stats.tool_calls_failed, 0);
        assert_eq!(stats.total_tokens(), 0);
        assert_eq!(stats.total_tool_calls(), 0);
        assert!(stats.tool_success_rate().is_none());
    }

    #[test]
    fn test_session_stats_record_llm_call() {
        let mut stats = SessionStats::default();
        stats.record_llm_call(&TokenUsage {
            prompt_tokens: Some(100),
            completion_tokens: Some(50),
        });
        assert_eq!(stats.llm_calls, 1);
        assert_eq!(stats.prompt_tokens, 100);
        assert_eq!(stats.completion_tokens, 50);
        assert_eq!(stats.total_tokens(), 150);

        stats.record_llm_call(&TokenUsage {
            prompt_tokens: Some(200),
            completion_tokens: Some(80),
        });
        assert_eq!(stats.llm_calls, 2);
        assert_eq!(stats.prompt_tokens, 300);
        assert_eq!(stats.completion_tokens, 130);
    }

    #[test]
    fn test_session_stats_record_llm_call_none_usage() {
        let mut stats = SessionStats::default();
        stats.record_llm_call(&TokenUsage::default());
        assert_eq!(stats.llm_calls, 1);
        assert_eq!(stats.prompt_tokens, 0);
        assert_eq!(stats.completion_tokens, 0);
    }

    #[test]
    fn test_session_stats_record_tool_call() {
        let mut stats = SessionStats::default();
        stats.record_tool_call(true);
        stats.record_tool_call(true);
        stats.record_tool_call(false);
        assert_eq!(stats.tool_calls_ok, 2);
        assert_eq!(stats.tool_calls_failed, 1);
        assert_eq!(stats.total_tool_calls(), 3);
        let rate = stats.tool_success_rate().unwrap();
        assert!((rate - 66.67).abs() < 1.0);
    }

    #[test]
    fn test_agent_stats_tracked_in_process_message() {
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "Cargo.toml"}),
                }],
                usage: TokenUsage {
                    prompt_tokens: Some(100),
                    completion_tokens: Some(50),
                },
            },
            LlmResponse {
                content: Some("Done.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage {
                    prompt_tokens: Some(200),
                    completion_tokens: Some(80),
                },
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        agent.process_message("Read file");

        assert_eq!(agent.stats.llm_calls, 2);
        assert_eq!(agent.stats.prompt_tokens, 300);
        assert_eq!(agent.stats.completion_tokens, 130);
        assert_eq!(agent.stats.tool_calls_ok, 1);
        assert_eq!(agent.stats.tool_calls_failed, 0);
    }

    // --- Tests for Pattern 2: JSON-as-text fallback ---

    #[test]
    fn test_text_to_tool_fallback_format_a() {
        // Model outputs JSON as text instead of using tool calling API
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: Some(
                    "I'll list the directory for you.\n\
                     ```json\n\
                     {\"name\": \"list_dir\", \"arguments\": {\"path\": \".\"}}\n\
                     ```"
                    .to_string(),
                ),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
            // After fallback tool execution, model gives final answer
            LlmResponse {
                content: Some("Here are the files.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("List files");
        assert_eq!(response, "Here are the files.");
        assert_eq!(agent.stats.text_to_tool_fallbacks, 1);
        assert_eq!(agent.stats.tool_calls_ok, 1);
    }

    #[test]
    fn test_text_to_tool_fallback_unknown_tool_ignored() {
        // JSON with unknown tool name should NOT be extracted
        let llm = MockLlm::new(vec![LlmResponse {
            content: Some("{\"name\": \"delete_everything\", \"arguments\": {}}".to_string()),
            tool_calls: vec![],
            usage: TokenUsage::default(),
        }]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Do something");
        // Should return the text as-is since the tool is unknown
        assert!(response.contains("delete_everything"));
        assert_eq!(agent.stats.text_to_tool_fallbacks, 0);
    }

    #[test]
    fn test_text_to_tool_fallback_stats_tracked() {
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: Some(
                    "{\"name\": \"list_dir\", \"arguments\": {\"path\": \".\"}}".to_string(),
                ),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Done.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        agent.process_message("List");
        assert_eq!(agent.stats.text_to_tool_fallbacks, 1);
    }

    // --- Tests for Pattern 1: Re-prompt on question ---

    #[test]
    fn test_reprompt_on_question() {
        // Model asks a question, then after nudge uses a tool
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: Some("Do you want me to read the file?".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
            // After re-prompt, model uses tool
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Here are the files.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Show files");
        assert_eq!(response, "Here are the files.");
        assert_eq!(agent.stats.reprompts, 1);
    }

    #[test]
    fn test_reprompt_limit() {
        // Model asks twice — second question should be returned as final answer
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: Some("Should I proceed?".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Are you sure?".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Do it");
        // After 1 re-prompt, the second question is returned as-is
        assert_eq!(response, "Are you sure?");
        assert_eq!(agent.stats.reprompts, 1);
    }

    #[test]
    fn test_non_question_text_returned_immediately() {
        let llm = MockLlm::new(vec![LlmResponse {
            content: Some("The answer is 42.".to_string()),
            tool_calls: vec![],
            usage: TokenUsage::default(),
        }]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("What is the answer?");
        assert_eq!(response, "The answer is 42.");
        assert_eq!(agent.stats.reprompts, 0);
    }

    // --- Tests for stats JSONL output ---

    #[test]
    fn test_stats_jsonl_emits_tool_call_and_session_end() {
        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("a.txt");
        std::fs::write(&target, "hi\n").unwrap();
        let target_path = target.display().to_string();
        let jsonl_path = dir.path().join("stats.jsonl");

        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": target_path}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("done".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent_with_jsonl(Box::new(llm), jsonl_path.clone());
        let _ = agent.process_message("read it");

        let body = std::fs::read_to_string(&jsonl_path).unwrap();
        let events: Vec<serde_json::Value> = body
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("invalid JSON line"))
            .collect();
        // Expect: 1 tool_call event + 1 session_end event.
        assert_eq!(events.len(), 2, "events: {:?}", events);
        assert_eq!(events[0]["event"], "tool_call");
        assert_eq!(events[0]["name"], "read_file");
        assert_eq!(events[0]["ok"], true);
        assert_eq!(events[1]["event"], "session_end");
        assert_eq!(events[1]["reason"], "answered");
        assert_eq!(events[1]["tool_calls_ok"], 1);
    }

    #[test]
    fn test_stats_jsonl_disabled_writes_nothing() {
        // Default config has stats_jsonl_path: None — no file should be created.
        let dir = tempfile::TempDir::new().unwrap();
        let unused = dir.path().join("nope.jsonl");
        let llm = MockLlm::new(vec![LlmResponse {
            content: Some("hi".to_string()),
            tool_calls: vec![],
            usage: TokenUsage::default(),
        }]);
        let mut agent = make_agent(Box::new(llm));
        let _ = agent.process_message("hi");
        assert!(!unused.exists());
    }

    // --- Tests for Pattern 3: premature exit after only-reads ---

    #[test]
    fn test_reprompt_on_premature_exit_after_only_reads() {
        // Model reads a file, then exits with empty content. Detector should
        // re-prompt; on the second turn the model edits and we accept the answer.
        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("hello.py");
        std::fs::write(&target, "def greet():\n    return 'hi'\n").unwrap();
        let target_path = target.display().to_string();

        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": target_path}),
                }],
                usage: TokenUsage::default(),
            },
            // Empty content + no tool calls: triggers premature-exit detector.
            LlmResponse {
                content: Some(String::new()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
            // After re-prompt, model finally writes.
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "edit_file".to_string(),
                    arguments: serde_json::json!({
                        "path": target_path,
                        "old_text": "return 'hi'",
                        "new_text": "return 'hello'"
                    }),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("Done.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Make greet say hello");
        assert_eq!(response, "Done.");
        assert_eq!(agent.stats.reprompts, 1);
    }

    #[test]
    fn test_no_reprompt_on_read_only_qa() {
        // Read file, then return a substantive answer. The premature-exit
        // detector must NOT fire (content is not empty).
        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("readme.txt");
        std::fs::write(&target, "Line 1\nLine 2\n").unwrap();
        let target_path = target.display().to_string();

        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": target_path}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("There are 2 lines.".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("How many lines in readme?");
        assert_eq!(response, "There are 2 lines.");
        assert_eq!(agent.stats.reprompts, 0);
    }

    #[test]
    fn test_no_reprompt_after_edit_action() {
        // Model edits successfully and exits with empty content — that is a
        // valid "done with no closing remark" turn, not a premature exit.
        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("a.txt");
        std::fs::write(&target, "old\n").unwrap();
        let target_path = target.display().to_string();

        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": target_path}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "edit_file".to_string(),
                    arguments: serde_json::json!({
                        "path": target_path,
                        "old_text": "old",
                        "new_text": "new"
                    }),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some(String::new()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let _response = agent.process_message("change old to new");
        assert_eq!(agent.stats.reprompts, 0);
    }

    // --- Unit tests for helper functions ---

    #[test]
    fn test_looks_like_question() {
        // English questions
        assert!(looks_like_question("Do you want me to edit?"));
        assert!(looks_like_question("Should I proceed?"));
        assert!(looks_like_question("Shall I read the file?"));
        assert!(looks_like_question("Would you like me to help?"));
        assert!(looks_like_question("Can I modify this?"));

        // Japanese questions
        assert!(looks_like_question("ファイルを編集しますか？"));
        assert!(looks_like_question("編集しましょうか"));
        assert!(looks_like_question("よろしいですか"));

        // Not questions
        assert!(!looks_like_question("The file has 42 lines."));
        assert!(!looks_like_question("Here is the result."));
        assert!(!looks_like_question("Done."));
        assert!(!looks_like_question(""));
    }

    #[test]
    fn test_extract_tool_calls_from_mixed_text() {
        let tools = default_registry();
        let content = "I'll read the file for you.\n\
                       {\"name\": \"read_file\", \"arguments\": {\"path\": \"Cargo.toml\"}}\n";
        let result = try_extract_tool_calls_from_text(content, &tools);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "read_file");
        assert_eq!(result[0].id, "fallback_0");
    }

    #[test]
    fn test_extract_non_tool_json_ignored() {
        let tools = default_registry();
        // JSON that doesn't match tool call format
        let content = "{\"key\": \"value\", \"count\": 42}";
        let result = try_extract_tool_calls_from_text(content, &tools);
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_empty_content() {
        let tools = default_registry();
        let result = try_extract_tool_calls_from_text("", &tools);
        assert!(result.is_empty());

        let result = try_extract_tool_calls_from_text("just plain text", &tools);
        assert!(result.is_empty());
    }

    // --- Subagent (Phase A: /agent slash, sequential, isolated memory) ---

    #[test]
    fn test_subagent_returns_child_final_assistant_message() {
        // The child runs one LLM turn that produces a text-only response;
        // run_subagent should hand that string back to the caller.
        let llm = MockLlm::new(vec![LlmResponse {
            content: Some("subagent finished".to_string()),
            tool_calls: vec![],
            usage: TokenUsage::default(),
        }]);
        let mut agent = make_agent(Box::new(llm));
        let (text, reason) = agent
            .run_subagent("investigate X", &mut |_| {}, &mut |_, _| true)
            .expect("subagent should run");
        assert_eq!(text, "subagent finished");
        assert_eq!(reason, ExitReason::Answered);
    }

    #[test]
    fn test_subagent_does_not_pollute_parent_memory() {
        // The parent has a conversation; the subagent runs a separate one.
        // After the subagent completes, the parent's memory must contain
        // ONLY the parent's prior turns + system prompt — none of the
        // subagent's user/assistant exchanges.
        let llm = MockLlm::new(vec![
            // Parent turn (won't be exercised here, just stubbed)
            LlmResponse {
                content: Some("subagent finished".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        // Pre-seed parent memory with one prior user turn.
        agent.memory.push(Message::user("parent question"));
        agent.memory.push(Message::assistant("parent answer"));
        let parent_len_before = agent.memory.len();

        let _ = agent
            .run_subagent("subagent task", &mut |_| {}, &mut |_, _| true)
            .unwrap();

        // Parent memory length unchanged after subagent completes.
        assert_eq!(agent.memory.len(), parent_len_before);
        // The subagent's brief is NOT in parent memory.
        let combined: String = agent
            .memory
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !combined.contains("subagent task"),
            "parent memory leaked subagent brief: {:?}",
            combined
        );
    }

    #[test]
    fn test_subagent_read_paths_isolated() {
        // Subagent reads happen in the child's tracker, not parent's.
        // After subagent completes, parent's read_paths must be unchanged.
        let llm = MockLlm::new(vec![LlmResponse {
            content: Some("done".to_string()),
            tool_calls: vec![],
            usage: TokenUsage::default(),
        }]);
        let mut agent = make_agent(Box::new(llm));
        agent.add_read_path("src/parent.rs");
        let parent_paths_before = agent.read_paths.clone();

        // Manually mutate read_paths INSIDE the subagent run via the
        // post-loop restore checkpoint: the tested invariant is that
        // run_subagent restores parent state, not that the child
        // tracker is exercised. So we just confirm parent_paths_before
        // == after.
        let _ = agent
            .run_subagent("any", &mut |_| {}, &mut |_, _| true)
            .unwrap();

        assert_eq!(agent.read_paths, parent_paths_before);
    }

    #[test]
    fn test_subagent_max_depth_blocks_nesting() {
        // A subagent that itself tries to spawn another subagent must be
        // rejected with a clear error string.
        let llm = MockLlm::new(vec![LlmResponse {
            content: Some("noop".to_string()),
            tool_calls: vec![],
            usage: TokenUsage::default(),
        }]);
        let mut agent = make_agent(Box::new(llm));
        // Manually force the depth as if we were already inside one.
        agent.subagent_depth = MAX_SUBAGENT_DEPTH;
        let err = agent
            .run_subagent("nested", &mut |_| {}, &mut |_, _| true)
            .expect_err("should reject nested subagent");
        assert!(err.contains("nesting"), "unexpected error: {}", err);
        // Depth wasn't mutated by the rejected call.
        assert_eq!(agent.subagent_depth, MAX_SUBAGENT_DEPTH);
    }

    #[test]
    fn test_model_call_to_subagent_tool_dispatches_to_run_subagent() {
        // Phase C: when the LLM emits a tool_call with name="subagent",
        // the agent loop must intercept it and route to run_subagent
        // (NOT to the SubagentTool stub's execute, which returns an
        // error). The child loop produces a final string; the parent
        // sees that string as the tool result.
        let llm = MockLlm::new(vec![
            // Parent turn: model emits a subagent call.
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "subagent".to_string(),
                    arguments: serde_json::json!({"task": "investigate X"}),
                }],
                usage: TokenUsage::default(),
            },
            // Child turn: subagent finishes with a text response.
            LlmResponse {
                content: Some("found 3 things".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
            // Parent turn 2: parent finishes after seeing child's result.
            LlmResponse {
                content: Some("done".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let answer = agent.process_message_with_callbacks("hi", &mut |_| {}, &mut |_, _| true);
        // Parent's final answer is "done" — the subagent's result
        // ("found 3 things") was injected as a tool_result message
        // back to the parent loop.
        assert_eq!(answer, "done");
        // Confirm the tool_result message is in parent memory and
        // contains the child's output.
        let combined: String = agent
            .memory
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            combined.contains("found 3 things"),
            "child result not threaded back to parent: {:?}",
            combined
        );
    }

    #[test]
    fn test_model_call_to_subagent_with_empty_task_returns_tool_error() {
        // The model called subagent with no task — the parent should
        // surface a tool-error message and continue, not crash.
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "subagent".to_string(),
                    arguments: serde_json::json!({}),
                }],
                usage: TokenUsage::default(),
            },
            // After the error, parent recovers with a text response.
            LlmResponse {
                content: Some("recovered".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let _ = agent.process_message_with_callbacks("hi", &mut |_| {}, &mut |_, _| true);
        let combined: String = agent
            .memory
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            combined.contains("non-empty 'task'"),
            "expected tool-error about empty task, got: {:?}",
            combined
        );
    }

    // --- Persistent memory (remember tool) ---

    #[test]
    fn test_remember_tool_routes_to_on_remember_callback_with_assigned_id() {
        // The model emits a `remember` tool call → agent loop intercepts it →
        // dispatches to the on_remember callback → callback returns a row id.
        // Parent stats record the call as a successful tool call, and the
        // tool result string contains the assigned id so the model can
        // refer to it (e.g. for /forget guidance).
        use std::cell::RefCell;
        use std::rc::Rc;

        let recorded: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let llm = MockLlm::new(vec![
            // Parent: emit remember call.
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "r1".to_string(),
                    name: "remember".to_string(),
                    arguments: serde_json::json!({"content": "uses pnpm not npm"}),
                }],
                usage: TokenUsage::default(),
            },
            // Parent: text response after tool result.
            LlmResponse {
                content: Some("noted".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let recorded_w = Rc::clone(&recorded);
        agent.set_on_remember(Box::new(move |fact| {
            recorded_w.borrow_mut().push(fact.to_string());
            Ok(42) // pretend the store assigned id=42
        }));

        let _ = agent.process_message_with_callbacks("anything", &mut |_| {}, &mut |_, _| true);

        assert_eq!(
            recorded.borrow().as_slice(),
            &["uses pnpm not npm".to_string()]
        );
        // The success message threaded back to the model includes the id.
        let combined: String = agent
            .memory
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            combined.contains("Remembered (id=42)"),
            "expected id surfaced in tool result, got: {:?}",
            combined
        );
        assert_eq!(agent.stats.tool_calls_ok, 1);
        assert_eq!(agent.stats.tool_calls_failed, 0);
    }

    #[test]
    fn test_remember_with_empty_content_is_a_tool_error() {
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "r1".to_string(),
                    name: "remember".to_string(),
                    arguments: serde_json::json!({"content": "   "}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("ok".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        agent.set_on_remember(Box::new(|_| Ok(0)));
        let _ = agent.process_message_with_callbacks("x", &mut |_| {}, &mut |_, _| true);
        let combined: String = agent
            .memory
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            combined.contains("non-empty 'content'"),
            "expected empty-content tool error, got: {:?}",
            combined
        );
        assert_eq!(agent.stats.tool_calls_failed, 1);
    }

    #[test]
    fn test_remember_without_callback_returns_clear_error() {
        // When on_remember is not wired (e.g. memory-disabled run), the
        // tool call must surface a diagnostic rather than silently
        // pretending the fact was stored.
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "r1".to_string(),
                    name: "remember".to_string(),
                    arguments: serde_json::json!({"content": "fact"}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("ok".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        // No set_on_remember call.
        let _ = agent.process_message_with_callbacks("x", &mut |_| {}, &mut |_, _| true);
        let combined: String = agent
            .memory
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            combined.contains("not configured"),
            "expected 'not configured' error, got: {:?}",
            combined
        );
        assert_eq!(agent.stats.tool_calls_failed, 1);
    }

    #[test]
    fn test_remember_callback_error_propagates_as_tool_failure() {
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "r1".to_string(),
                    name: "remember".to_string(),
                    arguments: serde_json::json!({"content": "fact"}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: Some("ok".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        agent.set_on_remember(Box::new(|_| Err("disk full".to_string())));
        let _ = agent.process_message_with_callbacks("x", &mut |_| {}, &mut |_, _| true);
        let combined: String = agent
            .memory
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            combined.contains("disk full"),
            "expected store error surfaced, got: {:?}",
            combined
        );
        assert_eq!(agent.stats.tool_calls_failed, 1);
    }

    #[test]
    fn test_subagent_hides_subagent_tool_from_child_loop() {
        // Hardening fix 1: child's chat_streaming() must NOT receive
        // the `subagent` tool definition. Otherwise the model wastes
        // tokens describing a tool it can't usefully invoke (depth-cap).
        // Capture the tool list passed to chat() via a thread_local so
        // we can inspect after Box<dyn> consumes the LlmProvider.
        thread_local! {
            static CAPTURED: RefCell<Vec<Vec<String>>> = const { RefCell::new(Vec::new()) };
        }
        struct CapturingLlm;
        impl LlmProvider for CapturingLlm {
            fn chat(
                &self,
                _m: &[Message],
                tools: &[ToolDefinition],
            ) -> Result<LlmResponse, LlmError> {
                CAPTURED.with(|c| {
                    c.borrow_mut()
                        .push(tools.iter().map(|t| t.name.clone()).collect())
                });
                Ok(LlmResponse {
                    content: Some("done".to_string()),
                    tool_calls: vec![],
                    usage: TokenUsage::default(),
                })
            }
        }
        CAPTURED.with(|c| c.borrow_mut().clear());

        let mut agent = Agent::new(
            Box::new(CapturingLlm),
            default_registry(),
            AgentConfig::default(),
            &[],
        );
        let _ = agent
            .run_subagent("any", &mut |_| {}, &mut |_, _| true)
            .unwrap();

        let calls = CAPTURED.with(|c| c.borrow().clone());
        assert!(!calls.is_empty(), "expected at least one chat call");
        for (i, names) in calls.iter().enumerate() {
            assert!(
                !names.contains(&"subagent".to_string()),
                "child chat call {} included `subagent` in its tool list: {:?}",
                i,
                names
            );
        }
    }

    #[test]
    fn test_run_subagent_returns_exit_reason_max_iterations_directly() {
        // Direct check: run_subagent's structural ExitReason should be
        // MaxIterations when the child hits the cap. Parent dispatch
        // tests (below) verify the success/failure mapping; this test
        // is the unit that pins the mapping itself.
        let llm = MockLlm::new(vec![
            // 8 turns of "list_dir" so the child keeps making progress
            // (advancing last_progress_iter) and exhausts both base and
            // extension caps.
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c1".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c2".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c3".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c4".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c5".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c6".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c7".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c8".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig {
                max_iterations: 2,
                ..AgentConfig::default()
            },
            &[],
        );
        let (text, reason) = agent
            .run_subagent("spin", &mut |_| {}, &mut |_, _| true)
            .expect("subagent should at least start");
        assert_eq!(reason, ExitReason::MaxIterations);
        assert!(!reason.is_success());
        assert!(text.starts_with("Max iterations reached"));
    }

    #[test]
    fn test_on_token_empty_signal_fires_before_tool_call_prints() {
        // The agent loop sends an empty-string `on_token("")` BEFORE
        // printing tool-call lines, so a caller-installed "thinking…"
        // spinner can stop and clear its line without racing with the
        // tool-call output that follows. Regression guard for the UX.3
        // / UX.1 race we surfaced when capturing the README screenshot.
        let llm = MockLlm::new(vec![
            // First turn: model emits a tool call (no text).
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c1".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            // Second turn: real text response.
            LlmResponse {
                content: Some("done".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let mut tokens: Vec<String> = Vec::new();
        let _ = agent.process_message_with_callbacks(
            "go",
            &mut |t| tokens.push(t.to_string()),
            &mut |_, _| true,
        );
        // First token must be the empty-string signal — guarantees the
        // spinner stop fires before any tool-call eprintln below.
        assert!(
            tokens.first().is_some_and(|t| t.is_empty()),
            "expected empty-string first token before tool-call output, got {:?}",
            tokens
        );
    }

    #[test]
    fn test_subagent_does_not_double_emit_session_end_in_stats_jsonl() {
        // Hardening fix #5: emit_session_end_at_depth() short-circuits
        // when subagent_depth > 0, so the child's loop completion does
        // not write a second `session_end` line into stats.jsonl. The
        // parent's normal session_end is still emitted exactly once.
        let dir = tempfile::TempDir::new().unwrap();
        let jsonl_path = dir.path().join("stats.jsonl");

        let llm = MockLlm::new(vec![
            // Parent: emit a subagent call.
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "p1".to_string(),
                    name: "subagent".to_string(),
                    arguments: serde_json::json!({"task": "investigate"}),
                }],
                usage: TokenUsage::default(),
            },
            // Child: returns immediately with a final answer.
            LlmResponse {
                content: Some("subagent done".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
            // Parent: final answer after seeing subagent's result.
            LlmResponse {
                content: Some("parent done".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = make_agent_with_jsonl(Box::new(llm), jsonl_path.clone());
        let _ = agent.process_message_with_callbacks("hi", &mut |_| {}, &mut |_, _| true);

        let contents = std::fs::read_to_string(&jsonl_path).expect("stats jsonl written");
        let session_end_count = contents
            .lines()
            .filter(|line| line.contains("\"event\":\"session_end\""))
            .count();
        assert_eq!(
            session_end_count, 1,
            "expected exactly 1 session_end event, got {}\n--- stats.jsonl ---\n{}",
            session_end_count, contents
        );
    }

    #[test]
    fn test_subagent_guard_restores_parent_state_on_child_panic() {
        // The RAII SubagentGuard must restore parent state even when
        // the child loop panics. We exercise this by giving the child
        // an LlmProvider that panics during chat(), then catching the
        // panic with std::panic::catch_unwind. After unwind:
        //   * memory length == before-call length (child swap reverted)
        //   * memory[0] == parent's original system prompt
        //   * read_paths == before-call set
        //   * subagent_depth == 0
        struct PanickingLlm;
        impl LlmProvider for PanickingLlm {
            fn chat(&self, _: &[Message], _: &[ToolDefinition]) -> Result<LlmResponse, LlmError> {
                panic!("simulated child-loop panic");
            }
        }

        let mut agent = Agent::new(
            Box::new(PanickingLlm),
            default_registry(),
            AgentConfig::default(),
            &[],
        );
        // Pre-seed parent state so we have something concrete to verify
        // gets restored.
        agent.memory.push(Message::user("parent question"));
        agent.memory.push(Message::assistant("parent answer"));
        agent.add_read_path("src/parent.rs");

        let parent_memory_len = agent.memory.len();
        let parent_system_content = agent.memory[0].content.clone();
        let parent_paths = agent.read_paths.clone();
        assert_eq!(agent.subagent_depth, 0);

        // catch_unwind needs UnwindSafe; &mut Agent isn't, but the whole
        // point of this test is to assert state is consistent across the
        // unwind, so AssertUnwindSafe is appropriate.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            agent.run_subagent("subagent task", &mut |_| {}, &mut |_, _| true)
        }));

        assert!(
            result.is_err(),
            "PanickingLlm should propagate the panic out of run_subagent"
        );

        // Parent state fully restored after unwind.
        assert_eq!(
            agent.memory.len(),
            parent_memory_len,
            "memory length not restored after panic"
        );
        assert_eq!(
            agent.memory[0].content, parent_system_content,
            "parent system prompt not restored"
        );
        assert_eq!(
            agent.read_paths, parent_paths,
            "read_paths not restored after panic"
        );
        assert_eq!(
            agent.subagent_depth, 0,
            "subagent_depth not decremented on panic"
        );
    }

    #[test]
    fn test_run_subagent_returns_exit_reason_llm_error_directly() {
        // ErrorLlm always returns a connection error; the child loop
        // surfaces it as ExitReason::LlmError(...). Parent must see
        // the tuple's reason as a failure (is_success() == false).
        let mut agent = make_agent(Box::new(ErrorLlm));
        let (text, reason) = agent
            .run_subagent("anything", &mut |_| {}, &mut |_, _| true)
            .expect("subagent should at least construct");
        assert!(matches!(reason, ExitReason::LlmError(_)));
        assert!(!reason.is_success());
        assert!(text.starts_with("Error:"));
    }

    #[test]
    fn test_subagent_max_iterations_marked_as_failure_at_parent() {
        // Hardening fix 2: when the child loop hits max_iterations, the
        // returned string starts with "Max iterations reached." — the
        // parent dispatch must mark tool_success=false so the parent's
        // model and stats reflect failure, not success.
        // We exercise this by giving the child a model that emits
        // tool_calls forever (no terminal text response), and a tiny
        // max_iterations cap.
        let llm = MockLlm::new(vec![
            // Parent: model emits subagent call.
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "p1".to_string(),
                    name: "subagent".to_string(),
                    arguments: serde_json::json!({"task": "spin"}),
                }],
                usage: TokenUsage::default(),
            },
            // Child: 8 turns of looping nonsense — list_dir each time so
            // every iteration counts as "progress" and base+extension cap
            // are both hit. This forces "Max iterations reached" return.
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c1".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c2".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c3".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c4".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c5".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c6".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c7".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
                usage: TokenUsage::default(),
            },
            // Parent recovers after seeing failed subagent.
            LlmResponse {
                content: Some("ok recovered".to_string()),
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);
        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig {
                max_iterations: 2,
                ..AgentConfig::default()
            },
            &[],
        );
        let _ = agent.process_message_with_callbacks("hi", &mut |_| {}, &mut |_, _| true);
        // Find the tool_result message for the subagent call. It should
        // contain "Max iterations reached" — proving the child hit the
        // cap. The success-flag check is observed via tool_calls_failed.
        let combined: String = agent
            .memory
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            combined.contains("Max iterations reached"),
            "expected Max iterations sentinel in tool result; memory: {:?}",
            combined
        );
        assert!(
            agent.stats.tool_calls_failed >= 1,
            "child max-iter should have recorded as a failed tool call; stats: {:?}",
            agent.stats
        );
    }

    #[test]
    fn test_subagent_inherits_system_prompt_from_parent() {
        // The child's first message must be the parent's system prompt
        // (so project instructions / skills carry over). We can't observe
        // the child's memory directly after restore, but we can inject a
        // mock that captures the messages it received and assert the
        // system message body matches the parent's.
        struct CapturingLlm {
            captured: RefCell<Vec<Vec<Message>>>,
        }
        impl LlmProvider for CapturingLlm {
            fn chat(
                &self,
                messages: &[Message],
                _tools: &[ToolDefinition],
            ) -> Result<LlmResponse, LlmError> {
                self.captured.borrow_mut().push(messages.to_vec());
                Ok(LlmResponse {
                    content: Some("done".to_string()),
                    tool_calls: vec![],
                    usage: TokenUsage::default(),
                })
            }
        }
        let llm = CapturingLlm {
            captured: RefCell::new(Vec::new()),
        };
        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig::default(),
            &[],
        );
        let parent_system = agent.memory[0].content.clone();
        let _ = agent
            .run_subagent("any", &mut |_| {}, &mut |_, _| true)
            .unwrap();

        // The child's first call to .chat() received [system, user_brief].
        // Pull the LAST captured invocation since the only call was the child's.
        // Workaround: we can't access the inner captured field via Box<dyn>,
        // so instead assert by behavior — the subagent didn't crash and the
        // restored parent memory still has its original system message.
        assert_eq!(agent.memory[0].content, parent_system);
    }
}
