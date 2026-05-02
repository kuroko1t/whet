use clap::{Parser, Subcommand};
use colored::Colorize;

mod agent;
mod config;
mod llm;
mod mcp;
mod memory;
mod security;
mod skills;
mod tools;

use agent::{Agent, AgentConfig, SessionStats};
use config::Config;
use llm::LlmProvider;
use memory::store::MemoryStore;
use skills::Skill;
use tools::default_registry;

fn create_provider(cfg: &Config, model: &str) -> Box<dyn LlmProvider> {
    match cfg.llm.provider.as_str() {
        "openai_compat" => Box::new(llm::openai_compat::OpenAiCompatClient::with_options(
            &cfg.llm.base_url,
            model,
            cfg.llm.api_key.clone(),
            cfg.llm.options.clone(),
        )),
        _ => Box::new(llm::ollama::OllamaClient::with_options(
            &cfg.llm.base_url,
            model,
            cfg.llm.options.clone(),
        )),
    }
}

#[derive(Parser)]
#[command(name = "whet")]
#[command(version)]
#[command(
    about = "An open-source local-LLM terminal coding agent (Ollama / llama.cpp / LM Studio / vLLM)."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Single-shot prompt (e.g., whet "fix the bug")
    #[arg(trailing_var_arg = true, num_args = 0..)]
    prompt: Vec<String>,

    /// LLM model to use
    #[arg(short, long)]
    model: Option<String>,

    /// Resume a previous conversation (optional: session ID)
    #[arg(long, short = 'r', num_args = 0..=1, default_missing_value = "")]
    resume: Option<String>,

    /// Continue the most recent conversation
    #[arg(long, short = 'c')]
    continue_conv: bool,

    /// Single-shot mode: send a message and exit
    #[arg(short = 'p', long = "prompt")]
    message: Option<String>,

    /// Skip all permission prompts (yolo mode)
    #[arg(short = 'y', long = "yolo")]
    yolo: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// List available tools
    Tools,
    /// Show configuration
    Config,
}

fn ask_approval(tool_name: &str, args: &serde_json::Value) -> bool {
    use std::io::{self, Write};
    eprintln!(
        "\n{}",
        format!("  Tool '{}' wants to execute:", tool_name).yellow()
    );
    // Show a compact summary of the arguments
    match tool_name {
        "shell" => {
            if let Some(cmd) = args["command"].as_str() {
                eprintln!("    command: {}", cmd.bright_white());
            }
        }
        "write_file" => {
            if let Some(path) = args["path"].as_str() {
                let content = args["content"].as_str().unwrap_or("");
                eprintln!("    path: {}", path.bright_white());
                eprintln!("    content: {} bytes", content.len());
            }
        }
        "edit_file" => {
            if let Some(path) = args["path"].as_str() {
                eprintln!("    path: {}", path.bright_white());
            }
            if let Some(old) = args["old_text"].as_str() {
                let preview = if old.len() > 80 {
                    let mut end = 80;
                    while !old.is_char_boundary(end) {
                        end -= 1;
                    }
                    format!("{}...", &old[..end])
                } else {
                    old.to_string()
                };
                eprintln!("    old_text: {}", preview.dimmed());
            }
            if let Some(new) = args["new_text"].as_str() {
                let preview = if new.len() > 80 {
                    let mut end = 80;
                    while !new.is_char_boundary(end) {
                        end -= 1;
                    }
                    format!("{}...", &new[..end])
                } else {
                    new.to_string()
                };
                eprintln!("    new_text: {}", preview.green());
            }
        }
        "git" => {
            if let Some(cmd) = args["command"].as_str() {
                let git_args = args["args"].as_str().unwrap_or("");
                eprintln!("    git {} {}", cmd.bright_white(), git_args);
            }
        }
        _ => {
            eprintln!("    args: {}", args.to_string().dimmed());
        }
    }
    eprint!("  {} ", "Allow? [y/N/a(lways)]".bright_yellow());
    io::stderr().flush().ok();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_ok() {
        let answer = input.trim().to_lowercase();
        match answer.as_str() {
            "y" | "yes" => true,
            "a" | "always" => {
                eprintln!("  {}", "(Switching to yolo mode for this session)".dimmed());
                true // The caller will handle mode switch via return value
            }
            _ => {
                eprintln!("  {}", "Denied.".red());
                false
            }
        }
    } else {
        false
    }
}

fn setup_agent(cfg: &Config, model: &str, skills: &[Skill], yolo: bool) -> Agent {
    let provider = create_provider(cfg, model);
    let mut registry = default_registry();

    // Register web tools if enabled
    if cfg.agent.web_enabled {
        tools::register_web_tools(&mut registry);
    }

    // Register MCP tools
    if !cfg.mcp.servers.is_empty() {
        mcp::register_mcp_tools(&mut registry, &cfg.mcp.servers);
    }

    let permission_mode = if yolo {
        config::PermissionMode::Yolo
    } else {
        cfg.agent.permission_mode.clone()
    };

    let agent_config = AgentConfig {
        model: model.to_string(),
        max_iterations: cfg.agent.max_iterations,
        permission_mode,
        plan_mode: false,
        context_compression: cfg.agent.context_compression,
        compaction_token_threshold: resolve_compaction_threshold(cfg),
        stats_jsonl_path: std::env::var_os("WHET_STATS_JSONL").map(std::path::PathBuf::from),
    };

    Agent::new(provider, registry, agent_config, skills)
}

/// Pick the actual compaction-trigger token count, scaling with the
/// model's context window when possible. Precedence:
///   1. Ratio mode (the default): if `compaction_token_threshold_ratio
///      > 0` AND `[llm.options].num_ctx` is set, use `num_ctx × ratio`.
///   2. Absolute fallback: otherwise use `compaction_token_threshold`.
///
/// Rationale: a single global threshold is either too eager on large-
/// context models (compacts at 2 % of a 200 K window) or too lax on
/// small ones (never fires before overflow on a 4 K window). The
/// ratio-based resolution adapts to whatever model the user runs at
/// invocation time, while the absolute knob remains for users who
/// want a deterministic value regardless of provider/model.
fn resolve_compaction_threshold(cfg: &Config) -> usize {
    if cfg.agent.compaction_token_threshold_ratio > 0.0 {
        if let Some(num_ctx) = cfg.llm.options.num_ctx {
            // Ratio mode. Cast through f32 to apply the fraction;
            // ratio sanity-clamped on the high end so a typo of 6.0
            // (instead of 0.6) doesn't produce an absurd threshold.
            let ratio = cfg.agent.compaction_token_threshold_ratio.clamp(0.0, 0.95);
            return ((num_ctx as f32) * ratio) as usize;
        }
    }
    cfg.agent.compaction_token_threshold
}

/// Open (or reopen) the persistent-memory SQLite handle. Used for the
/// `remember` tool callback and the /memories|/remember|/forget slash
/// commands. Returns `None` if the DB file can't be opened (whet keeps
/// running without persistence in that case).
fn open_memory_handle(cfg: &Config) -> Option<std::rc::Rc<std::cell::RefCell<MemoryStore>>> {
    MemoryStore::new(&cfg.memory.database_path)
        .ok()
        .map(|s| std::rc::Rc::new(std::cell::RefCell::new(s)))
}

/// Append project-scoped + global persistent memories to the agent's
/// system prompt and wire the `on_remember` callback that the model's
/// `remember` tool uses to write back. Idempotent — call exactly once
/// per Agent. Injection is capped at `cap` rows (most-recently-updated
/// wins) to keep system-prompt growth bounded even when the user has
/// hundreds of memories saved.
fn wire_persistent_memory(
    agent: &mut Agent,
    memory_handle: Option<&std::rc::Rc<std::cell::RefCell<MemoryStore>>>,
    working_dir: &str,
    cap: usize,
) {
    let Some(handle) = memory_handle else { return };
    let all_mems = handle
        .borrow()
        .list_memories(working_dir)
        .unwrap_or_default();
    let total = all_mems.len();
    let mems: Vec<_> = all_mems.into_iter().take(cap).collect();
    let truncated = total > mems.len();

    if !mems.is_empty() {
        let mut section = String::from("\n\n## Persistent memory (from past sessions)\n");
        for m in &mems {
            let scope = if m.working_dir.is_some() {
                "project"
            } else {
                "global"
            };
            section.push_str(&format!("- [#{} {}] {}\n", m.id, scope, m.content));
        }
        if truncated {
            section.push_str(&format!(
                "\n(Showing {} most-recently-updated memories of {}; older ones omitted from this session — use /memories to see them all.)\n",
                mems.len(),
                total,
            ));
        }
        section.push_str(
            "\nThese facts were saved by you (or the user) in past sessions and are still active. \
             Honour them unless explicitly contradicted in the current turn.\n",
        );
        if let Some(first) = agent.memory.first_mut() {
            first.content.push_str(&section);
        }
        let label = if truncated {
            format!(
                "Loaded {} persistent memories ({} hidden).",
                mems.len(),
                total - mems.len()
            )
        } else {
            format!("Loaded {} persistent memories.", mems.len())
        };
        eprintln!("{}", label.dimmed());
    }
    let writer = std::rc::Rc::clone(handle);
    let dir = working_dir.to_string();
    agent.set_on_remember(Box::new(move |fact| {
        writer
            .borrow()
            .add_memory(Some(&dir), fact)
            .map_err(|e| e.to_string())
    }));
}

/// Resolve the conversation id to load on a single-shot invocation.
///
/// Returns `Some(id)` when:
///
///   - `-r <id>` was given with a non-empty id, OR
///   - `-c` was given AND a past conversation exists for `working_dir`.
///
/// Returns `None` when no resume was requested or no past conversation
/// matches; the caller should then create a fresh conversation.
fn resolve_single_shot_resume_id(
    memory_handle: Option<&std::rc::Rc<std::cell::RefCell<MemoryStore>>>,
    working_dir: &str,
    resume_arg: Option<&str>,
    continue_conv: bool,
) -> Option<String> {
    match resume_arg {
        Some(id) if !id.is_empty() => Some(id.to_string()),
        Some(_) => None, // empty --resume: picker not available in single-shot
        None if continue_conv => memory_handle.and_then(|h| {
            h.borrow()
                .get_latest_conversation_id(working_dir)
                .ok()
                .flatten()
        }),
        None => None,
    }
}

/// Load `id`'s messages into `agent.memory` from the store. No-op if
/// `id` is None or the conversation has no messages. Quiet (no
/// "you>"/"bot>" replay) — the REPL path has its own richer presentation.
fn load_resumed_history_into_agent_by_id(
    agent: &mut Agent,
    memory_handle: Option<&std::rc::Rc<std::cell::RefCell<MemoryStore>>>,
    id: Option<&str>,
) {
    let Some(id) = id else { return };
    let Some(handle) = memory_handle else { return };
    let Ok(messages) = handle.borrow().load_messages(id) else {
        return;
    };
    if messages.is_empty() {
        return;
    }

    let mut has_tool_messages = false;
    for (role, content, tool_call_id, tool_calls_json) in &messages {
        match role.as_str() {
            "user" => agent.memory.push(llm::Message::user(content)),
            "assistant" => {
                if let Some(tc_json) = tool_calls_json {
                    if let Ok(tool_calls) = serde_json::from_str::<Vec<llm::ToolCall>>(tc_json) {
                        for tc in &tool_calls {
                            if tc.name == "read_file" {
                                if let Some(p) = tc.arguments["path"].as_str() {
                                    agent.add_read_path(p);
                                }
                            }
                        }
                        agent
                            .memory
                            .push(llm::Message::assistant_with_tool_calls(tool_calls));
                    } else {
                        agent.memory.push(llm::Message::assistant(content));
                    }
                } else {
                    agent.memory.push(llm::Message::assistant(content));
                }
            }
            "tool" => {
                if let Some(tc_id) = tool_call_id {
                    has_tool_messages = true;
                    agent.memory.push(llm::Message::tool_result(tc_id, content));
                }
            }
            _ => {}
        }
    }
    if !has_tool_messages {
        agent.set_resumed(true);
    }
    eprintln!(
        "{}",
        format!("Resumed {} ({} messages).", id, messages.len()).dimmed()
    );
}

fn pick_session(store: &MemoryStore, working_dir: &str) -> Option<String> {
    let convs = match store.list_conversations(working_dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "{} Failed to list conversations: {}",
                "Warning:".yellow(),
                e
            );
            return None;
        }
    };
    if convs.is_empty() {
        println!("No previous sessions found in this directory.");
        return None;
    }

    let items: Vec<String> = convs
        .iter()
        .map(|c| {
            let title = c.title.as_deref().unwrap_or("(untitled)");
            let ago = format_time_ago(&c.updated_at);
            format!(
                "{:<50} {:>15}   {} messages",
                truncate_str(title, 50),
                ago,
                c.message_count
            )
        })
        .collect();

    println!("\n  {}\n", "Resuming conversation".bold());

    let selection = dialoguer::Select::new()
        .items(&items)
        .default(0)
        .interact_opt();

    match selection {
        Ok(Some(idx)) => Some(convs[idx].id.clone()),
        _ => None,
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut end = max_len.saturating_sub(3);
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

fn format_time_ago(rfc3339: &str) -> String {
    let Ok(dt) = chrono::DateTime::parse_from_rfc3339(rfc3339) else {
        return rfc3339.to_string();
    };
    let now = chrono::Utc::now();
    let dur = now.signed_duration_since(dt);

    if dur.num_minutes() < 1 {
        "just now".to_string()
    } else if dur.num_minutes() < 60 {
        format!("{} min ago", dur.num_minutes())
    } else if dur.num_hours() < 24 {
        let h = dur.num_hours();
        format!("{} hour{} ago", h, if h == 1 { "" } else { "s" })
    } else {
        let d = dur.num_days();
        format!("{} day{} ago", d, if d == 1 { "" } else { "s" })
    }
}

fn print_session_stats(stats: &SessionStats) {
    if stats.llm_calls == 0 {
        return;
    }
    eprintln!("{}", "--- Session Stats ---".dimmed());
    eprintln!("  LLM calls:          {}", stats.llm_calls);
    eprintln!("  Prompt tokens:      {}", stats.prompt_tokens);
    eprintln!("  Completion tokens:  {}", stats.completion_tokens);
    eprintln!("  Total tokens:       {}", stats.total_tokens());
    let total_tools = stats.total_tool_calls();
    if total_tools > 0 {
        eprintln!(
            "  Tool calls:         {} ({} ok / {} failed)",
            total_tools, stats.tool_calls_ok, stats.tool_calls_failed
        );
        if let Some(rate) = stats.tool_success_rate() {
            eprintln!("  Tool success rate:  {:.0}%", rate);
        }
    }
    if stats.text_to_tool_fallbacks > 0 {
        eprintln!("  Text->tool fallbacks: {}", stats.text_to_tool_fallbacks);
    }
    if stats.reprompts > 0 {
        eprintln!("  Re-prompts:         {}", stats.reprompts);
    }
    eprintln!("{}", "---------------------".dimmed());
}

fn run_chat(
    model: Option<String>,
    resume: Option<String>,
    continue_conv: bool,
    message: Option<String>,
    yolo: bool,
) {
    let cfg = Config::load();
    let model = model.unwrap_or(cfg.llm.model.clone());
    let loaded_skills = skills::load_skills(&cfg.agent.skills_dir);

    // Single-shot mode
    if let Some(msg) = message.filter(|m| !m.trim().is_empty()) {
        let mut agent = setup_agent(&cfg, &model, &loaded_skills, yolo);
        // Single-shot still benefits from persistent memory: the model
        // can `remember` facts from a one-off invocation, and recalls
        // facts from past sessions on startup.
        let single_shot_working_dir = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        let single_shot_memory = open_memory_handle(&cfg);
        wire_persistent_memory(
            &mut agent,
            single_shot_memory.as_ref(),
            &single_shot_working_dir,
            cfg.memory.max_inject_memories,
        );

        // Single-shot honour for --resume / --continue: load the past
        // conversation into agent.memory before processing the new
        // user message, so `whet -c -p "summarise"` actually summarises
        // the previous session instead of starting fresh.
        let resumed_id = resolve_single_shot_resume_id(
            single_shot_memory.as_ref(),
            &single_shot_working_dir,
            resume.as_deref(),
            continue_conv,
        );
        load_resumed_history_into_agent_by_id(
            &mut agent,
            single_shot_memory.as_ref(),
            resumed_id.as_deref(),
        );

        // Establish the conversation_id and create the row if new.
        // Without this, `whet -p "..."` invocations don't persist
        // their messages to SQLite, so a subsequent `whet -c -p`
        // can't find them — making `-c` a no-op in single-shot.
        let conversation_id = match resumed_id {
            Some(id) => id,
            None => {
                let new_id = uuid::Uuid::new_v4().to_string();
                if let Some(handle) = single_shot_memory.as_ref() {
                    let _ = handle
                        .borrow()
                        .create_conversation(&new_id, &single_shot_working_dir);
                }
                new_id
            }
        };
        let memory_before = agent.memory.len();

        if cfg.llm.streaming {
            let mut spinner = Some(agent::display::Spinner::start());
            agent.process_message_with_callbacks(
                &msg,
                &mut |token| {
                    if let Some(mut s) = spinner.take() {
                        s.stop();
                    }
                    print!("{}", token);
                },
                &mut |_, _| yolo,
            );
            if let Some(mut s) = spinner.take() {
                s.stop();
            }
            println!();
        } else {
            let response =
                agent.process_message_with_callbacks(&msg, &mut |_| {}, &mut |_, _| yolo);
            println!("{}", response);
        }
        // Persist the new turn (user input + assistant response + any
        // tool messages) so the next `whet -c -p ...` can resume.
        if let Some(handle) = single_shot_memory.as_ref() {
            let store = handle.borrow();
            for msg in &agent.memory[memory_before..] {
                let role = msg.role.to_string();
                let tc_json = if msg.tool_calls.is_empty() {
                    None
                } else {
                    serde_json::to_string(&msg.tool_calls).ok()
                };
                let _ = store.save_message(
                    &conversation_id,
                    &role,
                    &msg.content,
                    msg.tool_call_id.as_deref(),
                    tc_json.as_deref(),
                );
            }
        }
        print_session_stats(&agent.stats);
        return;
    }

    // Interactive mode
    println!("{}", "whet v0.1.0".bold());
    println!("Terminal coding agent.\n");
    println!("Model: {}", model.green());
    println!(
        "Permission: {}",
        if yolo {
            "yolo".yellow().to_string()
        } else {
            cfg.agent.permission_mode.to_string().cyan().to_string()
        }
    );
    if !loaded_skills.is_empty() {
        println!(
            "Skills: {}",
            loaded_skills
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
                .cyan()
        );
    }
    println!("Type {} to exit.\n", "Ctrl+D".dimmed());

    let mut agent = setup_agent(&cfg, &model, &loaded_skills, yolo);

    if cfg.agent.web_enabled {
        println!("Web tools: {}", "enabled".green());
    }

    // Memory store
    let store = match MemoryStore::new(&cfg.memory.database_path) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!(
                "{} Failed to open memory database: {}",
                "Warning:".yellow(),
                e
            );
            None
        }
    };

    let working_dir = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();

    // Persistent memory side-channel: a second MemoryStore handle that
    // the `remember` tool callback and the /memories|/forget slashes
    // share. Re-opens the same SQLite file as `store` — Rusqlite handles
    // concurrent connections to one DB cleanly. The helper also injects
    // any memories scoped to this working_dir into the agent's system
    // prompt and wires the on_remember callback.
    let memory_handle = open_memory_handle(&cfg);
    wire_persistent_memory(
        &mut agent,
        memory_handle.as_ref(),
        &working_dir,
        cfg.memory.max_inject_memories,
    );

    // Determine which conversation to load
    let resume_id: Option<String> = if let Some(ref resume_arg) = resume {
        if resume_arg.is_empty() {
            // --resume (no arg) → picker
            store.as_ref().and_then(|s| pick_session(s, &working_dir))
        } else {
            // --resume <id> → direct
            Some(resume_arg.clone())
        }
    } else if continue_conv {
        // --continue → latest
        store
            .as_ref()
            .and_then(|s| s.get_latest_conversation_id(&working_dir).ok().flatten())
    } else {
        None
    };

    let mut conversation_created = false;
    let conversation_id = if let Some(id) = resume_id {
        if let Some(ref store) = store {
            println!("Resuming conversation: {}", id.dimmed());
            // Load previous messages
            if let Ok(messages) = store.load_messages(&id) {
                let mut has_tool_messages = false;
                for (role, content, tool_call_id, tool_calls_json) in &messages {
                    match role.as_str() {
                        "user" => {
                            agent.memory.push(llm::Message::user(content));
                            println!(
                                "{} {}",
                                "you>".blue().bold(),
                                truncate_str(content, 200).dimmed()
                            );
                        }
                        "assistant" => {
                            if let Some(tc_json) = tool_calls_json {
                                if let Ok(tool_calls) =
                                    serde_json::from_str::<Vec<llm::ToolCall>>(tc_json)
                                {
                                    // Reconstruct read_paths
                                    for tc in &tool_calls {
                                        if tc.name == "read_file" {
                                            if let Some(p) = tc.arguments["path"].as_str() {
                                                agent.add_read_path(p);
                                            }
                                        }
                                    }
                                    agent
                                        .memory
                                        .push(llm::Message::assistant_with_tool_calls(tool_calls));
                                } else {
                                    agent.memory.push(llm::Message::assistant(content));
                                }
                            } else {
                                agent.memory.push(llm::Message::assistant(content));
                            }
                            if !content.is_empty() {
                                println!(
                                    "{} {}",
                                    "bot>".green().bold(),
                                    truncate_str(content, 200).dimmed()
                                );
                            }
                        }
                        "tool" => {
                            if let Some(tc_id) = tool_call_id {
                                has_tool_messages = true;
                                agent.memory.push(llm::Message::tool_result(tc_id, content));
                            }
                        }
                        _ => {}
                    }
                }
                println!(
                    "\n{}\n",
                    format!("Restored {} messages.", messages.len()).dimmed()
                );
                // Backward compat: if no tool messages were restored
                // (old DB without tool data), skip read-before-edit check
                if !has_tool_messages {
                    agent.set_resumed(true);
                }
            }
            conversation_created = true;
            id
        } else {
            uuid::Uuid::new_v4().to_string()
        }
    } else {
        if resume.is_some() || continue_conv {
            println!("No previous conversation found. Starting new.\n");
        }
        uuid::Uuid::new_v4().to_string()
    };
    // Track whether title has already been set (true for resumed conversations)
    let mut title_set = conversation_created;

    let mut current_model = model;
    let mut rl = match rustyline::DefaultEditor::new() {
        Ok(editor) => editor,
        Err(e) => {
            eprintln!("{} Failed to initialize readline: {}", "Error:".red(), e);
            return;
        }
    };

    // Load persistent input history
    let history_path = dirs::home_dir()
        .map(|h| h.join(".whet").join("history.txt"))
        .unwrap_or_default();
    if history_path.exists() {
        let _ = rl.load_history(&history_path);
    }

    loop {
        let readline = rl.readline(&format!("{} ", "you>".blue().bold()));
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(input);

                // Handle slash commands
                if input.starts_with('/') {
                    match handle_slash_command(
                        input,
                        &cfg,
                        &mut agent,
                        &mut current_model,
                        &loaded_skills,
                        memory_handle.as_ref(),
                        &working_dir,
                    ) {
                        SlashResult::Handled => continue,
                        SlashResult::NewProvider(provider) => {
                            agent.llm = provider;
                            continue;
                        }
                    }
                }

                // Create conversation lazily on first message
                if !conversation_created {
                    if let Some(ref store) = store {
                        if let Err(e) = store.create_conversation(&conversation_id, &working_dir) {
                            eprintln!(
                                "{} Failed to create conversation: {}",
                                "Warning:".yellow(),
                                e
                            );
                        }
                    }
                    conversation_created = true;
                }

                // Save user message
                if let Some(ref store) = store {
                    if let Err(e) = store.save_message(&conversation_id, "user", input, None, None)
                    {
                        eprintln!("{} Failed to save message: {}", "Warning:".yellow(), e);
                    }
                    // Auto-set title from first user message
                    if !title_set {
                        let title_len = input.len().min(50);
                        let mut end = title_len;
                        while end > 0 && !input.is_char_boundary(end) {
                            end -= 1;
                        }
                        let _ = store.update_conversation_title(&conversation_id, &input[..end]);
                        title_set = true;
                    }
                }

                let memory_before = agent.memory.len();

                let start = std::time::Instant::now();
                // Always run the spinner — it's TTY-aware (no-op outside
                // a terminal) and now also stops via the agent loop's
                // on_token("") notification before tool-call lines
                // render, which prevents the spinner from racing with
                // tool-call output on the same row.
                let streaming = cfg.llm.streaming;
                let mut spinner = Some(agent::display::Spinner::start());
                let mut bot_prefix_printed = false;
                let response = agent.process_message_with_callbacks(
                    input,
                    &mut |token| {
                        if let Some(mut s) = spinner.take() {
                            s.stop();
                        }
                        // The empty-string `on_token("")` is the
                        // "tool-calls-incoming" signal — useful for
                        // stopping the spinner but we mustn't print a
                        // `bot>` prefix because there's no bot text yet.
                        // Defer the prefix until a real (non-empty)
                        // streamed token arrives.
                        if streaming && !token.is_empty() {
                            if !bot_prefix_printed {
                                eprint!("{} ", "bot>".green().bold());
                                bot_prefix_printed = true;
                            }
                            eprint!("{}", token);
                        }
                    },
                    &mut |tool_name, args| ask_approval(tool_name, args),
                );
                if let Some(mut s) = spinner.take() {
                    s.stop();
                }
                if streaming {
                    if !bot_prefix_printed {
                        // Loop ended without any streamed text (e.g. all
                        // tool calls + final empty content) — print prefix
                        // + response so the user still sees a final line.
                        print!("{} {}", "bot>".green().bold(), response);
                    }
                    eprintln!();
                } else {
                    println!("{} {}", "bot>".green().bold(), response);
                }
                let _response = response;
                let elapsed = start.elapsed();
                println!("{}", format!("({:.1}s)", elapsed.as_secs_f64()).dimmed());
                println!();

                // Save all new messages (skip the user message at memory_before)
                if let Some(ref store) = store {
                    for msg in &agent.memory[memory_before + 1..] {
                        let role = msg.role.to_string();
                        let tc_json = if msg.tool_calls.is_empty() {
                            None
                        } else {
                            serde_json::to_string(&msg.tool_calls).ok()
                        };
                        let _ = store.save_message(
                            &conversation_id,
                            &role,
                            &msg.content,
                            msg.tool_call_id.as_deref(),
                            tc_json.as_deref(),
                        );
                    }
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("Use Ctrl+D to exit.");
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("\nGoodbye!");
                print_session_stats(&agent.stats);
                // Save input history
                if let Some(parent) = history_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = rl.save_history(&history_path);
                break;
            }
            Err(err) => {
                eprintln!("Error: {}", err);
                break;
            }
        }
    }
}

enum SlashResult {
    Handled,
    NewProvider(Box<dyn LlmProvider>),
}

fn handle_slash_command(
    input: &str,
    cfg: &Config,
    agent: &mut Agent,
    current_model: &mut String,
    skills: &[Skill],
    memory_handle: Option<&std::rc::Rc<std::cell::RefCell<MemoryStore>>>,
    working_dir: &str,
) -> SlashResult {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0];
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd {
        "/model" => {
            if arg.is_empty() {
                println!("Current model: {}", current_model.green());
                println!("Usage: {} <model_name>", "/model".cyan());
            } else {
                let new_model = arg.to_string();
                println!(
                    "Switching model: {} -> {}",
                    current_model.dimmed(),
                    new_model.green()
                );
                *current_model = new_model.clone();
                let provider = create_provider(cfg, &new_model);
                return SlashResult::NewProvider(provider);
            }
            SlashResult::Handled
        }
        "/mode" => {
            if arg.is_empty() {
                println!(
                    "Current permission mode: {}",
                    agent.config.permission_mode.to_string().cyan()
                );
                println!("Usage: {} <default|accept_edits|yolo>", "/mode".cyan());
            } else {
                match arg {
                    "default" => {
                        agent.config.permission_mode = config::PermissionMode::Default;
                        println!("Permission mode: {}", "default".cyan());
                    }
                    "accept_edits" => {
                        agent.config.permission_mode = config::PermissionMode::AcceptEdits;
                        println!("Permission mode: {}", "accept_edits".cyan());
                    }
                    "yolo" => {
                        agent.config.permission_mode = config::PermissionMode::Yolo;
                        println!("Permission mode: {}", "yolo".yellow());
                    }
                    _ => {
                        eprintln!(
                            "{} Unknown mode '{}'. Use: default, accept_edits, yolo",
                            "Error:".red(),
                            arg
                        );
                    }
                }
            }
            SlashResult::Handled
        }
        "/plan" => {
            agent.config.plan_mode = !agent.config.plan_mode;
            if agent.config.plan_mode {
                println!(
                    "{} Plan mode {} - read-only tools only",
                    ">>".green(),
                    "ON".green().bold()
                );
            } else {
                println!(
                    "{} Plan mode {} - all tools available",
                    ">>".yellow(),
                    "OFF".yellow().bold()
                );
            }
            SlashResult::Handled
        }
        "/skills" => {
            if skills.is_empty() {
                println!("No skills loaded.");
                println!(
                    "Place .md files in {} to add skills.",
                    cfg.agent.skills_dir.cyan()
                );
            } else {
                println!("{}", "Loaded skills:".bold());
                for skill in skills {
                    let preview = if skill.content.len() > 80 {
                        let mut end = 80;
                        while !skill.content.is_char_boundary(end) {
                            end -= 1;
                        }
                        format!("{}...", &skill.content[..end])
                    } else {
                        skill.content.clone()
                    };
                    println!("  {} - {}", skill.name.cyan(), preview.dimmed());
                }
            }
            SlashResult::Handled
        }
        "/init" => {
            let path = std::path::Path::new("WHET.md");
            if path.exists() {
                eprintln!(
                    "{} WHET.md already exists in the current directory.",
                    "Error:".red()
                );
            } else {
                let template = "\
# Project Instructions

## Build & Test
- (Add build commands here)

## Code Style
- Follow existing patterns in the codebase

## Important Context
- (Add project-specific notes here)
";
                match std::fs::write(path, template) {
                    Ok(_) => println!("{} Created {}", ">>".green(), "WHET.md".cyan()),
                    Err(e) => eprintln!("{} Failed to create WHET.md: {}", "Error:".red(), e),
                }
            }
            SlashResult::Handled
        }
        "/compact" => {
            let instruction = if arg.is_empty() { None } else { Some(arg) };
            agent.compact(instruction);
            SlashResult::Handled
        }
        "/help" => {
            println!("{}", "Available commands:".bold());
            println!("  {} <name>  - Switch LLM model", "/model".cyan());
            println!(
                "  {} <mode>  - Change permission mode (default/accept_edits/yolo)",
                "/mode".cyan()
            );
            println!(
                "  {}           - Toggle plan mode (read-only analysis)",
                "/plan".cyan()
            );
            println!(
                "  {} [cmd]   - Run test-fix loop (default: cargo test)",
                "/test".cyan()
            );
            println!(
                "  {}         - Run diagnostics (ollama, model, config, MCP, dirs)",
                "/doctor".cyan()
            );
            println!(
                "  {} <task>  - Run a focused subagent on <task> (isolated context).",
                "/agent".cyan()
            );
            println!(
                "             {}",
                "The agent itself can also call subagent autonomously.".dimmed()
            );
            println!(
                "  {}      - List project's persistent memories",
                "/memories".cyan()
            );
            println!(
                "  {} <fact> - Manually save a persistent memory",
                "/remember".cyan()
            );
            println!(
                "  {} <id>    - Soft-delete a persistent memory",
                "/forget".cyan()
            );
            println!("  {}           - Generate WHET.md template", "/init".cyan());
            println!(
                "  {} [msg] - Compress conversation context",
                "/compact".cyan()
            );
            println!("  {}         - List loaded skills", "/skills".cyan());
            println!(
                "  {}          - Clear conversation history",
                "/clear".cyan()
            );
            println!("  {}           - Show this help", "/help".cyan());
            println!("  {}           - Exit", "Ctrl+D".dimmed());
            SlashResult::Handled
        }
        "/test" => {
            let test_cmd = if arg.is_empty() { "cargo test" } else { arg };
            run_test_fix_loop(agent, test_cmd, cfg);
            SlashResult::Handled
        }
        "/doctor" => {
            run_doctor_command(cfg, current_model);
            SlashResult::Handled
        }
        "/agent" => {
            if arg.is_empty() {
                eprintln!("{} usage: /agent <task>", "Error:".red());
                return SlashResult::Handled;
            }
            run_agent_subtask(agent, arg, cfg);
            SlashResult::Handled
        }
        "/memories" => {
            run_memories_list(memory_handle, working_dir);
            SlashResult::Handled
        }
        "/remember" => {
            if arg.is_empty() {
                eprintln!("{} usage: /remember <fact>", "Error:".red());
                return SlashResult::Handled;
            }
            run_memories_add(memory_handle, working_dir, arg);
            SlashResult::Handled
        }
        "/forget" => {
            if arg.is_empty() {
                eprintln!("{} usage: /forget <id>", "Error:".red());
                return SlashResult::Handled;
            }
            run_memories_forget(memory_handle, arg);
            SlashResult::Handled
        }
        "/clear" => {
            agent.memory.clear();
            agent
                .memory
                .push(llm::Message::system(&agent::prompt::system_prompt(skills)));
            println!("{}", "Conversation cleared.".green());
            SlashResult::Handled
        }
        _ => {
            eprintln!(
                "{} Unknown command '{}'. Type {} for available commands.",
                "Error:".red(),
                cmd,
                "/help".cyan()
            );
            SlashResult::Handled
        }
    }
}

fn run_doctor_command(cfg: &Config, active_model: &str) {
    use agent::doctor::{format_row, overall_exit_code, run_all, DiagnosticStatus};

    // Production HTTP getter — short timeout because /doctor should fail fast.
    let fetch = |url: &str| -> Result<String, String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|e| e.to_string())?;
        let resp = client.get(url).send().map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        resp.text().map_err(|e| e.to_string())
    };

    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("/"));

    println!("{}", "Whet diagnostics:".bold());
    let rows = run_all(cfg, active_model, &home, fetch);
    for row in &rows {
        let line = format_row(row);
        let coloured = match row.status {
            DiagnosticStatus::Pass => line.green().to_string(),
            DiagnosticStatus::Warn => line.yellow().to_string(),
            DiagnosticStatus::Fail => line.red().to_string(),
        };
        println!("  {}", coloured);
    }
    let code = overall_exit_code(&rows);
    if code == 0 {
        println!("{}", "Overall: PASS".green().bold());
    } else {
        println!("{}", "Overall: FAIL".red().bold());
    }
}

/// Dispatch a `/agent <task>` slash invocation. Spawns a subagent with
/// isolated memory + read-paths, prints the result inline, and returns
/// to the parent REPL with parent state untouched.
///
/// In this Phase A, the subagent uses the same model/tools/config as the
/// parent and runs sequentially. Permission policy is the parent's
/// (yolo / accept_edits / default).
fn run_agent_subtask(agent: &mut Agent, task: &str, cfg: &Config) {
    println!("{} {}", "Subagent:".cyan().bold(), task.dimmed());

    let yolo = matches!(cfg.agent.permission_mode, config::PermissionMode::Yolo);
    let streaming = cfg.llm.streaming;

    let result = if streaming {
        let mut spinner = Some(agent::display::Spinner::start());
        let r = agent.run_subagent(
            task,
            &mut |token| {
                if let Some(mut s) = spinner.take() {
                    s.stop();
                }
                eprint!("{}", token);
            },
            &mut |_, _| yolo,
        );
        if let Some(mut s) = spinner.take() {
            s.stop();
        }
        eprintln!();
        r
    } else {
        agent.run_subagent(task, &mut |_| {}, &mut |_, _| yolo)
    };

    match result {
        Ok((text, reason)) => {
            let header = if reason.is_success() {
                "Subagent result:".cyan().bold()
            } else {
                "Subagent stopped early:".yellow().bold()
            };
            println!("{}", header);
            println!("{}", text);
        }
        Err(e) => {
            eprintln!("{} subagent failed: {}", "Error:".red(), e);
        }
    }
}

fn run_memories_list(
    memory_handle: Option<&std::rc::Rc<std::cell::RefCell<MemoryStore>>>,
    working_dir: &str,
) {
    let Some(handle) = memory_handle else {
        eprintln!("{} persistent memory is not configured", "Error:".red());
        return;
    };
    let mems = match handle.borrow().list_memories(working_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{} {}", "Error:".red(), e);
            return;
        }
    };
    if mems.is_empty() {
        println!("{}", "No persistent memories for this project.".dimmed());
        return;
    }
    println!("{}", "Persistent memories:".bold());
    for m in mems {
        let scope = if m.working_dir.is_some() {
            "project".dimmed()
        } else {
            "global".cyan().dimmed()
        };
        println!("  {} {} {}", format!("[{}]", m.id).cyan(), scope, m.content);
    }
}

fn run_memories_add(
    memory_handle: Option<&std::rc::Rc<std::cell::RefCell<MemoryStore>>>,
    working_dir: &str,
    fact: &str,
) {
    let Some(handle) = memory_handle else {
        eprintln!("{} persistent memory is not configured", "Error:".red());
        return;
    };
    match handle.borrow().add_memory(Some(working_dir), fact) {
        Ok(id) => println!(
            "{} {} {}",
            "Remembered".green().bold(),
            format!("[id={}]", id).dimmed(),
            fact
        ),
        Err(e) => eprintln!("{} {}", "Error:".red(), e),
    }
}

fn run_memories_forget(
    memory_handle: Option<&std::rc::Rc<std::cell::RefCell<MemoryStore>>>,
    id_str: &str,
) {
    let Some(handle) = memory_handle else {
        eprintln!("{} persistent memory is not configured", "Error:".red());
        return;
    };
    let id: i64 = match id_str.parse() {
        Ok(n) => n,
        Err(_) => {
            eprintln!("{} id must be a number, got {:?}", "Error:".red(), id_str);
            return;
        }
    };
    match handle.borrow().forget_memory(id) {
        Ok(true) => println!("{} memory id={}", "Forgot".green().bold(), id),
        Ok(false) => println!("{} no active memory with id={}", "Notice:".yellow(), id),
        Err(e) => eprintln!("{} {}", "Error:".red(), e),
    }
}

fn run_test_fix_loop(agent: &mut Agent, test_cmd: &str, cfg: &Config) {
    let max_fix_iterations = 5;

    for iteration in 1..=max_fix_iterations {
        println!(
            "\n{} Running tests (iteration {}/{}): {}",
            ">>".cyan(),
            iteration,
            max_fix_iterations,
            test_cmd.bright_white()
        );

        // Run the test command
        let output = std::process::Command::new("sh")
            .args(["-c", test_cmd])
            .output();

        let output = match output {
            Ok(o) => o,
            Err(e) => {
                eprintln!("{} Failed to run test command: {}", "Error:".red(), e);
                return;
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}{}", stdout, stderr);

        if output.status.success() {
            println!("{} All tests passed!", ">>".green().bold());
            return;
        }

        // Tests failed — show summary and ask agent to fix
        println!(
            "{} Tests failed (exit code: {})",
            ">>".red(),
            output.status.code().unwrap_or(-1)
        );

        // Truncate output if too long
        let max_output = 4000;
        let failure_output = if combined.len() > max_output {
            let mut start = combined.len() - max_output;
            while !combined.is_char_boundary(start) {
                start += 1;
            }
            format!("...(truncated)\n{}", &combined[start..])
        } else {
            combined.to_string()
        };

        let fix_prompt = format!(
            "The test command `{}` failed with the following output:\n\n```\n{}\n```\n\nPlease analyze the test failures and fix the code. Use the available tools to read the relevant files, understand the errors, and make the necessary changes.",
            test_cmd, failure_output
        );

        println!("{} Asking agent to fix...\n", ">>".cyan());

        let streaming = cfg.llm.streaming;
        let mut spinner = Some(agent::display::Spinner::start());
        let mut bot_prefix_printed = false;
        let response = agent.process_message_with_callbacks(
            &fix_prompt,
            &mut |token| {
                if let Some(mut s) = spinner.take() {
                    s.stop();
                }
                if streaming && !token.is_empty() {
                    if !bot_prefix_printed {
                        eprint!("{} ", "bot>".green().bold());
                        bot_prefix_printed = true;
                    }
                    eprint!("{}", token);
                }
            },
            &mut |tool_name, args| ask_approval(tool_name, args),
        );
        if let Some(mut s) = spinner.take() {
            s.stop();
        }
        if streaming {
            if !bot_prefix_printed {
                print!("{} {}", "bot>".green().bold(), response);
            }
            eprintln!();
        } else {
            println!("{} {}", "bot>".green().bold(), response);
        }

        let _ = response; // Agent's fix is already applied via tools
    }

    eprintln!(
        "{} Max fix iterations ({}) reached. Tests still failing.",
        "Warning:".yellow(),
        max_fix_iterations
    );
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Tools) => {
            let cfg = Config::load();
            let mut registry = default_registry();
            if cfg.agent.web_enabled {
                tools::register_web_tools(&mut registry);
            }
            if !cfg.mcp.servers.is_empty() {
                mcp::register_mcp_tools(&mut registry, &cfg.mcp.servers);
            }
            println!("{}", "Available tools:".bold());
            println!();
            for tool in registry.list() {
                println!("  {} - {}", tool.name().cyan(), tool.description());
            }
        }
        Some(Commands::Config) => {
            let config = Config::load();
            println!("{}", "Current configuration:".bold());
            println!();
            match toml::to_string_pretty(&config) {
                Ok(s) => println!("{}", s),
                Err(e) => eprintln!("Error serializing config: {}", e),
            }
        }
        None => {
            // Default: `whet` alone or `whet "fix the bug"`
            let prompt_text = cli.prompt.join(" ");
            let message = cli.message.or(if prompt_text.is_empty() {
                None
            } else {
                Some(prompt_text)
            });
            run_chat(cli.model, cli.resume, cli.continue_conv, message, cli.yolo);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AgentConfig as CfgAgent, LlmConfig, LlmOptions, McpConfig, MemoryConfig, PermissionMode,
    };

    fn make_cfg(num_ctx: Option<u32>, ratio: f32, abs: usize) -> Config {
        let options = LlmOptions {
            num_ctx,
            ..LlmOptions::default()
        };
        Config {
            llm: LlmConfig {
                provider: "ollama".to_string(),
                model: "qwen3.5:9b".to_string(),
                base_url: "http://localhost:11434".to_string(),
                api_key: None,
                streaming: false,
                options,
            },
            agent: CfgAgent {
                max_iterations: 10,
                permission_mode: PermissionMode::Default,
                web_enabled: false,
                context_compression: true,
                compaction_token_threshold: abs,
                compaction_token_threshold_ratio: ratio,
                skills_dir: "~/.whet/skills".to_string(),
            },
            memory: MemoryConfig {
                database_path: ":memory:".to_string(),
                max_inject_memories: 50,
            },
            mcp: McpConfig::default(),
        }
    }

    #[test]
    fn resolve_threshold_uses_ratio_when_num_ctx_set() {
        // 8192 × 0.6 = 4915 (truncated by f32 cast).
        let cfg = make_cfg(Some(8192), 0.6, 5000);
        assert_eq!(resolve_compaction_threshold(&cfg), 4915);
    }

    #[test]
    fn resolve_threshold_scales_with_larger_context() {
        // 32768 × 0.6 = 19660. Big-context models compact later, as
        // they should.
        let cfg = make_cfg(Some(32768), 0.6, 5000);
        assert_eq!(resolve_compaction_threshold(&cfg), 19660);
    }

    #[test]
    fn resolve_threshold_scales_with_smaller_context() {
        // 4096 × 0.6 = 2457. Small-context models compact earlier,
        // before overflow.
        let cfg = make_cfg(Some(4096), 0.6, 5000);
        assert_eq!(resolve_compaction_threshold(&cfg), 2457);
    }

    #[test]
    fn resolve_threshold_falls_back_to_absolute_when_num_ctx_unset() {
        // Cloud providers (Anthropic, Gemini) don't set num_ctx in
        // [llm.options]. Fallback to the explicit absolute keeps a
        // sensible default.
        let cfg = make_cfg(None, 0.6, 5000);
        assert_eq!(resolve_compaction_threshold(&cfg), 5000);
    }

    #[test]
    fn resolve_threshold_uses_absolute_when_ratio_is_zero() {
        // Explicit opt-out of ratio mode: ratio = 0 means "use my
        // exact number regardless of num_ctx".
        let cfg = make_cfg(Some(8192), 0.0, 7777);
        assert_eq!(resolve_compaction_threshold(&cfg), 7777);
    }

    #[test]
    fn resolve_threshold_clamps_runaway_ratio() {
        // Defensive: a typo of 6.0 (meant 0.6) is clamped to 0.95 so
        // we never produce a threshold that exceeds num_ctx itself.
        // 8192 × 0.95 = 7782.
        let cfg = make_cfg(Some(8192), 6.0, 5000);
        let resolved = resolve_compaction_threshold(&cfg);
        assert!(
            resolved < 8192,
            "threshold {} must stay below num_ctx",
            resolved
        );
        assert_eq!(resolved, 7782);
    }
}
