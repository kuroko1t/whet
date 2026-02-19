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

use agent::{Agent, AgentConfig};
use config::Config;
use llm::LlmProvider;
use memory::store::MemoryStore;
use skills::Skill;
use tools::default_registry;

fn create_provider(cfg: &Config, model: &str) -> Box<dyn LlmProvider> {
    match cfg.llm.provider.as_str() {
        "openai_compat" => Box::new(llm::openai_compat::OpenAiCompatClient::new(
            &cfg.llm.base_url,
            model,
            cfg.llm.api_key.clone(),
        )),
        "anthropic" => {
            let api_key = cfg
                .llm
                .api_key
                .clone()
                .unwrap_or_else(|| std::env::var("ANTHROPIC_API_KEY").unwrap_or_default());
            let base_url =
                if cfg.llm.base_url.is_empty() || cfg.llm.base_url == "http://localhost:11434" {
                    "https://api.anthropic.com".to_string()
                } else {
                    cfg.llm.base_url.clone()
                };
            Box::new(llm::anthropic::AnthropicClient::new(
                &base_url, model, api_key,
            ))
        }
        "gemini" => {
            let api_key = cfg
                .llm
                .api_key
                .clone()
                .unwrap_or_else(|| std::env::var("GEMINI_API_KEY").unwrap_or_default());
            let base_url =
                if cfg.llm.base_url.is_empty() || cfg.llm.base_url == "http://localhost:11434" {
                    "https://generativelanguage.googleapis.com".to_string()
                } else {
                    cfg.llm.base_url.clone()
                };
            Box::new(llm::gemini::GeminiClient::new(&base_url, model, api_key))
        }
        _ => Box::new(llm::ollama::OllamaClient::new(&cfg.llm.base_url, model)),
    }
}

#[derive(Parser)]
#[command(name = "whet")]
#[command(version)]
#[command(about = "An open-source terminal coding agent. Powered by local or cloud LLMs.")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Single-shot prompt (e.g., whet "fix the bug")
    #[arg(trailing_var_arg = true, num_args = 0..)]
    prompt: Vec<String>,

    /// LLM model to use
    #[arg(short, long)]
    model: Option<String>,

    /// Resume the last conversation
    #[arg(long, short = 'r')]
    resume: bool,

    /// Force a new conversation (default)
    #[arg(long)]
    new: bool,

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
    };

    Agent::new(provider, registry, agent_config, skills)
}

fn run_chat(model: Option<String>, continue_conv: bool, message: Option<String>, yolo: bool) {
    let cfg = Config::load();
    let model = model.unwrap_or(cfg.llm.model.clone());
    let loaded_skills = skills::load_skills(&cfg.agent.skills_dir);

    // Single-shot mode
    if let Some(msg) = message.filter(|m| !m.trim().is_empty()) {
        let mut agent = setup_agent(&cfg, &model, &loaded_skills, yolo);

        if cfg.llm.streaming {
            agent.process_message_with_callbacks(
                &msg,
                &mut |token| {
                    print!("{}", token);
                },
                &mut |_, _| yolo,
            );
            println!();
        } else {
            let response =
                agent.process_message_with_callbacks(&msg, &mut |_| {}, &mut |_, _| yolo);
            println!("{}", response);
        }
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

    let mut conversation_created = false;
    let conversation_id = if continue_conv {
        if let Some(ref store) = store {
            match store.get_latest_conversation_id() {
                Ok(Some(id)) => {
                    println!("Resuming conversation: {}", id.dimmed());
                    // Load previous messages
                    if let Ok(messages) = store.load_messages(&id) {
                        for (role, content, _tool_call_id) in &messages {
                            match role.as_str() {
                                "user" => agent.memory.push(llm::Message::user(content)),
                                "assistant" => agent.memory.push(llm::Message::assistant(content)),
                                _ => {}
                            }
                        }
                        println!(
                            "Loaded {} previous messages.\n",
                            messages.len().to_string().cyan()
                        );
                    }
                    agent.set_resumed(true);
                    conversation_created = true;
                    id
                }
                _ => {
                    println!("No previous conversation found. Starting new.\n");
                    uuid::Uuid::new_v4().to_string()
                }
            }
        } else {
            uuid::Uuid::new_v4().to_string()
        }
    } else {
        uuid::Uuid::new_v4().to_string()
    };

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
                        if let Err(e) = store.create_conversation(&conversation_id) {
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
                    if let Err(e) = store.save_message(&conversation_id, "user", input, None) {
                        eprintln!("{} Failed to save message: {}", "Warning:".yellow(), e);
                    }
                }

                let start = std::time::Instant::now();
                let streaming = cfg.llm.streaming;
                let response = if streaming {
                    eprint!("{} ", "bot>".green().bold());
                    let response = agent.process_message_with_callbacks(
                        input,
                        &mut |token| {
                            eprint!("{}", token);
                        },
                        &mut |tool_name, args| ask_approval(tool_name, args),
                    );
                    eprintln!();
                    response
                } else {
                    eprint!("{}", "[thinking...]".dimmed());
                    let response = agent.process_message_with_callbacks(
                        input,
                        &mut |_| {},
                        &mut |tool_name, args| ask_approval(tool_name, args),
                    );
                    eprint!("\r{}\r", " ".repeat(20));
                    println!("{} {}", "bot>".green().bold(), response);
                    response
                };
                let elapsed = start.elapsed();
                println!("{}", format!("({:.1}s)", elapsed.as_secs_f64()).dimmed());
                println!();

                // Save assistant response
                if let Some(ref store) = store {
                    if let Err(e) =
                        store.save_message(&conversation_id, "assistant", &response, None)
                    {
                        eprintln!("{} Failed to save message: {}", "Warning:".yellow(), e);
                    }
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("Use Ctrl+D to exit.");
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("\nGoodbye!");
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
        let response = if streaming {
            eprint!("{} ", "bot>".green().bold());
            let response = agent.process_message_with_callbacks(
                &fix_prompt,
                &mut |token| {
                    eprint!("{}", token);
                },
                &mut |tool_name, args| ask_approval(tool_name, args),
            );
            eprintln!();
            response
        } else {
            eprint!("{}", "[thinking...]".dimmed());
            let response = agent.process_message_with_callbacks(
                &fix_prompt,
                &mut |_| {},
                &mut |tool_name, args| ask_approval(tool_name, args),
            );
            eprint!("\r{}\r", " ".repeat(20));
            println!("{} {}", "bot>".green().bold(), response);
            response
        };

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
            run_chat(cli.model, cli.resume, message, cli.yolo);
        }
    }
}
