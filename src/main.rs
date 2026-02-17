use clap::{Parser, Subcommand};
use colored::Colorize;

mod agent;
mod config;
mod llm;
mod mcp;
mod memory;
mod security;
mod tools;

use agent::{Agent, AgentConfig};
use config::Config;
use llm::LlmProvider;
use memory::store::MemoryStore;
use tools::default_registry;

fn create_provider(cfg: &Config, model: &str) -> Box<dyn LlmProvider> {
    match cfg.llm.provider.as_str() {
        "openai_compat" => Box::new(llm::openai_compat::OpenAiCompatClient::new(
            &cfg.llm.base_url,
            model,
            cfg.llm.api_key.clone(),
        )),
        _ => Box::new(llm::ollama::OllamaClient::new(&cfg.llm.base_url, model)),
    }
}

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
        #[arg(short, long)]
        model: Option<String>,
        /// Continue the last conversation
        #[arg(long, short = 'c')]
        r#continue: bool,
        /// Force a new conversation (default)
        #[arg(long)]
        new: bool,
    },
    /// List available tools
    Tools,
    /// Show configuration
    Config,
}

fn ask_approval(tool_name: &str, args: &serde_json::Value) -> bool {
    use std::io::{self, Write};
    eprintln!(
        "\n{}",
        format!(
            "  Tool '{}' wants to execute:",
            tool_name
        )
        .yellow()
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
                    format!("{}...", &old[..80])
                } else {
                    old.to_string()
                };
                eprintln!("    old_text: {}", preview.dimmed());
            }
            if let Some(new) = args["new_text"].as_str() {
                let preview = if new.len() > 80 {
                    format!("{}...", &new[..80])
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
    eprint!(
        "  {} ",
        "Allow? [y/N/a(lways)]".bright_yellow()
    );
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

fn run_chat(model: Option<String>, continue_conv: bool) {
    let cfg = Config::load();

    let model = model.unwrap_or(cfg.llm.model.clone());

    println!("{}", "hermitclaw v0.1.0".bold());
    println!("The hermit needs no network.\n");
    println!("Model: {}", model.green());
    println!(
        "Permission: {}",
        cfg.agent.permission_mode.to_string().cyan()
    );
    println!("Type {} to exit.\n", "Ctrl+D".dimmed());

    let provider = create_provider(&cfg, &model);
    let mut registry = default_registry();

    // Register MCP tools
    if !cfg.mcp.servers.is_empty() {
        mcp::register_mcp_tools(&mut registry, &cfg.mcp.servers);
    }
    let agent_config = AgentConfig {
        model: model.clone(),
        max_iterations: cfg.agent.max_iterations,
        permission_mode: cfg.agent.permission_mode.clone(),
        plan_mode: false,
    };

    let mut agent = Agent::new(provider, registry, agent_config);

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
                                "assistant" => {
                                    agent.memory.push(llm::Message::assistant(content))
                                }
                                _ => {}
                            }
                        }
                        println!(
                            "Loaded {} previous messages.\n",
                            messages.len().to_string().cyan()
                        );
                    }
                    id
                }
                _ => {
                    println!("No previous conversation found. Starting new.\n");
                    let id = uuid::Uuid::new_v4().to_string();
                    let _ = store.create_conversation(&id);
                    id
                }
            }
        } else {
            uuid::Uuid::new_v4().to_string()
        }
    } else {
        let id = uuid::Uuid::new_v4().to_string();
        if let Some(ref store) = store {
            let _ = store.create_conversation(&id);
        }
        id
    };

    let mut current_model = model;
    let mut rl = rustyline::DefaultEditor::new().expect("Failed to initialize readline");

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
                    match handle_slash_command(input, &cfg, &mut agent, &mut current_model) {
                        SlashResult::Handled => continue,
                        SlashResult::NewProvider(provider) => {
                            agent.llm = provider;
                            continue;
                        }
                        SlashResult::NotACommand => {} // Pass through to LLM
                    }
                }

                // Save user message
                if let Some(ref store) = store {
                    let _ = store.save_message(&conversation_id, "user", input, None);
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
                    let _ =
                        store.save_message(&conversation_id, "assistant", &response, None);
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("Use Ctrl+D to exit.");
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("\nGoodbye!");
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
    NotACommand,
}

fn handle_slash_command(
    input: &str,
    cfg: &Config,
    agent: &mut Agent,
    current_model: &mut String,
) -> SlashResult {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0];
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd {
        "/model" => {
            if arg.is_empty() {
                println!("Current model: {}", current_model.green());
                println!(
                    "Usage: {} <model_name>",
                    "/model".cyan()
                );
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
            println!("  {}          - Clear conversation history", "/clear".cyan());
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
                .push(llm::Message::system(&agent::prompt::system_prompt()));
            println!("{}", "Conversation cleared.".green());
            SlashResult::Handled
        }
        _ => SlashResult::NotACommand,
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
            format!("...(truncated)\n{}", &combined[combined.len() - max_output..])
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
        Commands::Chat {
            model,
            r#continue,
            new: _,
        } => {
            run_chat(model, r#continue);
        }
        Commands::Tools => {
            let cfg = Config::load();
            let mut registry = default_registry();
            if !cfg.mcp.servers.is_empty() {
                mcp::register_mcp_tools(&mut registry, &cfg.mcp.servers);
            }
            println!("{}", "Available tools:".bold());
            println!();
            for tool in registry.list() {
                println!("  {} - {}", tool.name().cyan(), tool.description());
            }
        }
        Commands::Config => {
            let config = Config::load();
            println!("{}", "Current configuration:".bold());
            println!();
            match toml::to_string_pretty(&config) {
                Ok(s) => println!("{}", s),
                Err(e) => eprintln!("Error serializing config: {}", e),
            }
        }
    }
}
