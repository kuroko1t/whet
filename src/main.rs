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

fn run_chat(model: Option<String>, continue_conv: bool) {
    let cfg = Config::load();

    let model = model.unwrap_or(cfg.llm.model.clone());

    println!("{}", "hermitclaw v0.1.0".bold());
    println!("The hermit needs no network.\n");
    println!("Model: {}", model.green());
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

                // Save user message
                if let Some(ref store) = store {
                    let _ = store.save_message(&conversation_id, "user", input, None);
                }

                let start = std::time::Instant::now();
                let streaming = cfg.llm.streaming;
                let response = if streaming {
                    eprint!("{} ", "bot>".green().bold());
                    let response = agent.process_message_with_callback(input, &mut |token| {
                        eprint!("{}", token);
                    });
                    eprintln!();
                    response
                } else {
                    eprint!("{}", "[thinking...]".dimmed());
                    let response = agent.process_message(input);
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
