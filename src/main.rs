use clap::{Parser, Subcommand};
use colored::Colorize;

mod agent;
mod config;
mod llm;
mod memory;
mod sandbox;
mod tools;

use agent::{Agent, AgentConfig};
use config::Config;
use llm::ollama::OllamaClient;
use memory::store::MemoryStore;
use tools::default_registry;

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
        /// Disable sandbox for tool execution
        #[arg(long)]
        no_sandbox: bool,
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

fn run_chat(model: Option<String>, no_sandbox: bool, continue_conv: bool) {
    let cfg = Config::load();

    let model = model.unwrap_or(cfg.llm.model.clone());
    let sandbox_enabled = if no_sandbox { false } else { cfg.agent.sandbox };

    println!("{}", "hermitclaw v0.1.0".bold());
    println!("The hermit needs no network.\n");
    println!("Model: {}", model.green());
    println!(
        "Sandbox: {}",
        if sandbox_enabled {
            "enabled".green().to_string()
        } else {
            "disabled".red().to_string()
        }
    );
    println!("Type {} to exit.\n", "Ctrl+D".dimmed());

    let client = OllamaClient::new(&cfg.llm.base_url, &model);
    let registry = default_registry();
    let agent_config = AgentConfig {
        model: model.clone(),
        max_iterations: cfg.agent.max_iterations,
        sandbox_enabled,
    };

    let mut agent = Agent::new(Box::new(client), registry, agent_config);

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

                eprint!("{}", "[thinking...]".dimmed());
                let start = std::time::Instant::now();
                let response = agent.process_message(input);
                let elapsed = start.elapsed();
                eprint!("\r{}\r", " ".repeat(20));

                println!("{} {}", "bot>".green().bold(), response);
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
            no_sandbox,
            r#continue,
            new: _,
        } => {
            run_chat(model, no_sandbox, r#continue);
        }
        Commands::Tools => {
            let registry = default_registry();
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
