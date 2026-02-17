use clap::{Parser, Subcommand};
use colored::Colorize;

mod agent;
mod config;
mod llm;
mod memory;
mod sandbox;
mod tools;

use agent::{Agent, AgentConfig};
use llm::ollama::OllamaClient;
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

fn run_chat(model: &str, no_sandbox: bool) {
    println!("{}", "hermitclaw v0.1.0".bold());
    println!("The hermit needs no network.\n");
    println!("Model: {}", model.green());
    println!("Sandbox: {}", if no_sandbox { "disabled".red().to_string() } else { "enabled".green().to_string() });
    println!("Type {} to exit.\n", "Ctrl+D".dimmed());

    let client = OllamaClient::new("http://localhost:11434", model);
    let registry = default_registry();
    let config = AgentConfig {
        model: model.to_string(),
        max_iterations: 10,
        sandbox_enabled: !no_sandbox,
    };

    let mut agent = Agent::new(Box::new(client), registry, config);

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

                eprint!("{}", "[thinking...]".dimmed());
                let start = std::time::Instant::now();
                let response = agent.process_message(input);
                let elapsed = start.elapsed();
                // Clear the "[thinking...]" line
                eprint!("\r{}\r", " ".repeat(20));

                println!(
                    "{} {}",
                    "bot>".green().bold(),
                    response
                );
                println!("{}", format!("({:.1}s)", elapsed.as_secs_f64()).dimmed());
                println!();
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
        Commands::Chat { model, no_sandbox } => {
            run_chat(&model, no_sandbox);
        }
        Commands::Tools => {
            let registry = default_registry();
            println!("{}", "Available tools:".bold());
            println!();
            for tool in registry.list() {
                println!(
                    "  {} - {}",
                    tool.name().cyan(),
                    tool.description()
                );
            }
        }
        Commands::Config => {
            let config = config::Config::default();
            println!("{}", "Current configuration:".bold());
            println!();
            match toml::to_string_pretty(&config) {
                Ok(s) => println!("{}", s),
                Err(e) => eprintln!("Error serializing config: {}", e),
            }
        }
    }
}
