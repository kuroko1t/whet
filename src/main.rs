use clap::{Parser, Subcommand};

mod agent;
mod config;
mod llm;
mod memory;
mod sandbox;
mod tools;

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

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { model, no_sandbox } => {
            println!("Starting chat with model: {} (sandbox: {})", model, !no_sandbox);
        }
        Commands::Tools => {
            println!("Available tools:");
        }
        Commands::Config => {
            println!("Configuration:");
        }
    }
}
