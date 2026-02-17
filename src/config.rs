use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub llm: LlmConfig,
    pub agent: AgentConfig,
    pub memory: MemoryConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    pub base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentConfig {
    pub max_iterations: usize,
    pub sandbox: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemoryConfig {
    pub database_path: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            llm: LlmConfig {
                provider: "ollama".to_string(),
                model: "qwen2.5:7b".to_string(),
                base_url: "http://localhost:11434".to_string(),
            },
            agent: AgentConfig {
                max_iterations: 10,
                sandbox: true,
            },
            memory: MemoryConfig {
                database_path: "~/.hermitclaw/memory.db".to_string(),
            },
        }
    }
}

impl Config {
    /// Load config from ~/.hermitclaw/config.toml, falling back to defaults.
    pub fn load() -> Self {
        let config_path = if let Some(home) = dirs::home_dir() {
            home.join(".hermitclaw").join("config.toml")
        } else {
            return Self::default();
        };

        if config_path.exists() {
            match std::fs::read_to_string(&config_path) {
                Ok(contents) => match toml::from_str(&contents) {
                    Ok(config) => config,
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to parse {}: {}. Using defaults.",
                            config_path.display(),
                            e
                        );
                        Self::default()
                    }
                },
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to read {}: {}. Using defaults.",
                        config_path.display(),
                        e
                    );
                    Self::default()
                }
            }
        } else {
            Self::default()
        }
    }
}
