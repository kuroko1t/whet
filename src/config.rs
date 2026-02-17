use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub llm: LlmConfig,
    pub agent: AgentConfig,
    pub memory: MemoryConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    pub base_url: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AgentConfig {
    pub max_iterations: usize,
    pub sandbox: bool,
}

#[derive(Debug, Serialize, Deserialize)]
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
