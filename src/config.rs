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
    /// Parse config from a TOML string.
    pub fn from_toml(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Serialize config to TOML string.
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_values() {
        let config = Config::default();
        assert_eq!(config.llm.provider, "ollama");
        assert_eq!(config.llm.model, "qwen2.5:7b");
        assert_eq!(config.llm.base_url, "http://localhost:11434");
        assert_eq!(config.agent.max_iterations, 10);
        assert!(config.agent.sandbox);
        assert_eq!(config.memory.database_path, "~/.hermitclaw/memory.db");
    }

    #[test]
    fn test_config_roundtrip_toml() {
        let config = Config::default();
        let toml_str = config.to_toml().unwrap();
        let parsed = Config::from_toml(&toml_str).unwrap();

        assert_eq!(parsed.llm.provider, config.llm.provider);
        assert_eq!(parsed.llm.model, config.llm.model);
        assert_eq!(parsed.llm.base_url, config.llm.base_url);
        assert_eq!(parsed.agent.max_iterations, config.agent.max_iterations);
        assert_eq!(parsed.agent.sandbox, config.agent.sandbox);
        assert_eq!(parsed.memory.database_path, config.memory.database_path);
    }

    #[test]
    fn test_config_parse_custom_values() {
        let toml_str = r#"
[llm]
provider = "ollama"
model = "llama3.2:3b"
base_url = "http://192.168.1.100:11434"

[agent]
max_iterations = 5
sandbox = false

[memory]
database_path = "/custom/path/memory.db"
"#;
        let config = Config::from_toml(toml_str).unwrap();
        assert_eq!(config.llm.model, "llama3.2:3b");
        assert_eq!(config.llm.base_url, "http://192.168.1.100:11434");
        assert_eq!(config.agent.max_iterations, 5);
        assert!(!config.agent.sandbox);
        assert_eq!(config.memory.database_path, "/custom/path/memory.db");
    }

    #[test]
    fn test_config_parse_invalid_toml() {
        let result = Config::from_toml("this is not valid toml {{{");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_parse_missing_section() {
        // Missing [memory] section
        let toml_str = r#"
[llm]
provider = "ollama"
model = "test"
base_url = "http://localhost:11434"

[agent]
max_iterations = 10
sandbox = true
"#;
        let result = Config::from_toml(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_parse_wrong_type() {
        // max_iterations should be usize, not string
        let toml_str = r#"
[llm]
provider = "ollama"
model = "test"
base_url = "http://localhost:11434"

[agent]
max_iterations = "not a number"
sandbox = true

[memory]
database_path = "test.db"
"#;
        let result = Config::from_toml(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_load_returns_defaults_when_no_file() {
        // load() should return defaults when config file doesn't exist
        let config = Config::load();
        assert_eq!(config.llm.model, "qwen2.5:7b");
    }
}
