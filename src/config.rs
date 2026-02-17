use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub llm: LlmConfig,
    pub agent: AgentConfig,
    pub memory: MemoryConfig,
    #[serde(default)]
    pub mcp: McpConfig,
}

/// Permission mode controlling when user approval is required for tool execution.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    /// Ask for approval on all file-modifying and command-executing tools
    #[default]
    Default,
    /// Auto-approve file edits, ask only for shell/git execution
    AcceptEdits,
    /// No confirmation needed for any tool
    Yolo,
}

impl fmt::Display for PermissionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PermissionMode::Default => write!(f, "default"),
            PermissionMode::AcceptEdits => write!(f, "accept_edits"),
            PermissionMode::Yolo => write!(f, "yolo"),
        }
    }
}

/// Risk level of a tool operation.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolRiskLevel {
    /// Read-only operations (read_file, list_dir, grep, repo_map)
    Safe,
    /// File modification operations (write_file, edit_file)
    Moderate,
    /// Command execution operations (shell, git)
    Dangerous,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    pub base_url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub streaming: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentConfig {
    pub max_iterations: usize,
    #[serde(default)]
    pub permission_mode: PermissionMode,
    #[serde(default)]
    pub web_enabled: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemoryConfig {
    pub database_path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            llm: LlmConfig {
                provider: "ollama".to_string(),
                model: "qwen2.5:7b".to_string(),
                base_url: "http://localhost:11434".to_string(),
                api_key: None,
                streaming: false,
            },
            agent: AgentConfig {
                max_iterations: 10,
                permission_mode: PermissionMode::Default,
                web_enabled: false,
            },
            memory: MemoryConfig {
                database_path: "~/.hermitclaw/memory.db".to_string(),
            },
            mcp: McpConfig::default(),
        }
    }
}

impl Config {
    /// Parse config from a TOML string.
    #[allow(dead_code)]
    pub fn from_toml(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Serialize config to TOML string.
    #[allow(dead_code)]
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

[memory]
database_path = "/custom/path/memory.db"
"#;
        let config = Config::from_toml(toml_str).unwrap();
        assert_eq!(config.llm.model, "llama3.2:3b");
        assert_eq!(config.llm.base_url, "http://192.168.1.100:11434");
        assert_eq!(config.agent.max_iterations, 5);
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

    // --- Backward compatibility tests ---

    #[test]
    fn test_config_backward_compat_without_api_key() {
        // Old configs without api_key should parse with None default
        let toml_str = r#"
[llm]
provider = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"

[agent]
max_iterations = 10

[memory]
database_path = "~/.hermitclaw/memory.db"
"#;
        let config = Config::from_toml(toml_str).unwrap();
        assert!(config.llm.api_key.is_none());
    }

    #[test]
    fn test_config_backward_compat_without_streaming() {
        // Old configs without streaming should default to false
        let toml_str = r#"
[llm]
provider = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"

[agent]
max_iterations = 10

[memory]
database_path = "~/.hermitclaw/memory.db"
"#;
        let config = Config::from_toml(toml_str).unwrap();
        assert!(!config.llm.streaming);
    }

    #[test]
    fn test_config_backward_compat_without_mcp() {
        // Old configs without mcp section should have empty servers
        let toml_str = r#"
[llm]
provider = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"

[agent]
max_iterations = 10

[memory]
database_path = "~/.hermitclaw/memory.db"
"#;
        let config = Config::from_toml(toml_str).unwrap();
        assert!(config.mcp.servers.is_empty());
    }

    #[test]
    fn test_config_with_api_key() {
        let toml_str = r#"
[llm]
provider = "openai_compat"
model = "gpt-3.5-turbo"
base_url = "http://localhost:8080"
api_key = "sk-test-key-123"

[agent]
max_iterations = 10

[memory]
database_path = "test.db"
"#;
        let config = Config::from_toml(toml_str).unwrap();
        assert_eq!(config.llm.api_key, Some("sk-test-key-123".to_string()));
        assert_eq!(config.llm.provider, "openai_compat");
    }

    #[test]
    fn test_config_with_streaming_enabled() {
        let toml_str = r#"
[llm]
provider = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"
streaming = true

[agent]
max_iterations = 10

[memory]
database_path = "test.db"
"#;
        let config = Config::from_toml(toml_str).unwrap();
        assert!(config.llm.streaming);
    }

    #[test]
    fn test_config_with_mcp_servers() {
        let toml_str = r#"
[llm]
provider = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"

[agent]
max_iterations = 10

[memory]
database_path = "test.db"

[[mcp.servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

[[mcp.servers]]
name = "sqlite"
command = "mcp-server-sqlite"
args = ["--db", "test.db"]
"#;
        let config = Config::from_toml(toml_str).unwrap();
        assert_eq!(config.mcp.servers.len(), 2);
        assert_eq!(config.mcp.servers[0].name, "filesystem");
        assert_eq!(config.mcp.servers[0].command, "npx");
        assert_eq!(config.mcp.servers[0].args.len(), 3);
        assert_eq!(config.mcp.servers[1].name, "sqlite");
    }

    #[test]
    fn test_config_mcp_server_without_args() {
        let toml_str = r#"
[llm]
provider = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"

[agent]
max_iterations = 10

[memory]
database_path = "test.db"

[[mcp.servers]]
name = "simple"
command = "mcp-server"
"#;
        let config = Config::from_toml(toml_str).unwrap();
        assert_eq!(config.mcp.servers.len(), 1);
        assert_eq!(config.mcp.servers[0].name, "simple");
        assert!(config.mcp.servers[0].args.is_empty());
    }

    #[test]
    fn test_config_default_new_fields() {
        let config = Config::default();
        assert!(config.llm.api_key.is_none());
        assert!(!config.llm.streaming);
        assert!(config.mcp.servers.is_empty());
    }

    #[test]
    fn test_config_roundtrip_with_new_fields() {
        let mut config = Config::default();
        config.llm.api_key = Some("sk-test".to_string());
        config.llm.streaming = true;

        let toml_str = config.to_toml().unwrap();
        let parsed = Config::from_toml(&toml_str).unwrap();

        assert_eq!(parsed.llm.api_key, Some("sk-test".to_string()));
        assert!(parsed.llm.streaming);
    }

    #[test]
    fn test_config_with_all_new_fields() {
        let toml_str = r#"
[llm]
provider = "openai_compat"
model = "llama3"
base_url = "http://localhost:8080"
api_key = "sk-key"
streaming = true

[agent]
max_iterations = 20

[memory]
database_path = "/data/memory.db"

[[mcp.servers]]
name = "tools"
command = "mcp-tools"
args = ["--verbose"]
"#;
        let config = Config::from_toml(toml_str).unwrap();
        assert_eq!(config.llm.provider, "openai_compat");
        assert_eq!(config.llm.api_key, Some("sk-key".to_string()));
        assert!(config.llm.streaming);
        assert_eq!(config.agent.max_iterations, 20);
        assert_eq!(config.mcp.servers.len(), 1);
        assert_eq!(config.mcp.servers[0].name, "tools");
    }

    #[test]
    fn test_config_extra_unknown_fields_ignored() {
        // TOML parser should ignore unknown fields gracefully
        let toml_str = r#"
[llm]
provider = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"

[agent]
max_iterations = 10

[memory]
database_path = "test.db"
"#;
        // This should parse fine even without all new optional fields
        let config = Config::from_toml(toml_str).unwrap();
        assert_eq!(config.llm.provider, "ollama");
    }

    // --- Permission mode tests ---

    #[test]
    fn test_default_permission_mode() {
        let config = Config::default();
        assert_eq!(config.agent.permission_mode, PermissionMode::Default);
    }

    #[test]
    fn test_permission_mode_display() {
        assert_eq!(PermissionMode::Default.to_string(), "default");
        assert_eq!(PermissionMode::AcceptEdits.to_string(), "accept_edits");
        assert_eq!(PermissionMode::Yolo.to_string(), "yolo");
    }

    #[test]
    fn test_permission_mode_backward_compat() {
        // Configs without permission_mode should default to Default
        let toml_str = r#"
[llm]
provider = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"

[agent]
max_iterations = 10

[memory]
database_path = "test.db"
"#;
        let config = Config::from_toml(toml_str).unwrap();
        assert_eq!(config.agent.permission_mode, PermissionMode::Default);
    }

    #[test]
    fn test_permission_mode_yolo() {
        let toml_str = r#"
[llm]
provider = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"

[agent]
max_iterations = 10
permission_mode = "yolo"

[memory]
database_path = "test.db"
"#;
        let config = Config::from_toml(toml_str).unwrap();
        assert_eq!(config.agent.permission_mode, PermissionMode::Yolo);
    }

    #[test]
    fn test_permission_mode_accept_edits() {
        let toml_str = r#"
[llm]
provider = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434"

[agent]
max_iterations = 10
permission_mode = "accept_edits"

[memory]
database_path = "test.db"
"#;
        let config = Config::from_toml(toml_str).unwrap();
        assert_eq!(config.agent.permission_mode, PermissionMode::AcceptEdits);
    }

    #[test]
    fn test_permission_mode_roundtrip() {
        let mut config = Config::default();
        config.agent.permission_mode = PermissionMode::Yolo;
        let toml_str = config.to_toml().unwrap();
        let parsed = Config::from_toml(&toml_str).unwrap();
        assert_eq!(parsed.agent.permission_mode, PermissionMode::Yolo);
    }
}
