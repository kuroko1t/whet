pub mod list_dir;
pub mod read_file;
pub mod shell;
pub mod write_file;

use crate::llm::ToolDefinition;
use std::fmt;

#[derive(Debug)]
pub enum ToolError {
    InvalidArguments(String),
    ExecutionFailed(String),
    PermissionDenied(String),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::InvalidArguments(msg) => write!(f, "Invalid arguments: {}", msg),
            ToolError::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
            ToolError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
        }
    }
}

impl std::error::Error for ToolError {}

pub struct ToolPermissions {
    pub filesystem_read: bool,
    pub filesystem_write: bool,
    pub network: bool,
    pub subprocess: bool,
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError>;
    fn permissions(&self) -> ToolPermissions;
}

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| t.as_ref())
    }

    pub fn list(&self) -> Vec<&dyn Tool> {
        self.tools.iter().map(|t| t.as_ref()).collect()
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a ToolRegistry with all built-in tools registered.
pub fn default_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Box::new(read_file::ReadFileTool));
    registry.register(Box::new(list_dir::ListDirTool));
    registry.register(Box::new(write_file::WriteFileTool));
    registry.register(Box::new(shell::ShellTool));
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_registry_register_and_list() {
        let registry = default_registry();
        let tools = registry.list();
        assert_eq!(tools.len(), 4);
    }

    #[test]
    fn test_registry_get_by_name() {
        let registry = default_registry();
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("list_dir").is_some());
        assert!(registry.get("write_file").is_some());
        assert!(registry.get("shell").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_definitions() {
        let registry = default_registry();
        let defs = registry.definitions();
        assert_eq!(defs.len(), 4);
        for def in &defs {
            assert!(!def.name.is_empty());
            assert!(!def.description.is_empty());
            assert!(def.parameters.is_object());
        }
    }

    #[test]
    fn test_read_file_tool() {
        let tool = read_file::ReadFileTool;
        let result = tool.execute(json!({"path": "Cargo.toml"}));
        assert!(result.is_ok());
        assert!(result.unwrap().contains("hermitclaw"));
    }

    #[test]
    fn test_read_file_tool_missing() {
        let tool = read_file::ReadFileTool;
        let result = tool.execute(json!({"path": "/nonexistent/file.txt"}));
        assert!(result.is_err());
    }

    #[test]
    fn test_list_dir_tool() {
        let tool = list_dir::ListDirTool;
        let result = tool.execute(json!({"path": "src"}));
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("main.rs"));
    }

    #[test]
    fn test_list_dir_tool_missing() {
        let tool = list_dir::ListDirTool;
        let result = tool.execute(json!({"path": "/nonexistent/dir"}));
        assert!(result.is_err());
    }
}
