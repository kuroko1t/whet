pub mod apply_diff;
pub mod edit_file;
pub mod git;
pub mod grep;
pub mod list_dir;
pub mod read_file;
pub mod repo_map;
pub mod shell;
pub mod web_fetch;
pub mod web_search;
pub mod write_file;

use crate::config::ToolRiskLevel;
use crate::llm::ToolDefinition;
use std::collections::HashMap;
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

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError>;

    /// Risk level of this tool. Determines when user approval is required.
    /// Default: classify by tool name.
    fn risk_level(&self) -> ToolRiskLevel {
        match self.name() {
            "write_file" | "edit_file" => ToolRiskLevel::Moderate,
            "shell" | "git" => ToolRiskLevel::Dangerous,
            _ => ToolRiskLevel::Safe,
        }
    }
}

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
    /// Insertion order for deterministic listing
    order: Vec<String>,
    /// Cached definitions — built once, reused across iterations
    cached_defs: Vec<ToolDefinition>,
    cached_safe_defs: Vec<ToolDefinition>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            order: Vec::new(),
            cached_defs: Vec::new(),
            cached_safe_defs: Vec::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.name().to_string();
        self.order.push(name.clone());
        self.tools.insert(name, tool);
        // Invalidate caches
        self.rebuild_caches();
    }

    /// O(1) lookup by name.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn list(&self) -> Vec<&dyn Tool> {
        self.order
            .iter()
            .filter_map(|name| self.tools.get(name).map(|t| t.as_ref()))
            .collect()
    }

    /// Return cached tool definitions. No allocation on repeated calls.
    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.cached_defs
    }

    /// Return cached safe-only definitions. No allocation on repeated calls.
    pub fn safe_definitions(&self) -> &[ToolDefinition] {
        &self.cached_safe_defs
    }

    fn rebuild_caches(&mut self) {
        self.cached_defs = self
            .order
            .iter()
            .filter_map(|name| self.tools.get(name))
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect();

        self.cached_safe_defs = self
            .order
            .iter()
            .filter_map(|name| self.tools.get(name))
            .filter(|t| t.risk_level() == ToolRiskLevel::Safe)
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect();
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
    registry.register(Box::new(grep::GrepTool));
    registry.register(Box::new(edit_file::EditFileTool));
    registry.register(Box::new(git::GitTool));
    registry.register(Box::new(repo_map::RepoMapTool));
    registry.register(Box::new(apply_diff::ApplyDiffTool));
    registry
}

/// Register web tools (web_fetch, web_search). Call this when web features are enabled.
pub fn register_web_tools(registry: &mut ToolRegistry) {
    registry.register(Box::new(web_fetch::WebFetchTool));
    registry.register(Box::new(web_search::WebSearchTool));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_registry_register_and_list() {
        let registry = default_registry();
        let tools = registry.list();
        assert_eq!(tools.len(), 9);
    }

    #[test]
    fn test_registry_get_by_name() {
        let registry = default_registry();
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("list_dir").is_some());
        assert!(registry.get("write_file").is_some());
        assert!(registry.get("shell").is_some());
        assert!(registry.get("grep").is_some());
        assert!(registry.get("edit_file").is_some());
        assert!(registry.get("git").is_some());
        assert!(registry.get("repo_map").is_some());
        assert!(registry.get("apply_diff").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_definitions() {
        let registry = default_registry();
        let defs = registry.definitions();
        assert_eq!(defs.len(), 9);
        for def in defs {
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
        assert!(result.unwrap().contains("whet"));
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

    #[test]
    fn test_tool_error_display_messages() {
        let err = ToolError::InvalidArguments("bad arg".to_string());
        assert_eq!(err.to_string(), "Invalid arguments: bad arg");

        let err = ToolError::ExecutionFailed("cmd failed".to_string());
        assert_eq!(err.to_string(), "Execution failed: cmd failed");

        let err = ToolError::PermissionDenied("no access".to_string());
        assert_eq!(err.to_string(), "Permission denied: no access");
    }

    #[test]
    fn test_tool_error_is_std_error() {
        let err: Box<dyn std::error::Error> =
            Box::new(ToolError::ExecutionFailed("test".to_string()));
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_registry_empty() {
        let registry = ToolRegistry::new();
        assert_eq!(registry.list().len(), 0);
        assert!(registry.get("anything").is_none());
        assert_eq!(registry.definitions().len(), 0);
    }

    #[test]
    fn test_registry_default_is_empty() {
        let registry = ToolRegistry::default();
        assert_eq!(registry.list().len(), 0);
    }

    #[test]
    fn test_all_tools_have_valid_schemas() {
        let registry = default_registry();
        for tool in registry.list() {
            let schema = tool.parameters_schema();
            assert!(
                schema.is_object(),
                "Tool '{}' schema should be an object",
                tool.name()
            );
            assert_eq!(
                schema["type"],
                "object",
                "Tool '{}' schema type should be 'object'",
                tool.name()
            );
            assert!(
                schema.get("properties").is_some(),
                "Tool '{}' schema should have 'properties'",
                tool.name()
            );
        }
    }

    #[test]
    fn test_all_tool_names_are_unique() {
        let registry = default_registry();
        let tools = registry.list();
        let mut names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        let original_len = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), original_len, "Tool names should be unique");
    }

    #[test]
    fn test_safe_definitions_only_safe_tools() {
        use crate::config::ToolRiskLevel;
        let registry = default_registry();
        let safe_defs = registry.safe_definitions();

        // Should only include Safe tools
        for def in safe_defs {
            let tool = registry.get(&def.name).unwrap();
            assert_eq!(
                tool.risk_level(),
                ToolRiskLevel::Safe,
                "Tool '{}' should be Safe to appear in safe_definitions",
                def.name
            );
        }

        // Should not include write_file, edit_file, shell, git, apply_diff
        let safe_names: Vec<&str> = safe_defs.iter().map(|d| d.name.as_str()).collect();
        assert!(!safe_names.contains(&"write_file"));
        assert!(!safe_names.contains(&"edit_file"));
        assert!(!safe_names.contains(&"shell"));
        assert!(!safe_names.contains(&"git"));
        assert!(!safe_names.contains(&"apply_diff"));

        // Should include read_file, list_dir, grep, repo_map
        assert!(safe_names.contains(&"read_file"));
        assert!(safe_names.contains(&"list_dir"));
        assert!(safe_names.contains(&"grep"));
        assert!(safe_names.contains(&"repo_map"));
    }

    #[test]
    fn test_tool_risk_levels() {
        use crate::config::ToolRiskLevel;

        let registry = default_registry();
        // Safe tools
        assert_eq!(
            registry.get("read_file").unwrap().risk_level(),
            ToolRiskLevel::Safe
        );
        assert_eq!(
            registry.get("list_dir").unwrap().risk_level(),
            ToolRiskLevel::Safe
        );
        assert_eq!(
            registry.get("grep").unwrap().risk_level(),
            ToolRiskLevel::Safe
        );
        assert_eq!(
            registry.get("repo_map").unwrap().risk_level(),
            ToolRiskLevel::Safe
        );
        // Moderate tools
        assert_eq!(
            registry.get("write_file").unwrap().risk_level(),
            ToolRiskLevel::Moderate
        );
        assert_eq!(
            registry.get("edit_file").unwrap().risk_level(),
            ToolRiskLevel::Moderate
        );
        assert_eq!(
            registry.get("apply_diff").unwrap().risk_level(),
            ToolRiskLevel::Moderate
        );
        // Dangerous tools
        assert_eq!(
            registry.get("shell").unwrap().risk_level(),
            ToolRiskLevel::Dangerous
        );
        // Git uses dynamic risk level per-command; default is Moderate
        assert_eq!(
            registry.get("git").unwrap().risk_level(),
            ToolRiskLevel::Moderate
        );
    }
}
