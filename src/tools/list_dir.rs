use super::{Tool, ToolError, ToolPermissions};
use crate::sandbox::namespace::is_path_safe;
use serde_json::json;

pub struct ListDirTool;

impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "List the contents of a directory"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to list recursively (default: false)"
                }
            },
            "required": ["path"]
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'path' argument".to_string()))?;
        let recursive = args["recursive"].as_bool().unwrap_or(false);

        if !is_path_safe(path) {
            return Err(ToolError::PermissionDenied(format!(
                "Access to '{}' is blocked for security",
                path
            )));
        }

        let mut entries = Vec::new();
        list_entries(path, recursive, &mut entries)?;
        Ok(entries.join("\n"))
    }

    fn permissions(&self) -> ToolPermissions {
        ToolPermissions {
            filesystem_read: true,
            filesystem_write: false,
            network: false,
            subprocess: false,
        }
    }
}

fn list_entries(path: &str, recursive: bool, entries: &mut Vec<String>) -> Result<(), ToolError> {
    let dir = std::fs::read_dir(path)
        .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read '{}': {}", path, e)))?;

    for entry in dir {
        let entry = entry
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read entry: {}", e)))?;
        let path_buf = entry.path();
        let display = path_buf.display().to_string();

        if path_buf.is_dir() {
            entries.push(format!("{}/", display));
            if recursive {
                list_entries(&display, true, entries)?;
            }
        } else {
            entries.push(display);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_list_src_directory() {
        let tool = ListDirTool;
        let result = tool.execute(json!({"path": "src"})).unwrap();
        assert!(result.contains("main.rs"));
        assert!(result.contains("lib.rs"));
    }

    #[test]
    fn test_list_nonexistent_directory() {
        let tool = ListDirTool;
        let result = tool.execute(json!({"path": "/nonexistent_dir_xyz"}));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::ExecutionFailed(_)));
    }

    #[test]
    fn test_list_recursive() {
        let tool = ListDirTool;
        let result = tool
            .execute(json!({"path": "src", "recursive": true}))
            .unwrap();
        // Recursive should find files inside subdirectories
        assert!(result.contains("main.rs"));
        assert!(result.contains("ollama.rs"));
        assert!(result.contains("prompt.rs"));
    }

    #[test]
    fn test_list_non_recursive_default() {
        let tool = ListDirTool;
        let result = tool.execute(json!({"path": "src"})).unwrap();
        // Non-recursive should show top-level only
        // llm/ is a directory, should have trailing /
        assert!(result.contains("llm/") || result.contains("llm"));
        // Should NOT contain files inside subdirectories
        assert!(!result.contains("ollama.rs"));
    }

    #[test]
    fn test_list_empty_directory() {
        let dir = "/tmp/hermitclaw_test_empty_dir";
        fs::create_dir_all(dir).ok();
        // Remove all contents
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }

        let tool = ListDirTool;
        let result = tool.execute(json!({"path": dir})).unwrap();
        assert!(result.is_empty());

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_list_directories_have_trailing_slash() {
        let tool = ListDirTool;
        let result = tool.execute(json!({"path": "src"})).unwrap();
        // All subdirectories should end with /
        for line in result.lines() {
            if std::path::Path::new(line.trim_end_matches('/')).is_dir() {
                assert!(line.ends_with('/'), "Directory entry should end with /: {}", line);
            }
        }
    }

    #[test]
    fn test_list_missing_path_arg() {
        let tool = ListDirTool;
        let result = tool.execute(json!({}));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_list_file_instead_of_dir() {
        let tool = ListDirTool;
        // Passing a file path instead of directory should fail
        let result = tool.execute(json!({"path": "Cargo.toml"}));
        assert!(result.is_err());
    }

    #[test]
    fn test_list_permissions() {
        let tool = ListDirTool;
        let perms = tool.permissions();
        assert!(perms.filesystem_read);
        assert!(!perms.filesystem_write);
        assert!(!perms.network);
        assert!(!perms.subprocess);
    }
}
