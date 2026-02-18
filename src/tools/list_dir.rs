use super::{Tool, ToolError};
use crate::security::path::is_path_safe;
use serde_json::json;

const MAX_DEPTH: usize = 10;
const MAX_ENTRIES: usize = 5000;

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
        let mut truncated = false;
        list_entries(path, recursive, &mut entries, 0, &mut truncated)?;
        if truncated {
            entries.push("...[truncated]".to_string());
        }
        Ok(entries.join("\n"))
    }
}

fn list_entries(
    path: &str,
    recursive: bool,
    entries: &mut Vec<String>,
    depth: usize,
    truncated: &mut bool,
) -> Result<(), ToolError> {
    if entries.len() >= MAX_ENTRIES {
        *truncated = true;
        return Ok(());
    }

    if recursive && depth >= MAX_DEPTH {
        *truncated = true;
        return Ok(());
    }

    let dir = std::fs::read_dir(path)
        .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read '{}': {}", path, e)))?;

    for entry in dir {
        if entries.len() >= MAX_ENTRIES {
            *truncated = true;
            return Ok(());
        }

        let entry = entry
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read entry: {}", e)))?;
        let path_buf = entry.path();
        let display = path_buf.display().to_string();

        if path_buf.is_dir() {
            entries.push(format!("{}/", display));
            if recursive {
                // Skip symlinks to prevent infinite recursion from cycles
                let is_symlink = path_buf.symlink_metadata()
                    .map(|m| m.file_type().is_symlink())
                    .unwrap_or(false);
                if !is_symlink {
                    list_entries(&display, true, entries, depth + 1, truncated)?;
                }
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
                assert!(
                    line.ends_with('/'),
                    "Directory entry should end with /: {}",
                    line
                );
            }
        }
    }

    #[test]
    fn test_list_missing_path_arg() {
        let tool = ListDirTool;
        let result = tool.execute(json!({}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[test]
    fn test_list_file_instead_of_dir() {
        let tool = ListDirTool;
        // Passing a file path instead of directory should fail
        let result = tool.execute(json!({"path": "Cargo.toml"}));
        assert!(result.is_err());
    }
}
