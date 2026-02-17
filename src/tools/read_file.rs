use super::{Tool, ToolError, ToolPermissions};
use crate::sandbox::namespace::is_path_safe;
use serde_json::json;

pub struct ReadFileTool;

impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file at the given path"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read"
                }
            },
            "required": ["path"]
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'path' argument".to_string()))?;

        if !is_path_safe(path) {
            return Err(ToolError::PermissionDenied(format!(
                "Access to '{}' is blocked for security",
                path
            )));
        }

        std::fs::read_to_string(path)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read '{}': {}", path, e)))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_existing_file() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": "Cargo.toml"})).unwrap();
        assert!(result.contains("hermitclaw"));
        assert!(result.contains("[package]"));
    }

    #[test]
    fn test_read_nonexistent_file() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": "/nonexistent/file.txt"}));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::ExecutionFailed(_)));
    }

    #[test]
    fn test_read_blocked_etc_shadow() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": "/etc/shadow"}));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::PermissionDenied(_)));
    }

    #[test]
    fn test_read_blocked_etc_gshadow() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": "/etc/gshadow"}));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::PermissionDenied(_)));
    }

    #[test]
    fn test_read_blocked_sudoers() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": "/etc/sudoers"}));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::PermissionDenied(_)));
    }

    #[test]
    fn test_read_missing_path_arg() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({}));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_read_directory_fails() {
        let tool = ReadFileTool;
        // Reading a directory should fail
        let result = tool.execute(json!({"path": "src"}));
        assert!(result.is_err());
    }

    #[test]
    fn test_read_write_roundtrip() {
        // Write a file, then read it back
        let path = "/tmp/hermitclaw_test_roundtrip.txt";
        let content = "roundtrip test content\nline 2";
        std::fs::write(path, content).unwrap();

        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": path})).unwrap();
        assert_eq!(result, content);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_read_permissions() {
        let tool = ReadFileTool;
        let perms = tool.permissions();
        assert!(perms.filesystem_read);
        assert!(!perms.filesystem_write);
        assert!(!perms.network);
        assert!(!perms.subprocess);
    }
}
