use super::{Tool, ToolError};
use crate::security::path::is_path_safe;
use serde_json::json;

pub struct WriteFileTool;

impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file at the given path"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write"
                }
            },
            "required": ["path", "content"]
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'path' argument".to_string()))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'content' argument".to_string()))?;

        if !is_path_safe(path) {
            return Err(ToolError::PermissionDenied(format!(
                "Access to '{}' is blocked for security",
                path
            )));
        }

        // Protect against emptying existing files
        if std::path::Path::new(path).exists() {
            let existing_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            if content.is_empty() && existing_size > 0 {
                return Err(ToolError::PermissionDenied(format!(
                    "Refusing to overwrite '{}' ({} bytes) with empty content",
                    path, existing_size
                )));
            }
        }

        std::fs::write(path, content).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to write '{}': {}", path, e))
        })?;

        Ok(format!("Successfully wrote to '{}'", path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_write_and_verify() {
        let tool = WriteFileTool;
        let path = "/tmp/whet_test_write.txt";
        let content = "hello from whet test";
        let result = tool
            .execute(json!({"path": path, "content": content}))
            .unwrap();
        assert!(result.contains("Successfully wrote"));

        // Verify the file was actually written
        let read_back = fs::read_to_string(path).unwrap();
        assert_eq!(read_back, content);

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_write_overwrites_existing() {
        let tool = WriteFileTool;
        let path = "/tmp/whet_test_overwrite.txt";

        tool.execute(json!({"path": path, "content": "first"}))
            .unwrap();
        tool.execute(json!({"path": path, "content": "second"}))
            .unwrap();

        let read_back = fs::read_to_string(path).unwrap();
        assert_eq!(read_back, "second");

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_write_blocked_sensitive_path() {
        let tool = WriteFileTool;
        let result = tool.execute(json!({"path": "/etc/shadow", "content": "bad"}));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
    }

    #[test]
    fn test_write_missing_path_arg() {
        let tool = WriteFileTool;
        let result = tool.execute(json!({"content": "hello"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[test]
    fn test_write_missing_content_arg() {
        let tool = WriteFileTool;
        let result = tool.execute(json!({"path": "/tmp/test.txt"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[test]
    fn test_write_nonexistent_parent_dir() {
        let tool = WriteFileTool;
        let result = tool.execute(json!({
            "path": "/tmp/whet_nonexistent_dir_xyz/file.txt",
            "content": "hello"
        }));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::ExecutionFailed(_)));
    }

    #[test]
    fn test_write_path_traversal_blocked() {
        let tool = WriteFileTool;
        let result = tool.execute(json!({"path": "/tmp/../etc/shadow", "content": "bad"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_write_ssh_key_blocked() {
        let tool = WriteFileTool;
        let result = tool.execute(json!({"path": "~/.ssh/id_rsa", "content": "bad"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_write_allows_empty_new_file() {
        let tool = WriteFileTool;
        let path = "/tmp/whet_test_empty_new.txt";
        // Ensure file does not exist
        fs::remove_file(path).ok();

        let result = tool.execute(json!({"path": path, "content": ""})).unwrap();
        assert!(result.contains("Successfully wrote"));

        let read_back = fs::read_to_string(path).unwrap();
        assert_eq!(read_back, "");

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_write_blocks_emptying_existing_file() {
        let tool = WriteFileTool;
        let path = "/tmp/whet_test_no_empty_overwrite.txt";

        // Create a file with content first
        fs::write(path, "important content").unwrap();

        // Attempt to overwrite with empty content should fail
        let result = tool.execute(json!({"path": path, "content": ""}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));

        // Original content should be preserved
        let read_back = fs::read_to_string(path).unwrap();
        assert_eq!(read_back, "important content");

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_write_allows_overwrite_with_content() {
        let tool = WriteFileTool;
        let path = "/tmp/whet_test_overwrite_with_content.txt";

        // Create a file with content
        fs::write(path, "old content").unwrap();

        // Overwrite with new non-empty content should succeed
        let result = tool
            .execute(json!({"path": path, "content": "new content"}))
            .unwrap();
        assert!(result.contains("Successfully wrote"));

        let read_back = fs::read_to_string(path).unwrap();
        assert_eq!(read_back, "new content");

        fs::remove_file(path).ok();
    }
}
