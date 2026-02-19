use super::{Tool, ToolError};
use crate::security::path::is_path_safe;
use serde_json::json;

const MAX_FILE_SIZE: u64 = 10_000_000; // 10MB

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

        let file_size = std::fs::metadata(path)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read '{}': {}", path, e)))?
            .len();
        if file_size > MAX_FILE_SIZE {
            return Err(ToolError::ExecutionFailed(format!(
                "File '{}' is too large ({} bytes, max {} bytes)",
                path, file_size, MAX_FILE_SIZE
            )));
        }

        std::fs::read_to_string(path)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read '{}': {}", path, e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_existing_file() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": "Cargo.toml"})).unwrap();
        assert!(result.contains("whet"));
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
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_read_blocked_etc_gshadow() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": "/etc/gshadow"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_read_blocked_sudoers() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": "/etc/sudoers"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_read_missing_path_arg() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[test]
    fn test_read_directory_fails() {
        let tool = ReadFileTool;
        // Reading a directory should fail
        let result = tool.execute(json!({"path": "src"}));
        assert!(result.is_err());
    }

    #[test]
    fn test_read_path_traversal_blocked() {
        let tool = ReadFileTool;
        // Attempt to reach /etc/shadow via path traversal
        let result = tool.execute(json!({"path": "/tmp/../etc/shadow"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_read_path_traversal_deep() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": "/tmp/../../etc/shadow"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_read_ssh_via_absolute_home_path() {
        let tool = ReadFileTool;
        if let Some(home) = dirs::home_dir() {
            let path = format!("{}/.ssh/id_rsa", home.display());
            let result = tool.execute(json!({"path": path}));
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                ToolError::PermissionDenied(_)
            ));
        }
    }

    #[test]
    fn test_read_write_roundtrip() {
        // Write a file, then read it back
        let path = "/tmp/whet_test_roundtrip.txt";
        let content = "roundtrip test content\nline 2";
        std::fs::write(path, content).unwrap();

        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": path})).unwrap();
        assert_eq!(result, content);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_read_via_symlink_to_sensitive_blocked() {
        let dir = tempfile::TempDir::new().unwrap();
        let link_path = dir.path().join("read_shadow_link");
        std::os::unix::fs::symlink("/etc/shadow", &link_path).unwrap();

        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": link_path.display().to_string()}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_read_via_symlink_to_safe_allowed() {
        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("read_target.txt");
        let link_path = dir.path().join("read_safe_link");
        std::fs::write(&target, "safe content via symlink").unwrap();
        std::os::unix::fs::symlink(&target, &link_path).unwrap();

        let tool = ReadFileTool;
        let result = tool
            .execute(json!({"path": link_path.display().to_string()}))
            .unwrap();
        assert_eq!(result, "safe content via symlink");
    }

    #[test]
    fn test_read_file_size_limit() {
        let path = "/tmp/whet_test_large_file.txt";
        // Create a file slightly larger than MAX_FILE_SIZE (10MB)
        {
            use std::io::Write;
            let mut f = std::fs::File::create(path).unwrap();
            let chunk = vec![b'x'; 1_000_000]; // 1MB chunks
            for _ in 0..11 {
                f.write_all(&chunk).unwrap();
            }
        }

        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": path}));
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("too large"));

        std::fs::remove_file(path).ok();
    }
}
