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
