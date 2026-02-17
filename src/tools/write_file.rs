use super::{Tool, ToolError, ToolPermissions};
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

        std::fs::write(path, content)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to write '{}': {}", path, e)))?;

        Ok(format!("Successfully wrote to '{}'", path))
    }

    fn permissions(&self) -> ToolPermissions {
        ToolPermissions {
            filesystem_read: true,
            filesystem_write: true,
            network: false,
            subprocess: false,
        }
    }
}
