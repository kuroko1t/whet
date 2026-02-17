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
