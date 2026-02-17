use super::{Tool, ToolError};
use crate::security::path::is_path_safe;
use serde_json::json;

pub struct EditFileTool;

impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing an exact text match with new text"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to edit"
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace (must appear exactly once)"
                },
                "new_text": {
                    "type": "string",
                    "description": "The replacement text"
                }
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'path' argument".to_string()))?;
        let old_text = args["old_text"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'old_text' argument".to_string()))?;
        let new_text = args["new_text"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'new_text' argument".to_string()))?;

        if !is_path_safe(path) {
            return Err(ToolError::PermissionDenied(format!(
                "Access to '{}' is blocked for security",
                path
            )));
        }

        let content = std::fs::read_to_string(path)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read '{}': {}", path, e)))?;

        let count = content.matches(old_text).count();

        match count {
            0 => Err(ToolError::ExecutionFailed(
                "old_text not found in file".to_string(),
            )),
            1 => {
                let new_content = content.replacen(old_text, new_text, 1);
                std::fs::write(path, &new_content).map_err(|e| {
                    ToolError::ExecutionFailed(format!("Failed to write '{}': {}", path, e))
                })?;

                // Show context around the change
                let change_pos = new_content.find(new_text).unwrap_or(0);
                let context = get_context(&new_content, change_pos, new_text.len());

                Ok(format!(
                    "Successfully edited '{}'. Context around change:\n{}",
                    path, context
                ))
            }
            n => Err(ToolError::ExecutionFailed(format!(
                "old_text appears {} times; provide more context to make it unique",
                n
            ))),
        }
    }
}

fn get_context(content: &str, pos: usize, replacement_len: usize) -> String {
    let before_start = content[..pos]
        .rmatch_indices('\n')
        .nth(2)
        .map(|(i, _)| i + 1)
        .unwrap_or(0);

    let after_end_search_start = pos + replacement_len;
    let after_end = content[after_end_search_start..]
        .match_indices('\n')
        .nth(2)
        .map(|(i, _)| after_end_search_start + i)
        .unwrap_or(content.len());

    content[before_start..after_end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn setup_test_file(path: &str, content: &str) {
        fs::write(path, content).unwrap();
    }

    fn cleanup(path: &str) {
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_edit_file_basic_replacement() {
        let path = "/tmp/hermitclaw_test_edit_basic.txt";
        setup_test_file(path, "Hello World\nGoodbye World\n");

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path,
                "old_text": "Hello World",
                "new_text": "Hi World"
            }))
            .unwrap();

        assert!(result.contains("Successfully edited"));
        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("Hi World"));
        assert!(content.contains("Goodbye World"));

        cleanup(path);
    }

    #[test]
    fn test_edit_file_old_text_not_found() {
        let path = "/tmp/hermitclaw_test_edit_notfound.txt";
        setup_test_file(path, "Hello World\n");

        let tool = EditFileTool;
        let result = tool.execute(json!({
            "path": path,
            "old_text": "Nonexistent text",
            "new_text": "Replacement"
        }));

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("old_text not found"));

        cleanup(path);
    }

    #[test]
    fn test_edit_file_multiple_matches() {
        let path = "/tmp/hermitclaw_test_edit_multi.txt";
        setup_test_file(path, "foo bar foo\nbaz foo\n");

        let tool = EditFileTool;
        let result = tool.execute(json!({
            "path": path,
            "old_text": "foo",
            "new_text": "qux"
        }));

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("appears 3 times"));

        cleanup(path);
    }

    #[test]
    fn test_edit_file_missing_args() {
        let tool = EditFileTool;

        let result = tool.execute(json!({"old_text": "a", "new_text": "b"}));
        assert!(matches!(result.unwrap_err(), ToolError::InvalidArguments(_)));

        let result = tool.execute(json!({"path": "/tmp/test.txt", "new_text": "b"}));
        assert!(matches!(result.unwrap_err(), ToolError::InvalidArguments(_)));

        let result = tool.execute(json!({"path": "/tmp/test.txt", "old_text": "a"}));
        assert!(matches!(result.unwrap_err(), ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_edit_file_blocked_path() {
        let tool = EditFileTool;
        let result = tool.execute(json!({
            "path": "/etc/shadow",
            "old_text": "a",
            "new_text": "b"
        }));
        assert!(matches!(result.unwrap_err(), ToolError::PermissionDenied(_)));
    }

    #[test]
    fn test_edit_file_nonexistent_file() {
        let tool = EditFileTool;
        let result = tool.execute(json!({
            "path": "/nonexistent/file.txt",
            "old_text": "a",
            "new_text": "b"
        }));
        assert!(matches!(result.unwrap_err(), ToolError::ExecutionFailed(_)));
    }

    #[test]
    fn test_edit_file_preserves_rest() {
        let path = "/tmp/hermitclaw_test_edit_preserve.txt";
        let original = "line1\nline2\nline3\nline4\nline5\n";
        setup_test_file(path, original);

        let tool = EditFileTool;
        tool.execute(json!({
            "path": path,
            "old_text": "line3",
            "new_text": "REPLACED"
        }))
        .unwrap();

        let content = fs::read_to_string(path).unwrap();
        assert_eq!(content, "line1\nline2\nREPLACED\nline4\nline5\n");

        cleanup(path);
    }

    #[test]
    fn test_edit_file_unicode_content() {
        let path = "/tmp/hermitclaw_test_edit_unicode.txt";
        setup_test_file(path, "こんにちは世界\nRust言語\n");

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path,
                "old_text": "こんにちは世界",
                "new_text": "Hello World"
            }))
            .unwrap();

        assert!(result.contains("Successfully edited"));
        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("Hello World"));
        assert!(content.contains("Rust言語"));

        cleanup(path);
    }

    #[test]
    fn test_edit_file_empty_replacement() {
        let path = "/tmp/hermitclaw_test_edit_empty.txt";
        setup_test_file(path, "keep this\nremove this\nkeep too\n");

        let tool = EditFileTool;
        tool.execute(json!({
            "path": path,
            "old_text": "remove this\n",
            "new_text": ""
        }))
        .unwrap();

        let content = fs::read_to_string(path).unwrap();
        assert_eq!(content, "keep this\nkeep too\n");

        cleanup(path);
    }

    #[test]
    fn test_edit_file_multiline_replacement() {
        let path = "/tmp/hermitclaw_test_edit_multiline.txt";
        setup_test_file(path, "start\nold line 1\nold line 2\nend\n");

        let tool = EditFileTool;
        tool.execute(json!({
            "path": path,
            "old_text": "old line 1\nold line 2",
            "new_text": "new single line"
        }))
        .unwrap();

        let content = fs::read_to_string(path).unwrap();
        assert_eq!(content, "start\nnew single line\nend\n");

        cleanup(path);
    }

    #[test]
    fn test_edit_file_context_output() {
        let path = "/tmp/hermitclaw_test_edit_context.txt";
        setup_test_file(
            path,
            "line1\nline2\nline3\nTARGET\nline5\nline6\nline7\n",
        );

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path,
                "old_text": "TARGET",
                "new_text": "REPLACED"
            }))
            .unwrap();

        // Result should contain context around the change
        assert!(result.contains("REPLACED"));

        cleanup(path);
    }

    #[test]
    fn test_get_context_beginning_of_file() {
        let content = "REPLACED\nline2\nline3\nline4\n";
        let ctx = get_context(content, 0, "REPLACED".len());
        assert!(ctx.contains("REPLACED"));
    }

    #[test]
    fn test_get_context_end_of_file() {
        let content = "line1\nline2\nline3\nREPLACED";
        let pos = content.find("REPLACED").unwrap();
        let ctx = get_context(content, pos, "REPLACED".len());
        assert!(ctx.contains("REPLACED"));
    }
}
