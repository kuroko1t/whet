use super::{Tool, ToolError};
use crate::security::path::is_path_safe;
use serde_json::json;

pub struct WriteFileTool;

impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Create or overwrite a file. Use a path relative to the current working directory (e.g. `app.py`, `src/main.rs`) unless you have a specific reason to write outside the project. Parent directories are created automatically — you do NOT need to call `shell(mkdir -p …)` before writing into a new subdirectory."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to write to. Prefer a path relative to the current working directory (e.g. `app.py`, `src/main.rs`) — avoid absolute paths like `/app.py` (which writes to the filesystem root) unless that is the explicit intent."
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

        // Auto-create the parent directory if it doesn't exist yet.
        // Spares the model from having to chain `shell(mkdir -p …)` →
        // `write_file(…)` whenever a multi-file scaffold lands a file
        // into a fresh subdirectory (e.g. `templates/index.html`,
        // `src/main.rs`). Safe because `is_path_safe(path)` already
        // gated the full target path above, so any parent created
        // here is a prefix of an already-allowed path.
        let mut created_parent = false;
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    ToolError::ExecutionFailed(format!(
                        "Failed to create parent directory for '{}': {}",
                        path, e
                    ))
                })?;
                created_parent = true;
            }
        }

        std::fs::write(path, content).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to write '{}': {}", path, e))
        })?;

        let suffix = if created_parent {
            " (created parent directory)"
        } else {
            ""
        };
        Ok(format!("Successfully wrote to '{}'{}", path, suffix))
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
    fn test_write_creates_missing_parent_dir() {
        // Behaviour change (was ExecutionFailed pre-auto-mkdir):
        // a single missing parent directory is now created on the
        // fly. This eliminates the dog-food failure pattern where the
        // model wrote `templates/index.html` before chaining a mkdir.
        let tool = WriteFileTool;
        let dir = tempfile::TempDir::new().unwrap();
        let nested = dir.path().join("brand_new_subdir/file.txt");

        let result = tool
            .execute(json!({
                "path": nested.display().to_string(),
                "content": "hello"
            }))
            .unwrap();
        assert!(result.contains("Successfully wrote"));
        assert!(
            result.contains("(created parent directory)"),
            "success message should announce the auto-mkdir, got: {result:?}"
        );

        let read_back = fs::read_to_string(&nested).unwrap();
        assert_eq!(read_back, "hello");
        assert!(nested.parent().unwrap().is_dir());
    }

    #[test]
    fn test_write_creates_deeply_nested_parents() {
        // Auto-mkdir is recursive — a 3-level miss is created in one
        // call. Same generic fix covers `src/components/foo/bar.rs`,
        // `templates/admin/users.html`, etc.
        let tool = WriteFileTool;
        let dir = tempfile::TempDir::new().unwrap();
        let deep = dir.path().join("a/b/c/file.txt");

        let result = tool
            .execute(json!({
                "path": deep.display().to_string(),
                "content": "deep"
            }))
            .unwrap();
        assert!(result.contains("Successfully wrote"));
        assert!(result.contains("(created parent directory)"));
        assert!(deep.parent().unwrap().is_dir());
    }

    #[test]
    fn test_write_no_mkdir_message_when_parent_already_exists() {
        // Negative companion: when the parent already exists, the
        // success message must NOT claim it was created. Guards
        // against false-positive logging on every write.
        let tool = WriteFileTool;
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("file.txt");

        let result = tool
            .execute(json!({
                "path": path.display().to_string(),
                "content": "x"
            }))
            .unwrap();
        assert!(result.contains("Successfully wrote"));
        assert!(
            !result.contains("(created parent directory)"),
            "must not claim mkdir when parent already existed, got: {result:?}"
        );
    }

    #[test]
    fn test_write_creates_no_dir_for_bare_filename() {
        // A path with no parent component (e.g. `foo.txt` written into
        // the cwd) should NOT trigger the create_dir_all branch — the
        // empty parent path would be a degenerate input.
        let tool = WriteFileTool;
        let dir = tempfile::TempDir::new().unwrap();
        let prev_cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let result = tool.execute(json!({
            "path": "bare_filename.txt",
            "content": "x"
        }));
        std::env::set_current_dir(prev_cwd).unwrap();
        let result = result.unwrap();
        assert!(result.contains("Successfully wrote"));
        assert!(!result.contains("(created parent directory)"));
    }

    #[test]
    fn test_write_does_not_mkdir_into_blocked_path() {
        // Auto-mkdir must NOT happen when is_path_safe rejects the
        // target — the safety check is the gate, not an afterthought.
        // `/etc/sudoers.d/*` is on the prefix blocklist, so even a
        // never-seen-before child path is rejected before the mkdir
        // branch runs.
        let tool = WriteFileTool;
        let result = tool.execute(json!({
            "path": "/etc/sudoers.d/whet_should_not_create/file.txt",
            "content": "bad"
        }));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
        assert!(
            !std::path::Path::new("/etc/sudoers.d/whet_should_not_create").exists(),
            "blocked path must not have been mkdir'd as a side effect"
        );
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

    #[test]
    fn test_write_via_symlink_to_sensitive_blocked() {
        let dir = tempfile::TempDir::new().unwrap();
        let link_path = dir.path().join("write_shadow_link");
        std::os::unix::fs::symlink("/etc/shadow", &link_path).unwrap();
        if !link_path
            .symlink_metadata()
            .is_ok_and(|m| m.file_type().is_symlink())
        {
            return;
        }

        let tool = WriteFileTool;
        let result =
            tool.execute(json!({"path": link_path.display().to_string(), "content": "bad"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_write_via_symlink_to_safe_allowed() {
        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("write_target.txt");
        let link_path = dir.path().join("write_safe_link");
        fs::write(&target, "").unwrap();
        std::os::unix::fs::symlink(&target, &link_path).unwrap();

        let tool = WriteFileTool;
        let result = tool
            .execute(
                json!({"path": link_path.display().to_string(), "content": "written via symlink"}),
            )
            .unwrap();
        assert!(result.contains("Successfully wrote"));

        let read_back = fs::read_to_string(&target).unwrap();
        assert_eq!(read_back, "written via symlink");
    }

    #[test]
    fn test_description_recommends_relative_path() {
        // Regression guard: the dog-food run on 2026-05-02 found the
        // model wrote to `/app.py` (filesystem root) twice before
        // recovering with `./app.py`, costing 2 wasted tool calls.
        // The fix is purely descriptive — the schema's path-parameter
        // doc must steer the model toward relative paths so the same
        // class of mistake doesn't repeat across languages / projects.
        let tool = WriteFileTool;
        let schema = tool.parameters_schema();
        let path_desc = schema["properties"]["path"]["description"]
            .as_str()
            .expect("path description must exist");
        assert!(
            path_desc.contains("relative to the current working directory"),
            "path description must explicitly recommend a relative path (got: {path_desc:?})"
        );
        assert!(
            path_desc.contains("absolute"),
            "path description must call out absolute paths as the trap to avoid (got: {path_desc:?})"
        );
        // Top-level tool description should also mention it so models
        // that read tool descriptions but skim parameter schemas still
        // see the hint.
        assert!(
            tool.description()
                .contains("relative to the current working directory"),
            "tool description must also mention relative paths (got: {:?})",
            tool.description()
        );
    }

    #[test]
    fn test_write_unicode_content() {
        let tool = WriteFileTool;
        let path = "/tmp/whet_test_write_unicode.txt";
        let content = "日本語テスト 🦀 émojis & spëcial chars";
        let result = tool
            .execute(json!({"path": path, "content": content}))
            .unwrap();
        assert!(result.contains("Successfully wrote"));

        let read_back = fs::read_to_string(path).unwrap();
        assert_eq!(read_back, content);

        fs::remove_file(path).ok();
    }
}
