use super::{Tool, ToolError};
use crate::security::path::is_path_safe;
use serde_json::json;

const MAX_DEPTH: usize = 10;
const MAX_ENTRIES: usize = 5000;

const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
];

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
                // Skip well-known build/dependency directories to avoid huge output
                let dir_name = path_buf.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if SKIP_DIRS.contains(&dir_name) {
                    continue;
                }
                // Skip symlinks to prevent infinite recursion from cycles
                let is_symlink = path_buf
                    .symlink_metadata()
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
        let dir = "/tmp/whet_test_empty_dir";
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

    #[test]
    fn test_list_max_entries_truncated() {
        let dir = "/tmp/whet_test_max_entries";
        fs::create_dir_all(dir).ok();
        // Create MAX_ENTRIES + 10 files
        for i in 0..MAX_ENTRIES + 10 {
            let path = format!("{}/file_{:06}.txt", dir, i);
            fs::write(&path, "x").ok();
        }

        let tool = ListDirTool;
        let result = tool.execute(json!({"path": dir})).unwrap();
        assert!(
            result.contains("...[truncated]"),
            "Should indicate truncation when exceeding MAX_ENTRIES"
        );
        // Count entries (excluding truncated marker)
        let count = result.lines().filter(|l| !l.contains("truncated")).count();
        assert!(
            count <= MAX_ENTRIES,
            "Should not exceed MAX_ENTRIES (got {})",
            count
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_list_blocked_sensitive_path() {
        let tool = ListDirTool;
        let result = tool.execute(json!({"path": "~/.ssh"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_list_hidden_files_shown() {
        let dir = "/tmp/whet_test_hidden";
        fs::create_dir_all(dir).ok();
        fs::write(format!("{}/.hidden_file", dir), "hidden").ok();
        fs::write(format!("{}/visible_file", dir), "visible").ok();

        let tool = ListDirTool;
        let result = tool.execute(json!({"path": dir})).unwrap();
        assert!(
            result.contains(".hidden_file"),
            "Hidden files should be listed"
        );
        assert!(result.contains("visible_file"));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_list_recursive_depth_limit() {
        let base = "/tmp/whet_test_depth";
        fs::remove_dir_all(base).ok();
        // Create deeply nested directories (deeper than MAX_DEPTH)
        let mut path = base.to_string();
        for i in 0..MAX_DEPTH + 3 {
            path = format!("{}/level_{}", path, i);
            fs::create_dir_all(&path).ok();
            fs::write(format!("{}/file.txt", path), "x").ok();
        }

        let tool = ListDirTool;
        let result = tool
            .execute(json!({"path": base, "recursive": true}))
            .unwrap();
        assert!(
            result.contains("...[truncated]"),
            "Should truncate at MAX_DEPTH"
        );

        fs::remove_dir_all(base).ok();
    }

    #[test]
    fn test_recursive_skips_skip_dirs_contents() {
        let base = "/tmp/whet_test_skip_dirs";
        fs::remove_dir_all(base).ok();
        fs::create_dir_all(format!("{}/target/debug", base)).ok();
        fs::write(format!("{}/target/debug/binary", base), "x").ok();
        fs::create_dir_all(format!("{}/node_modules/pkg", base)).ok();
        fs::write(format!("{}/node_modules/pkg/index.js", base), "x").ok();
        fs::create_dir_all(format!("{}/src", base)).ok();
        fs::write(format!("{}/src/main.rs", base), "fn main() {}").ok();

        let tool = ListDirTool;
        let result = tool
            .execute(json!({"path": base, "recursive": true}))
            .unwrap();

        // target/ and node_modules/ directories should be listed
        assert!(
            result.contains("target/"),
            "target/ directory should appear in listing"
        );
        assert!(
            result.contains("node_modules/"),
            "node_modules/ directory should appear in listing"
        );
        // But their contents should NOT be listed
        assert!(
            !result.contains("debug"),
            "target/debug should not be listed (SKIP_DIRS)"
        );
        assert!(
            !result.contains("index.js"),
            "node_modules/pkg/index.js should not be listed (SKIP_DIRS)"
        );
        // Normal directories should still be recursed
        assert!(
            result.contains("main.rs"),
            "src/main.rs should be listed normally"
        );

        fs::remove_dir_all(base).ok();
    }

    #[test]
    fn test_non_recursive_unaffected_by_skip_dirs() {
        let base = "/tmp/whet_test_skip_dirs_nonrec";
        fs::remove_dir_all(base).ok();
        fs::create_dir_all(format!("{}/target", base)).ok();
        fs::create_dir_all(format!("{}/src", base)).ok();

        let tool = ListDirTool;
        let result = tool.execute(json!({"path": base})).unwrap();

        // Non-recursive should list both target/ and src/ without filtering
        assert!(
            result.contains("target/"),
            "target/ should appear in non-recursive listing"
        );
        assert!(
            result.contains("src/"),
            "src/ should appear in non-recursive listing"
        );

        fs::remove_dir_all(base).ok();
    }

    #[test]
    fn test_list_symlink_dir_not_followed_recursively() {
        let base = "/tmp/whet_test_symlink_dir";
        let target = "/tmp/whet_test_symlink_target_dir";
        fs::remove_dir_all(base).ok();
        fs::remove_dir_all(target).ok();

        fs::create_dir_all(base).ok();
        fs::create_dir_all(target).ok();
        fs::write(format!("{}/target_file.txt", target), "x").ok();

        // Create symlink inside base pointing to target
        std::os::unix::fs::symlink(target, format!("{}/link", base)).ok();

        let tool = ListDirTool;
        let result = tool
            .execute(json!({"path": base, "recursive": true}))
            .unwrap();
        // The symlink directory should be listed but not recursed into
        assert!(result.contains("link/"), "Symlink dir should be listed");
        assert!(
            !result.contains("target_file.txt"),
            "Should NOT recurse into symlink directory"
        );

        fs::remove_dir_all(base).ok();
        fs::remove_dir_all(target).ok();
    }
}
