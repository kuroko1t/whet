use super::{Tool, ToolError};
use crate::security::path::is_path_safe;
use serde_json::json;
use std::fs;
use std::io::Read;
use std::path::Path;

pub struct GrepTool;

const MAX_RESULTS: usize = 100;
const MAX_FILE_SIZE: u64 = 1_048_576; // 1MB
const BINARY_CHECK_SIZE: usize = 512;

const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules"];

impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a pattern in files recursively"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The text pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "The directory or file to search in (default: current directory)"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Whether to ignore case (default: false)"
                }
            },
            "required": ["pattern"]
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'pattern' argument".to_string()))?;
        let path = args["path"].as_str().unwrap_or(".");
        let case_insensitive = args["case_insensitive"].as_bool().unwrap_or(false);

        if !is_path_safe(path) {
            return Err(ToolError::PermissionDenied(format!(
                "Access to '{}' is blocked for security",
                path
            )));
        }

        let search_pattern = if case_insensitive {
            pattern.to_lowercase()
        } else {
            pattern.to_string()
        };

        let mut results = Vec::new();
        search_path(
            Path::new(path),
            &search_pattern,
            case_insensitive,
            &mut results,
        )?;

        if results.is_empty() {
            Ok("No matches found.".to_string())
        } else {
            let truncated = results.len() >= MAX_RESULTS;
            let mut output = results.join("\n");
            if truncated {
                output.push_str(&format!(
                    "\n\n(Results truncated at {} matches)",
                    MAX_RESULTS
                ));
            }
            Ok(output)
        }
    }
}

fn is_binary(path: &Path) -> bool {
    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut buf = [0u8; BINARY_CHECK_SIZE];
    let n = match file.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return false,
    };
    buf[..n].contains(&0)
}

fn search_path(
    path: &Path,
    pattern: &str,
    case_insensitive: bool,
    results: &mut Vec<String>,
) -> Result<(), ToolError> {
    if results.len() >= MAX_RESULTS {
        return Ok(());
    }

    if path.is_file() {
        search_file(path, pattern, case_insensitive, results)?;
    } else if path.is_dir() {
        let entries = fs::read_dir(path).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to read '{}': {}", path.display(), e))
        })?;

        let mut sorted_entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        sorted_entries.sort_by_key(|e| e.file_name());

        for entry in sorted_entries {
            if results.len() >= MAX_RESULTS {
                break;
            }
            let entry_path = entry.path();

            if entry_path.is_dir() {
                // Skip symlinks to prevent infinite recursion from cycles
                if entry_path
                    .symlink_metadata()
                    .map(|m| m.file_type().is_symlink())
                    .unwrap_or(false)
                {
                    continue;
                }
                if let Some(name) = entry_path.file_name().and_then(|n| n.to_str()) {
                    if SKIP_DIRS.contains(&name) {
                        continue;
                    }
                }
                search_path(&entry_path, pattern, case_insensitive, results)?;
            } else {
                search_file(&entry_path, pattern, case_insensitive, results)?;
            }
        }
    }

    Ok(())
}

fn search_file(
    path: &Path,
    pattern: &str,
    case_insensitive: bool,
    results: &mut Vec<String>,
) -> Result<(), ToolError> {
    // Skip large files
    if let Ok(metadata) = fs::metadata(path) {
        if metadata.len() > MAX_FILE_SIZE {
            return Ok(());
        }
    }

    // Skip binary files
    if is_binary(path) {
        return Ok(());
    }

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Ok(()), // Skip unreadable files
    };

    let path_str = path.display().to_string();

    for (line_num, line) in content.lines().enumerate() {
        if results.len() >= MAX_RESULTS {
            break;
        }

        let matches = if case_insensitive {
            line.to_lowercase().contains(pattern)
        } else {
            line.contains(pattern)
        };

        if matches {
            results.push(format!("{}:{}: {}", path_str, line_num + 1, line));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grep_finds_pattern_in_file() {
        let tool = GrepTool;
        let result = tool
            .execute(json!({"pattern": "whet", "path": "Cargo.toml"}))
            .unwrap();
        assert!(result.contains("whet"));
        assert!(result.contains("Cargo.toml:"));
    }

    #[test]
    fn test_grep_finds_pattern_in_directory() {
        let tool = GrepTool;
        let result = tool
            .execute(json!({"pattern": "fn main", "path": "src"}))
            .unwrap();
        assert!(result.contains("main.rs"));
    }

    #[test]
    fn test_grep_no_matches() {
        let tool = GrepTool;
        let result = tool
            .execute(json!({"pattern": "zzz_truly_unique_missing_42", "path": "Cargo.toml"}))
            .unwrap();
        assert_eq!(result, "No matches found.");
    }

    #[test]
    fn test_grep_case_insensitive() {
        let tool = GrepTool;
        let result = tool
            .execute(json!({"pattern": "WHET", "path": "Cargo.toml", "case_insensitive": true}))
            .unwrap();
        assert!(result.contains("whet"));
    }

    #[test]
    fn test_grep_case_sensitive_no_match() {
        let tool = GrepTool;
        let result = tool
            .execute(json!({"pattern": "WHET", "path": "Cargo.toml", "case_insensitive": false}))
            .unwrap();
        assert_eq!(result, "No matches found.");
    }

    #[test]
    fn test_grep_default_path() {
        let tool = GrepTool;
        // Default path should be "."
        let result = tool.execute(json!({"pattern": "whet"})).unwrap();
        assert!(result.contains("whet"));
    }

    #[test]
    fn test_grep_missing_pattern() {
        let tool = GrepTool;
        let result = tool.execute(json!({"path": "src"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[test]
    fn test_grep_nonexistent_path() {
        let tool = GrepTool;
        let result = tool
            .execute(json!({"pattern": "test", "path": "/nonexistent_dir_xyz"}))
            .unwrap();
        // Nonexistent path is neither file nor dir, so no matches
        assert_eq!(result, "No matches found.");
    }

    #[test]
    fn test_grep_line_numbers() {
        // Create a test file with known content
        let path = "/tmp/whet_test_grep.txt";
        fs::write(path, "line one\nline two\nfind me\nline four\n").unwrap();

        let tool = GrepTool;
        let result = tool
            .execute(json!({"pattern": "find me", "path": path}))
            .unwrap();
        assert!(result.contains(":3:"));
        assert!(result.contains("find me"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_grep_skips_binary_files() {
        let path = "/tmp/whet_test_binary.bin";
        let mut content = vec![0u8; 100];
        content[50] = 0; // NULL byte
        fs::write(path, &content).unwrap();

        let tool = GrepTool;
        let result = tool
            .execute(json!({"pattern": "anything", "path": path}))
            .unwrap();
        assert_eq!(result, "No matches found.");

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_grep_max_results_truncation() {
        // Create a file with more than MAX_RESULTS matching lines
        let dir = "/tmp/whet_test_grep_max";
        fs::create_dir_all(dir).ok();
        let path = format!("{}/many_matches.txt", dir);
        let content: String = (0..150).map(|i| format!("match_line {}\n", i)).collect();
        fs::write(&path, &content).unwrap();

        let tool = GrepTool;
        let result = tool
            .execute(json!({"pattern": "match_line", "path": &path}))
            .unwrap();
        assert!(result.contains("Results truncated at 100 matches"));
        // Count the actual result lines (excluding the truncation message)
        let match_lines = result.lines().filter(|l| l.contains("match_line")).count();
        assert_eq!(match_lines, MAX_RESULTS);

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_grep_skips_git_directory() {
        // Search in the project root - should find matches but NOT in .git/
        let tool = GrepTool;
        let result = tool
            .execute(json!({"pattern": "HEAD", "path": "."}))
            .unwrap();
        // Should not contain any .git/ paths
        for line in result.lines() {
            assert!(
                !line.starts_with(".git/") && !line.contains("/.git/"),
                "Should skip .git directory, found: {}",
                line
            );
        }
    }

    #[test]
    fn test_grep_skips_large_files() {
        let dir = "/tmp/whet_test_grep_large";
        fs::create_dir_all(dir).ok();
        let path = format!("{}/large.txt", dir);
        // Create a file larger than 1MB
        let content = "findme\n".repeat(200_000); // ~1.4MB
        fs::write(&path, &content).unwrap();

        let tool = GrepTool;
        let result = tool
            .execute(json!({"pattern": "findme", "path": &path}))
            .unwrap();
        // Large file should be skipped
        assert_eq!(result, "No matches found.");

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_grep_single_file_search() {
        let path = "/tmp/whet_test_grep_single.txt";
        fs::write(path, "alpha\nbeta\ngamma\n").unwrap();

        let tool = GrepTool;
        let result = tool
            .execute(json!({"pattern": "beta", "path": path}))
            .unwrap();
        assert!(result.contains(":2:"));
        assert!(result.contains("beta"));
        // Should only have one match
        assert_eq!(result.lines().count(), 1);

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_grep_blocked_path() {
        let tool = GrepTool;
        let result = tool.execute(json!({"pattern": "test", "path": "/etc/shadow"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_is_binary_with_text_file() {
        let path = "/tmp/whet_test_not_binary.txt";
        fs::write(path, "just plain text\nwith newlines\n").unwrap();
        assert!(!is_binary(Path::new(path)));
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_is_binary_with_null_bytes() {
        let path = "/tmp/whet_test_is_binary.bin";
        let mut data = b"some text".to_vec();
        data.push(0); // NULL byte
        data.extend_from_slice(b"more text");
        fs::write(path, &data).unwrap();
        assert!(is_binary(Path::new(path)));
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_is_binary_nonexistent_file() {
        assert!(!is_binary(Path::new("/nonexistent_file_xyz")));
    }
}
