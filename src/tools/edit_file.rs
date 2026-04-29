use super::{Tool, ToolError};
use crate::security::path::is_path_safe;
use serde_json::json;

const MAX_FILE_SIZE: u64 = 10_000_000; // 10MB

pub struct EditFileTool;

/// Outcome of locating `old_text` inside the file content.
enum MatchResult {
    /// Exact byte-for-byte match at a unique position.
    Exact { byte_start: usize },
    /// Per-line whitespace-normalized match at a unique line range.
    Fuzzy {
        line_start: usize,
        line_count: usize,
    },
    /// Multiple exact matches — caller must add more context.
    AmbiguousExact(usize),
    /// Multiple fuzzy matches — caller must add more context.
    AmbiguousFuzzy(usize),
    /// No match at any tier.
    NotFound,
}

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
        let old_text = args["old_text"].as_str().ok_or_else(|| {
            ToolError::InvalidArguments("missing 'old_text' argument".to_string())
        })?;
        let new_text = args["new_text"].as_str().ok_or_else(|| {
            ToolError::InvalidArguments("missing 'new_text' argument".to_string())
        })?;

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

        let content = std::fs::read_to_string(path)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read '{}': {}", path, e)))?;

        match locate_match(&content, old_text) {
            MatchResult::Exact { byte_start } => {
                let mut new_content = String::with_capacity(content.len() + new_text.len());
                new_content.push_str(&content[..byte_start]);
                new_content.push_str(new_text);
                new_content.push_str(&content[byte_start + old_text.len()..]);
                std::fs::write(path, &new_content).map_err(|e| {
                    ToolError::ExecutionFailed(format!("Failed to write '{}': {}", path, e))
                })?;
                let context = get_context(&new_content, byte_start, new_text.len());
                Ok(format!(
                    "Successfully edited '{}'. Context around change:\n{}",
                    path, context
                ))
            }
            MatchResult::Fuzzy {
                line_start,
                line_count,
            } => {
                let mut content_lines: Vec<&str> = content.split('\n').collect();
                let new_text_lines: Vec<&str> = new_text.split('\n').collect();
                content_lines.splice(
                    line_start..line_start + line_count,
                    new_text_lines.iter().copied(),
                );
                let new_content = content_lines.join("\n");
                std::fs::write(path, &new_content).map_err(|e| {
                    ToolError::ExecutionFailed(format!("Failed to write '{}': {}", path, e))
                })?;
                let preview_pos = new_content
                    .split('\n')
                    .take(line_start)
                    .map(|l| l.len() + 1)
                    .sum::<usize>()
                    .min(new_content.len());
                let preview_len = new_text_lines.iter().map(|l| l.len() + 1).sum::<usize>();
                let context = get_context(&new_content, preview_pos, preview_len);
                Ok(format!(
                    "Successfully edited '{}' (fuzzy whitespace match). Context around change:\n{}",
                    path, context
                ))
            }
            MatchResult::AmbiguousExact(n) => Err(ToolError::ExecutionFailed(format!(
                "old_text appears {} times; provide more context to make it unique",
                n
            ))),
            MatchResult::AmbiguousFuzzy(n) => Err(ToolError::ExecutionFailed(format!(
                "old_text matched {} locations after whitespace normalization; provide more context to disambiguate",
                n
            ))),
            MatchResult::NotFound => Err(ToolError::ExecutionFailed(
                "old_text not found in file (tried exact and whitespace-normalized matching)".to_string(),
            )),
        }
    }
}

/// Locate `old_text` in `content`, falling back from exact to per-line trim match.
fn locate_match(content: &str, old_text: &str) -> MatchResult {
    // Tier 1: exact byte match.
    let exact_count = content.matches(old_text).count();
    match exact_count {
        1 => {
            let byte_start = content
                .find(old_text)
                .expect("count==1 implies find returns Some");
            return MatchResult::Exact { byte_start };
        }
        n if n > 1 => return MatchResult::AmbiguousExact(n),
        _ => {} // 0 → fall through to fuzzy
    }

    // Tier 2: whitespace-normalized per-line match.
    let content_lines: Vec<&str> = content.split('\n').collect();
    let old_lines: Vec<&str> = old_text.split('\n').collect();

    // Skip fuzzy if old_text has no usable content (avoid matching anywhere).
    if old_lines.is_empty() || old_lines.iter().all(|l| l.trim().is_empty()) {
        return MatchResult::NotFound;
    }
    if old_lines.len() > content_lines.len() {
        return MatchResult::NotFound;
    }

    let n = old_lines.len();
    let mut matches: Vec<usize> = Vec::new();
    for start in 0..=content_lines.len() - n {
        let all_match = content_lines[start..start + n]
            .iter()
            .zip(old_lines.iter())
            .all(|(c, o)| c.trim() == o.trim());
        if all_match {
            matches.push(start);
        }
    }

    match matches.len() {
        0 => MatchResult::NotFound,
        1 => MatchResult::Fuzzy {
            line_start: matches[0],
            line_count: n,
        },
        m => MatchResult::AmbiguousFuzzy(m),
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
        let path = "/tmp/whet_test_edit_basic.txt";
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
        let path = "/tmp/whet_test_edit_notfound.txt";
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
        let path = "/tmp/whet_test_edit_multi.txt";
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
        assert!(matches!(
            result.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));

        let result = tool.execute(json!({"path": "/tmp/test.txt", "new_text": "b"}));
        assert!(matches!(
            result.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));

        let result = tool.execute(json!({"path": "/tmp/test.txt", "old_text": "a"}));
        assert!(matches!(
            result.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[test]
    fn test_edit_file_blocked_path() {
        let tool = EditFileTool;
        let result = tool.execute(json!({
            "path": "/etc/shadow",
            "old_text": "a",
            "new_text": "b"
        }));
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
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
        let path = "/tmp/whet_test_edit_preserve.txt";
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
        let path = "/tmp/whet_test_edit_unicode.txt";
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
        let path = "/tmp/whet_test_edit_empty.txt";
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
        let path = "/tmp/whet_test_edit_multiline.txt";
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
        let path = "/tmp/whet_test_edit_context.txt";
        setup_test_file(path, "line1\nline2\nline3\nTARGET\nline5\nline6\nline7\n");

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

    #[test]
    fn test_edit_file_via_symlink_to_sensitive_blocked() {
        let dir = tempfile::TempDir::new().unwrap();
        let link_path = dir.path().join("edit_shadow_link");
        std::os::unix::fs::symlink("/etc/shadow", &link_path).unwrap();
        if !link_path
            .symlink_metadata()
            .is_ok_and(|m| m.file_type().is_symlink())
        {
            return;
        }

        let tool = EditFileTool;
        let result = tool.execute(json!({
            "path": link_path.display().to_string(),
            "old_text": "root",
            "new_text": "hacked"
        }));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_edit_file_via_symlink_to_safe_allowed() {
        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("edit_target.txt");
        let link_path = dir.path().join("edit_safe_link");
        setup_test_file(
            &target.display().to_string(),
            "Hello World\nGoodbye World\n",
        );
        std::os::unix::fs::symlink(&target, &link_path).unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": link_path.display().to_string(),
                "old_text": "Hello World",
                "new_text": "Hi World"
            }))
            .unwrap();
        assert!(result.contains("Successfully edited"));

        let content = fs::read_to_string(&target).unwrap();
        assert!(content.contains("Hi World"));
    }

    #[test]
    fn test_fuzzy_tabs_vs_spaces() {
        // File uses 4-space indent; model emits tab indent. Tier-2 should rescue.
        let path = "/tmp/whet_test_fuzzy_tabs.txt";
        setup_test_file(
            path,
            "def greet():\n    print(\"hi\")\n\ndef farewell():\n    print(\"bye\")\n",
        );

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path,
                "old_text": "def greet():\n\tprint(\"hi\")",
                "new_text": "def greet():\n    print(\"hello\")"
            }))
            .unwrap();
        assert!(result.contains("fuzzy whitespace match"));

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("print(\"hello\")"));
        assert!(content.contains("def farewell():"));
        assert!(content.contains("print(\"bye\")"));
        cleanup(path);
    }

    #[test]
    fn test_fuzzy_trailing_whitespace() {
        // Model emits old_text with trailing space the file doesn't have.
        let path = "/tmp/whet_test_fuzzy_trailing.txt";
        setup_test_file(path, "alpha\nbeta\ngamma\n");

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path,
                "old_text": "beta   ",
                "new_text": "BETA"
            }))
            .unwrap();
        assert!(result.contains("fuzzy whitespace match"));
        assert_eq!(fs::read_to_string(path).unwrap(), "alpha\nBETA\ngamma\n");
        cleanup(path);
    }

    #[test]
    fn test_fuzzy_indent_depth_difference() {
        // Model dropped a level of indentation (4 spaces → 0 spaces).
        let path = "/tmp/whet_test_fuzzy_indent.txt";
        setup_test_file(path, "class A:\n    def m(self):\n        return 1\n");

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path,
                "old_text": "def m(self):\n    return 1",
                "new_text": "def m(self):\n    return 2"
            }))
            .unwrap();
        assert!(result.contains("fuzzy whitespace match"));
        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("return 2"));
        assert!(!content.contains("return 1"));
        cleanup(path);
    }

    #[test]
    fn test_fuzzy_ambiguous_match_errors() {
        // Two trim-equivalent locations → reject rather than picking blindly.
        let path = "/tmp/whet_test_fuzzy_ambig.txt";
        setup_test_file(path, "    foo\n    bar\nbaz\n\tfoo\n\tbar\n");

        let tool = EditFileTool;
        let result = tool.execute(json!({
            "path": path,
            "old_text": "foo\nbar",
            "new_text": "X"
        }));
        let err = result.unwrap_err();
        assert!(err.to_string().contains("after whitespace normalization"));
        cleanup(path);
    }

    #[test]
    fn test_fuzzy_falls_back_only_when_exact_misses() {
        // Exact match exists → fuzzy must NOT engage (otherwise an unexpected
        // second fuzzy match could shadow the unique exact one).
        let path = "/tmp/whet_test_exact_priority.txt";
        setup_test_file(path, "foo\nbar\n  foo\n  bar\n");

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path,
                "old_text": "foo\nbar",
                "new_text": "X\nY"
            }))
            .unwrap();
        // Exact-match path is taken — message must not mention the fuzzy fallback.
        assert!(!result.contains("fuzzy whitespace match"));
        let content = fs::read_to_string(path).unwrap();
        assert_eq!(content, "X\nY\n  foo\n  bar\n");
        cleanup(path);
    }

    #[test]
    fn test_fuzzy_not_found_message_is_explicit() {
        let path = "/tmp/whet_test_fuzzy_nomatch.txt";
        setup_test_file(path, "hello\n");

        let tool = EditFileTool;
        let err = tool
            .execute(json!({
                "path": path,
                "old_text": "completely different",
                "new_text": "x"
            }))
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not found"));
        assert!(msg.contains("whitespace-normalized"));
        cleanup(path);
    }

    #[test]
    fn test_fuzzy_blank_old_text_does_not_match_anywhere() {
        // Whitespace-only old_text would otherwise match every line — guard against it.
        let path = "/tmp/whet_test_fuzzy_blank.txt";
        setup_test_file(path, "alpha\nbeta\n");

        let tool = EditFileTool;
        let err = tool
            .execute(json!({
                "path": path,
                "old_text": "   ",
                "new_text": "X"
            }))
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
        // File untouched.
        assert_eq!(fs::read_to_string(path).unwrap(), "alpha\nbeta\n");
        cleanup(path);
    }

    #[test]
    fn test_edit_file_size_limit() {
        let path = "/tmp/whet_test_edit_large.txt";
        // Create a file larger than MAX_FILE_SIZE (10MB)
        {
            use std::io::Write;
            let mut f = std::fs::File::create(path).unwrap();
            let chunk = vec![b'x'; 1_000_000];
            for _ in 0..11 {
                f.write_all(&chunk).unwrap();
            }
        }

        let tool = EditFileTool;
        let result = tool.execute(json!({
            "path": path,
            "old_text": "x",
            "new_text": "y"
        }));
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("too large"));

        cleanup(path);
    }
}
