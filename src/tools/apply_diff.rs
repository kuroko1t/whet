use super::{Tool, ToolError};
use crate::security::path::is_path_safe;
use serde_json::json;

pub struct ApplyDiffTool;

impl Tool for ApplyDiffTool {
    fn name(&self) -> &str {
        "apply_diff"
    }

    fn description(&self) -> &str {
        "Apply a unified diff to a file. Supports multi-hunk patches. The diff should use standard unified diff format with @@ line markers."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to patch"
                },
                "diff": {
                    "type": "string",
                    "description": "The unified diff to apply (with @@ hunk headers, - for removals, + for additions)"
                }
            },
            "required": ["path", "diff"]
        })
    }

    fn risk_level(&self) -> crate::config::ToolRiskLevel {
        crate::config::ToolRiskLevel::Moderate
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'path' argument".to_string()))?;
        let diff = args["diff"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'diff' argument".to_string()))?;

        if !is_path_safe(path) {
            return Err(ToolError::PermissionDenied(format!(
                "Access to '{}' is blocked for security",
                path
            )));
        }

        let content = std::fs::read_to_string(path)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read '{}': {}", path, e)))?;

        let hunks = parse_unified_diff(diff)?;
        if hunks.is_empty() {
            return Err(ToolError::InvalidArguments(
                "No valid hunks found in diff".to_string(),
            ));
        }

        let new_content = apply_hunks(&content, &hunks)?;

        std::fs::write(path, &new_content).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to write '{}': {}", path, e))
        })?;

        Ok(format!(
            "Successfully applied {} hunk(s) to '{}'",
            hunks.len(),
            path
        ))
    }
}

#[derive(Debug)]
struct DiffHunk {
    old_start: usize, // 1-based line number
    old_count: usize,
    lines: Vec<DiffLine>,
}

#[derive(Debug, Clone)]
enum DiffLine {
    Context(String),
    Remove(String),
    Add(String),
}

fn parse_unified_diff(diff: &str) -> Result<Vec<DiffHunk>, ToolError> {
    let mut hunks = Vec::new();
    let lines: Vec<&str> = diff.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Skip --- and +++ header lines
        if line.starts_with("---") || line.starts_with("+++") {
            i += 1;
            continue;
        }

        // Look for @@ hunk header
        if line.starts_with("@@") {
            let (old_start, old_count) = parse_hunk_header(line)?;
            let mut hunk_lines = Vec::new();
            i += 1;

            while i < lines.len() && !lines[i].starts_with("@@") {
                let l = lines[i];
                if let Some(rest) = l.strip_prefix('-') {
                    hunk_lines.push(DiffLine::Remove(rest.to_string()));
                } else if let Some(rest) = l.strip_prefix('+') {
                    hunk_lines.push(DiffLine::Add(rest.to_string()));
                } else if let Some(rest) = l.strip_prefix(' ') {
                    hunk_lines.push(DiffLine::Context(rest.to_string()));
                } else if l.is_empty() {
                    // Treat empty line as context with empty content
                    hunk_lines.push(DiffLine::Context(String::new()));
                } else if l.starts_with("---") || l.starts_with("+++") {
                    // Skip file header lines that might appear between hunks
                    i += 1;
                    continue;
                } else {
                    // Treat as context line (no prefix = context)
                    hunk_lines.push(DiffLine::Context(l.to_string()));
                }
                i += 1;
            }

            hunks.push(DiffHunk {
                old_start,
                old_count,
                lines: hunk_lines,
            });
        } else {
            i += 1;
        }
    }

    Ok(hunks)
}

fn parse_hunk_header(line: &str) -> Result<(usize, usize), ToolError> {
    // Parse @@ -old_start,old_count +new_start,new_count @@
    // or @@ -old_start +new_start @@
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(ToolError::InvalidArguments(format!(
            "Invalid hunk header: {}",
            line
        )));
    }

    let old_range = parts[1].trim_start_matches('-');
    let (old_start, old_count) = if let Some((start, count)) = old_range.split_once(',') {
        let s = start.parse::<usize>().map_err(|_| {
            ToolError::InvalidArguments(format!("Invalid line number in: {}", line))
        })?;
        let c = count
            .parse::<usize>()
            .map_err(|_| ToolError::InvalidArguments(format!("Invalid count in: {}", line)))?;
        (s, c)
    } else {
        let s = old_range.parse::<usize>().map_err(|_| {
            ToolError::InvalidArguments(format!("Invalid line number in: {}", line))
        })?;
        (s, 1)
    };

    Ok((old_start, old_count))
}

fn apply_hunks(content: &str, hunks: &[DiffHunk]) -> Result<String, ToolError> {
    let original_lines: Vec<&str> = content.lines().collect();
    let mut result_lines: Vec<String> = Vec::new();
    let mut current_line = 0usize; // 0-based index into original_lines

    // Hunks should be applied in order (top to bottom)
    for hunk in hunks {
        if hunk.old_start == 0 {
            return Err(ToolError::InvalidArguments(
                "Invalid hunk: old_start must be >= 1 in unified diff format".to_string(),
            ));
        }
        let hunk_start = hunk.old_start - 1; // Convert to 0-based

        // Copy all lines before this hunk
        while current_line < hunk_start && current_line < original_lines.len() {
            result_lines.push(original_lines[current_line].to_string());
            current_line += 1;
        }

        // Process hunk lines
        let mut old_consumed = 0;
        for diff_line in &hunk.lines {
            match diff_line {
                DiffLine::Context(text) => {
                    // Verify context matches, then preserve the original line
                    if current_line < original_lines.len() {
                        let orig = original_lines[current_line];
                        if orig.trim() != text.trim() && !text.is_empty() {
                            return Err(ToolError::ExecutionFailed(format!(
                                "Context mismatch at line {}: expected '{}', found '{}'",
                                current_line + 1,
                                text,
                                orig
                            )));
                        }
                        // Use the original file's line to preserve exact whitespace
                        result_lines.push(orig.to_string());
                    } else {
                        result_lines.push(text.clone());
                    }
                    current_line += 1;
                    old_consumed += 1;
                }
                DiffLine::Remove(_text) => {
                    // Skip this line from original
                    current_line += 1;
                    old_consumed += 1;
                }
                DiffLine::Add(text) => {
                    result_lines.push(text.clone());
                }
            }
        }

        // If the hunk consumed fewer old lines than expected, advance
        while old_consumed < hunk.old_count && current_line < original_lines.len() {
            result_lines.push(original_lines[current_line].to_string());
            current_line += 1;
            old_consumed += 1;
        }
    }

    // Copy remaining lines after the last hunk
    while current_line < original_lines.len() {
        result_lines.push(original_lines[current_line].to_string());
        current_line += 1;
    }

    // Preserve trailing newline if original had one
    let mut result = result_lines.join("\n");
    if content.ends_with('\n') {
        result.push('\n');
    }

    Ok(result)
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
    fn test_apply_simple_diff() {
        let path = "/tmp/hermitclaw_test_diff_simple.txt";
        setup_test_file(path, "line1\nline2\nline3\nline4\n");

        let tool = ApplyDiffTool;
        let result = tool
            .execute(json!({
                "path": path,
                "diff": "@@ -2,2 +2,2 @@\n-line2\n-line3\n+modified2\n+modified3"
            }))
            .unwrap();

        assert!(result.contains("1 hunk(s)"));
        let content = fs::read_to_string(path).unwrap();
        assert_eq!(content, "line1\nmodified2\nmodified3\nline4\n");

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_with_context() {
        let path = "/tmp/hermitclaw_test_diff_ctx.txt";
        setup_test_file(path, "a\nb\nc\nd\ne\n");

        let tool = ApplyDiffTool;
        let result = tool
            .execute(json!({
                "path": path,
                "diff": "@@ -2,3 +2,3 @@\n b\n-c\n+C\n d"
            }))
            .unwrap();

        assert!(result.contains("1 hunk(s)"));
        let content = fs::read_to_string(path).unwrap();
        assert_eq!(content, "a\nb\nC\nd\ne\n");

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_add_lines() {
        let path = "/tmp/hermitclaw_test_diff_add.txt";
        setup_test_file(path, "line1\nline2\nline3\n");

        let tool = ApplyDiffTool;
        let result = tool
            .execute(json!({
                "path": path,
                "diff": "@@ -2,1 +2,3 @@\n line2\n+new_line_a\n+new_line_b"
            }))
            .unwrap();

        assert!(result.contains("1 hunk(s)"));
        let content = fs::read_to_string(path).unwrap();
        assert_eq!(content, "line1\nline2\nnew_line_a\nnew_line_b\nline3\n");

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_remove_lines() {
        let path = "/tmp/hermitclaw_test_diff_rm.txt";
        setup_test_file(path, "a\nb\nc\nd\ne\n");

        let tool = ApplyDiffTool;
        let result = tool
            .execute(json!({
                "path": path,
                "diff": "@@ -2,3 +2,1 @@\n-b\n-c\n-d"
            }))
            .unwrap();

        assert!(result.contains("1 hunk(s)"));
        let content = fs::read_to_string(path).unwrap();
        assert_eq!(content, "a\ne\n");

        cleanup(path);
    }

    #[test]
    fn test_apply_multi_hunk_diff() {
        let path = "/tmp/hermitclaw_test_diff_multi.txt";
        setup_test_file(path, "a\nb\nc\nd\ne\nf\ng\nh\n");

        let tool = ApplyDiffTool;
        let result = tool
            .execute(json!({
                "path": path,
                "diff": "@@ -1,2 +1,2 @@\n-a\n+A\n b\n@@ -6,2 +6,2 @@\n-f\n+F\n g"
            }))
            .unwrap();

        assert!(result.contains("2 hunk(s)"));
        let content = fs::read_to_string(path).unwrap();
        assert_eq!(content, "A\nb\nc\nd\ne\nF\ng\nh\n");

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_with_file_headers() {
        let path = "/tmp/hermitclaw_test_diff_headers.txt";
        setup_test_file(path, "old\n");

        let tool = ApplyDiffTool;
        let result = tool
            .execute(json!({
                "path": path,
                "diff": "--- a/file.txt\n+++ b/file.txt\n@@ -1,1 +1,1 @@\n-old\n+new"
            }))
            .unwrap();

        assert!(result.contains("1 hunk(s)"));
        let content = fs::read_to_string(path).unwrap();
        assert_eq!(content, "new\n");

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_blocked_path() {
        let tool = ApplyDiffTool;
        let result = tool.execute(json!({
            "path": "/etc/shadow",
            "diff": "@@ -1,1 +1,1 @@\n-old\n+new"
        }));
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_apply_diff_nonexistent_file() {
        let tool = ApplyDiffTool;
        let result = tool.execute(json!({
            "path": "/nonexistent/file.txt",
            "diff": "@@ -1,1 +1,1 @@\n-old\n+new"
        }));
        assert!(matches!(result.unwrap_err(), ToolError::ExecutionFailed(_)));
    }

    #[test]
    fn test_apply_diff_no_hunks() {
        let path = "/tmp/hermitclaw_test_diff_nohunks.txt";
        setup_test_file(path, "content\n");

        let tool = ApplyDiffTool;
        let result = tool.execute(json!({
            "path": path,
            "diff": "not a valid diff"
        }));
        assert!(matches!(
            result.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_missing_args() {
        let tool = ApplyDiffTool;
        assert!(matches!(
            tool.execute(json!({"path": "/tmp/test.txt"})).unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
        assert!(matches!(
            tool.execute(json!({"diff": "@@ -1,1 +1,1 @@\n-a\n+b"}))
                .unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[test]
    fn test_apply_diff_old_start_zero_rejected() {
        let path = "/tmp/hermitclaw_test_diff_zero.txt";
        setup_test_file(path, "line1\nline2\n");

        let tool = ApplyDiffTool;
        let result = tool.execute(json!({
            "path": path,
            "diff": "@@ -0,1 +0,1 @@\n-line1\n+changed"
        }));
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("old_start must be >= 1"));

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_preserves_original_whitespace() {
        // Context lines should preserve the original file's whitespace, not the diff's
        let path = "/tmp/hermitclaw_test_diff_ws.txt";
        setup_test_file(path, "  indented\nnormal\n  also indented\n");

        let tool = ApplyDiffTool;
        let result = tool
            .execute(json!({
                "path": path,
                "diff": "@@ -1,3 +1,3 @@\n   indented\n-normal\n+CHANGED\n   also indented"
            }))
            .unwrap();

        assert!(result.contains("1 hunk(s)"));
        let content = std::fs::read_to_string(path).unwrap();
        // Should preserve original "  indented" (2 spaces), not diff's "  indented" (which might differ)
        assert!(content.starts_with("  indented\n"));
        assert!(content.contains("CHANGED"));

        cleanup(path);
    }

    #[test]
    fn test_parse_hunk_header_with_count() {
        let (start, count) = parse_hunk_header("@@ -10,5 +12,7 @@").unwrap();
        assert_eq!(start, 10);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_parse_hunk_header_without_count() {
        let (start, count) = parse_hunk_header("@@ -3 +5 @@").unwrap();
        assert_eq!(start, 3);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_parse_hunk_header_with_context_text() {
        let (start, count) = parse_hunk_header("@@ -10,3 +10,5 @@ fn main() {").unwrap();
        assert_eq!(start, 10);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_risk_level_is_moderate() {
        let tool = ApplyDiffTool;
        use crate::config::ToolRiskLevel;
        assert_eq!(tool.risk_level(), ToolRiskLevel::Moderate);
    }
}
