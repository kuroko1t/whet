use super::{Tool, ToolError};
use crate::security::path::is_path_safe;
use serde_json::json;

const MAX_FILE_SIZE: u64 = 10_000_000; // 10MB

pub struct ApplyDiffTool;

impl Tool for ApplyDiffTool {
    fn name(&self) -> &str {
        "apply_diff"
    }

    fn description(&self) -> &str {
        "Apply a unified diff to one or more files. Supports multi-hunk patches and \
         multi-file diffs that include `--- path` / `+++ path` headers between hunks. \
         Hunk @@ line numbers are treated as hints — the actual location is determined \
         by matching the context/removal lines, so slightly off-by-N anchors still apply."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Default file path to patch. Used when the diff has no `--- path` header. \
                                    For multi-file diffs the per-hunk header takes precedence."
                },
                "diff": {
                    "type": "string",
                    "description": "Unified diff to apply. Use `--- path` / `+++ path` headers to target \
                                    multiple files in one call. Each hunk uses `@@ -old +new @@` markers; \
                                    line numbers are best-effort and matched fuzzily against context."
                }
            },
            "required": ["path", "diff"]
        })
    }

    fn risk_level(&self) -> crate::config::ToolRiskLevel {
        crate::config::ToolRiskLevel::Moderate
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let default_path = args["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'path' argument".to_string()))?;
        let diff = args["diff"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'diff' argument".to_string()))?;

        let groups = parse_unified_diff(diff)?;
        if groups.is_empty() || groups.iter().all(|g| g.hunks.is_empty()) {
            return Err(ToolError::InvalidArguments(
                "No valid hunks found in diff".to_string(),
            ));
        }

        // Per-file structured report.
        // - `applied_paths`: files that we successfully wrote.
        // - `failed_files`: files where at least one hunk could not be applied
        //   (we DID NOT write these — atomic-per-file semantics).
        let mut applied_paths: Vec<(String, usize)> = Vec::new(); // (path, hunks_applied)
        let mut failed_files: Vec<(String, Vec<HunkOutcome>)> = Vec::new();

        for group in &groups {
            if group.hunks.is_empty() {
                continue;
            }
            // Use the per-group `--- path` when it resolves to a real file; otherwise
            // fall back to the JSON `path` argument. This preserves backwards-compat
            // with diffs that use dummy `a/file.txt` markers while still letting
            // multi-file diffs route hunks to their real files.
            let target = match group.file_path.as_deref() {
                Some(p) if std::fs::metadata(p).is_ok() => p,
                _ => default_path,
            };

            if !is_path_safe(target) {
                return Err(ToolError::PermissionDenied(format!(
                    "Access to '{}' is blocked for security",
                    target
                )));
            }

            let file_size = std::fs::metadata(target).map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to read '{}': {}", target, e))
            })?;
            if file_size.len() > MAX_FILE_SIZE {
                return Err(ToolError::ExecutionFailed(format!(
                    "File '{}' is too large ({} bytes, max {} bytes)",
                    target,
                    file_size.len(),
                    MAX_FILE_SIZE
                )));
            }

            let content = std::fs::read_to_string(target).map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to read '{}': {}", target, e))
            })?;

            let (outcomes, new_content) = apply_hunks(&content, &group.hunks);

            match new_content {
                Some(new) => {
                    std::fs::write(target, &new).map_err(|e| {
                        ToolError::ExecutionFailed(format!("Failed to write '{}': {}", target, e))
                    })?;
                    applied_paths.push((target.to_string(), group.hunks.len()));
                }
                None => {
                    // At least one hunk failed; leave the file untouched.
                    failed_files.push((target.to_string(), outcomes));
                }
            }
        }

        if failed_files.is_empty() {
            let total_hunks: usize = applied_paths.iter().map(|(_, n)| n).sum();
            if applied_paths.len() == 1 {
                Ok(format!(
                    "Successfully applied {} hunk(s) to '{}'",
                    total_hunks, applied_paths[0].0
                ))
            } else {
                let names: Vec<&str> = applied_paths.iter().map(|(p, _)| p.as_str()).collect();
                Ok(format!(
                    "Successfully applied {} hunk(s) across {} files: {}",
                    total_hunks,
                    applied_paths.len(),
                    names.join(", ")
                ))
            }
        } else {
            // Build a partial-failure report so the model can retry only the
            // failing hunks. Files with no failures are listed at the top.
            let mut msg = String::new();
            if !applied_paths.is_empty() {
                msg.push_str("Applied successfully:\n");
                for (path, n) in &applied_paths {
                    msg.push_str(&format!(
                        "  ✓ {} ({} hunk{})\n",
                        path,
                        n,
                        if *n == 1 { "" } else { "s" }
                    ));
                }
            }
            msg.push_str("Failed (file left unchanged):\n");
            for (path, outcomes) in &failed_files {
                msg.push_str(&format!("  ✗ {}:\n", path));
                for (i, outcome) in outcomes.iter().enumerate() {
                    let label = match outcome {
                        HunkOutcome::Applied { old_start } => format!(
                            "    hunk {} (line {}): anchor found (would apply, but rolled back because another hunk in this file failed)",
                            i + 1,
                            old_start
                        ),
                        HunkOutcome::Failed { old_start, reason } => format!(
                            "    hunk {} (line {}): {}",
                            i + 1,
                            old_start,
                            reason
                        ),
                    };
                    msg.push_str(&label);
                    msg.push('\n');
                }
            }
            msg.push_str(
                "Retry only the failing hunks with corrected anchors. \
                 Successful files have already been written.",
            );
            Err(ToolError::ExecutionFailed(msg))
        }
    }
}

#[derive(Debug)]
struct FileDiff {
    file_path: Option<String>,
    hunks: Vec<DiffHunk>,
}

#[derive(Debug)]
struct DiffHunk {
    old_start: usize, // 1-based line number, treated as a hint
    lines: Vec<DiffLine>,
}

#[derive(Debug, Clone)]
enum DiffLine {
    Context(String),
    Remove(String),
    Add(String),
}

/// Per-hunk apply outcome, used to build a report when some hunks fail.
#[derive(Debug)]
enum HunkOutcome {
    Applied { old_start: usize },
    Failed { old_start: usize, reason: String },
}

/// Parse a unified diff into groups by file. Each `--- <path>` header starts a new
/// group; hunks before the first header (or in diffs without headers) end up in a
/// single group with `file_path = None`.
fn parse_unified_diff(diff: &str) -> Result<Vec<FileDiff>, ToolError> {
    let mut groups: Vec<FileDiff> = Vec::new();
    let mut current_file: Option<String> = None;
    let mut current_hunks: Vec<DiffHunk> = Vec::new();

    let lines: Vec<&str> = diff.lines().collect();
    let mut i = 0;

    let flush =
        |file: &mut Option<String>, hunks: &mut Vec<DiffHunk>, groups: &mut Vec<FileDiff>| {
            if !hunks.is_empty() || file.is_some() {
                groups.push(FileDiff {
                    file_path: file.take(),
                    hunks: std::mem::take(hunks),
                });
            }
        };

    while i < lines.len() {
        let line = lines[i];

        // File header `--- path` starts a new group.
        if let Some(rest) = line.strip_prefix("--- ") {
            flush(&mut current_file, &mut current_hunks, &mut groups);
            current_file = Some(strip_path_prefix(rest.trim()));
            i += 1;
            continue;
        }

        // `+++ path` is informational; we trust `---` for the path.
        if line.starts_with("+++") {
            i += 1;
            continue;
        }

        if line.starts_with("@@") {
            let old_start = parse_hunk_header(line)?;
            let mut hunk_lines = Vec::new();
            i += 1;

            while i < lines.len() && !lines[i].starts_with("@@") && !lines[i].starts_with("--- ") {
                let l = lines[i];
                if let Some(rest) = l.strip_prefix('-') {
                    if rest.starts_with("--") {
                        // `---` inside a hunk: stop the hunk so the caller restarts on the header.
                        break;
                    }
                    hunk_lines.push(DiffLine::Remove(rest.to_string()));
                } else if let Some(rest) = l.strip_prefix('+') {
                    if rest.starts_with("++") {
                        break;
                    }
                    hunk_lines.push(DiffLine::Add(rest.to_string()));
                } else if let Some(rest) = l.strip_prefix(' ') {
                    hunk_lines.push(DiffLine::Context(rest.to_string()));
                } else if l.is_empty() {
                    hunk_lines.push(DiffLine::Context(String::new()));
                } else {
                    // Unprefixed line — treat as context to be lenient.
                    hunk_lines.push(DiffLine::Context(l.to_string()));
                }
                i += 1;
            }

            current_hunks.push(DiffHunk {
                old_start,
                lines: hunk_lines,
            });
        } else {
            i += 1;
        }
    }

    flush(&mut current_file, &mut current_hunks, &mut groups);

    Ok(groups)
}

/// Strip common diff path prefixes like `a/`, `b/`, `./`.
fn strip_path_prefix(p: &str) -> String {
    let trimmed = p
        .strip_prefix("a/")
        .or_else(|| p.strip_prefix("b/"))
        .or_else(|| p.strip_prefix("./"))
        .unwrap_or(p);
    trimmed.trim().to_string()
}

fn parse_hunk_header(line: &str) -> Result<usize, ToolError> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(ToolError::InvalidArguments(format!(
            "Invalid hunk header: {}",
            line
        )));
    }

    let old_range = parts[1].trim_start_matches('-');
    let old_start = if let Some((start, _count)) = old_range.split_once(',') {
        start
            .parse::<usize>()
            .map_err(|_| ToolError::InvalidArguments(format!("Invalid line number in: {}", line)))?
    } else {
        old_range
            .parse::<usize>()
            .map_err(|_| ToolError::InvalidArguments(format!("Invalid line number in: {}", line)))?
    };

    if old_start == 0 {
        return Err(ToolError::InvalidArguments(
            "Invalid hunk: old_start must be >= 1 in unified diff format".to_string(),
        ));
    }

    Ok(old_start)
}

/// Try to apply every hunk and report per-hunk outcome. The returned content
/// is only valid when every hunk applied; it's `None` if any hunk failed (the
/// caller should not write a partially-applied file). The `outcomes` vector
/// always has one entry per input hunk, in order.
fn apply_hunks(content: &str, hunks: &[DiffHunk]) -> (Vec<HunkOutcome>, Option<String>) {
    let original_lines: Vec<&str> = content.lines().collect();
    let mut result_lines: Vec<String> = Vec::new();
    let mut current_line = 0usize; // 0-based index into original_lines
    let mut outcomes: Vec<HunkOutcome> = Vec::with_capacity(hunks.len());
    let mut all_applied = true;

    for hunk in hunks {
        // Build the anchor: the sequence of context+remove lines we must find in the file.
        let anchor: Vec<&str> = hunk
            .lines
            .iter()
            .filter_map(|l| match l {
                DiffLine::Context(t) | DiffLine::Remove(t) => Some(t.as_str()),
                DiffLine::Add(_) => None,
            })
            .collect();

        let hint = hunk.old_start.saturating_sub(1);
        let actual_start = if anchor.is_empty() {
            // Pure-add hunk: drop the new lines at the hint position, clamped to file end.
            Some(hint.min(original_lines.len()).max(current_line))
        } else {
            find_anchor(&original_lines, &anchor, current_line, hint)
        };

        let Some(actual_start) = actual_start else {
            outcomes.push(HunkOutcome::Failed {
                old_start: hunk.old_start,
                reason: "context lines do not match the file".to_string(),
            });
            all_applied = false;
            // Skip this hunk; continue trying the rest so the model gets a full report.
            continue;
        };

        // Copy lines between previous cursor and the anchor.
        while current_line < actual_start && current_line < original_lines.len() {
            result_lines.push(original_lines[current_line].to_string());
            current_line += 1;
        }

        // Apply the hunk's lines.
        for diff_line in &hunk.lines {
            match diff_line {
                DiffLine::Context(_) => {
                    if current_line < original_lines.len() {
                        result_lines.push(original_lines[current_line].to_string());
                        current_line += 1;
                    }
                }
                DiffLine::Remove(_) => {
                    if current_line < original_lines.len() {
                        current_line += 1;
                    }
                }
                DiffLine::Add(text) => {
                    result_lines.push(text.clone());
                }
            }
        }
        outcomes.push(HunkOutcome::Applied {
            old_start: hunk.old_start,
        });
    }

    if !all_applied {
        return (outcomes, None);
    }

    // Tail of the original file.
    while current_line < original_lines.len() {
        result_lines.push(original_lines[current_line].to_string());
        current_line += 1;
    }

    let mut result = result_lines.join("\n");
    if content.ends_with('\n') && !result.ends_with('\n') {
        result.push('\n');
    }

    (outcomes, Some(result))
}

/// Locate the position in `lines` where `anchor` matches (after trim). Search starts
/// from `min_pos`. When multiple matches exist, the one closest to `hint` wins. The
/// first match at or after `hint` is checked first as a fast path.
fn find_anchor(lines: &[&str], anchor: &[&str], min_pos: usize, hint: usize) -> Option<usize> {
    if anchor.is_empty() {
        return Some(hint.min(lines.len()).max(min_pos));
    }

    let max_start = lines.len().saturating_sub(anchor.len());

    let try_at = |start: usize| -> bool {
        if start + anchor.len() > lines.len() {
            return false;
        }
        anchor
            .iter()
            .zip(&lines[start..start + anchor.len()])
            .all(|(a, l)| a.trim() == l.trim())
    };

    // Fast path: try the hint position first.
    if hint >= min_pos && hint <= max_start && try_at(hint) {
        return Some(hint);
    }

    // Otherwise search the whole valid range and pick the closest match to the hint.
    let mut best: Option<usize> = None;
    let mut best_distance = usize::MAX;
    for start in min_pos..=max_start {
        if try_at(start) {
            let distance = start.abs_diff(hint);
            if distance < best_distance {
                best_distance = distance;
                best = Some(start);
            }
        }
    }
    best
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
        let path = "/tmp/whet_test_diff_simple.txt";
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
        let path = "/tmp/whet_test_diff_ctx.txt";
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
        let path = "/tmp/whet_test_diff_add.txt";
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
        let path = "/tmp/whet_test_diff_rm.txt";
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
        let path = "/tmp/whet_test_diff_multi.txt";
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
        let path = "/tmp/whet_test_diff_headers.txt";
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
        let path = "/tmp/whet_test_diff_nohunks.txt";
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
        let path = "/tmp/whet_test_diff_zero.txt";
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
        let path = "/tmp/whet_test_diff_ws.txt";
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
        assert!(content.starts_with("  indented\n"));
        assert!(content.contains("CHANGED"));

        cleanup(path);
    }

    #[test]
    fn test_parse_hunk_header_with_count() {
        let start = parse_hunk_header("@@ -10,5 +12,7 @@").unwrap();
        assert_eq!(start, 10);
    }

    #[test]
    fn test_parse_hunk_header_without_count() {
        let start = parse_hunk_header("@@ -3 +5 @@").unwrap();
        assert_eq!(start, 3);
    }

    #[test]
    fn test_parse_hunk_header_with_context_text() {
        let start = parse_hunk_header("@@ -10,3 +10,5 @@ fn main() {").unwrap();
        assert_eq!(start, 10);
    }

    #[test]
    fn test_risk_level_is_moderate() {
        let tool = ApplyDiffTool;
        use crate::config::ToolRiskLevel;
        assert_eq!(tool.risk_level(), ToolRiskLevel::Moderate);
    }

    #[test]
    fn test_apply_diff_file_size_limit() {
        let path = "/tmp/whet_test_diff_large.txt";
        {
            use std::io::Write;
            let mut f = std::fs::File::create(path).unwrap();
            let chunk = vec![b'x'; 1_000_000];
            for _ in 0..11 {
                f.write_all(&chunk).unwrap();
            }
        }

        let tool = ApplyDiffTool;
        let result = tool.execute(json!({
            "path": path,
            "diff": "@@ -1,1 +1,1 @@\n-x\n+y"
        }));
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("too large"));

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_trailing_newline_preserved() {
        let path = "/tmp/whet_test_diff_newline.txt";
        setup_test_file(path, "line1\nline2\n");

        let tool = ApplyDiffTool;
        tool.execute(json!({
            "path": path,
            "diff": "@@ -2,1 +2,1 @@\n-line2\n+changed"
        }))
        .unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.ends_with('\n'));
        assert_eq!(content, "line1\nchanged\n");

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_no_trailing_newline_preserved() {
        let path = "/tmp/whet_test_diff_no_newline.txt";
        setup_test_file(path, "line1\nline2");

        let tool = ApplyDiffTool;
        tool.execute(json!({
            "path": path,
            "diff": "@@ -2,1 +2,1 @@\n-line2\n+changed"
        }))
        .unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        assert!(!content.ends_with('\n'));
        assert_eq!(content, "line1\nchanged");

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_malformed_hunk_header() {
        let path = "/tmp/whet_test_diff_malformed.txt";
        setup_test_file(path, "content\n");

        let tool = ApplyDiffTool;
        let result = tool.execute(json!({
            "path": path,
            "diff": "@@ invalid header @@\n-old\n+new"
        }));
        assert!(result.is_err());

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_context_mismatch() {
        let path = "/tmp/whet_test_diff_ctx_mismatch.txt";
        setup_test_file(path, "alpha\nbeta\ngamma\n");

        let tool = ApplyDiffTool;
        let result = tool.execute(json!({
            "path": path,
            "diff": "@@ -1,3 +1,3 @@\n wrong_context\n-beta\n+BETA\n gamma"
        }));
        // wrong_context is not in the file at all and gamma isn't adjacent in a way that
        // makes a complete anchor — fuzzy match fails, error is raised.
        assert!(result.is_err());

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_multi_hunk_with_gap() {
        let path = "/tmp/whet_test_diff_gap.txt";
        setup_test_file(
            path,
            "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\n",
        );

        let tool = ApplyDiffTool;
        tool.execute(json!({
            "path": path,
            "diff": "@@ -1,1 +1,1 @@\n-line1\n+FIRST\n@@ -7,1 +7,1 @@\n-line7\n+SEVENTH"
        }))
        .unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("FIRST"));
        assert!(content.contains("SEVENTH"));
        assert!(content.contains("line4"));
        assert!(content.contains("line5"));

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_add_lines_only() {
        let path = "/tmp/whet_test_diff_add_only.txt";
        setup_test_file(path, "line1\nline2\n");

        let tool = ApplyDiffTool;
        tool.execute(json!({
            "path": path,
            "diff": "@@ -1,1 +1,3 @@\n line1\n+new_line_a\n+new_line_b"
        }))
        .unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        assert_eq!(content, "line1\nnew_line_a\nnew_line_b\nline2\n");

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_remove_lines_only() {
        let path = "/tmp/whet_test_diff_remove_only.txt";
        setup_test_file(path, "keep\nremove_me\nalso_remove\nkeep_too\n");

        let tool = ApplyDiffTool;
        tool.execute(json!({
            "path": path,
            "diff": "@@ -1,4 +1,2 @@\n keep\n-remove_me\n-also_remove\n keep_too"
        }))
        .unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        assert_eq!(content, "keep\nkeep_too\n");

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_with_file_headers_replace() {
        let path = "/tmp/whet_test_diff_headers2.txt";
        setup_test_file(path, "old_line\n");

        let tool = ApplyDiffTool;
        tool.execute(json!({
            "path": path,
            "diff": "--- a/file.txt\n+++ b/file.txt\n@@ -1,1 +1,1 @@\n-old_line\n+new_line"
        }))
        .unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        assert_eq!(content, "new_line\n");

        cleanup(path);
    }

    // --- New tests for fuzzy line matching and multi-file support ---

    #[test]
    fn test_apply_diff_wrong_line_number_fuzzy_matches() {
        // The hunk header points at line 99 but the actual context is on line 2.
        // The fuzzy matcher should find it anyway.
        let path = "/tmp/whet_test_diff_wrong_lineno.txt";
        setup_test_file(path, "alpha\nbeta\ngamma\ndelta\n");

        let tool = ApplyDiffTool;
        tool.execute(json!({
            "path": path,
            "diff": "@@ -99,2 +99,2 @@\n-beta\n+BETA"
        }))
        .unwrap();

        let content = fs::read_to_string(path).unwrap();
        assert_eq!(content, "alpha\nBETA\ngamma\ndelta\n");

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_append_at_end_of_file() {
        // The classic "model uses apply_diff to append a new function" case from task8.
        // The hunk anchors near the end and the new lines should be appended.
        let path = "/tmp/whet_test_diff_append.txt";
        setup_test_file(path, "def existing():\n    return 1\n");

        let tool = ApplyDiffTool;
        tool.execute(json!({
            "path": path,
            "diff": "@@ -2,1 +2,5 @@\n     return 1\n+\n+\n+def added():\n+    return 2"
        }))
        .unwrap();

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("def existing()"));
        assert!(content.contains("def added()"));
        assert!(content.contains("return 1"));
        assert!(content.contains("return 2"));

        cleanup(path);
    }

    #[test]
    fn test_apply_diff_multi_file() {
        // Two files patched in one apply_diff call. This is the task2_typo failure mode.
        let path_a = "/tmp/whet_test_diff_multi_a.txt";
        let path_b = "/tmp/whet_test_diff_multi_b.txt";
        setup_test_file(path_a, "this is file a, with a typo recieve\n");
        setup_test_file(path_b, "file b also recieve\n");

        let tool = ApplyDiffTool;
        let diff = format!(
            "--- {pa}\n+++ {pa}\n@@ -1,1 +1,1 @@\n-this is file a, with a typo recieve\n+this is file a, with a typo receive\n--- {pb}\n+++ {pb}\n@@ -1,1 +1,1 @@\n-file b also recieve\n+file b also receive\n",
            pa = path_a,
            pb = path_b
        );
        let res = tool
            .execute(json!({
                "path": path_a,  // first file as default; multi-file diff overrides per group
                "diff": diff,
            }))
            .unwrap();

        assert!(
            res.contains("2 file"),
            "expected multi-file message: {}",
            res
        );
        assert_eq!(
            fs::read_to_string(path_a).unwrap(),
            "this is file a, with a typo receive\n"
        );
        assert_eq!(fs::read_to_string(path_b).unwrap(), "file b also receive\n");

        cleanup(path_a);
        cleanup(path_b);
    }

    #[test]
    fn test_apply_diff_multi_file_inconsistent_default() {
        // The default `path` argument and the diff's first file header may disagree.
        // The diff's header should win (this matches what the model usually intends).
        let path_a = "/tmp/whet_test_diff_overrides_a.txt";
        let path_b = "/tmp/whet_test_diff_overrides_b.txt";
        setup_test_file(path_a, "alpha\n");
        setup_test_file(path_b, "beta\n");

        let tool = ApplyDiffTool;
        let diff = format!(
            "--- {pb}\n+++ {pb}\n@@ -1,1 +1,1 @@\n-beta\n+BETA\n",
            pb = path_b
        );
        tool.execute(json!({
            "path": path_a,
            "diff": diff,
        }))
        .unwrap();

        // path_a must be untouched.
        assert_eq!(fs::read_to_string(path_a).unwrap(), "alpha\n");
        assert_eq!(fs::read_to_string(path_b).unwrap(), "BETA\n");

        cleanup(path_a);
        cleanup(path_b);
    }

    #[test]
    fn test_apply_diff_anchor_not_found_errors() {
        // No fuzzy match exists — the anchor's content is nowhere in the file.
        let path = "/tmp/whet_test_diff_anchor_missing.txt";
        setup_test_file(path, "alpha\nbeta\n");

        let tool = ApplyDiffTool;
        let result = tool.execute(json!({
            "path": path,
            "diff": "@@ -1,1 +1,1 @@\n-NEVER_PRESENT\n+REPLACEMENT"
        }));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("context lines do not match"),
            "msg was: {}",
            msg
        );
        // File must be left untouched.
        let content = fs::read_to_string(path).unwrap();
        assert_eq!(content, "alpha\nbeta\n");

        cleanup(path);
    }

    #[test]
    fn test_partial_failure_reports_per_hunk_outcomes() {
        // Two hunks: the first has a valid anchor, the second's anchor is bogus.
        // The file must be left unchanged (atomic-per-file) and the error
        // must enumerate every hunk so the model can retry only the failing one.
        let path = "/tmp/whet_test_diff_partial.txt";
        setup_test_file(path, "alpha\nbeta\ngamma\n");

        let diff = "@@ -1,1 +1,1 @@\n-alpha\n+ALPHA\n@@ -3,1 +3,1 @@\n-NEVER\n+ZZZ";
        let tool = ApplyDiffTool;
        let result = tool.execute(json!({"path": path, "diff": diff}));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();

        assert!(msg.contains("hunk 1"), "msg was: {}", msg);
        assert!(msg.contains("hunk 2"), "msg was: {}", msg);
        assert!(msg.contains("anchor found"), "msg was: {}", msg);
        assert!(
            msg.contains("context lines do not match"),
            "msg was: {}",
            msg
        );

        // File must be untouched — atomic-per-file rollback.
        let content = fs::read_to_string(path).unwrap();
        assert_eq!(content, "alpha\nbeta\ngamma\n");

        cleanup(path);
    }

    #[test]
    fn test_multi_file_partial_writes_succeeded_skips_failed() {
        // File A's hunk applies cleanly; File B's hunk has a bogus anchor.
        // File A must be written; File B must be untouched. The error must
        // mention both files distinctly.
        let dir = tempfile::TempDir::new().unwrap();
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        std::fs::write(&a, "one\n").unwrap();
        std::fs::write(&b, "two\n").unwrap();

        let diff = format!(
            "--- {}\n@@ -1,1 +1,1 @@\n-one\n+ONE\n--- {}\n@@ -1,1 +1,1 @@\n-NEVER\n+ZZZ",
            a.display(),
            b.display()
        );
        let tool = ApplyDiffTool;
        let result = tool.execute(json!({"path": a.display().to_string(), "diff": diff}));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();

        // File A was written.
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "ONE\n");
        // File B was left alone.
        assert_eq!(std::fs::read_to_string(&b).unwrap(), "two\n");

        assert!(msg.contains("Applied successfully"), "msg was: {}", msg);
        assert!(msg.contains("a.txt"), "msg was: {}", msg);
        assert!(msg.contains("Failed"), "msg was: {}", msg);
        assert!(msg.contains("b.txt"), "msg was: {}", msg);
    }
}
