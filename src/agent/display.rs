//! Compact, Claude-Code-style display helpers for agent surfaces.
//!
//! UX.1 — `format_tool_call_compact` turns the verbose
//! `[tool: read_file] {"path":"src/lib.rs"}` line into the much more
//! scannable `Read(src/lib.rs)` form. The mapping is per-tool: each known
//! tool extracts its primary argument and presents it inline. Unknown
//! tools (e.g. MCP-registered ones at runtime) fall back to a short
//! `name(args)` rendering.
//!
//! The rendering function is pure (no I/O, no colour) so it's
//! unit-testable. The caller is responsible for ANSI colouring.

use serde_json::Value;

/// Maximum characters of an inline argument shown before a `…` ellipsis.
/// Picked so a typical shell command or path fits on one terminal line.
const MAX_ARG_LEN: usize = 70;

/// Render a single tool call as a Claude-Code-style compact line.
///
/// The output never includes ANSI colour — the caller adds it.
///
/// # Examples
///
/// ```
/// use serde_json::json;
/// use whet::agent::display::format_tool_call_compact;
///
/// assert_eq!(
///     format_tool_call_compact("read_file", &json!({"path": "src/lib.rs"})),
///     "Read(src/lib.rs)",
/// );
/// ```
pub fn format_tool_call_compact(name: &str, args: &Value) -> String {
    let s = |k: &str| args.get(k).and_then(|v| v.as_str()).unwrap_or("");

    match name {
        "read_file" => format!("Read({})", s("path")),
        "edit_file" => format!("Edit({})", s("path")),
        "apply_diff" => format!("Apply({})", s("path")),
        "write_file" => format!("Write({})", s("path")),
        "list_dir" => {
            let p = s("path");
            if p.is_empty() {
                "LS(.)".to_string()
            } else {
                format!("LS({})", p)
            }
        }
        "shell" => format!("Shell({})", truncate_arg(s("command"))),
        "grep" => {
            let pattern = s("pattern");
            let path = s("path");
            if path.is_empty() || path == "." {
                format!("Grep({})", truncate_arg(pattern))
            } else {
                format!("Grep({} in {})", truncate_arg(pattern), path)
            }
        }
        "repo_map" => {
            let p = s("path");
            if p.is_empty() {
                "RepoMap()".to_string()
            } else {
                format!("RepoMap({})", p)
            }
        }
        "web_fetch" => format!("Fetch({})", truncate_arg(s("url"))),
        "web_search" => format!("Search({})", truncate_arg(s("query"))),
        "git" => format!("Git({})", truncate_arg(s("command"))),
        // Unknown / MCP-registered tools: keep the tool name and show a
        // condensed JSON of args so the user can still see what was sent.
        _ => {
            let json = args.to_string();
            let short = truncate_arg(&json);
            if short.is_empty() || short == "{}" {
                format!("{}()", name)
            } else {
                format!("{}({})", name, short)
            }
        }
    }
}

/// Truncate `s` to at most `MAX_ARG_LEN` characters, appending `…` when
/// truncation occurred. Char-boundary safe.
fn truncate_arg(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= MAX_ARG_LEN {
        s.to_string()
    } else {
        let kept: String = chars.into_iter().take(MAX_ARG_LEN).collect();
        format!("{}…", kept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn read_file_compact() {
        assert_eq!(
            format_tool_call_compact("read_file", &json!({"path": "src/lib.rs"})),
            "Read(src/lib.rs)"
        );
    }

    #[test]
    fn edit_file_compact() {
        assert_eq!(
            format_tool_call_compact(
                "edit_file",
                &json!({"path": "src/main.rs", "old_text": "x", "new_text": "y"})
            ),
            "Edit(src/main.rs)"
        );
    }

    #[test]
    fn apply_diff_compact_uses_path_when_present() {
        assert_eq!(
            format_tool_call_compact(
                "apply_diff",
                &json!({"path": "src/foo.rs", "diff": "@@ ..."})
            ),
            "Apply(src/foo.rs)"
        );
    }

    #[test]
    fn write_file_compact() {
        assert_eq!(
            format_tool_call_compact(
                "write_file",
                &json!({"path": "out.txt", "content": "hello"})
            ),
            "Write(out.txt)"
        );
    }

    #[test]
    fn list_dir_compact_with_and_without_path() {
        assert_eq!(
            format_tool_call_compact("list_dir", &json!({"path": "src"})),
            "LS(src)"
        );
        assert_eq!(format_tool_call_compact("list_dir", &json!({})), "LS(.)");
    }

    #[test]
    fn shell_compact() {
        assert_eq!(
            format_tool_call_compact("shell", &json!({"command": "ls -la"})),
            "Shell(ls -la)"
        );
    }

    #[test]
    fn shell_command_truncated_when_long() {
        let long = "x".repeat(200);
        let out = format_tool_call_compact("shell", &json!({"command": long}));
        // Must start with "Shell(" and end with ")"
        assert!(out.starts_with("Shell("));
        assert!(out.ends_with(")"));
        // Total length should be far below the original 200-char command.
        assert!(out.contains("…"));
        assert!(out.len() < 100);
    }

    #[test]
    fn grep_compact_with_path_and_without() {
        assert_eq!(
            format_tool_call_compact("grep", &json!({"pattern": "TODO", "path": "src"})),
            "Grep(TODO in src)"
        );
        assert_eq!(
            format_tool_call_compact("grep", &json!({"pattern": "TODO"})),
            "Grep(TODO)"
        );
        assert_eq!(
            format_tool_call_compact("grep", &json!({"pattern": "TODO", "path": "."})),
            "Grep(TODO)"
        );
    }

    #[test]
    fn repo_map_compact() {
        assert_eq!(
            format_tool_call_compact("repo_map", &json!({})),
            "RepoMap()"
        );
        assert_eq!(
            format_tool_call_compact("repo_map", &json!({"path": "src"})),
            "RepoMap(src)"
        );
    }

    #[test]
    fn web_fetch_and_search_compact() {
        assert_eq!(
            format_tool_call_compact("web_fetch", &json!({"url": "https://example.com/foo"})),
            "Fetch(https://example.com/foo)"
        );
        assert_eq!(
            format_tool_call_compact(
                "web_search",
                &json!({"query": "rust async runtime comparison"})
            ),
            "Search(rust async runtime comparison)"
        );
    }

    #[test]
    fn git_compact() {
        assert_eq!(
            format_tool_call_compact("git", &json!({"command": "status"})),
            "Git(status)"
        );
    }

    #[test]
    fn unknown_tool_falls_back_to_name_and_short_args() {
        let out =
            format_tool_call_compact("mcp_some_server_action", &json!({"a": 1, "b": "hello"}));
        assert!(out.starts_with("mcp_some_server_action("));
        assert!(out.contains("\"a\":1"));
    }

    #[test]
    fn unknown_tool_with_empty_args_renders_empty_parens() {
        assert_eq!(format_tool_call_compact("mystery", &json!({})), "mystery()");
    }

    #[test]
    fn does_not_include_legacy_tool_prefix() {
        // The whole point of UX.1: no `[tool:` prefix anywhere.
        let out = format_tool_call_compact("read_file", &json!({"path": "x"}));
        assert!(!out.contains("[tool:"));
        assert!(!out.contains("tool:"));
    }

    #[test]
    fn truncation_is_char_boundary_safe() {
        // A multi-byte string that would slice mid-char if we used bytes.
        let multibyte = "あいうえお".repeat(20); // 100 chars, 300 bytes
        let truncated = truncate_arg(&multibyte);
        // Must be valid UTF-8 (would have panicked otherwise on str slice).
        assert!(truncated.ends_with("…"));
        // Should keep MAX_ARG_LEN chars (70) + ellipsis.
        assert_eq!(truncated.chars().count(), MAX_ARG_LEN + 1);
    }

    #[test]
    fn missing_required_arg_does_not_panic() {
        // If the model emits malformed args (no "path"), we still produce
        // something — empty parens — rather than crashing.
        let out = format_tool_call_compact("read_file", &json!({}));
        assert_eq!(out, "Read()");
    }
}
