use super::{Tool, ToolError};
use serde_json::json;
use std::process::Command;

pub struct GitTool;

const ALLOWED_COMMANDS: &[&str] = &[
    "status", "diff", "log", "add", "commit", "branch", "show", "stash",
];

const BLOCKED_COMMANDS: &[&str] = &[
    "push", "reset", "clean", "checkout", "rebase", "merge", "pull", "fetch", "remote", "clone",
    "force-push",
];

impl Tool for GitTool {
    fn name(&self) -> &str {
        "git"
    }

    fn description(&self) -> &str {
        "Execute a git command (safe commands only: status, diff, log, add, commit, branch, show, stash)"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The git subcommand (e.g., 'status', 'diff', 'log')"
                },
                "args": {
                    "type": "string",
                    "description": "Additional arguments for the git command (optional)"
                }
            },
            "required": ["command"]
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'command' argument".to_string()))?;
        let extra_args = args["args"].as_str().unwrap_or("");

        // Check if command is blocked
        if BLOCKED_COMMANDS.contains(&command) {
            return Err(ToolError::PermissionDenied(format!(
                "git {} is blocked for safety. Allowed commands: {}",
                command,
                ALLOWED_COMMANDS.join(", ")
            )));
        }

        // Check if command is in allowlist
        if !ALLOWED_COMMANDS.contains(&command) {
            return Err(ToolError::PermissionDenied(format!(
                "git {} is not allowed. Allowed commands: {}",
                command,
                ALLOWED_COMMANDS.join(", ")
            )));
        }

        // Special check: commit requires -m flag to prevent interactive editor
        if command == "commit" && !extra_args.contains("-m") {
            return Err(ToolError::InvalidArguments(
                "git commit requires -m flag (e.g., args: \"-m 'commit message'\"). Interactive editor is not supported.".to_string(),
            ));
        }

        let mut cmd = Command::new("git");
        cmd.arg(command);

        // Parse extra args by splitting on whitespace (respecting quotes)
        if !extra_args.is_empty() {
            for arg in shell_split(extra_args) {
                cmd.arg(arg);
            }
        }

        let output = cmd.output().map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to execute git: {}", e))
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("[stderr] ");
            result.push_str(&stderr);
        }

        if result.is_empty() {
            result = "(no output)".to_string();
        }

        Ok(result)
    }
}

/// Simple shell-like argument splitting that respects single and double quotes.
fn shell_split(s: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    for ch in s.chars() {
        match ch {
            '\'' if !in_double_quote => {
                in_single_quote = !in_single_quote;
            }
            '"' if !in_single_quote => {
                in_double_quote = !in_double_quote;
            }
            ' ' | '\t' if !in_single_quote && !in_double_quote => {
                if !current.is_empty() {
                    args.push(current.clone());
                    current.clear();
                }
            }
            _ => {
                current.push(ch);
            }
        }
    }
    if !current.is_empty() {
        args.push(current);
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_status() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "status"}));
        // Should succeed in a git repo
        assert!(result.is_ok());
    }

    #[test]
    fn test_git_log() {
        let tool = GitTool;
        let result = tool
            .execute(json!({"command": "log", "args": "--oneline -5"}))
            .unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_git_diff() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "diff"}));
        assert!(result.is_ok());
    }

    #[test]
    fn test_git_branch() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "branch"})).unwrap();
        assert!(result.contains("main") || result.contains("master") || !result.is_empty());
    }

    #[test]
    fn test_git_push_blocked() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "push"}));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::PermissionDenied(_)));
    }

    #[test]
    fn test_git_reset_blocked() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "reset"}));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::PermissionDenied(_)));
    }

    #[test]
    fn test_git_clean_blocked() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "clean"}));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::PermissionDenied(_)));
    }

    #[test]
    fn test_git_unknown_command() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "bisect"}));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::PermissionDenied(_)));
    }

    #[test]
    fn test_git_commit_without_m_flag() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "commit"}));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("-m flag"));
    }

    #[test]
    fn test_git_missing_command_arg() {
        let tool = GitTool;
        let result = tool.execute(json!({}));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_shell_split_basic() {
        let args = shell_split("--oneline -5");
        assert_eq!(args, vec!["--oneline", "-5"]);
    }

    #[test]
    fn test_shell_split_quoted() {
        let args = shell_split("-m 'initial commit'");
        assert_eq!(args, vec!["-m", "initial commit"]);
    }

    #[test]
    fn test_shell_split_double_quoted() {
        let args = shell_split("-m \"initial commit\"");
        assert_eq!(args, vec!["-m", "initial commit"]);
    }

    #[test]
    fn test_shell_split_empty() {
        let args = shell_split("");
        assert!(args.is_empty());
    }

    // --- Test all blocked commands individually ---

    #[test]
    fn test_git_all_blocked_commands() {
        let tool = GitTool;
        for cmd in BLOCKED_COMMANDS {
            let result = tool.execute(json!({"command": cmd}));
            assert!(
                result.is_err(),
                "git {} should be blocked",
                cmd
            );
            assert!(
                matches!(result.unwrap_err(), ToolError::PermissionDenied(_)),
                "git {} should return PermissionDenied",
                cmd
            );
        }
    }

    #[test]
    fn test_git_commit_with_m_flag_is_accepted() {
        let tool = GitTool;
        // This should NOT error on the -m check (may error on git itself if nothing staged)
        let result = tool.execute(json!({"command": "commit", "args": "-m 'test message'"}));
        // We're just checking it doesn't get rejected for missing -m
        match result {
            Ok(_) => {} // Succeeded (something was staged)
            Err(ToolError::ExecutionFailed(_)) => {} // git itself errored (nothing to commit)
            Err(ToolError::InvalidArguments(msg)) => {
                panic!("Should not get InvalidArguments with -m flag: {}", msg);
            }
            Err(other) => {
                panic!("Unexpected error: {}", other);
            }
        }
    }

    #[test]
    fn test_git_show() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "show", "args": "--stat HEAD"}));
        assert!(result.is_ok());
    }

    #[test]
    fn test_git_stash_list() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "stash", "args": "list"}));
        assert!(result.is_ok());
    }

    #[test]
    fn test_shell_split_multiple_spaces() {
        let args = shell_split("  --flag   value  ");
        assert_eq!(args, vec!["--flag", "value"]);
    }

    #[test]
    fn test_shell_split_mixed_quotes() {
        let args = shell_split(r#"-m "hello 'world'""#);
        assert_eq!(args, vec!["-m", "hello 'world'"]);
    }

    #[test]
    fn test_shell_split_single_arg() {
        let args = shell_split("--oneline");
        assert_eq!(args, vec!["--oneline"]);
    }

    #[test]
    fn test_git_error_messages_contain_allowed_list() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "push"}));
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("status"));
        assert!(err_msg.contains("diff"));
        assert!(err_msg.contains("log"));
    }
}
