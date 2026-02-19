use super::{Tool, ToolError};
use crate::config::ToolRiskLevel;
use serde_json::json;
use std::process::Command;

pub struct GitTool;

/// Commands that are always allowed without approval (read-only).
const SAFE_COMMANDS: &[&str] = &["status", "diff", "log", "show", "branch", "stash"];

/// Commands that require user approval before execution.
const APPROVAL_COMMANDS: &[&str] = &[
    "add",
    "commit",
    "checkout",
    "switch",
    "pull",
    "fetch",
    "push",
    "merge",
    "tag",
    "cherry-pick",
    "remote",
];

/// Commands that are always blocked regardless of approval.
const BLOCKED_COMMANDS: &[&str] = &["clean", "rebase"];

/// Detect dangerous argument patterns that should always be blocked.
fn has_dangerous_args(command: &str, args: &[String]) -> Option<String> {
    match command {
        "reset" => {
            if args.iter().any(|a| a == "--hard") {
                return Some("git reset --hard is blocked for safety".to_string());
            }
            // soft/mixed reset is allowed with approval
            None
        }
        "push" => {
            if args
                .iter()
                .any(|a| a == "--force" || a == "-f" || a == "--force-with-lease")
            {
                return Some("git push --force is blocked for safety".to_string());
            }
            None
        }
        "clean" => Some("git clean is blocked for safety (can delete untracked files)".to_string()),
        "rebase" => {
            if args.iter().any(|a| a == "-i" || a == "--interactive") {
                return Some(
                    "git rebase -i is blocked (interactive mode not supported)".to_string(),
                );
            }
            Some("git rebase is blocked for safety".to_string())
        }
        _ => None,
    }
}

impl Tool for GitTool {
    fn name(&self) -> &str {
        "git"
    }

    fn description(&self) -> &str {
        "Execute a git command. Safe commands (status, diff, log, show, branch, stash) run freely. \
         Others (add, commit, checkout, switch, pull, fetch, push, merge, tag, cherry-pick, remote, reset) require approval."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The git subcommand (e.g., 'status', 'diff', 'log', 'add', 'commit', 'push')"
                },
                "args": {
                    "type": "string",
                    "description": "Additional arguments for the git command (optional)"
                }
            },
            "required": ["command"]
        })
    }

    fn risk_level(&self) -> ToolRiskLevel {
        // Default risk level — actual risk is determined per-command in execute().
        // We return Moderate so that approval is requested for non-safe commands in default permission mode.
        // Safe commands bypass this via the agent's permission check since we override at execute time.
        ToolRiskLevel::Moderate
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'command' argument".to_string()))?;
        let extra_args = args["args"].as_str().unwrap_or("");
        let parsed_args = shell_split(extra_args);

        // Always-blocked commands
        if BLOCKED_COMMANDS.contains(&command) {
            if let Some(reason) = has_dangerous_args(command, &parsed_args) {
                return Err(ToolError::PermissionDenied(reason));
            }
            return Err(ToolError::PermissionDenied(format!(
                "git {} is blocked for safety",
                command
            )));
        }

        // Check for dangerous argument patterns on approval commands
        if let Some(reason) = has_dangerous_args(command, &parsed_args) {
            return Err(ToolError::PermissionDenied(reason));
        }

        // Validate the command is in one of the known categories
        let is_safe = SAFE_COMMANDS.contains(&command);
        let is_approval = APPROVAL_COMMANDS.contains(&command);
        // Allow "reset" as approval-required (dangerous patterns already caught above)
        let is_reset = command == "reset";

        if !is_safe && !is_approval && !is_reset {
            return Err(ToolError::PermissionDenied(format!(
                "git {} is not allowed. Allowed commands: {}, {}",
                command,
                SAFE_COMMANDS.join(", "),
                APPROVAL_COMMANDS.join(", ")
            )));
        }

        // Special check: commit requires -m flag to prevent interactive editor
        if command == "commit" {
            let has_m_flag = parsed_args.iter().any(|a| a == "-m" || a.starts_with("-m"));
            if !has_m_flag {
                return Err(ToolError::InvalidArguments(
                    "git commit requires -m flag (e.g., args: \"-m 'commit message'\"). Interactive editor is not supported.".to_string(),
                ));
            }
        }

        let mut cmd = Command::new("git");
        cmd.arg(command);

        // Parse extra args by splitting on whitespace (respecting quotes)
        if !extra_args.is_empty() {
            for arg in &parsed_args {
                cmd.arg(arg);
            }
        }

        let output = cmd
            .output()
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to execute git: {}", e)))?;

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

/// Returns the risk level for a specific git subcommand.
/// Safe commands return Safe, approval commands return Moderate.
pub fn git_command_risk_level(command: &str) -> ToolRiskLevel {
    if SAFE_COMMANDS.contains(&command) {
        ToolRiskLevel::Safe
    } else {
        ToolRiskLevel::Moderate
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

    // --- Approval-required commands ---

    #[test]
    fn test_git_push_is_approval_required() {
        // push is now allowed (with approval), not blocked
        assert!(APPROVAL_COMMANDS.contains(&"push"));
    }

    #[test]
    fn test_git_push_force_blocked() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "push", "args": "--force"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_git_push_force_f_blocked() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "push", "args": "-f origin main"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    // --- Always-blocked commands ---

    #[test]
    fn test_git_reset_hard_blocked() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "reset", "args": "--hard HEAD~1"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_git_reset_soft_allowed() {
        // reset --soft is allowed (with approval) — it just might not change anything
        let tool = GitTool;
        let result = tool.execute(json!({"command": "reset", "args": "--soft HEAD"}));
        // Should not be PermissionDenied
        match result {
            Ok(_) => {}
            Err(ToolError::PermissionDenied(msg)) => {
                panic!("reset --soft should not be blocked: {}", msg)
            }
            Err(_) => {} // git itself may error, that's fine
        }
    }

    #[test]
    fn test_git_clean_blocked() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "clean"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_git_rebase_blocked() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "rebase"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_git_rebase_interactive_blocked() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "rebase", "args": "-i HEAD~3"}));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("interactive"));
    }

    #[test]
    fn test_git_unknown_command() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "bisect"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
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
        assert!(matches!(
            result.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
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

    #[test]
    fn test_git_commit_without_m_false_positive_rejected() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "commit", "args": "--allow-multiple"}));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("-m flag"));
    }

    #[test]
    fn test_git_commit_with_m_flag_is_accepted() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "commit", "args": "-m 'test message'"}));
        match result {
            Ok(_) => {}
            Err(ToolError::ExecutionFailed(_)) => {}
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
        let result = tool.execute(json!({"command": "bisect"}));
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("status"));
        assert!(err_msg.contains("diff"));
        assert!(err_msg.contains("log"));
    }

    // --- Risk level tests ---

    #[test]
    fn test_git_command_risk_level_safe() {
        assert_eq!(git_command_risk_level("status"), ToolRiskLevel::Safe);
        assert_eq!(git_command_risk_level("diff"), ToolRiskLevel::Safe);
        assert_eq!(git_command_risk_level("log"), ToolRiskLevel::Safe);
        assert_eq!(git_command_risk_level("show"), ToolRiskLevel::Safe);
        assert_eq!(git_command_risk_level("branch"), ToolRiskLevel::Safe);
        assert_eq!(git_command_risk_level("stash"), ToolRiskLevel::Safe);
    }

    #[test]
    fn test_git_command_risk_level_moderate() {
        assert_eq!(git_command_risk_level("add"), ToolRiskLevel::Moderate);
        assert_eq!(git_command_risk_level("commit"), ToolRiskLevel::Moderate);
        assert_eq!(git_command_risk_level("push"), ToolRiskLevel::Moderate);
        assert_eq!(git_command_risk_level("checkout"), ToolRiskLevel::Moderate);
        assert_eq!(git_command_risk_level("merge"), ToolRiskLevel::Moderate);
    }

    // --- Approval commands are accepted ---

    #[test]
    fn test_git_checkout_accepted() {
        assert!(APPROVAL_COMMANDS.contains(&"checkout"));
    }

    #[test]
    fn test_git_switch_accepted() {
        assert!(APPROVAL_COMMANDS.contains(&"switch"));
    }

    #[test]
    fn test_git_fetch_accepted() {
        assert!(APPROVAL_COMMANDS.contains(&"fetch"));
    }

    #[test]
    fn test_git_merge_accepted() {
        assert!(APPROVAL_COMMANDS.contains(&"merge"));
    }

    #[test]
    fn test_git_tag_accepted() {
        assert!(APPROVAL_COMMANDS.contains(&"tag"));
    }

    #[test]
    fn test_git_cherry_pick_accepted() {
        assert!(APPROVAL_COMMANDS.contains(&"cherry-pick"));
    }

    #[test]
    fn test_git_remote_accepted() {
        assert!(APPROVAL_COMMANDS.contains(&"remote"));
    }

    // --- Edge case tests ---

    #[test]
    fn test_git_empty_output_shows_no_output() {
        let tool = GitTool;
        // `git diff` with no changes produces empty output
        // Use a known-clean state by diffing HEAD with itself
        let result = tool
            .execute(json!({"command": "diff", "args": "HEAD HEAD"}))
            .unwrap();
        assert_eq!(result, "(no output)");
    }

    #[test]
    fn test_shell_split_unclosed_single_quote() {
        // Unclosed quote should include remaining text as one token
        let args = shell_split("-m 'unclosed message");
        assert_eq!(args, vec!["-m", "unclosed message"]);
    }

    #[test]
    fn test_shell_split_unclosed_double_quote() {
        let args = shell_split("-m \"unclosed message");
        assert_eq!(args, vec!["-m", "unclosed message"]);
    }

    #[test]
    fn test_shell_split_tab_separator() {
        let args = shell_split("-m\t'message'");
        assert_eq!(args, vec!["-m", "message"]);
    }

    #[test]
    fn test_shell_split_only_whitespace() {
        let args = shell_split("   \t  ");
        assert!(args.is_empty());
    }

    #[test]
    fn test_git_push_force_with_lease_blocked() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "push", "args": "--force-with-lease"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[test]
    fn test_git_risk_level_unknown_command() {
        // Unknown commands default to Moderate
        assert_eq!(git_command_risk_level("bisect"), ToolRiskLevel::Moderate);
        assert_eq!(git_command_risk_level(""), ToolRiskLevel::Moderate);
    }

    #[test]
    fn test_git_commit_with_long_form_m_flag() {
        let tool = GitTool;
        // --message should NOT be accepted (only -m and -m* are checked)
        let result = tool.execute(json!({"command": "commit", "args": "--message 'test'"}));
        // The current code checks for -m or starts_with("-m"), so --message won't match
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("-m flag"));
    }

    #[test]
    fn test_git_reset_hard_with_extra_args() {
        let tool = GitTool;
        let result = tool.execute(json!({"command": "reset", "args": "--hard --quiet HEAD~1"}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }
}
