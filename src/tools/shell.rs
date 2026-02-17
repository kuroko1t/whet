use super::{Tool, ToolError};
use serde_json::json;

pub struct ShellTool;

impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "The working directory (optional)"
                }
            },
            "required": ["command"]
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'command' argument".to_string()))?;
        let working_dir = args["working_dir"].as_str();

        let mut cmd = std::process::Command::new("sh");
        cmd.args(["-c", command]);

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        let output = cmd
            .output()
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to execute command: {}", e)))?;

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

        // Append exit code so the agent can distinguish success from failure
        let exit_code = output.status.code().unwrap_or(-1);
        if exit_code != 0 {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(&format!("[exit code: {}]", exit_code));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_echo() {
        let tool = ShellTool;
        let result = tool.execute(json!({"command": "echo hello"})).unwrap();
        assert_eq!(result.trim(), "hello");
    }

    #[test]
    fn test_shell_stderr_output() {
        let tool = ShellTool;
        let result = tool
            .execute(json!({"command": "echo error_msg >&2"}))
            .unwrap();
        assert!(result.contains("[stderr]"));
        assert!(result.contains("error_msg"));
    }

    #[test]
    fn test_shell_mixed_stdout_stderr() {
        let tool = ShellTool;
        let result = tool
            .execute(json!({"command": "echo out && echo err >&2"}))
            .unwrap();
        assert!(result.contains("out"));
        assert!(result.contains("[stderr]"));
        assert!(result.contains("err"));
    }

    #[test]
    fn test_shell_working_dir() {
        let tool = ShellTool;
        let result = tool
            .execute(json!({"command": "pwd", "working_dir": "/tmp"}))
            .unwrap();
        assert!(result.trim().starts_with("/tmp"));
    }

    #[test]
    fn test_shell_invalid_working_dir() {
        let tool = ShellTool;
        let result =
            tool.execute(json!({"command": "echo hi", "working_dir": "/nonexistent_dir_12345"}));
        assert!(result.is_err());
    }

    #[test]
    fn test_shell_missing_command_arg() {
        let tool = ShellTool;
        let result = tool.execute(json!({}));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_shell_failing_command_reports_exit_code() {
        let tool = ShellTool;
        let result = tool.execute(json!({"command": "false"})).unwrap();
        assert!(result.contains("[exit code: 1]"));
    }

    #[test]
    fn test_shell_success_no_exit_code_shown() {
        let tool = ShellTool;
        let result = tool.execute(json!({"command": "true"})).unwrap();
        assert!(!result.contains("[exit code"));
    }

    #[test]
    fn test_shell_multiline_output() {
        let tool = ShellTool;
        let result = tool
            .execute(json!({"command": "echo line1 && echo line2 && echo line3"}))
            .unwrap();
        let lines: Vec<&str> = result.trim().lines().collect();
        assert_eq!(lines.len(), 3);
    }
}
