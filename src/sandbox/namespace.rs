use super::{Sandbox, SandboxError, SandboxResult};
use crate::tools::ToolPermissions;

pub struct NamespaceSandbox;

impl NamespaceSandbox {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NamespaceSandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl Sandbox for NamespaceSandbox {
    fn execute(
        &self,
        command: &str,
        permissions: &ToolPermissions,
        working_dir: Option<&str>,
    ) -> Result<SandboxResult, SandboxError> {
        let mut cmd = std::process::Command::new("unshare");

        if !permissions.network {
            cmd.arg("--net");
        }

        cmd.args(["--", "sh", "-c", command]);

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        let output = cmd
            .output()
            .map_err(|e| SandboxError::ExecutionFailed(format!("Failed to run unshare: {}", e)))?;

        Ok(SandboxResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }
}

pub struct NoOpSandbox;

impl Sandbox for NoOpSandbox {
    fn execute(
        &self,
        command: &str,
        _permissions: &ToolPermissions,
        working_dir: Option<&str>,
    ) -> Result<SandboxResult, SandboxError> {
        let mut cmd = std::process::Command::new("sh");
        cmd.args(["-c", command]);

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        let output = cmd
            .output()
            .map_err(|e| SandboxError::ExecutionFailed(format!("Failed to run command: {}", e)))?;

        Ok(SandboxResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }
}
