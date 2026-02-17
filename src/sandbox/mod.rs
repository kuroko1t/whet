pub mod namespace;

use crate::tools::ToolPermissions;
use std::fmt;

#[derive(Debug)]
pub struct SandboxResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

#[derive(Debug)]
pub enum SandboxError {
    PermissionDenied(String),
    ExecutionFailed(String),
    Timeout,
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SandboxError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            SandboxError::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
            SandboxError::Timeout => write!(f, "Execution timed out"),
        }
    }
}

impl std::error::Error for SandboxError {}

pub trait Sandbox {
    fn execute(
        &self,
        command: &str,
        permissions: &ToolPermissions,
        working_dir: Option<&str>,
    ) -> Result<SandboxResult, SandboxError>;
}
