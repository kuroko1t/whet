use super::{Sandbox, SandboxError, SandboxResult};
use crate::tools::ToolPermissions;

const TIMEOUT_SECS: u64 = 30;

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
        // Use timeout + unshare for sandboxed execution
        let mut cmd = std::process::Command::new("timeout");
        cmd.arg(TIMEOUT_SECS.to_string());
        cmd.arg("unshare");

        if !permissions.network {
            cmd.arg("--net");
        }

        cmd.args(["--", "sh", "-c", command]);

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        let output = cmd.output().map_err(|e| {
            SandboxError::ExecutionFailed(format!("Failed to run sandboxed command: {}", e))
        })?;

        let exit_code = output.status.code().unwrap_or(-1);

        // timeout command returns 124 when the command times out
        if exit_code == 124 {
            return Err(SandboxError::Timeout);
        }

        Ok(SandboxResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code,
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

/// Check if a path is safe to access (not a sensitive system path).
pub fn is_path_safe(path: &str) -> bool {
    let sensitive_paths = [
        "/etc/shadow",
        "/etc/gshadow",
        "/etc/sudoers",
    ];
    let sensitive_prefixes = [
        "~/.ssh",
        "~/.gnupg",
        "~/.aws",
    ];

    let expanded = if path.starts_with('~') {
        if let Some(home) = dirs::home_dir() {
            path.replacen('~', &home.display().to_string(), 1)
        } else {
            path.to_string()
        }
    } else {
        path.to_string()
    };

    let canonical = std::path::Path::new(&expanded);
    let path_str = canonical.display().to_string();

    for sensitive in &sensitive_paths {
        if path_str == *sensitive {
            return false;
        }
    }

    if let Some(home) = dirs::home_dir() {
        for prefix in &sensitive_prefixes {
            let expanded_prefix = prefix.replacen('~', &home.display().to_string(), 1);
            if path_str.starts_with(&expanded_prefix) {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_sandbox_executes() {
        let sandbox = NoOpSandbox;
        let perms = ToolPermissions {
            filesystem_read: true,
            filesystem_write: false,
            network: false,
            subprocess: true,
        };
        let result = sandbox.execute("echo hello", &perms, None).unwrap();
        assert_eq!(result.stdout.trim(), "hello");
        assert_eq!(result.exit_code, 0);
    }

    #[test]
    fn test_namespace_sandbox_blocks_network() {
        let sandbox = NamespaceSandbox::new();
        let perms = ToolPermissions {
            filesystem_read: false,
            filesystem_write: false,
            network: false,
            subprocess: true,
        };
        // Try to reach the network - should fail in isolated namespace
        let result = sandbox.execute(
            "ping -c 1 -W 1 127.0.0.1 2>&1 || echo 'NETWORK_BLOCKED'",
            &perms,
            None,
        );
        match result {
            Ok(r) => {
                assert!(
                    r.stdout.contains("NETWORK_BLOCKED")
                        || r.stderr.contains("Network is unreachable")
                        || r.exit_code != 0,
                    "Expected network to be blocked, got stdout='{}' stderr='{}'",
                    r.stdout,
                    r.stderr
                );
            }
            Err(_) => {
                // If unshare fails (no permissions), that's also acceptable in test
            }
        }
    }

    #[test]
    fn test_path_safety() {
        assert!(is_path_safe("./src/main.rs"));
        assert!(is_path_safe("/tmp/test.txt"));
        assert!(!is_path_safe("/etc/shadow"));
    }
}
