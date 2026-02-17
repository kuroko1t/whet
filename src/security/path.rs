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
    fn test_path_safety_safe_paths() {
        assert!(is_path_safe("./src/main.rs"));
        assert!(is_path_safe("/tmp/test.txt"));
        assert!(is_path_safe("Cargo.toml"));
        assert!(is_path_safe("/home/user/project/file.txt"));
        assert!(is_path_safe("/etc/hostname"));
        assert!(is_path_safe("/var/log/syslog"));
    }

    #[test]
    fn test_path_safety_blocks_etc_shadow() {
        assert!(!is_path_safe("/etc/shadow"));
    }

    #[test]
    fn test_path_safety_blocks_etc_gshadow() {
        assert!(!is_path_safe("/etc/gshadow"));
    }

    #[test]
    fn test_path_safety_blocks_etc_sudoers() {
        assert!(!is_path_safe("/etc/sudoers"));
    }

    #[test]
    fn test_path_safety_blocks_ssh_dir() {
        assert!(!is_path_safe("~/.ssh/id_rsa"));
        assert!(!is_path_safe("~/.ssh/authorized_keys"));
        assert!(!is_path_safe("~/.ssh/config"));
    }

    #[test]
    fn test_path_safety_blocks_gnupg() {
        assert!(!is_path_safe("~/.gnupg/private-keys-v1.d/key.gpg"));
    }

    #[test]
    fn test_path_safety_blocks_aws() {
        assert!(!is_path_safe("~/.aws/credentials"));
        assert!(!is_path_safe("~/.aws/config"));
    }

    #[test]
    fn test_path_safety_allows_etc_other() {
        assert!(is_path_safe("/etc/hosts"));
        assert!(is_path_safe("/etc/resolv.conf"));
    }
}
