use std::path::{Component, PathBuf};

/// Normalize a path by resolving `.` and `..` components without touching the filesystem.
/// Unlike `canonicalize()`, this works even if the path doesn't exist.
fn normalize_path(path: &str) -> PathBuf {
    let p = std::path::Path::new(path);
    let mut components = Vec::new();

    for component in p.components() {
        match component {
            Component::CurDir => {} // skip "."
            Component::ParentDir => {
                // Pop the last component if possible
                if let Some(last) = components.last() {
                    match last {
                        Component::RootDir | Component::Prefix(_) => {} // can't go above root
                        Component::ParentDir => {
                            components.push(component);
                        }
                        _ => {
                            components.pop();
                        }
                    }
                } else {
                    components.push(component);
                }
            }
            _ => {
                components.push(component);
            }
        }
    }

    let mut result = PathBuf::new();
    for c in &components {
        result.push(c.as_os_str());
    }
    if result.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        result
    }
}

/// Check if a path is safe to access (not a sensitive system path).
/// Uses path normalization to prevent traversal attacks.
pub fn is_path_safe(path: &str) -> bool {
    let sensitive_paths = [
        "/etc/shadow",
        "/etc/gshadow",
        "/etc/sudoers",
        "/etc/sudoers.d",
    ];
    let sensitive_prefixes_system = ["/etc/sudoers.d/"];
    let sensitive_prefixes_home = [
        ".ssh",
        ".gnupg",
        ".aws",
        ".config/gcloud",
        ".docker/config.json",
        ".kube/config",
    ];

    // Expand ~ to home directory
    let expanded = if path.starts_with('~') {
        if let Some(home) = dirs::home_dir() {
            path.replacen('~', &home.display().to_string(), 1)
        } else {
            path.to_string()
        }
    } else {
        path.to_string()
    };

    // Build a set of path representations to check against the blocklist.
    // We check both the logically-normalized path (resolves . and ..) and
    // the filesystem-canonicalized path (resolves symlinks), to catch both
    // logical traversal attacks and symlink-based bypasses.
    let logical = normalize_path(&expanded);

    let mut paths_to_check = vec![logical.display().to_string()];
    if let Ok(canonical) = std::fs::canonicalize(&expanded) {
        let canonical_str = canonical.display().to_string();
        if !paths_to_check.contains(&canonical_str) {
            paths_to_check.push(canonical_str);
        }
    }

    for path_str in &paths_to_check {
        // Block exact sensitive paths
        for sensitive in &sensitive_paths {
            if path_str == sensitive {
                return false;
            }
        }

        // Block system sensitive prefixes
        for prefix in &sensitive_prefixes_system {
            if path_str.starts_with(prefix) {
                return false;
            }
        }
    }

    // Block home-relative sensitive paths.
    // Canonicalize home dir too for consistent comparison when symlinks are involved.
    if let Some(home) = dirs::home_dir() {
        let home_canonical = std::fs::canonicalize(&home).unwrap_or_else(|_| home.clone());
        let home_variants = [
            home.display().to_string(),
            home_canonical.display().to_string(),
        ];

        for path_str in &paths_to_check {
            for home_str in &home_variants {
                for suffix in &sensitive_prefixes_home {
                    let blocked = format!("{}/{}", home_str, suffix);
                    if path_str == &blocked || path_str.starts_with(&format!("{}/", blocked)) {
                        return false;
                    }
                }
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
    fn test_path_traversal_blocked() {
        // Attempt to reach /etc/shadow via traversal from /tmp
        assert!(!is_path_safe("/tmp/../etc/shadow"));
        assert!(!is_path_safe("/tmp/../../etc/shadow"));
    }

    #[test]
    fn test_path_traversal_home_sensitive() {
        // Attempt to reach ~/.ssh via traversal
        if let Some(home) = dirs::home_dir() {
            let traversal = format!("{}/Documents/../.ssh/id_rsa", home.display());
            assert!(!is_path_safe(&traversal));
        }
    }

    #[test]
    fn test_dot_dot_in_safe_path() {
        // /tmp/../tmp/safe.txt should be OK
        assert!(is_path_safe("/tmp/../tmp/safe.txt"));
    }

    #[test]
    fn test_path_safety_blocks_docker_config() {
        assert!(!is_path_safe("~/.docker/config.json"));
    }

    #[test]
    fn test_path_safety_blocks_kube_config() {
        assert!(!is_path_safe("~/.kube/config"));
    }

    #[test]
    fn test_path_safety_allows_etc_other() {
        assert!(is_path_safe("/etc/hosts"));
        assert!(is_path_safe("/etc/resolv.conf"));
    }

    #[test]
    fn test_normalize_path_resolves_dot_dot() {
        assert_eq!(normalize_path("/a/b/../c"), PathBuf::from("/a/c"));
        assert_eq!(normalize_path("/a/./b"), PathBuf::from("/a/b"));
        assert_eq!(normalize_path("/a/b/../../c"), PathBuf::from("/c"));
    }

    #[test]
    fn test_normalize_path_root() {
        assert_eq!(normalize_path("/"), PathBuf::from("/"));
        assert_eq!(normalize_path("/.."), PathBuf::from("/"));
    }
}
