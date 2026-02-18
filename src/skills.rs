use std::path::Path;

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub content: String,
}

/// Load skill files (.md) from the given directory path.
/// Returns an empty vec if the directory doesn't exist or is unreadable.
pub fn load_skills(skills_dir: &str) -> Vec<Skill> {
    let expanded = expand_tilde(skills_dir);
    let dir = Path::new(&expanded);

    if !dir.is_dir() {
        return Vec::new();
    }

    let mut skills = Vec::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let name = match path.file_stem().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        skills.push(Skill { name, content });
    }

    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

fn expand_tilde(path: &str) -> String {
    if path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            return format!("{}{}", home.display(), &path[1..]);
        }
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_skills_missing_dir() {
        let skills = load_skills("/nonexistent/path/skills");
        assert!(skills.is_empty());
    }

    #[test]
    fn test_load_skills_empty_dir() {
        let dir = std::env::temp_dir().join("hermitclaw_test_skills_empty");
        std::fs::create_dir_all(&dir).ok();
        let skills = load_skills(dir.to_str().unwrap());
        assert!(skills.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_skills_with_md_files() {
        let dir = std::env::temp_dir().join("hermitclaw_test_skills_md");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("coding.md"), "You are a coding assistant.").ok();
        std::fs::write(dir.join("review.md"), "Review code carefully.").ok();
        std::fs::write(dir.join("notes.txt"), "This should be ignored.").ok();

        let skills = load_skills(dir.to_str().unwrap());
        assert_eq!(skills.len(), 2);
        assert_eq!(skills[0].name, "coding");
        assert_eq!(skills[0].content, "You are a coding assistant.");
        assert_eq!(skills[1].name, "review");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_expand_tilde() {
        let expanded = expand_tilde("~/test");
        assert!(!expanded.starts_with("~/"));

        let no_tilde = expand_tilde("/absolute/path");
        assert_eq!(no_tilde, "/absolute/path");
    }
}
