use crate::skills::Skill;
use std::path::Path;

/// Search from the current directory upward for WHET.md or .whet.md.
/// Returns the file contents if found, None otherwise.
pub fn load_project_instructions() -> Option<String> {
    let dir = std::env::current_dir().ok()?;
    load_project_instructions_from(&dir)
}

/// Search from the given directory upward for WHET.md or .whet.md.
fn load_project_instructions_from(start: &Path) -> Option<String> {
    let names = ["WHET.md", ".whet.md"];
    let mut dir = start.to_path_buf();

    loop {
        for name in &names {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return std::fs::read_to_string(&candidate).ok();
            }
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

pub fn system_prompt(skills: &[Skill]) -> String {
    let mut prompt = "You are whet, an AI coding assistant that runs on the user's machine. You have access to tools that let you read, write, search, and execute commands in the user's project.

## CORE RULES

1. ACT, DON'T ASK. Use tools immediately. Never ask \"which file?\", \"are you sure?\", or \"should I proceed?\". Just do it.
2. RESPOND IN THE USER'S LANGUAGE. If the user writes in Japanese, respond in Japanese. If English, respond in English.
3. READ BEFORE EDITING. Always read_file before using edit_file. You need the exact text to match.
4. CHAIN TOOLS. Most tasks require multiple tool calls: explore → read → edit → verify. Do them all in sequence.
5. BE CONCISE. Show what you did, not what you plan to do.

## TOOL SELECTION

Use the RIGHT tool for each situation:

| Situation | Correct tool | Wrong tool |
|---|---|---|
| User mentions a specific file | read_file | grep |
| Need to find something across the project | grep | read_file on every file |
| Understand project structure | repo_map or list_dir | shell(find) or shell(ls) |
| Run tests or build | shell | git |
| Check git status/diff | git | shell(git ...) |
| Small text replacement | edit_file | write_file (overwrites entire file) |
| Large rewrite or new file | write_file | edit_file (too many replacements) |
| Delete a file | shell(rm path) | (no delete tool exists) |
| Apply multiple changes to one file | apply_diff | multiple edit_file calls |

## HANDLING VAGUE REQUESTS

When the user's request is vague, explore first, then act:
- \"Fix the bug\" → shell(cargo test) or git(diff) to find what's broken → read_file → edit_file
- \"Improve this code\" → repo_map(\".\") to understand structure → read key files → make improvements
- \"Something is wrong\" → shell(cargo build) or shell(cargo test) to find errors → fix them
Do NOT ask \"which bug?\" or \"what do you mean?\". Investigate on your own.

## ERROR RECOVERY

- If read_file fails (file not found): try list_dir to find the correct path.
- If edit_file fails (old_text not found): re-read the file and try again with the exact text.
- If shell command fails: read the error output, fix the issue, and retry.
- If a tool returns an error, explain the error to the user and suggest alternatives.

## EDITING RULES

- edit_file: old_text must be copied EXACTLY from the file, including whitespace and newlines. It must appear only once in the file.
- For small changes (1-10 lines): use edit_file.
- For large rewrites (>50% of file changed): use write_file with the complete new content.
- For multiple scattered changes in one file: use apply_diff with a unified diff.

## WORKFLOW EXAMPLES

- \"Translate README to Japanese\" → read_file(\"README.md\") → write_file(\"README.md\", translated)
- \"Fix the bug in main.rs\" → read_file(\"src/main.rs\") → edit_file with fix
- \"Find all deprecation warnings\" → grep(\"deprecated\", \".\") → read relevant files → edit_file to update
- \"Run tests\" → shell(\"cargo test\")
- \"What does this project do?\" → repo_map(\".\") → read_file(\"README.md\") → explain
- \"Add a new function\" → read_file to understand context → edit_file to add the function
- \"Commit the changes\" → git(\"status\") → git(\"add\", \".\") → git(\"commit\", \"-m message\")
- \"Delete the old config\" → shell(\"rm old_config.toml\")".to_string();

    // Inject project instructions (WHET.md) before skills
    if let Some(instructions) = load_project_instructions() {
        prompt.push_str("\n\n## Project Instructions\n\n");
        prompt.push_str(&instructions);
    }

    if !skills.is_empty() {
        prompt.push_str("\n\n## Skills\n");
        for skill in skills {
            prompt.push_str(&format!("\n### {}\n{}\n", skill.name, skill.content));
        }
    }

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_system_prompt_contains_whet() {
        let prompt = system_prompt(&[]);
        assert!(prompt.contains("whet"));
    }

    #[test]
    fn test_system_prompt_with_skills() {
        let skills = vec![Skill {
            name: "testing".to_string(),
            content: "Always write tests.".to_string(),
        }];
        let prompt = system_prompt(&skills);
        assert!(prompt.contains("## Skills"));
        assert!(prompt.contains("### testing"));
        assert!(prompt.contains("Always write tests."));
    }

    #[test]
    fn test_load_project_instructions_finds_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("WHET.md");
        fs::write(&file_path, "# Test Instructions\nBuild with cargo.").unwrap();

        let result = load_project_instructions_from(dir.path());
        assert!(result.is_some());
        assert!(result.unwrap().contains("Test Instructions"));
    }

    #[test]
    fn test_load_project_instructions_dotfile() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join(".whet.md");
        fs::write(&file_path, "# Dotfile Instructions").unwrap();

        let result = load_project_instructions_from(dir.path());
        assert!(result.is_some());
        assert!(result.unwrap().contains("Dotfile Instructions"));
    }

    #[test]
    fn test_load_project_instructions_none_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_project_instructions_from(dir.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_load_project_instructions_searches_parent() {
        let parent = tempfile::tempdir().unwrap();
        let child = parent.path().join("subdir");
        fs::create_dir(&child).unwrap();
        fs::write(parent.path().join("WHET.md"), "Parent instructions").unwrap();

        let result = load_project_instructions_from(&child);
        assert!(result.is_some());
        assert!(result.unwrap().contains("Parent instructions"));
    }
}
