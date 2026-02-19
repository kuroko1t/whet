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

## CORE RULES (MUST FOLLOW)

1. MATCH THE USER'S LANGUAGE. This is mandatory. If the user writes in Japanese, you MUST respond entirely in Japanese. If Chinese, respond in Chinese. Always mirror the language of the user's latest message. NEVER default to English unless the user writes in English.
2. ACT, DON'T ASK. Use tools immediately. Never ask \"which file?\", \"are you sure?\", or \"should I proceed?\". Just do it.
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
| Learn what the project does | read_file(\"README.md\") first, then repo_map | repo_map alone (shows only code symbols) |
| Run tests or build | shell | git |
| Check git status/diff | git | shell(git ...) |
| Small text replacement | edit_file | write_file (overwrites entire file) |
| Large rewrite or new file | write_file | edit_file (too many replacements) |
| Delete a file | shell(rm path) | (no delete tool exists) |
| Apply multiple changes to one file | apply_diff | multiple edit_file calls |

## HANDLING VAGUE REQUESTS

ALWAYS start by finding concrete errors — NEVER give generic advice without investigating first.
- \"Fix the bug\" → FIRST: shell(\"cargo test\") or shell(\"cargo build\") to find actual errors → read_file on failing code → edit_file to fix
- \"Improve this code\" → repo_map(\".\") to understand structure → read key files → make improvements
- \"Something is wrong\" → shell(\"cargo build\") or shell(\"cargo test\") to find errors → fix them
- \"Tell me about this project\" → read_file(\"README.md\") → repo_map(\".\") for structure → explain
Do NOT ask \"which bug?\" or \"what do you mean?\". Investigate on your own.
Do NOT give generic advice like \"you should check...\". Use tools to investigate and act.

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
- \"What does this project do?\" → read_file(\"README.md\") → repo_map(\".\") for structure → explain
- \"Add a new function\" → read_file to understand context → edit_file to add the function
- \"Commit the changes\" → git(\"status\") → git(\"add\", \".\") → git(\"commit\", \"-m message\")
- \"Delete the old config\" → shell(\"rm old_config.toml\")

## CRITICAL: LANGUAGE RULE
You MUST reply in the SAME language as the user's latest message. Do NOT reply in English if the user wrote in another language. This rule overrides all other formatting preferences.
例: ユーザーが日本語で質問 → 日本語で回答。英語で質問 → 英語で回答。".to_string();

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

    // --- System prompt content verification tests ---

    #[test]
    fn test_prompt_has_core_rules_section() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("## CORE RULES"),
            "Prompt must have CORE RULES section"
        );
    }

    #[test]
    fn test_prompt_language_rule_is_first_and_mandatory() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("1. MATCH THE USER'S LANGUAGE"),
            "Language rule must be rule #1"
        );
        assert!(
            prompt.contains("MUST"),
            "Language rule must use MUST for enforcement"
        );
        assert!(
            prompt.contains("NEVER default to English"),
            "Language rule must explicitly warn against defaulting to English"
        );
    }

    #[test]
    fn test_prompt_language_reminder_at_end() {
        let prompt = system_prompt(&[]);
        // The CRITICAL language reminder should be near the end of the base prompt
        assert!(
            prompt.contains("CRITICAL: LANGUAGE RULE"),
            "Prompt must have critical language reminder"
        );
        assert!(
            prompt.contains("日本語"),
            "Language reminder must include Japanese example"
        );
    }

    #[test]
    fn test_prompt_act_dont_ask_rule() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("ACT, DON'T ASK"),
            "Prompt must have ACT DON'T ASK rule"
        );
        assert!(
            prompt.contains("Never ask"),
            "Rule must explicitly forbid asking"
        );
    }

    #[test]
    fn test_prompt_read_before_editing_rule() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("READ BEFORE EDITING"),
            "Prompt must require reading before editing"
        );
        assert!(
            prompt.contains("read_file before using edit_file"),
            "Rule must specify read_file → edit_file order"
        );
    }

    #[test]
    fn test_prompt_chain_tools_rule() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("CHAIN TOOLS"),
            "Prompt must have CHAIN TOOLS rule"
        );
    }

    #[test]
    fn test_prompt_be_concise_rule() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("BE CONCISE"),
            "Prompt must have BE CONCISE rule"
        );
    }

    #[test]
    fn test_prompt_has_tool_selection_table() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("## TOOL SELECTION"),
            "Prompt must have TOOL SELECTION section"
        );
        // Verify key tool mappings exist
        assert!(
            prompt.contains("read_file"),
            "Tool selection must mention read_file"
        );
        assert!(
            prompt.contains("edit_file"),
            "Tool selection must mention edit_file"
        );
        assert!(
            prompt.contains("write_file"),
            "Tool selection must mention write_file"
        );
        assert!(prompt.contains("grep"), "Tool selection must mention grep");
        assert!(
            prompt.contains("repo_map"),
            "Tool selection must mention repo_map"
        );
        assert!(
            prompt.contains("apply_diff"),
            "Tool selection must mention apply_diff"
        );
        assert!(
            prompt.contains("shell"),
            "Tool selection must mention shell"
        );
        assert!(prompt.contains("git"), "Tool selection must mention git");
    }

    #[test]
    fn test_prompt_has_vague_request_handling() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("## HANDLING VAGUE REQUESTS"),
            "Prompt must have HANDLING VAGUE REQUESTS section"
        );
        assert!(
            prompt.contains("Do NOT ask"),
            "Must explicitly forbid asking for clarification on vague requests"
        );
    }

    #[test]
    fn test_prompt_has_error_recovery() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("## ERROR RECOVERY"),
            "Prompt must have ERROR RECOVERY section"
        );
        assert!(
            prompt.contains("read_file fails"),
            "Must cover read_file failure"
        );
        assert!(
            prompt.contains("edit_file fails"),
            "Must cover edit_file failure"
        );
        assert!(
            prompt.contains("shell command fails"),
            "Must cover shell failure"
        );
    }

    #[test]
    fn test_prompt_has_editing_rules() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("## EDITING RULES"),
            "Prompt must have EDITING RULES section"
        );
        assert!(
            prompt.contains("old_text must be copied EXACTLY"),
            "Must stress exact matching for edit_file"
        );
    }

    #[test]
    fn test_prompt_has_workflow_examples() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("## WORKFLOW EXAMPLES"),
            "Prompt must have WORKFLOW EXAMPLES section"
        );
    }

    #[test]
    fn test_prompt_all_sections_present() {
        let prompt = system_prompt(&[]);
        let required_sections = [
            "## CORE RULES",
            "## TOOL SELECTION",
            "## HANDLING VAGUE REQUESTS",
            "## ERROR RECOVERY",
            "## EDITING RULES",
            "## WORKFLOW EXAMPLES",
            "## CRITICAL: LANGUAGE RULE",
        ];
        for section in &required_sections {
            assert!(
                prompt.contains(section),
                "Missing required section: {}",
                section
            );
        }
    }

    #[test]
    fn test_prompt_no_old_project_names() {
        let prompt = system_prompt(&[]);
        assert!(
            !prompt.contains("hermit"),
            "Prompt must not contain old project name 'hermit'"
        );
        assert!(
            !prompt.contains("clawbot"),
            "Prompt must not contain old project name 'clawbot'"
        );
    }

    #[test]
    fn test_prompt_identifies_as_whet() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.starts_with("You are whet"),
            "Prompt must identify itself as whet"
        );
    }

    #[test]
    fn test_prompt_skills_appended_after_base() {
        let skills = vec![
            Skill {
                name: "skill_a".to_string(),
                content: "Do A.".to_string(),
            },
            Skill {
                name: "skill_b".to_string(),
                content: "Do B.".to_string(),
            },
        ];
        let prompt = system_prompt(&skills);
        // Skills section should come after the base prompt
        let base_end = prompt.find("## CRITICAL: LANGUAGE RULE").unwrap();
        let skills_start = prompt.find("## Skills").unwrap();
        assert!(
            skills_start > base_end,
            "Skills must be appended after the base prompt"
        );
        assert!(prompt.contains("### skill_a"));
        assert!(prompt.contains("### skill_b"));
        assert!(prompt.contains("Do A."));
        assert!(prompt.contains("Do B."));
    }

    #[test]
    fn test_prompt_workflow_read_file_before_repo_map() {
        let prompt = system_prompt(&[]);
        // The "What does this project do?" workflow should use read_file before repo_map
        let workflow_line = "\"What does this project do?\"";
        let workflow_pos = prompt
            .find(workflow_line)
            .expect("Workflow must include project overview example");
        // Extract the line containing this workflow
        let line_end = prompt[workflow_pos..]
            .find('\n')
            .unwrap_or(prompt.len() - workflow_pos);
        let line = &prompt[workflow_pos..workflow_pos + line_end];
        let read_pos = line
            .find("read_file")
            .expect("Workflow must mention read_file");
        let repo_map_pos = line
            .find("repo_map")
            .expect("Workflow must mention repo_map");
        assert!(
            read_pos < repo_map_pos,
            "read_file must come before repo_map in project overview workflow"
        );
    }

    #[test]
    fn test_prompt_vague_requests_mention_cargo_test() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("cargo test") || prompt.contains("cargo build"),
            "HANDLING VAGUE REQUESTS must mention cargo test or cargo build for investigation"
        );
    }

    #[test]
    fn test_prompt_no_generic_advice_rule() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("NEVER give generic advice"),
            "Prompt must explicitly forbid generic advice without investigation"
        );
    }

    #[test]
    fn test_prompt_readme_mentioned_in_tool_selection() {
        let prompt = system_prompt(&[]);
        assert!(
            prompt.contains("read_file(\"README.md\")"),
            "Tool selection table must mention reading README.md for project overview"
        );
    }

    #[test]
    fn test_prompt_no_skills_means_no_skills_section() {
        let prompt = system_prompt(&[]);
        assert!(
            !prompt.contains("## Skills"),
            "Empty skills should not add Skills section"
        );
    }

    #[test]
    fn test_prompt_tool_selection_correct_vs_wrong() {
        let prompt = system_prompt(&[]);
        // Verify the table explicitly warns against wrong tool usage
        assert!(
            prompt.contains("Wrong tool"),
            "Tool selection table must have Wrong tool column"
        );
        assert!(
            prompt.contains("shell(git ...)"),
            "Must warn against using shell for git commands"
        );
    }

    #[test]
    fn test_prompt_language_rule_appears_twice() {
        let prompt = system_prompt(&[]);
        // Language instruction should appear both in CORE RULES and at the end
        let first = prompt.find("MATCH THE USER'S LANGUAGE").unwrap();
        let last = prompt.find("CRITICAL: LANGUAGE RULE").unwrap();
        assert!(
            last > first,
            "Language rule must appear both at start and end of prompt"
        );
    }
}
