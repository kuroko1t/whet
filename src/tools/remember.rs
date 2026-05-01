//! `remember` tool — model-callable handle into the persistent memory
//! store.
//!
//! The tool's `execute()` is a stub: the agent loop intercepts
//! `tool_call.name == "remember"` and routes it to a callback that
//! the application installed on the `Agent` (see
//! `Agent::set_on_remember`). Stub returns a diagnostic error if it
//! ever gets called via the generic tool dispatch path so a future
//! refactor can't silently regress the route.
//!
//! Design rationale:
//! - The `Tool::execute(args)` signature can't see Agent state, so a
//!   pure trait impl can't read or write the memory store. Same
//!   pattern as `subagent`.
//! - Memories are project-scoped by default (the `working_dir` is
//!   captured by the callback closure at construction). Globals are
//!   reachable via slash commands but not directly through this tool.

use super::{Tool, ToolError};
use crate::config::ToolRiskLevel;
use serde_json::json;

pub struct RememberTool;

impl Tool for RememberTool {
    fn name(&self) -> &str {
        "remember"
    }

    fn description(&self) -> &str {
        "Save a durable fact about this project so future sessions \
         start with it. Use ONLY for things that will still be true \
         next time (project conventions, env quirks, user preferences) \
         — e.g. \"uses pnpm not npm\", \"tests run via cargo nextest\", \
         \"don't reformat foo.rs\". Do NOT use for transient state, \
         conversation summary, or anything tied to the current task."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact to remember. One short imperative sentence."
                }
            },
            "required": ["content"]
        })
    }

    fn execute(&self, _args: serde_json::Value) -> Result<String, ToolError> {
        Err(ToolError::ExecutionFailed(
            "remember tool reached the generic dispatch path; \
             this should be intercepted by the agent loop. \
             Please report this as a bug."
                .to_string(),
        ))
    }

    fn risk_level(&self) -> ToolRiskLevel {
        // Writes only to the local SQLite memory file. No destructive
        // side effects on the user's filesystem or shell — auto-allow.
        ToolRiskLevel::Safe
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_remember() {
        assert_eq!(RememberTool.name(), "remember");
    }

    #[test]
    fn schema_requires_content() {
        let schema = RememberTool.parameters_schema();
        let required = schema["required"].as_array().expect("required is array");
        let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(names, vec!["content"]);
    }

    #[test]
    fn execute_returns_diagnostic_error() {
        let r = RememberTool.execute(serde_json::json!({"content": "x"}));
        assert!(r.is_err());
        let msg = r.unwrap_err().to_string();
        assert!(msg.contains("remember"));
        assert!(msg.contains("intercepted"));
    }

    #[test]
    fn description_warns_against_transient_use() {
        let desc = RememberTool.description();
        assert!(desc.contains("Do NOT"));
        assert!(desc.contains("transient") || desc.contains("conversation"));
    }
}
