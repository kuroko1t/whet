//! Subagent tool — exposes `Agent::run_subagent` to the LLM.
//!
//! `execute()` is **never called** under normal operation: the agent
//! loop intercepts `tool_call.name == "subagent"` and routes to
//! `Agent::run_subagent` directly, because the child loop needs access
//! to the parent's LLM client + tools + memory snapshot — none of
//! which are available through the `Tool::execute(args)` signature.
//!
//! The stub exists so the model SEES `subagent` in the tool list with
//! a proper schema and description. If the loop's special-case ever
//! fails to fire (e.g. a future refactor), `execute()` returns a
//! diagnostic error rather than silently doing nothing.

use super::{Tool, ToolError};
use crate::config::ToolRiskLevel;
use serde_json::json;

pub struct SubagentTool;

impl Tool for SubagentTool {
    fn name(&self) -> &str {
        "subagent"
    }

    fn description(&self) -> &str {
        "Spawn a focused subagent to handle an isolated subtask. The subagent runs \
         with a fresh conversation context and read-tracking but shares your tools \
         and model. Use this when:\n\
         \n\
         - You need to investigate or read many files whose contents would clutter \
           your context (e.g. \"find all callers of X across the project\").\n\
         - The subtask is self-contained and you only need its conclusion, not its \
           intermediate steps (e.g. \"run cargo test and summarise failures\").\n\
         - The work is fundamentally independent from the main task (e.g. \"draft \
           module A's tests while I work on module B\").\n\
         \n\
         Do NOT use this for trivial tasks the main loop can do in one or two tool \
         calls — the cost of spinning up a subagent is not worth it. Subagents \
         cannot spawn further subagents.\n\
         \n\
         The `task` argument is what the subagent is asked to do. Optional `context` \
         passes any pre-known state (file paths it should look at, constraints, etc.) \
         so the subagent doesn't have to rediscover. The subagent returns its final \
         summary as a single string."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The subtask description, written as a clear instruction \
                                    to the subagent. Be specific about what success looks like."
                },
                "context": {
                    "type": "string",
                    "description": "Optional pre-known context (file paths, constraints, prior \
                                    findings) so the subagent doesn't repeat your discovery work."
                }
            },
            "required": ["task"]
        })
    }

    /// Stub — never called. The agent loop special-cases this tool name
    /// and dispatches to `Agent::run_subagent` directly. If we ever
    /// observe this branch firing, it means a refactor broke the
    /// interception path.
    fn execute(&self, _args: serde_json::Value) -> Result<String, ToolError> {
        Err(ToolError::ExecutionFailed(
            "subagent tool reached the generic dispatch path; \
             this should be intercepted by the agent loop. \
             Please report this as a bug."
                .to_string(),
        ))
    }

    fn risk_level(&self) -> ToolRiskLevel {
        // The subagent inherits parent's tools, so any risk it carries
        // bubbles up through the child's own approval prompts. Marking
        // `subagent` itself as Safe avoids double-prompting on every
        // delegation.
        ToolRiskLevel::Safe
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_subagent() {
        assert_eq!(SubagentTool.name(), "subagent");
    }

    #[test]
    fn schema_requires_task() {
        let schema = SubagentTool.parameters_schema();
        let required = schema["required"].as_array().expect("required is array");
        let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"task"));
    }

    #[test]
    fn execute_returns_diagnostic_error() {
        // The stub must surface a clear error if the loop ever fails to
        // intercept the call — silent no-op would mask a regression.
        let r = SubagentTool.execute(serde_json::json!({"task": "x"}));
        assert!(r.is_err());
        let msg = r.unwrap_err().to_string();
        assert!(msg.contains("subagent"));
        assert!(msg.contains("intercepted"));
    }

    #[test]
    fn description_mentions_when_to_use() {
        // Guidance text is the whole point of registering this tool.
        let desc = SubagentTool.description();
        assert!(desc.contains("isolated"));
        assert!(desc.contains("Do NOT use this for trivial"));
    }
}
