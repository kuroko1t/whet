pub mod prompt;

use crate::config::{PermissionMode, ToolRiskLevel};
use crate::llm::{LlmProvider, Message};
use crate::skills::Skill;
use crate::tools::ToolRegistry;
use colored::Colorize;

const MAX_TOOL_OUTPUT_CHARS: usize = 50_000;
const MAX_CONTEXT_MESSAGES: usize = 40;
const SUMMARIZE_KEEP_RECENT: usize = 10;

pub struct Agent {
    pub llm: Box<dyn LlmProvider>,
    pub tools: ToolRegistry,
    pub memory: Vec<Message>,
    pub config: AgentConfig,
}

pub struct AgentConfig {
    #[allow(dead_code)]
    pub model: String,
    pub max_iterations: usize,
    pub permission_mode: PermissionMode,
    pub plan_mode: bool,
    pub context_compression: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "qwen2.5:7b".to_string(),
            max_iterations: 10,
            permission_mode: PermissionMode::Default,
            plan_mode: false,
            context_compression: true,
        }
    }
}

impl Agent {
    pub fn new(
        llm: Box<dyn LlmProvider>,
        tools: ToolRegistry,
        config: AgentConfig,
        skills: &[Skill],
    ) -> Self {
        let memory = vec![Message::system(&prompt::system_prompt(skills))];
        Self {
            llm,
            tools,
            memory,
            config,
        }
    }

    #[allow(dead_code)]
    pub fn process_message(&mut self, user_input: &str) -> String {
        self.process_message_with_callbacks(user_input, &mut |_| {}, &mut |_, _| true)
    }

    #[allow(dead_code)]
    pub fn process_message_with_callback(
        &mut self,
        user_input: &str,
        on_token: &mut dyn FnMut(&str),
    ) -> String {
        self.process_message_with_callbacks(user_input, on_token, &mut |_, _| true)
    }

    /// Compress context by summarizing old messages when the memory exceeds the threshold.
    fn compress_context(&mut self) {
        if self.memory.len() <= MAX_CONTEXT_MESSAGES {
            return;
        }
        self.compress_context_with_instruction(None);
    }

    /// Manually compress the conversation context.
    /// If `instruction` is provided, it is appended to the summarization prompt.
    pub fn compact(&mut self, instruction: Option<&str>) {
        if self.memory.len() <= 2 {
            eprintln!("  Nothing to compress.");
            return;
        }
        self.compress_context_with_instruction(instruction);
    }

    fn compress_context_with_instruction(&mut self, instruction: Option<&str>) {
        // Split: [system_prompt] + [old_messages] + [recent_messages]
        // Keep system prompt (index 0) and the last SUMMARIZE_KEEP_RECENT messages
        let keep_recent = SUMMARIZE_KEEP_RECENT.min(self.memory.len().saturating_sub(1));
        let keep_from = self.memory.len() - keep_recent;

        if keep_from <= 1 {
            return;
        }

        // Build summarization request by draining old messages (avoids cloning)
        // First, split off the recent messages
        let recent_messages = self.memory.split_off(keep_from);
        // Now self.memory = [system_prompt, old_messages...]
        // Drain old messages, keeping system_prompt at index 0
        let system_prompt = self.memory[0].clone();
        // Use the remaining messages (including system prompt) as summarization input
        let summarize_prompt = match instruction {
            Some(inst) => format!(
                "Summarize the conversation so far concisely, preserving key facts, decisions, and context needed for future turns. Additional instructions: {}",
                inst
            ),
            None => "Summarize the conversation so far concisely, preserving key facts, decisions, and context needed for future turns:".to_string(),
        };
        self.memory.push(Message::user(&summarize_prompt));

        // Call LLM without tools for summarization
        let summary = match self.llm.chat(&self.memory, &[]) {
            Ok(resp) => resp.content.unwrap_or_default(),
            Err(_) => {
                // Restore memory on error
                self.memory.pop(); // remove summarize request
                self.memory.extend(recent_messages);
                return;
            }
        };

        if summary.is_empty() {
            self.memory.pop();
            self.memory.extend(recent_messages);
            return;
        }

        // Rebuild memory: [system_prompt, summary, recent_messages...]
        self.memory.clear();
        self.memory.push(system_prompt);
        self.memory.push(Message::system(&format!(
            "Previous conversation summary: {}",
            summary
        )));
        self.memory.extend(recent_messages);

        eprintln!(
            "  {}",
            "[context compressed: conversation summarized]".dimmed()
        );
    }

    /// Process a message with streaming callback and approval callback.
    /// `on_approve` is called before executing a tool that requires approval.
    /// It receives (tool_name, &arguments) and returns true to allow, false to deny.
    pub fn process_message_with_callbacks(
        &mut self,
        user_input: &str,
        on_token: &mut dyn FnMut(&str),
        on_approve: &mut dyn FnMut(&str, &serde_json::Value) -> bool,
    ) -> String {
        self.memory.push(Message::user(user_input));

        // Compress context if enabled and threshold exceeded
        if self.config.context_compression {
            self.compress_context();
        }

        let tool_defs = if self.config.plan_mode {
            self.tools.safe_definitions()
        } else {
            self.tools.definitions()
        };

        for _iteration in 0..self.config.max_iterations {
            let response = match self.llm.chat_streaming(&self.memory, tool_defs, on_token) {
                Ok(resp) => resp,
                Err(e) => return format!("Error: {}", e),
            };

            // If no tool calls, return the content
            if response.tool_calls.is_empty() {
                let content = response.content.unwrap_or_default();
                self.memory.push(Message::assistant(&content));
                return content;
            }

            // Separate content from tool calls to avoid borrow issues
            let response_content = response.content;
            let tool_calls = response.tool_calls;

            // Store tool calls in memory — move instead of clone
            self.memory
                .push(Message::assistant_with_tool_calls(tool_calls.clone()));

            for tool_call in &tool_calls {
                eprintln!(
                    "  {} {}",
                    format!("[tool: {}]", tool_call.name).cyan(),
                    tool_call.arguments.to_string().dimmed()
                );

                let result = if let Some(tool) = self.tools.get(&tool_call.name) {
                    // Determine effective risk level (dynamic for git)
                    let effective_risk = if tool_call.name == "git" {
                        let git_cmd = tool_call.arguments["command"].as_str().unwrap_or("");
                        crate::tools::git::git_command_risk_level(git_cmd)
                    } else {
                        tool.risk_level()
                    };

                    // In plan mode, block non-safe tools
                    if self.config.plan_mode && effective_risk != ToolRiskLevel::Safe {
                        "Tool blocked: plan mode is active (read-only). Use /plan to toggle."
                            .to_string()
                    } else if self.needs_approval(effective_risk) {
                        if !on_approve(&tool_call.name, &tool_call.arguments) {
                            "Tool execution denied by user.".to_string()
                        } else {
                            match tool.execute(tool_call.arguments.clone()) {
                                Ok(output) => output,
                                Err(e) => format!("Tool error: {}", e),
                            }
                        }
                    } else {
                        match tool.execute(tool_call.arguments.clone()) {
                            Ok(output) => output,
                            Err(e) => format!("Tool error: {}", e),
                        }
                    }
                } else {
                    format!("Unknown tool: {}", tool_call.name)
                };

                let result = if result.len() > MAX_TOOL_OUTPUT_CHARS {
                    let mut end = MAX_TOOL_OUTPUT_CHARS;
                    while !result.is_char_boundary(end) {
                        end -= 1;
                    }
                    let mut truncated = String::with_capacity(end + 40);
                    truncated.push_str(&result[..end]);
                    truncated.push_str("\n...[output truncated to 50KB]");
                    truncated
                } else {
                    result
                };

                self.memory
                    .push(Message::tool_result(&tool_call.id, &result));
            }

            // Content alongside tool calls is already streamed to the user.
            // Don't add it as a separate memory message — the model would repeat itself.
            let _ = response_content;
        }

        "Max iterations reached. The agent could not complete the task.".to_string()
    }

    /// Determine if a tool at the given risk level needs user approval.
    fn needs_approval(&self, risk_level: ToolRiskLevel) -> bool {
        match self.config.permission_mode {
            PermissionMode::Yolo => false,
            PermissionMode::AcceptEdits => risk_level == ToolRiskLevel::Dangerous,
            PermissionMode::Default => {
                risk_level == ToolRiskLevel::Moderate || risk_level == ToolRiskLevel::Dangerous
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmError, LlmResponse, Role, ToolCall, ToolDefinition};
    use crate::tools::default_registry;
    use std::cell::RefCell;

    /// A mock LLM that returns pre-scripted responses in sequence.
    struct MockLlm {
        responses: RefCell<Vec<LlmResponse>>,
    }

    impl MockLlm {
        fn new(responses: Vec<LlmResponse>) -> Self {
            // Reverse so we can pop from the end
            let mut r = responses;
            r.reverse();
            Self {
                responses: RefCell::new(r),
            }
        }
    }

    impl LlmProvider for MockLlm {
        fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
        ) -> Result<LlmResponse, LlmError> {
            let mut responses = self.responses.borrow_mut();
            if let Some(resp) = responses.pop() {
                Ok(resp)
            } else {
                // If we run out of scripted responses, return empty content
                Ok(LlmResponse {
                    content: Some("(no more scripted responses)".to_string()),
                    tool_calls: vec![],
                })
            }
        }
    }

    /// A mock LLM that always returns an error.
    struct ErrorLlm;

    impl LlmProvider for ErrorLlm {
        fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
        ) -> Result<LlmResponse, LlmError> {
            Err(LlmError::ConnectionError(
                "Cannot connect to Ollama".to_string(),
            ))
        }
    }

    fn make_agent(llm: Box<dyn LlmProvider>) -> Agent {
        Agent::new(llm, default_registry(), AgentConfig::default(), &[])
    }

    #[test]
    fn test_simple_text_response() {
        let llm = MockLlm::new(vec![LlmResponse {
            content: Some("Hello! How can I help?".to_string()),
            tool_calls: vec![],
        }]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Hi there");
        assert_eq!(response, "Hello! How can I help?");
    }

    #[test]
    fn test_empty_content_response() {
        let llm = MockLlm::new(vec![LlmResponse {
            content: None,
            tool_calls: vec![],
        }]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Hi");
        // None content should become empty string
        assert_eq!(response, "");
    }

    #[test]
    fn test_tool_call_then_response() {
        // First response: call read_file tool
        // Second response: use tool result to answer
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "Cargo.toml"}),
                }],
            },
            LlmResponse {
                content: Some("The project is named whet.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("What is this project?");
        assert_eq!(response, "The project is named whet.");
    }

    #[test]
    fn test_multiple_tool_calls_in_one_response() {
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![
                    ToolCall {
                        id: "call_0".to_string(),
                        name: "read_file".to_string(),
                        arguments: serde_json::json!({"path": "Cargo.toml"}),
                    },
                    ToolCall {
                        id: "call_1".to_string(),
                        name: "list_dir".to_string(),
                        arguments: serde_json::json!({"path": "src"}),
                    },
                ],
            },
            LlmResponse {
                content: Some("I found 2 things.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Analyze the project");
        assert_eq!(response, "I found 2 things.");

        // Verify memory contains system + user + assistant(tool_calls) + 2 tool results + assistant
        // system(1) + user(1) + assistant_tool_calls(1) + tool_result(2) + assistant(1) = 6
        assert_eq!(agent.memory.len(), 6);
    }

    #[test]
    fn test_unknown_tool_handled_gracefully() {
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "nonexistent_tool".to_string(),
                    arguments: serde_json::json!({}),
                }],
            },
            LlmResponse {
                content: Some("Sorry, that tool doesn't exist.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Use nonexistent tool");
        assert_eq!(response, "Sorry, that tool doesn't exist.");

        // Check that "Unknown tool" message was stored in memory
        let tool_result_msg = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have a tool result message");
        assert!(tool_result_msg
            .content
            .contains("Unknown tool: nonexistent_tool"));
    }

    #[test]
    fn test_tool_execution_error_handled() {
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "/nonexistent/file.txt"}),
                }],
            },
            LlmResponse {
                content: Some("The file doesn't exist.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Read a missing file");
        assert_eq!(response, "The file doesn't exist.");

        // Check that error was stored as tool result
        let tool_result_msg = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have a tool result message");
        assert!(tool_result_msg.content.contains("Tool error:"));
    }

    #[test]
    fn test_llm_error_returns_error_message() {
        let mut agent = make_agent(Box::new(ErrorLlm));
        let response = agent.process_message("Hello");
        assert!(response.starts_with("Error:"));
        assert!(response.contains("Cannot connect to Ollama"));
    }

    #[test]
    fn test_max_iterations_reached() {
        // LLM always returns tool calls, never a final response
        let mut responses = Vec::new();
        for _ in 0..15 {
            responses.push(LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                }],
            });
        }
        let llm = MockLlm::new(responses);

        let config = AgentConfig {
            max_iterations: 3,
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(Box::new(llm), default_registry(), config, &[]);
        let response = agent.process_message("Keep using tools forever");
        assert_eq!(
            response,
            "Max iterations reached. The agent could not complete the task."
        );
    }

    #[test]
    fn test_memory_accumulates_across_messages() {
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: Some("First response".to_string()),
                tool_calls: vec![],
            },
            LlmResponse {
                content: Some("Second response".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = make_agent(Box::new(llm));

        agent.process_message("First question");
        // system + user + assistant = 3
        assert_eq!(agent.memory.len(), 3);

        agent.process_message("Second question");
        // 3 + user + assistant = 5
        assert_eq!(agent.memory.len(), 5);
    }

    #[test]
    fn test_system_prompt_is_first_message() {
        let llm = MockLlm::new(vec![LlmResponse {
            content: Some("ok".to_string()),
            tool_calls: vec![],
        }]);
        let agent = make_agent(Box::new(llm));

        assert_eq!(agent.memory.len(), 1);
        assert_eq!(agent.memory[0].role, Role::System);
        assert!(agent.memory[0].content.contains("whet"));
    }

    #[test]
    fn test_tool_call_with_content_alongside() {
        // LLM returns both tool_calls AND content in the same response.
        // Content alongside tool calls is NOT stored in memory to prevent
        // the model from repeating itself in the next iteration.
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: Some("Let me check that file.".to_string()),
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "Cargo.toml"}),
                }],
            },
            LlmResponse {
                content: Some("Done reading.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("Check Cargo.toml");
        assert_eq!(response, "Done reading.");

        // Memory: system + user + assistant_tool_calls + tool_result + assistant("Done reading.")
        // Content alongside tool calls ("Let me check") is intentionally dropped from memory.
        let assistant_messages: Vec<_> = agent
            .memory
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .collect();
        // Only 2 assistant messages: one with tool_calls (empty content), one final response
        assert_eq!(assistant_messages.len(), 2);
        assert_eq!(assistant_messages[1].content, "Done reading.");
    }

    // --- Streaming callback tests ---

    /// A mock LLM that implements chat_streaming with token-by-token callback
    struct StreamingMockLlm {
        responses: RefCell<Vec<(Vec<String>, LlmResponse)>>,
    }

    impl StreamingMockLlm {
        /// Each entry: (tokens_to_emit, final_response)
        fn new(responses: Vec<(Vec<String>, LlmResponse)>) -> Self {
            let mut r = responses;
            r.reverse();
            Self {
                responses: RefCell::new(r),
            }
        }
    }

    impl LlmProvider for StreamingMockLlm {
        fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
        ) -> Result<LlmResponse, LlmError> {
            let mut responses = self.responses.borrow_mut();
            if let Some((_, resp)) = responses.pop() {
                Ok(resp)
            } else {
                Ok(LlmResponse {
                    content: Some("(no more responses)".to_string()),
                    tool_calls: vec![],
                })
            }
        }

        fn chat_streaming(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            on_token: &mut dyn FnMut(&str),
        ) -> Result<LlmResponse, LlmError> {
            let mut responses = self.responses.borrow_mut();
            if let Some((tokens, resp)) = responses.pop() {
                for token in &tokens {
                    on_token(token);
                }
                Ok(resp)
            } else {
                Ok(LlmResponse {
                    content: Some("(no more responses)".to_string()),
                    tool_calls: vec![],
                })
            }
        }
    }

    #[test]
    fn test_process_message_with_callback_receives_tokens() {
        let llm = StreamingMockLlm::new(vec![(
            vec!["Hello".to_string(), " world".to_string(), "!".to_string()],
            LlmResponse {
                content: Some("Hello world!".to_string()),
                tool_calls: vec![],
            },
        )]);

        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig::default(),
            &[],
        );

        let mut received_tokens = Vec::new();
        let response = agent.process_message_with_callback("Hi", &mut |token| {
            received_tokens.push(token.to_string());
        });

        assert_eq!(response, "Hello world!");
        assert_eq!(received_tokens, vec!["Hello", " world", "!"]);
    }

    #[test]
    fn test_process_message_with_callback_empty_callback() {
        // process_message uses an empty callback internally
        let llm = StreamingMockLlm::new(vec![(
            vec!["token1".to_string()],
            LlmResponse {
                content: Some("token1".to_string()),
                tool_calls: vec![],
            },
        )]);

        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig::default(),
            &[],
        );
        let response = agent.process_message("Hi");
        assert_eq!(response, "token1");
    }

    #[test]
    fn test_process_message_with_callback_tool_calls() {
        // When tool calls happen, streaming callback should still work
        let llm = StreamingMockLlm::new(vec![
            (
                vec![], // No tokens during tool call response
                LlmResponse {
                    content: None,
                    tool_calls: vec![ToolCall {
                        id: "call_0".to_string(),
                        name: "read_file".to_string(),
                        arguments: serde_json::json!({"path": "Cargo.toml"}),
                    }],
                },
            ),
            (
                vec!["The ".to_string(), "project".to_string(), ".".to_string()],
                LlmResponse {
                    content: Some("The project.".to_string()),
                    tool_calls: vec![],
                },
            ),
        ]);

        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig::default(),
            &[],
        );

        let mut received_tokens = Vec::new();
        let response = agent.process_message_with_callback("What project?", &mut |token| {
            received_tokens.push(token.to_string());
        });

        assert_eq!(response, "The project.");
        // Tokens should come from the second call (after tool execution)
        assert_eq!(received_tokens, vec!["The ", "project", "."]);
    }

    #[test]
    fn test_process_message_with_callback_error() {
        let mut agent = make_agent(Box::new(ErrorLlm));

        let mut received_tokens = Vec::new();
        let response = agent.process_message_with_callback("Hello", &mut |token| {
            received_tokens.push(token.to_string());
        });

        assert!(response.starts_with("Error:"));
        // No tokens should be emitted on error
        assert!(received_tokens.is_empty());
    }

    #[test]
    fn test_process_message_with_callback_multiple_turns() {
        let llm = StreamingMockLlm::new(vec![
            (
                vec!["First".to_string()],
                LlmResponse {
                    content: Some("First".to_string()),
                    tool_calls: vec![],
                },
            ),
            (
                vec!["Second".to_string()],
                LlmResponse {
                    content: Some("Second".to_string()),
                    tool_calls: vec![],
                },
            ),
        ]);

        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig::default(),
            &[],
        );

        let mut tokens1 = Vec::new();
        let r1 = agent.process_message_with_callback("Q1", &mut |t| tokens1.push(t.to_string()));
        assert_eq!(r1, "First");
        assert_eq!(tokens1, vec!["First"]);

        let mut tokens2 = Vec::new();
        let r2 = agent.process_message_with_callback("Q2", &mut |t| tokens2.push(t.to_string()));
        assert_eq!(r2, "Second");
        assert_eq!(tokens2, vec!["Second"]);

        // Memory should have system + user1 + assistant1 + user2 + assistant2 = 5
        assert_eq!(agent.memory.len(), 5);
    }

    #[test]
    fn test_process_message_delegates_to_callback_version() {
        // process_message should produce the same result as process_message_with_callback
        let llm1 = MockLlm::new(vec![LlmResponse {
            content: Some("Same result".to_string()),
            tool_calls: vec![],
        }]);
        let llm2 = MockLlm::new(vec![LlmResponse {
            content: Some("Same result".to_string()),
            tool_calls: vec![],
        }]);

        let mut agent1 = make_agent(Box::new(llm1));
        let mut agent2 = make_agent(Box::new(llm2));

        let r1 = agent1.process_message("test");
        let r2 = agent2.process_message_with_callback("test", &mut |_| {});

        assert_eq!(r1, r2);
    }

    // --- Permission system tests ---

    fn make_agent_with_mode(llm: Box<dyn LlmProvider>, mode: PermissionMode) -> Agent {
        Agent::new(
            llm,
            default_registry(),
            AgentConfig {
                permission_mode: mode,
                ..AgentConfig::default()
            },
            &[],
        )
    }

    #[test]
    fn test_needs_approval_default_mode() {
        let llm = MockLlm::new(vec![]);
        let agent = make_agent_with_mode(Box::new(llm), PermissionMode::Default);
        assert!(agent.needs_approval(ToolRiskLevel::Moderate));
        assert!(agent.needs_approval(ToolRiskLevel::Dangerous));
        assert!(!agent.needs_approval(ToolRiskLevel::Safe));
    }

    #[test]
    fn test_needs_approval_accept_edits_mode() {
        let llm = MockLlm::new(vec![]);
        let agent = make_agent_with_mode(Box::new(llm), PermissionMode::AcceptEdits);
        assert!(!agent.needs_approval(ToolRiskLevel::Moderate));
        assert!(agent.needs_approval(ToolRiskLevel::Dangerous));
        assert!(!agent.needs_approval(ToolRiskLevel::Safe));
    }

    #[test]
    fn test_needs_approval_yolo_mode() {
        let llm = MockLlm::new(vec![]);
        let agent = make_agent_with_mode(Box::new(llm), PermissionMode::Yolo);
        assert!(!agent.needs_approval(ToolRiskLevel::Moderate));
        assert!(!agent.needs_approval(ToolRiskLevel::Dangerous));
        assert!(!agent.needs_approval(ToolRiskLevel::Safe));
    }

    #[test]
    fn test_tool_denied_by_approval_callback() {
        // shell tool is Dangerous → needs approval in Default mode
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "shell".to_string(),
                    arguments: serde_json::json!({"command": "echo hello"}),
                }],
            },
            LlmResponse {
                content: Some("I couldn't execute the command.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = make_agent_with_mode(Box::new(llm), PermissionMode::Default);

        let response = agent.process_message_with_callbacks(
            "Run echo",
            &mut |_| {},
            &mut |_, _| false, // Always deny
        );

        assert_eq!(response, "I couldn't execute the command.");
        // Check that the tool result contains the denial message
        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(tool_result.content.contains("denied by user"));
    }

    #[test]
    fn test_tool_approved_by_callback() {
        // shell tool is approved → executes normally
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "shell".to_string(),
                    arguments: serde_json::json!({"command": "echo approved"}),
                }],
            },
            LlmResponse {
                content: Some("Command executed.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = make_agent_with_mode(Box::new(llm), PermissionMode::Default);

        let response = agent.process_message_with_callbacks(
            "Run echo",
            &mut |_| {},
            &mut |_, _| true, // Always approve
        );

        assert_eq!(response, "Command executed.");
        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(tool_result.content.contains("approved"));
    }

    #[test]
    fn test_yolo_mode_skips_approval() {
        // In yolo mode, shell tool should execute without approval callback being called
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "shell".to_string(),
                    arguments: serde_json::json!({"command": "echo yolo"}),
                }],
            },
            LlmResponse {
                content: Some("Done.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = make_agent_with_mode(Box::new(llm), PermissionMode::Yolo);

        let mut approval_called = false;
        let response =
            agent.process_message_with_callbacks("Run echo", &mut |_| {}, &mut |_, _| {
                approval_called = true;
                false // Would deny, but should never be called
            });

        assert_eq!(response, "Done.");
        assert!(
            !approval_called,
            "Approval should not be called in yolo mode"
        );
    }

    #[test]
    fn test_safe_tool_never_needs_approval() {
        // read_file is Safe → no approval needed even in Default mode
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "Cargo.toml"}),
                }],
            },
            LlmResponse {
                content: Some("Read the file.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = make_agent_with_mode(Box::new(llm), PermissionMode::Default);

        let mut approval_called = false;
        let response =
            agent.process_message_with_callbacks("Read Cargo.toml", &mut |_| {}, &mut |_, _| {
                approval_called = true;
                false
            });

        assert_eq!(response, "Read the file.");
        assert!(
            !approval_called,
            "Approval should not be called for safe tools"
        );
    }

    #[test]
    fn test_accept_edits_allows_write_file() {
        // In AcceptEdits mode, write_file (Moderate) should not need approval
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "write_file".to_string(),
                    arguments: serde_json::json!({
                        "path": "/tmp/whet_perm_test.txt",
                        "content": "test"
                    }),
                }],
            },
            LlmResponse {
                content: Some("File written.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = make_agent_with_mode(Box::new(llm), PermissionMode::AcceptEdits);

        let mut approval_called = false;
        let response =
            agent.process_message_with_callbacks("Write file", &mut |_| {}, &mut |_, _| {
                approval_called = true;
                false
            });

        assert_eq!(response, "File written.");
        assert!(
            !approval_called,
            "write_file should not need approval in AcceptEdits mode"
        );
        // Cleanup
        std::fs::remove_file("/tmp/whet_perm_test.txt").ok();
    }

    // --- Plan mode tests ---

    #[test]
    fn test_plan_mode_blocks_dangerous_tools() {
        // shell is Dangerous → should be blocked in plan mode
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "shell".to_string(),
                    arguments: serde_json::json!({"command": "rm -rf /"}),
                }],
            },
            LlmResponse {
                content: Some("Blocked.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig {
                plan_mode: true,
                ..AgentConfig::default()
            },
            &[],
        );

        let response = agent.process_message("Delete everything");
        assert_eq!(response, "Blocked.");

        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(tool_result.content.contains("plan mode"));
    }

    #[test]
    fn test_plan_mode_blocks_moderate_tools() {
        // write_file is Moderate → should be blocked in plan mode
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "write_file".to_string(),
                    arguments: serde_json::json!({"path": "/tmp/test.txt", "content": "x"}),
                }],
            },
            LlmResponse {
                content: Some("Can't write.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig {
                plan_mode: true,
                ..AgentConfig::default()
            },
            &[],
        );

        let response = agent.process_message("Write file");
        assert_eq!(response, "Can't write.");

        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        assert!(tool_result.content.contains("plan mode"));
    }

    #[test]
    fn test_plan_mode_allows_safe_tools() {
        // read_file is Safe → should work in plan mode
        let llm = MockLlm::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "Cargo.toml"}),
                }],
            },
            LlmResponse {
                content: Some("Read successfully.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = Agent::new(
            Box::new(llm),
            default_registry(),
            AgentConfig {
                plan_mode: true,
                ..AgentConfig::default()
            },
            &[],
        );

        let response = agent.process_message("Read Cargo.toml");
        assert_eq!(response, "Read successfully.");

        let tool_result = agent
            .memory
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have tool result");
        // Should contain actual file content, not a "blocked" message
        assert!(!tool_result.content.contains("plan mode"));
        assert!(tool_result.content.contains("whet"));
    }
}
