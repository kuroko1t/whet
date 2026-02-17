pub mod prompt;

use crate::llm::{LlmProvider, Message};
use crate::tools::ToolRegistry;
use colored::Colorize;

pub struct Agent {
    pub llm: Box<dyn LlmProvider>,
    pub tools: ToolRegistry,
    pub memory: Vec<Message>,
    pub config: AgentConfig,
}

pub struct AgentConfig {
    pub model: String,
    pub max_iterations: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "qwen2.5:7b".to_string(),
            max_iterations: 10,
        }
    }
}

impl Agent {
    pub fn new(
        llm: Box<dyn LlmProvider>,
        tools: ToolRegistry,
        config: AgentConfig,
    ) -> Self {
        let mut memory = Vec::new();
        memory.push(Message::system(&prompt::system_prompt()));
        Self {
            llm,
            tools,
            memory,
            config,
        }
    }

    pub fn process_message(&mut self, user_input: &str) -> String {
        self.process_message_with_callback(user_input, &mut |_| {})
    }

    pub fn process_message_with_callback(
        &mut self,
        user_input: &str,
        on_token: &mut dyn FnMut(&str),
    ) -> String {
        self.memory.push(Message::user(user_input));

        let tool_defs = self.tools.definitions();

        for _iteration in 0..self.config.max_iterations {
            let response = match self.llm.chat_streaming(&self.memory, &tool_defs, on_token) {
                Ok(resp) => resp,
                Err(e) => return format!("Error: {}", e),
            };

            // If no tool calls, return the content
            if response.tool_calls.is_empty() {
                let content = response.content.unwrap_or_default();
                self.memory.push(Message::assistant(&content));
                return content;
            }

            // Process tool calls
            self.memory
                .push(Message::assistant_with_tool_calls(response.tool_calls.clone()));

            for tool_call in &response.tool_calls {
                eprintln!(
                    "  {} {}",
                    format!("[tool: {}]", tool_call.name).cyan(),
                    tool_call.arguments.to_string().dimmed()
                );

                let result = if let Some(tool) = self.tools.get(&tool_call.name) {
                    match tool.execute(tool_call.arguments.clone()) {
                        Ok(output) => output,
                        Err(e) => format!("Tool error: {}", e),
                    }
                } else {
                    format!("Unknown tool: {}", tool_call.name)
                };

                self.memory
                    .push(Message::tool_result(&tool_call.id, &result));
            }

            // If the LLM also returned content alongside tool calls, include it
            if let Some(content) = &response.content {
                if !content.is_empty() {
                    self.memory.push(Message::assistant(content));
                }
            }
        }

        "Max iterations reached. The agent could not complete the task.".to_string()
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
        Agent::new(llm, default_registry(), AgentConfig::default())
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
                content: Some("The project is named hermitclaw.".to_string()),
                tool_calls: vec![],
            },
        ]);
        let mut agent = make_agent(Box::new(llm));
        let response = agent.process_message("What is this project?");
        assert_eq!(response, "The project is named hermitclaw.");
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
        assert!(tool_result_msg.content.contains("Unknown tool: nonexistent_tool"));
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
        let mut agent = Agent::new(Box::new(llm), default_registry(), config);
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
        assert!(agent.memory[0].content.contains("hermitclaw"));
    }

    #[test]
    fn test_tool_call_with_content_alongside() {
        // LLM returns both tool_calls AND content in the same response
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

        // Memory should have: system + user + assistant_tool_calls + tool_result + assistant("Let me check") + assistant("Done reading")
        let assistant_messages: Vec<_> = agent
            .memory
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .collect();
        assert!(assistant_messages.len() >= 2);
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

        let mut agent = Agent::new(Box::new(llm), default_registry(), AgentConfig::default());

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

        let mut agent = Agent::new(Box::new(llm), default_registry(), AgentConfig::default());
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

        let mut agent = Agent::new(Box::new(llm), default_registry(), AgentConfig::default());

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

        let mut agent = Agent::new(Box::new(llm), default_registry(), AgentConfig::default());

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
}
