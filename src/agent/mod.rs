pub mod prompt;

use crate::llm::{LlmProvider, Message, Role};
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
    pub sandbox_enabled: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "qwen2.5:7b".to_string(),
            max_iterations: 10,
            sandbox_enabled: true,
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
        self.memory.push(Message::user(user_input));

        let tool_defs = self.tools.definitions();

        for _iteration in 0..self.config.max_iterations {
            let response = match self.llm.chat(&self.memory, &tool_defs) {
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
