pub mod prompt;

use crate::llm::{LlmProvider, Message};
use crate::tools::ToolRegistry;

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
        Self {
            llm,
            tools,
            memory: Vec::new(),
            config,
        }
    }
}
