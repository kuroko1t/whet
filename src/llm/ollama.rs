use super::{LlmError, LlmProvider, LlmResponse, Message, ToolCall, ToolDefinition};

pub struct OllamaClient {
    pub base_url: String,
    pub model: String,
    client: reqwest::blocking::Client,
}

impl OllamaClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }
}

impl LlmProvider for OllamaClient {
    fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
    ) -> Result<LlmResponse, LlmError> {
        // Placeholder - will be implemented in Phase 2
        Ok(LlmResponse {
            content: Some("Not yet implemented".to_string()),
            tool_calls: vec![],
        })
    }
}
