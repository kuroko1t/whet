pub mod ollama;

use std::fmt;

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub tool_call_id: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug)]
pub enum LlmError {
    ConnectionError(String),
    RequestError(String),
    ParseError(String),
    ModelNotFound(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            LlmError::RequestError(msg) => write!(f, "Request error: {}", msg),
            LlmError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            LlmError::ModelNotFound(model) => write!(f, "Model not found: {}", model),
        }
    }
}

impl std::error::Error for LlmError {}

pub trait LlmProvider {
    fn chat(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<LlmResponse, LlmError>;
}
