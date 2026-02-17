use super::{LlmError, LlmProvider, LlmResponse, Message, Role, ToolCall, ToolDefinition};
use serde::{Deserialize, Serialize};

pub struct OllamaClient {
    pub base_url: String,
    pub model: String,
    client: reqwest::blocking::Client,
}

// --- Ollama API request/response types ---

#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OllamaTool>,
}

#[derive(Serialize, Deserialize, Debug)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Serialize, Deserialize, Debug)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Serialize, Deserialize, Debug)]
struct OllamaFunctionCall {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Serialize)]
struct OllamaTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OllamaFunctionDef,
}

#[derive(Serialize)]
struct OllamaFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize, Debug)]
struct OllamaChatResponse {
    message: OllamaMessage,
}

// --- Implementation ---

impl OllamaClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    fn convert_messages(messages: &[Message]) -> Vec<OllamaMessage> {
        messages
            .iter()
            .map(|m| {
                let mut ollama_msg = OllamaMessage {
                    role: m.role.to_string(),
                    content: m.content.clone(),
                    tool_calls: None,
                };

                if !m.tool_calls.is_empty() {
                    ollama_msg.tool_calls = Some(
                        m.tool_calls
                            .iter()
                            .map(|tc| OllamaToolCall {
                                function: OllamaFunctionCall {
                                    name: tc.name.clone(),
                                    arguments: tc.arguments.clone(),
                                },
                            })
                            .collect(),
                    );
                }

                ollama_msg
            })
            .collect()
    }

    fn convert_tools(tools: &[ToolDefinition]) -> Vec<OllamaTool> {
        tools
            .iter()
            .map(|t| OllamaTool {
                tool_type: "function".to_string(),
                function: OllamaFunctionDef {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                },
            })
            .collect()
    }
}

impl LlmProvider for OllamaClient {
    fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse, LlmError> {
        let url = format!("{}/api/chat", self.base_url);

        let request = OllamaChatRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            stream: false,
            tools: Self::convert_tools(tools),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .map_err(|e| {
                if e.is_connect() {
                    LlmError::ConnectionError(
                        "Cannot connect to Ollama. Is it running? Start with: ollama serve"
                            .to_string(),
                    )
                } else if e.is_timeout() {
                    LlmError::RequestError("Request timed out".to_string())
                } else {
                    LlmError::RequestError(e.to_string())
                }
            })?;

        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(LlmError::ModelNotFound(format!(
                "Model '{}' not found. Pull it with: ollama pull {}",
                self.model, self.model
            )));
        }
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            return Err(LlmError::RequestError(format!(
                "Ollama returned status {}: {}",
                status, body
            )));
        }

        let resp_body: OllamaChatResponse = response.json().map_err(|e| {
            LlmError::ParseError(format!("Failed to parse Ollama response: {}", e))
        })?;

        let tool_calls = resp_body
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(i, tc)| ToolCall {
                id: format!("call_{}", i),
                name: tc.function.name,
                arguments: tc.function.arguments,
            })
            .collect::<Vec<_>>();

        let content = if resp_body.message.content.is_empty() {
            None
        } else {
            Some(resp_body.message.content)
        };

        Ok(LlmResponse {
            content,
            tool_calls,
        })
    }
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: vec![],
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: vec![],
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: vec![],
        }
    }

    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: String::new(),
            tool_call_id: None,
            tool_calls,
        }
    }

    pub fn tool_result(tool_call_id: &str, content: &str) -> Self {
        Self {
            role: Role::Tool,
            content: content.to_string(),
            tool_call_id: Some(tool_call_id.to_string()),
            tool_calls: vec![],
        }
    }
}
