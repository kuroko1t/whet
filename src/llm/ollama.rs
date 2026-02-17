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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- Message conversion tests ---

    #[test]
    fn test_convert_user_message() {
        let messages = vec![Message::user("Hello")];
        let converted = OllamaClient::convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
        assert_eq!(converted[0].content, "Hello");
        assert!(converted[0].tool_calls.is_none());
    }

    #[test]
    fn test_convert_system_message() {
        let messages = vec![Message::system("You are helpful.")];
        let converted = OllamaClient::convert_messages(&messages);
        assert_eq!(converted[0].role, "system");
        assert_eq!(converted[0].content, "You are helpful.");
    }

    #[test]
    fn test_convert_assistant_with_tool_calls() {
        let tc = vec![ToolCall {
            id: "call_0".to_string(),
            name: "read_file".to_string(),
            arguments: json!({"path": "/tmp/test.txt"}),
        }];
        let messages = vec![Message::assistant_with_tool_calls(tc)];
        let converted = OllamaClient::convert_messages(&messages);
        assert_eq!(converted[0].role, "assistant");
        let tool_calls = converted[0].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "read_file");
        assert_eq!(tool_calls[0].function.arguments, json!({"path": "/tmp/test.txt"}));
    }

    #[test]
    fn test_convert_tool_result_message() {
        let messages = vec![Message::tool_result("call_0", "file contents here")];
        let converted = OllamaClient::convert_messages(&messages);
        assert_eq!(converted[0].role, "tool");
        assert_eq!(converted[0].content, "file contents here");
    }

    #[test]
    fn test_convert_mixed_conversation() {
        let messages = vec![
            Message::system("sys prompt"),
            Message::user("question"),
            Message::assistant("answer"),
            Message::user("follow-up"),
        ];
        let converted = OllamaClient::convert_messages(&messages);
        assert_eq!(converted.len(), 4);
        assert_eq!(converted[0].role, "system");
        assert_eq!(converted[1].role, "user");
        assert_eq!(converted[2].role, "assistant");
        assert_eq!(converted[3].role, "user");
    }

    #[test]
    fn test_convert_empty_messages() {
        let messages: Vec<Message> = vec![];
        let converted = OllamaClient::convert_messages(&messages);
        assert!(converted.is_empty());
    }

    // --- Tool definition conversion tests ---

    #[test]
    fn test_convert_tools_single() {
        let tools = vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }),
        }];
        let converted = OllamaClient::convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].tool_type, "function");
        assert_eq!(converted[0].function.name, "read_file");
        assert_eq!(converted[0].function.description, "Read a file");
    }

    #[test]
    fn test_convert_tools_empty() {
        let tools: Vec<ToolDefinition> = vec![];
        let converted = OllamaClient::convert_tools(&tools);
        assert!(converted.is_empty());
    }

    // --- Request serialization tests ---

    #[test]
    fn test_request_serialization_no_tools() {
        let request = OllamaChatRequest {
            model: "qwen2.5:7b".to_string(),
            messages: vec![OllamaMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                tool_calls: None,
            }],
            stream: false,
            tools: vec![],
        };
        let json_str = serde_json::to_string(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(parsed["model"], "qwen2.5:7b");
        assert_eq!(parsed["stream"], false);
        assert_eq!(parsed["messages"][0]["role"], "user");
        assert_eq!(parsed["messages"][0]["content"], "Hello");
        // tools field should be absent (skip_serializing_if = "Vec::is_empty")
        assert!(parsed.get("tools").is_none());
    }

    #[test]
    fn test_request_serialization_with_tools() {
        let request = OllamaChatRequest {
            model: "test-model".to_string(),
            messages: vec![],
            stream: false,
            tools: vec![OllamaTool {
                tool_type: "function".to_string(),
                function: OllamaFunctionDef {
                    name: "test_tool".to_string(),
                    description: "A test tool".to_string(),
                    parameters: json!({"type": "object"}),
                },
            }],
        };
        let json_str = serde_json::to_string(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(parsed["tools"][0]["type"], "function");
        assert_eq!(parsed["tools"][0]["function"]["name"], "test_tool");
    }

    // --- Response deserialization tests ---

    #[test]
    fn test_response_parse_text_only() {
        let json = json!({
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help?"
            }
        });
        let resp: OllamaChatResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.message.content, "Hello! How can I help?");
        assert!(resp.message.tool_calls.is_none());
    }

    #[test]
    fn test_response_parse_with_tool_calls() {
        let json = json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "/tmp/test.txt"}
                        }
                    }
                ]
            }
        });
        let resp: OllamaChatResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.message.content, "");
        let tool_calls = resp.message.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "read_file");
    }

    #[test]
    fn test_response_parse_multiple_tool_calls() {
        let json = json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "tool_a", "arguments": {}}},
                    {"function": {"name": "tool_b", "arguments": {"key": "value"}}}
                ]
            }
        });
        let resp: OllamaChatResponse = serde_json::from_value(json).unwrap();
        let tool_calls = resp.message.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].function.name, "tool_a");
        assert_eq!(tool_calls[1].function.name, "tool_b");
    }

    #[test]
    fn test_response_parse_empty_content_becomes_none() {
        // Simulate what our LlmProvider impl does with empty content
        let json = json!({
            "message": {
                "role": "assistant",
                "content": ""
            }
        });
        let resp: OllamaChatResponse = serde_json::from_value(json).unwrap();
        // Our code treats empty string as None
        let content = if resp.message.content.is_empty() {
            None
        } else {
            Some(resp.message.content)
        };
        assert!(content.is_none());
    }

    // --- Message constructor tests ---

    #[test]
    fn test_message_constructors() {
        let sys = Message::system("sys");
        assert_eq!(sys.role, Role::System);
        assert_eq!(sys.content, "sys");
        assert!(sys.tool_call_id.is_none());
        assert!(sys.tool_calls.is_empty());

        let usr = Message::user("usr");
        assert_eq!(usr.role, Role::User);

        let asst = Message::assistant("asst");
        assert_eq!(asst.role, Role::Assistant);
        assert!(asst.tool_calls.is_empty());

        let tool = Message::tool_result("id_0", "result");
        assert_eq!(tool.role, Role::Tool);
        assert_eq!(tool.tool_call_id, Some("id_0".to_string()));

        let asst_tc = Message::assistant_with_tool_calls(vec![ToolCall {
            id: "c0".to_string(),
            name: "test".to_string(),
            arguments: json!({}),
        }]);
        assert_eq!(asst_tc.role, Role::Assistant);
        assert_eq!(asst_tc.tool_calls.len(), 1);
        assert!(asst_tc.content.is_empty());
    }

    // --- OllamaClient constructor test ---

    #[test]
    fn test_client_stores_config() {
        let client = OllamaClient::new("http://localhost:11434", "test-model");
        assert_eq!(client.base_url, "http://localhost:11434");
        assert_eq!(client.model, "test-model");
    }
}
