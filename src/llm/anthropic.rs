use super::{LlmError, LlmProvider, LlmResponse, Message, Role, ToolCall, ToolDefinition};
use serde::{Deserialize, Serialize};

pub struct AnthropicClient {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
    client: reqwest::blocking::Client,
}

// --- Anthropic Messages API request/response types ---

#[derive(Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    stream: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Deserialize, Debug)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    #[allow(dead_code)]
    stop_reason: Option<String>,
}

// --- Implementation ---

impl AnthropicClient {
    pub fn new(base_url: &str, model: &str, api_key: String) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            client,
        }
    }

    fn convert_messages(messages: &[Message]) -> (Option<String>, Vec<AnthropicMessage>) {
        let mut system_prompt = None;
        let mut result = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    // Anthropic uses a top-level `system` field, not a system message
                    if system_prompt.is_none() {
                        system_prompt = Some(msg.content.clone());
                    } else {
                        // Append additional system messages (e.g., conversation summary)
                        if let Some(ref mut sp) = system_prompt {
                            sp.push_str("\n\n");
                            sp.push_str(&msg.content);
                        }
                    }
                }
                Role::User => {
                    result.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Text(msg.content.clone()),
                    });
                }
                Role::Assistant => {
                    if !msg.tool_calls.is_empty() {
                        let mut blocks: Vec<ContentBlock> = Vec::new();
                        if !msg.content.is_empty() {
                            blocks.push(ContentBlock::Text {
                                text: msg.content.clone(),
                            });
                        }
                        for tc in &msg.tool_calls {
                            blocks.push(ContentBlock::ToolUse {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                input: tc.arguments.clone(),
                            });
                        }
                        result.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContent::Blocks(blocks),
                        });
                    } else {
                        result.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContent::Text(msg.content.clone()),
                        });
                    }
                }
                Role::Tool => {
                    let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
                    result.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Blocks(vec![ContentBlock::ToolResult {
                            tool_use_id,
                            content: msg.content.clone(),
                        }]),
                    });
                }
            }
        }

        (system_prompt, result)
    }

    fn convert_tools(tools: &[ToolDefinition]) -> Vec<AnthropicTool> {
        tools
            .iter()
            .map(|t| AnthropicTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.parameters.clone(),
            })
            .collect()
    }

    fn parse_response(resp: MessagesResponse) -> LlmResponse {
        let mut content_text = String::new();
        let mut tool_calls = Vec::new();

        for block in resp.content {
            match block {
                ContentBlock::Text { text } => {
                    content_text.push_str(&text);
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments: input,
                    });
                }
                ContentBlock::ToolResult { .. } => {}
            }
        }

        let content = if content_text.is_empty() {
            None
        } else {
            Some(content_text)
        };

        LlmResponse {
            content,
            tool_calls,
        }
    }
}

impl LlmProvider for AnthropicClient {
    fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse, LlmError> {
        let url = format!("{}/v1/messages", self.base_url);
        let (system, api_messages) = Self::convert_messages(messages);

        let request = MessagesRequest {
            model: self.model.clone(),
            max_tokens: 4096,
            system,
            messages: api_messages,
            tools: Self::convert_tools(tools),
            stream: false,
        };

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| {
                if e.is_connect() {
                    LlmError::ConnectionError(format!(
                        "Cannot connect to Anthropic API at {}",
                        self.base_url
                    ))
                } else if e.is_timeout() {
                    LlmError::RequestError("Request timed out".to_string())
                } else {
                    LlmError::RequestError(e.to_string())
                }
            })?;

        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(LlmError::ModelNotFound(format!(
                "Model '{}' not found on Anthropic API",
                self.model
            )));
        }
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(LlmError::RequestError(
                "Authentication failed. Check your api_key in config.".to_string(),
            ));
        }
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            return Err(LlmError::RequestError(format!(
                "Anthropic API returned status {}: {}",
                status, body
            )));
        }

        let resp_body: MessagesResponse = response
            .json()
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {}", e)))?;

        Ok(Self::parse_response(resp_body))
    }

    fn chat_streaming(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        on_token: &mut dyn FnMut(&str),
    ) -> Result<LlmResponse, LlmError> {
        let url = format!("{}/v1/messages", self.base_url);
        let (system, api_messages) = Self::convert_messages(messages);

        let request = MessagesRequest {
            model: self.model.clone(),
            max_tokens: 4096,
            system,
            messages: api_messages,
            tools: Self::convert_tools(tools),
            stream: true,
        };

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| {
                if e.is_connect() {
                    LlmError::ConnectionError(format!(
                        "Cannot connect to Anthropic API at {}",
                        self.base_url
                    ))
                } else if e.is_timeout() {
                    LlmError::RequestError("Request timed out".to_string())
                } else {
                    LlmError::RequestError(e.to_string())
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            return Err(LlmError::RequestError(format!(
                "Anthropic API returned status {}: {}",
                status, body
            )));
        }

        let reader = std::io::BufReader::new(response);
        let mut accumulated_content = String::with_capacity(1024);
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        // Track the current tool call being assembled from streaming events
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_input = String::with_capacity(256);

        use std::io::BufRead;
        for line_result in reader.lines() {
            let line = line_result
                .map_err(|e| LlmError::ParseError(format!("Failed to read stream: {}", e)))?;

            let trimmed = line.trim();

            if !trimmed.starts_with("data: ") {
                continue;
            }

            let data = &trimmed[6..];

            let event: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let event_type = event.get("type").and_then(|t| t.as_str()).unwrap_or("");

            match event_type {
                "content_block_start" => {
                    if let Some(block) = event.get("content_block") {
                        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if block_type == "tool_use" {
                            current_tool_id = block
                                .get("id")
                                .and_then(|i| i.as_str())
                                .unwrap_or("")
                                .to_string();
                            current_tool_name = block
                                .get("name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("")
                                .to_string();
                            current_tool_input.clear();
                        }
                    }
                }
                "content_block_delta" => {
                    if let Some(delta) = event.get("delta") {
                        let delta_type = delta.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        match delta_type {
                            "text_delta" => {
                                if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                    if !text.is_empty() {
                                        on_token(text);
                                        accumulated_content.push_str(text);
                                    }
                                }
                            }
                            "input_json_delta" => {
                                if let Some(partial) =
                                    delta.get("partial_json").and_then(|p| p.as_str())
                                {
                                    current_tool_input.push_str(partial);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "content_block_stop" => {
                    if !current_tool_name.is_empty() {
                        let arguments: serde_json::Value =
                            serde_json::from_str(&current_tool_input).unwrap_or_else(|_| {
                                serde_json::Value::Object(serde_json::Map::new())
                            });
                        tool_calls.push(ToolCall {
                            id: current_tool_id.clone(),
                            name: current_tool_name.clone(),
                            arguments,
                        });
                        current_tool_id.clear();
                        current_tool_name.clear();
                        current_tool_input.clear();
                    }
                }
                "message_stop" => {
                    break;
                }
                _ => {}
            }
        }

        let content = if accumulated_content.is_empty() {
            None
        } else {
            Some(accumulated_content)
        };

        Ok(LlmResponse {
            content,
            tool_calls,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_client_stores_config() {
        let client = AnthropicClient::new(
            "https://api.anthropic.com",
            "claude-3-haiku",
            "sk-test".into(),
        );
        assert_eq!(client.base_url, "https://api.anthropic.com");
        assert_eq!(client.model, "claude-3-haiku");
        assert_eq!(client.api_key, "sk-test");
    }

    #[test]
    fn test_client_trims_trailing_slash() {
        let client = AnthropicClient::new("https://api.anthropic.com/", "test", "key".into());
        assert_eq!(client.base_url, "https://api.anthropic.com");
    }

    #[test]
    fn test_convert_messages_system() {
        let messages = vec![Message::system("You are helpful.")];
        let (system, msgs) = AnthropicClient::convert_messages(&messages);
        assert_eq!(system, Some("You are helpful.".to_string()));
        assert!(msgs.is_empty()); // System is extracted, not a message
    }

    #[test]
    fn test_convert_messages_user() {
        let messages = vec![Message::user("Hello")];
        let (system, msgs) = AnthropicClient::convert_messages(&messages);
        assert!(system.is_none());
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
    }

    #[test]
    fn test_convert_messages_assistant_with_tool_calls() {
        let tc = vec![ToolCall {
            id: "toolu_01".to_string(),
            name: "read_file".to_string(),
            arguments: json!({"path": "/tmp/test.txt"}),
        }];
        let messages = vec![Message::assistant_with_tool_calls(tc)];
        let (_, msgs) = AnthropicClient::convert_messages(&messages);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "assistant");
        match &msgs[0].content {
            AnthropicContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    ContentBlock::ToolUse { id, name, .. } => {
                        assert_eq!(id, "toolu_01");
                        assert_eq!(name, "read_file");
                    }
                    _ => panic!("Expected ToolUse block"),
                }
            }
            _ => panic!("Expected Blocks content"),
        }
    }

    #[test]
    fn test_convert_messages_tool_result() {
        let messages = vec![Message::tool_result("toolu_01", "file contents")];
        let (_, msgs) = AnthropicClient::convert_messages(&messages);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user"); // Tool results are sent as user messages
        match &msgs[0].content {
            AnthropicContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                    } => {
                        assert_eq!(tool_use_id, "toolu_01");
                        assert_eq!(content, "file contents");
                    }
                    _ => panic!("Expected ToolResult block"),
                }
            }
            _ => panic!("Expected Blocks content"),
        }
    }

    #[test]
    fn test_convert_tools() {
        let tools = vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        }];
        let converted = AnthropicClient::convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].name, "read_file");
    }

    #[test]
    fn test_parse_response_text() {
        let resp = MessagesResponse {
            content: vec![ContentBlock::Text {
                text: "Hello!".to_string(),
            }],
            stop_reason: Some("end_turn".to_string()),
        };
        let parsed = AnthropicClient::parse_response(resp);
        assert_eq!(parsed.content, Some("Hello!".to_string()));
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn test_parse_response_tool_use() {
        let resp = MessagesResponse {
            content: vec![ContentBlock::ToolUse {
                id: "toolu_01".to_string(),
                name: "read_file".to_string(),
                input: json!({"path": "/tmp/test"}),
            }],
            stop_reason: Some("tool_use".to_string()),
        };
        let parsed = AnthropicClient::parse_response(resp);
        assert!(parsed.content.is_none());
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "read_file");
    }

    #[test]
    fn test_convert_multiple_system_messages() {
        let messages = vec![
            Message::system("System prompt"),
            Message::system("Previous conversation summary: ..."),
            Message::user("Hello"),
        ];
        let (system, msgs) = AnthropicClient::convert_messages(&messages);
        assert!(system.unwrap().contains("Previous conversation summary"));
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_sse_content_block_delta_text() {
        let event_json = json!({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"}
        });
        let delta_type = event_json["delta"]["type"].as_str().unwrap();
        assert_eq!(delta_type, "text_delta");
        let text = event_json["delta"]["text"].as_str().unwrap();
        assert_eq!(text, "Hello");
    }

    #[test]
    fn test_sse_content_block_start_tool_use() {
        let event_json = json!({
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "read_file"
            }
        });
        let block = &event_json["content_block"];
        assert_eq!(block["type"].as_str().unwrap(), "tool_use");
        assert_eq!(block["id"].as_str().unwrap(), "toolu_01");
        assert_eq!(block["name"].as_str().unwrap(), "read_file");
    }

    #[test]
    fn test_sse_input_json_delta() {
        let event_json = json!({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": "{\"path\":"}
        });
        let partial = event_json["delta"]["partial_json"].as_str().unwrap();
        assert_eq!(partial, "{\"path\":");
    }

    #[test]
    fn test_request_serialization() {
        let request = MessagesRequest {
            model: "claude-3-haiku".to_string(),
            max_tokens: 4096,
            system: Some("You are helpful.".to_string()),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicContent::Text("Hello".to_string()),
            }],
            tools: vec![],
            stream: false,
        };
        let json_str = serde_json::to_string(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["model"], "claude-3-haiku");
        assert_eq!(parsed["max_tokens"], 4096);
        assert_eq!(parsed["system"], "You are helpful.");
        assert!(parsed.get("tools").is_none()); // skip_serializing_if empty
    }
}
