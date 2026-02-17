use super::{LlmError, LlmProvider, LlmResponse, Message, ToolCall, ToolDefinition};
use serde::{Deserialize, Serialize};

pub struct OpenAiCompatClient {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    client: reqwest::blocking::Client,
}

// --- OpenAI-compatible API request/response types ---

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ChatTool>,
    stream: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ChatMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ChatToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: ChatFunctionCall,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ChatFunctionCall {
    name: String,
    arguments: String, // OpenAI sends arguments as a JSON string
}

#[derive(Serialize)]
struct ChatTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: ChatFunctionDef,
}

#[derive(Serialize)]
struct ChatFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize, Debug)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize, Debug)]
struct ChatChoice {
    message: ChatMessage,
}

// --- Implementation ---

impl OpenAiCompatClient {
    pub fn new(base_url: &str, model: &str, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    fn convert_messages(messages: &[Message]) -> Vec<ChatMessage> {
        messages
            .iter()
            .map(|m| {
                let mut msg = ChatMessage {
                    role: m.role.to_string(),
                    content: if m.content.is_empty() && !m.tool_calls.is_empty() {
                        None
                    } else {
                        Some(m.content.clone())
                    },
                    tool_calls: None,
                    tool_call_id: m.tool_call_id.clone(),
                };

                if !m.tool_calls.is_empty() {
                    msg.tool_calls = Some(
                        m.tool_calls
                            .iter()
                            .map(|tc| ChatToolCall {
                                id: tc.id.clone(),
                                call_type: "function".to_string(),
                                function: ChatFunctionCall {
                                    name: tc.name.clone(),
                                    arguments: tc.arguments.to_string(),
                                },
                            })
                            .collect(),
                    );
                }

                msg
            })
            .collect()
    }

    fn convert_tools(tools: &[ToolDefinition]) -> Vec<ChatTool> {
        tools
            .iter()
            .map(|t| ChatTool {
                tool_type: "function".to_string(),
                function: ChatFunctionDef {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                },
            })
            .collect()
    }
}

impl LlmProvider for OpenAiCompatClient {
    fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse, LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let request = ChatRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            tools: Self::convert_tools(tools),
            stream: false,
        };

        let mut req_builder = self.client.post(&url).json(&request);

        if let Some(ref key) = self.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        let response = req_builder.send().map_err(|e| {
            if e.is_connect() {
                LlmError::ConnectionError(format!(
                    "Cannot connect to OpenAI-compatible server at {}. Is it running?",
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
                "Model '{}' not found on server at {}",
                self.model, self.base_url
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
                "Server returned status {}: {}",
                status, body
            )));
        }

        let resp_body: ChatResponse = response.json().map_err(|e| {
            LlmError::ParseError(format!("Failed to parse response: {}", e))
        })?;

        let choice = resp_body
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::ParseError("No choices in response".to_string()))?;

        let tool_calls = choice
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| {
                // OpenAI sends arguments as a JSON string, parse it
                let arguments: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or_else(|_| {
                        serde_json::Value::Object(serde_json::Map::new())
                    });
                ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments,
                }
            })
            .collect::<Vec<_>>();

        let content = choice
            .message
            .content
            .filter(|c| !c.is_empty());

        Ok(LlmResponse {
            content,
            tool_calls,
        })
    }

    fn chat_streaming(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        on_token: &mut dyn FnMut(&str),
    ) -> Result<LlmResponse, LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let request = ChatRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            tools: Self::convert_tools(tools),
            stream: true,
        };

        let mut req_builder = self.client.post(&url).json(&request);

        if let Some(ref key) = self.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        let response = req_builder.send().map_err(|e| {
            if e.is_connect() {
                LlmError::ConnectionError(format!(
                    "Cannot connect to OpenAI-compatible server at {}. Is it running?",
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
                "Server returned status {}: {}",
                status, body
            )));
        }

        let reader = std::io::BufReader::new(response);
        let mut accumulated_content = String::new();
        // Track tool calls by index for incremental assembly
        let mut tool_call_map: std::collections::HashMap<usize, (String, String, String)> =
            std::collections::HashMap::new();

        use std::io::BufRead;
        for line_result in reader.lines() {
            let line = line_result
                .map_err(|e| LlmError::ParseError(format!("Failed to read stream: {}", e)))?;

            let line = line.trim().to_string();

            // SSE format: "data: {...}" or "data: [DONE]"
            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }

            let chunk: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
                if let Some(choice) = choices.first() {
                    let delta = &choice["delta"];

                    // Content delta
                    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                        if !content.is_empty() {
                            on_token(content);
                            accumulated_content.push_str(content);
                        }
                    }

                    // Tool call deltas
                    if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array())
                    {
                        for tc in tool_calls {
                            let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0)
                                as usize;
                            let entry = tool_call_map
                                .entry(index)
                                .or_insert_with(|| (String::new(), String::new(), String::new()));

                            if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                                entry.0 = id.to_string();
                            }
                            if let Some(func) = tc.get("function") {
                                if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                                    entry.1 = name.to_string();
                                }
                                if let Some(args) =
                                    func.get("arguments").and_then(|a| a.as_str())
                                {
                                    entry.2.push_str(args);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Assemble final tool calls
        let mut tool_calls: Vec<ToolCall> = tool_call_map
            .into_iter()
            .map(|(_, (id, name, args_str))| {
                let arguments: serde_json::Value =
                    serde_json::from_str(&args_str).unwrap_or_else(|_| {
                        serde_json::Value::Object(serde_json::Map::new())
                    });
                ToolCall {
                    id,
                    name,
                    arguments,
                }
            })
            .collect();
        tool_calls.sort_by(|a, b| a.id.cmp(&b.id));

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
        let client = OpenAiCompatClient::new(
            "http://localhost:8080",
            "test-model",
            Some("sk-test".to_string()),
        );
        assert_eq!(client.base_url, "http://localhost:8080");
        assert_eq!(client.model, "test-model");
        assert_eq!(client.api_key, Some("sk-test".to_string()));
    }

    #[test]
    fn test_client_no_api_key() {
        let client = OpenAiCompatClient::new("http://localhost:8080", "test-model", None);
        assert!(client.api_key.is_none());
    }

    #[test]
    fn test_client_trims_trailing_slash() {
        let client = OpenAiCompatClient::new("http://localhost:8080/", "test-model", None);
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[test]
    fn test_convert_user_message() {
        let messages = vec![Message::user("Hello")];
        let converted = OpenAiCompatClient::convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
        assert_eq!(converted[0].content, Some("Hello".to_string()));
        assert!(converted[0].tool_calls.is_none());
    }

    #[test]
    fn test_convert_system_message() {
        let messages = vec![Message::system("You are helpful.")];
        let converted = OpenAiCompatClient::convert_messages(&messages);
        assert_eq!(converted[0].role, "system");
        assert_eq!(converted[0].content, Some("You are helpful.".to_string()));
    }

    #[test]
    fn test_convert_assistant_with_tool_calls() {
        let tc = vec![ToolCall {
            id: "call_0".to_string(),
            name: "read_file".to_string(),
            arguments: json!({"path": "/tmp/test.txt"}),
        }];
        let messages = vec![Message::assistant_with_tool_calls(tc)];
        let converted = OpenAiCompatClient::convert_messages(&messages);
        assert_eq!(converted[0].role, "assistant");
        assert!(converted[0].content.is_none()); // empty content with tool calls -> None
        let tool_calls = converted[0].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "read_file");
        // Arguments should be JSON string
        assert_eq!(
            tool_calls[0].function.arguments,
            "{\"path\":\"/tmp/test.txt\"}"
        );
    }

    #[test]
    fn test_convert_tool_result_message() {
        let messages = vec![Message::tool_result("call_0", "file contents")];
        let converted = OpenAiCompatClient::convert_messages(&messages);
        assert_eq!(converted[0].role, "tool");
        assert_eq!(converted[0].content, Some("file contents".to_string()));
        assert_eq!(converted[0].tool_call_id, Some("call_0".to_string()));
    }

    #[test]
    fn test_convert_tools() {
        let tools = vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        }];
        let converted = OpenAiCompatClient::convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].tool_type, "function");
        assert_eq!(converted[0].function.name, "read_file");
    }

    #[test]
    fn test_convert_empty_tools() {
        let tools: Vec<ToolDefinition> = vec![];
        let converted = OpenAiCompatClient::convert_tools(&tools);
        assert!(converted.is_empty());
    }

    #[test]
    fn test_response_parse_text_only() {
        let json_val = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                }
            }]
        });
        let resp: ChatResponse = serde_json::from_value(json_val).unwrap();
        assert_eq!(resp.choices[0].message.content, Some("Hello!".to_string()));
        assert!(resp.choices[0].message.tool_calls.is_none());
    }

    #[test]
    fn test_response_parse_with_tool_calls() {
        let json_val = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"/tmp/test.txt\"}"
                        }
                    }]
                }
            }]
        });
        let resp: ChatResponse = serde_json::from_value(json_val).unwrap();
        let tool_calls = resp.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc123");
        assert_eq!(tool_calls[0].function.name, "read_file");
        // Verify we can parse the arguments string back to JSON
        let args: serde_json::Value =
            serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["path"], "/tmp/test.txt");
    }

    #[test]
    fn test_request_serialization() {
        let request = ChatRequest {
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some("Hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: vec![],
            stream: false,
        };
        let json_str = serde_json::to_string(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["model"], "test-model");
        assert_eq!(parsed["stream"], false);
        assert!(parsed.get("tools").is_none()); // skip_serializing_if empty
    }

    // --- Streaming SSE parsing tests ---

    #[test]
    fn test_sse_line_parsing_content_delta() {
        // Simulate an SSE "data: {...}" line from chat/completions streaming
        let sse_line = r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#;
        let data = &sse_line[6..]; // strip "data: "
        let chunk: serde_json::Value = serde_json::from_str(data).unwrap();

        let content = chunk["choices"][0]["delta"]["content"].as_str();
        assert_eq!(content, Some("Hello"));
    }

    #[test]
    fn test_sse_line_parsing_done() {
        let sse_line = "data: [DONE]";
        let data = &sse_line[6..];
        assert_eq!(data, "[DONE]");
    }

    #[test]
    fn test_sse_content_accumulation_simulation() {
        let sse_lines = vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}",
            "data: {\"choices\":[{\"delta\":{\"content\":\"!\"}}]}",
            "data: [DONE]",
        ];

        let mut accumulated = String::new();
        let mut callback_tokens = Vec::new();

        for line in sse_lines {
            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }
            let chunk: serde_json::Value = serde_json::from_str(data).unwrap();
            if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
                if let Some(choice) = choices.first() {
                    if let Some(content) = choice["delta"]["content"].as_str() {
                        if !content.is_empty() {
                            callback_tokens.push(content.to_string());
                            accumulated.push_str(content);
                        }
                    }
                }
            }
        }

        assert_eq!(accumulated, "Hello world!");
        assert_eq!(callback_tokens, vec!["Hello", " world", "!"]);
    }

    #[test]
    fn test_sse_tool_call_delta_assembly() {
        // OpenAI sends tool calls incrementally across multiple deltas
        let sse_lines = vec![
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"read_file","arguments":""}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"pa"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"/tmp/test\"}"}}]}}]}"#,
            "data: [DONE]",
        ];

        let mut tool_call_map: std::collections::HashMap<usize, (String, String, String)> =
            std::collections::HashMap::new();

        for line in sse_lines {
            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }
            let chunk: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
                if let Some(choice) = choices.first() {
                    if let Some(tool_calls) =
                        choice["delta"].get("tool_calls").and_then(|tc| tc.as_array())
                    {
                        for tc in tool_calls {
                            let index =
                                tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                            let entry = tool_call_map
                                .entry(index)
                                .or_insert_with(|| (String::new(), String::new(), String::new()));
                            if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                                entry.0 = id.to_string();
                            }
                            if let Some(func) = tc.get("function") {
                                if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                                    entry.1 = name.to_string();
                                }
                                if let Some(args) = func.get("arguments").and_then(|a| a.as_str())
                                {
                                    entry.2.push_str(args);
                                }
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(tool_call_map.len(), 1);
        let (id, name, args_str) = &tool_call_map[&0];
        assert_eq!(id, "call_abc");
        assert_eq!(name, "read_file");
        let args: serde_json::Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(args["path"], "/tmp/test");
    }

    #[test]
    fn test_sse_multiple_tool_calls_delta() {
        // Two tool calls streamed incrementally
        let sse_lines = vec![
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_1","type":"function","function":{"name":"list_dir","arguments":"{\"path\":\".\"}"}}]}}]}"#,
            "data: [DONE]",
        ];

        let mut tool_call_map: std::collections::HashMap<usize, (String, String, String)> =
            std::collections::HashMap::new();

        for line in sse_lines {
            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }
            let chunk: serde_json::Value = serde_json::from_str(data).unwrap();
            if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
                if let Some(choice) = choices.first() {
                    if let Some(tcs) =
                        choice["delta"].get("tool_calls").and_then(|tc| tc.as_array())
                    {
                        for tc in tcs {
                            let index =
                                tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                            let entry = tool_call_map
                                .entry(index)
                                .or_insert_with(|| (String::new(), String::new(), String::new()));
                            if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                                entry.0 = id.to_string();
                            }
                            if let Some(func) = tc.get("function") {
                                if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                                    entry.1 = name.to_string();
                                }
                                if let Some(args) = func.get("arguments").and_then(|a| a.as_str())
                                {
                                    entry.2.push_str(args);
                                }
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(tool_call_map.len(), 2);
        assert_eq!(tool_call_map[&0].1, "read_file");
        assert_eq!(tool_call_map[&1].1, "list_dir");
    }

    #[test]
    fn test_sse_non_data_lines_skipped() {
        // SSE streams can contain event: lines, comments, etc.
        let lines = vec![
            ": this is a comment",
            "event: message",
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}",
            "",
            "data: [DONE]",
        ];

        let mut found_content = false;
        for line in lines {
            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }
            if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) {
                if chunk["choices"][0]["delta"]["content"].as_str() == Some("hi") {
                    found_content = true;
                }
            }
        }
        assert!(found_content);
    }

    #[test]
    fn test_sse_empty_content_delta_skipped() {
        let sse_line = r#"data: {"choices":[{"delta":{"content":""}}]}"#;
        let data = &sse_line[6..];
        let chunk: serde_json::Value = serde_json::from_str(data).unwrap();
        let content = chunk["choices"][0]["delta"]["content"].as_str().unwrap();
        assert!(content.is_empty());
    }

    #[test]
    fn test_sse_malformed_json_skipped() {
        // Malformed JSON should not cause panic in the streaming parser
        let lines = vec![
            "data: {invalid json}",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}",
            "data: [DONE]",
        ];

        let mut accumulated = String::new();
        for line in lines {
            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }
            // Malformed JSON should be skipped (continue)
            let chunk: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(content) = chunk["choices"][0]["delta"]["content"].as_str() {
                accumulated.push_str(content);
            }
        }
        assert_eq!(accumulated, "ok");
    }

    #[test]
    fn test_response_parse_multiple_tool_calls() {
        let json_val = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\":\"a.txt\"}"
                            }
                        },
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "list_dir",
                                "arguments": "{\"path\":\".\"}"
                            }
                        }
                    ]
                }
            }]
        });
        let resp: ChatResponse = serde_json::from_value(json_val).unwrap();
        let tcs = resp.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0].function.name, "read_file");
        assert_eq!(tcs[1].function.name, "list_dir");
    }

    #[test]
    fn test_response_parse_empty_content_becomes_none() {
        let json_val = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": ""
                }
            }]
        });
        let resp: ChatResponse = serde_json::from_value(json_val).unwrap();
        let content = resp.choices[0].message.content.as_ref();
        // Our code filters empty content
        let filtered = content.filter(|c| !c.is_empty());
        assert!(filtered.is_none());
    }

    #[test]
    fn test_response_parse_null_content() {
        let json_val = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null
                }
            }]
        });
        let resp: ChatResponse = serde_json::from_value(json_val).unwrap();
        assert!(resp.choices[0].message.content.is_none());
    }

    #[test]
    fn test_request_serialization_with_stream_true() {
        let request = ChatRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: vec![],
            stream: true,
        };
        let json_str = serde_json::to_string(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["stream"], true);
    }

    #[test]
    fn test_convert_mixed_conversation() {
        let messages = vec![
            Message::system("sys"),
            Message::user("question"),
            Message::assistant("answer"),
            Message::user("follow-up"),
        ];
        let converted = OpenAiCompatClient::convert_messages(&messages);
        assert_eq!(converted.len(), 4);
        assert_eq!(converted[0].role, "system");
        assert_eq!(converted[1].role, "user");
        assert_eq!(converted[2].role, "assistant");
        assert_eq!(converted[3].role, "user");
    }

    #[test]
    fn test_tool_call_arguments_json_string_parsing() {
        // OpenAI sends arguments as a JSON string, verify we can parse it
        let args_str = r#"{"path":"/tmp/test.txt","recursive":true}"#;
        let parsed: serde_json::Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(parsed["path"], "/tmp/test.txt");
        assert_eq!(parsed["recursive"], true);
    }

    #[test]
    fn test_tool_call_arguments_invalid_json_fallback() {
        // If arguments is invalid JSON, should fallback to empty object
        let args_str = "invalid json {{{";
        let parsed: serde_json::Value = serde_json::from_str(args_str)
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
        assert!(parsed.is_object());
        assert_eq!(parsed.as_object().unwrap().len(), 0);
    }
}
