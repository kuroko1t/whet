use super::{LlmError, LlmProvider, LlmResponse, Message, Role, ToolCall, ToolDefinition};
use serde::{Deserialize, Serialize};

pub struct GeminiClient {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
    client: reqwest::blocking::Client,
}

// --- Gemini API request/response types ---

#[derive(Serialize)]
struct GenerateContentRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<GeminiToolDeclaration>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GeminiFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Serialize)]
struct GeminiToolDeclaration {
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<GeminiFunctionDef>,
}

#[derive(Serialize)]
struct GeminiFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize, Debug)]
struct GenerateContentResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Deserialize, Debug)]
struct GeminiCandidate {
    content: GeminiContent,
}

// --- Implementation ---

impl GeminiClient {
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

    fn convert_messages(messages: &[Message]) -> (Option<GeminiContent>, Vec<GeminiContent>) {
        let mut system_instruction: Option<GeminiContent> = None;
        let mut contents = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    // Gemini uses system_instruction
                    let text = msg.content.clone();
                    if let Some(ref mut si) = system_instruction {
                        si.parts.push(GeminiPart::Text { text });
                    } else {
                        system_instruction = Some(GeminiContent {
                            role: None,
                            parts: vec![GeminiPart::Text { text }],
                        });
                    }
                }
                Role::User => {
                    contents.push(GeminiContent {
                        role: Some("user".to_string()),
                        parts: vec![GeminiPart::Text {
                            text: msg.content.clone(),
                        }],
                    });
                }
                Role::Assistant => {
                    if !msg.tool_calls.is_empty() {
                        let mut parts = Vec::new();
                        if !msg.content.is_empty() {
                            parts.push(GeminiPart::Text {
                                text: msg.content.clone(),
                            });
                        }
                        for tc in &msg.tool_calls {
                            parts.push(GeminiPart::FunctionCall {
                                function_call: GeminiFunctionCall {
                                    name: tc.name.clone(),
                                    args: tc.arguments.clone(),
                                },
                            });
                        }
                        contents.push(GeminiContent {
                            role: Some("model".to_string()),
                            parts,
                        });
                    } else {
                        contents.push(GeminiContent {
                            role: Some("model".to_string()),
                            parts: vec![GeminiPart::Text {
                                text: msg.content.clone(),
                            }],
                        });
                    }
                }
                Role::Tool => {
                    // Gemini expects tool results as functionResponse
                    // We need the tool name, which we don't have directly on the Message.
                    // Use tool_call_id as a fallback name identifier.
                    let name = msg.tool_call_id.clone().unwrap_or_default();
                    contents.push(GeminiContent {
                        role: Some("user".to_string()),
                        parts: vec![GeminiPart::FunctionResponse {
                            function_response: GeminiFunctionResponse {
                                name,
                                response: serde_json::json!({ "result": msg.content }),
                            },
                        }],
                    });
                }
            }
        }

        (system_instruction, contents)
    }

    fn convert_tools(tools: &[ToolDefinition]) -> Vec<GeminiToolDeclaration> {
        if tools.is_empty() {
            return Vec::new();
        }
        vec![GeminiToolDeclaration {
            function_declarations: tools
                .iter()
                .map(|t| GeminiFunctionDef {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                })
                .collect(),
        }]
    }

    fn parse_response(resp: GenerateContentResponse) -> Result<LlmResponse, LlmError> {
        let candidate = resp
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::ParseError("No candidates in response".to_string()))?;

        let mut content_text = String::new();
        let mut tool_calls = Vec::new();

        for part in candidate.content.parts {
            match part {
                GeminiPart::Text { text } => {
                    content_text.push_str(&text);
                }
                GeminiPart::FunctionCall { function_call } => {
                    // Use function name as ID — Gemini has no call IDs,
                    // and functionResponse requires the function name to match.
                    tool_calls.push(ToolCall {
                        id: function_call.name.clone(),
                        name: function_call.name,
                        arguments: function_call.args,
                    });
                }
                GeminiPart::FunctionResponse { .. } => {}
            }
        }

        let content = if content_text.is_empty() {
            None
        } else {
            Some(content_text)
        };

        Ok(LlmResponse {
            content,
            tool_calls,
        })
    }
}

impl LlmProvider for GeminiClient {
    fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse, LlmError> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, self.model, self.api_key
        );

        let (system_instruction, contents) = Self::convert_messages(messages);

        let request = GenerateContentRequest {
            contents,
            system_instruction,
            tools: Self::convert_tools(tools),
        };

        let response = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| {
                if e.is_connect() {
                    LlmError::ConnectionError(format!(
                        "Cannot connect to Gemini API at {}",
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
                "Model '{}' not found on Gemini API",
                self.model
            )));
        }
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(LlmError::RequestError(
                "Authentication failed. Check your api_key in config.".to_string(),
            ));
        }
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            return Err(LlmError::RequestError(format!(
                "Gemini API returned status {}: {}",
                status, body
            )));
        }

        let resp_body: GenerateContentResponse = response
            .json()
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {}", e)))?;

        Self::parse_response(resp_body)
    }

    fn chat_streaming(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        on_token: &mut dyn FnMut(&str),
    ) -> Result<LlmResponse, LlmError> {
        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url, self.model, self.api_key
        );

        let (system_instruction, contents) = Self::convert_messages(messages);

        let request = GenerateContentRequest {
            contents,
            system_instruction,
            tools: Self::convert_tools(tools),
        };

        let response = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| {
                if e.is_connect() {
                    LlmError::ConnectionError(format!(
                        "Cannot connect to Gemini API at {}",
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
                "Gemini API returned status {}: {}",
                status, body
            )));
        }

        let reader = std::io::BufReader::new(response);
        let mut accumulated_content = String::with_capacity(1024);
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        use std::io::BufRead;
        for line_result in reader.lines() {
            let line = line_result
                .map_err(|e| LlmError::ParseError(format!("Failed to read stream: {}", e)))?;

            let trimmed = line.trim();

            if !trimmed.starts_with("data: ") {
                continue;
            }

            let data = &trimmed[6..];

            let chunk: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Gemini streaming returns GenerateContentResponse-shaped chunks
            if let Some(candidates) = chunk.get("candidates").and_then(|c| c.as_array()) {
                if let Some(candidate) = candidates.first() {
                    if let Some(parts) = candidate
                        .get("content")
                        .and_then(|c| c.get("parts"))
                        .and_then(|p| p.as_array())
                    {
                        for part in parts {
                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                if !text.is_empty() {
                                    on_token(text);
                                    accumulated_content.push_str(text);
                                }
                            }
                            if let Some(fc) = part.get("functionCall") {
                                let name = fc
                                    .get("name")
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let args = fc
                                    .get("args")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                                tool_calls.push(ToolCall {
                                    id: name.clone(),
                                    name,
                                    arguments: args,
                                });
                            }
                        }
                    }
                }
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
        let client = GeminiClient::new(
            "https://generativelanguage.googleapis.com",
            "gemini-pro",
            "test-key".into(),
        );
        assert_eq!(client.base_url, "https://generativelanguage.googleapis.com");
        assert_eq!(client.model, "gemini-pro");
        assert_eq!(client.api_key, "test-key");
    }

    #[test]
    fn test_client_trims_trailing_slash() {
        let client = GeminiClient::new(
            "https://generativelanguage.googleapis.com/",
            "test",
            "key".into(),
        );
        assert_eq!(client.base_url, "https://generativelanguage.googleapis.com");
    }

    #[test]
    fn test_convert_messages_system() {
        let messages = vec![Message::system("You are helpful.")];
        let (si, contents) = GeminiClient::convert_messages(&messages);
        assert!(si.is_some());
        assert!(contents.is_empty());
        let si = si.unwrap();
        assert_eq!(si.parts.len(), 1);
    }

    #[test]
    fn test_convert_messages_user() {
        let messages = vec![Message::user("Hello")];
        let (si, contents) = GeminiClient::convert_messages(&messages);
        assert!(si.is_none());
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, Some("user".to_string()));
    }

    #[test]
    fn test_convert_messages_assistant() {
        let messages = vec![Message::assistant("Hi there")];
        let (_, contents) = GeminiClient::convert_messages(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, Some("model".to_string()));
    }

    #[test]
    fn test_convert_messages_assistant_with_tool_calls() {
        let tc = vec![ToolCall {
            id: "call_0".to_string(),
            name: "read_file".to_string(),
            arguments: json!({"path": "/tmp/test.txt"}),
        }];
        let messages = vec![Message::assistant_with_tool_calls(tc)];
        let (_, contents) = GeminiClient::convert_messages(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, Some("model".to_string()));
        assert_eq!(contents[0].parts.len(), 1);
    }

    #[test]
    fn test_convert_messages_tool_result() {
        let messages = vec![Message::tool_result("read_file", "file contents")];
        let (_, contents) = GeminiClient::convert_messages(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, Some("user".to_string()));
    }

    #[test]
    fn test_convert_tools() {
        let tools = vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        }];
        let converted = GeminiClient::convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].function_declarations.len(), 1);
        assert_eq!(converted[0].function_declarations[0].name, "read_file");
    }

    #[test]
    fn test_convert_tools_empty() {
        let tools: Vec<ToolDefinition> = vec![];
        let converted = GeminiClient::convert_tools(&tools);
        assert!(converted.is_empty());
    }

    #[test]
    fn test_parse_response_text() {
        let resp = GenerateContentResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![GeminiPart::Text {
                        text: "Hello!".to_string(),
                    }],
                },
            }],
        };
        let parsed = GeminiClient::parse_response(resp).unwrap();
        assert_eq!(parsed.content, Some("Hello!".to_string()));
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn test_parse_response_function_call() {
        let resp = GenerateContentResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![GeminiPart::FunctionCall {
                        function_call: GeminiFunctionCall {
                            name: "read_file".to_string(),
                            args: json!({"path": "/tmp/test"}),
                        },
                    }],
                },
            }],
        };
        let parsed = GeminiClient::parse_response(resp).unwrap();
        assert!(parsed.content.is_none());
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "read_file");
    }

    #[test]
    fn test_parse_response_no_candidates() {
        let resp = GenerateContentResponse { candidates: vec![] };
        let result = GeminiClient::parse_response(resp);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_multiple_system_messages() {
        let messages = vec![
            Message::system("System prompt"),
            Message::system("Summary of previous conversation"),
            Message::user("Hello"),
        ];
        let (si, contents) = GeminiClient::convert_messages(&messages);
        assert!(si.is_some());
        let si = si.unwrap();
        assert_eq!(si.parts.len(), 2); // Both system messages combined
        assert_eq!(contents.len(), 1); // Only the user message
    }

    #[test]
    fn test_request_serialization() {
        let request = GenerateContentRequest {
            contents: vec![GeminiContent {
                role: Some("user".to_string()),
                parts: vec![GeminiPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            system_instruction: None,
            tools: vec![],
        };
        let json_str = serde_json::to_string(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["contents"][0]["role"], "user");
        assert!(parsed.get("system_instruction").is_none()); // skip_serializing_if None
        assert!(parsed.get("tools").is_none()); // skip_serializing_if empty
    }

    #[test]
    fn test_streaming_sse_text_chunk() {
        let sse_data = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello"}]
                }
            }]
        });
        let text = sse_data["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap();
        assert_eq!(text, "Hello");
    }

    #[test]
    fn test_streaming_sse_function_call_chunk() {
        let sse_data = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "read_file",
                            "args": {"path": "/tmp/test"}
                        }
                    }]
                }
            }]
        });
        let fc = &sse_data["candidates"][0]["content"]["parts"][0]["functionCall"];
        assert_eq!(fc["name"].as_str().unwrap(), "read_file");
        assert_eq!(fc["args"]["path"].as_str().unwrap(), "/tmp/test");
    }
}
