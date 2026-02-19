use serde::{Deserialize, Serialize};

/// JSON-RPC 2.0 request
#[derive(Serialize, Debug)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    pub fn new(id: u64, method: &str, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        }
    }
}

/// JSON-RPC 2.0 notification (no id)
#[derive(Serialize, Debug)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcNotification {
    pub fn new(method: &str, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
        }
    }
}

/// JSON-RPC 2.0 response
#[derive(Deserialize, Debug)]
pub struct JsonRpcResponse {
    #[allow(dead_code)]
    pub jsonrpc: String,
    #[allow(dead_code)]
    pub id: Option<u64>,
    pub result: Option<serde_json::Value>,
    pub error: Option<JsonRpcError>,
}

#[derive(Deserialize, Debug)]
pub struct JsonRpcError {
    #[allow(dead_code)]
    pub code: i64,
    pub message: String,
}

/// MCP tool information from tools/list
#[derive(Deserialize, Debug, Clone)]
pub struct McpToolInfo {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    #[serde(default)]
    pub input_schema: Option<serde_json::Value>,
}

/// MCP tool call result content
#[derive(Deserialize, Debug)]
pub struct McpToolCallResult {
    pub content: Vec<McpContent>,
    #[serde(default)]
    #[serde(alias = "isError")]
    pub is_error: bool,
}

#[derive(Deserialize, Debug)]
pub struct McpContent {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub content_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_request_serialization() {
        let req = JsonRpcRequest::new(1, "initialize", Some(json!({"capabilities": {}})));
        let json_str = serde_json::to_string(&req).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["id"], 1);
        assert_eq!(parsed["method"], "initialize");
    }

    #[test]
    fn test_request_no_params() {
        let req = JsonRpcRequest::new(2, "tools/list", None);
        let json_str = serde_json::to_string(&req).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(parsed.get("params").is_none());
    }

    #[test]
    fn test_notification_serialization() {
        let notif = JsonRpcNotification::new("notifications/initialized", None);
        let json_str = serde_json::to_string(&notif).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["method"], "notifications/initialized");
        assert!(parsed.get("id").is_none());
    }

    #[test]
    fn test_response_deserialization_success() {
        let json_val = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"protocolVersion": "2024-11-05"}
        });
        let resp: JsonRpcResponse = serde_json::from_value(json_val).unwrap();
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_response_deserialization_error() {
        let json_val = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32600, "message": "Invalid request"}
        });
        let resp: JsonRpcResponse = serde_json::from_value(json_val).unwrap();
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32600);
        assert_eq!(err.message, "Invalid request");
    }

    #[test]
    fn test_tool_info_deserialization() {
        let json_val = json!({
            "name": "read_file",
            "description": "Read a file",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        });
        let info: McpToolInfo = serde_json::from_value(json_val).unwrap();
        assert_eq!(info.name, "read_file");
        assert_eq!(info.description, Some("Read a file".to_string()));
        assert!(info.input_schema.is_some());
    }

    #[test]
    fn test_tool_info_minimal() {
        let json_val = json!({"name": "test_tool"});
        let info: McpToolInfo = serde_json::from_value(json_val).unwrap();
        assert_eq!(info.name, "test_tool");
        assert!(info.description.is_none());
        assert!(info.input_schema.is_none());
    }

    #[test]
    fn test_tool_call_result_deserialization() {
        let json_val = json!({
            "content": [{"type": "text", "text": "file contents here"}],
            "is_error": false
        });
        let result: McpToolCallResult = serde_json::from_value(json_val).unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
        assert_eq!(
            result.content[0].text,
            Some("file contents here".to_string())
        );
    }

    #[test]
    fn test_tool_call_result_error() {
        let json_val = json!({
            "content": [{"type": "text", "text": "something went wrong"}],
            "is_error": true
        });
        let result: McpToolCallResult = serde_json::from_value(json_val).unwrap();
        assert!(result.is_error);
    }

    #[test]
    fn test_request_large_id() {
        let req = JsonRpcRequest::new(u64::MAX, "test", None);
        let json_str = serde_json::to_string(&req).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["id"], u64::MAX);
    }

    #[test]
    fn test_request_with_nested_params() {
        let params = json!({
            "name": "read_file",
            "arguments": {
                "path": "/tmp/test.txt",
                "options": {"encoding": "utf-8"}
            }
        });
        let req = JsonRpcRequest::new(1, "tools/call", Some(params.clone()));
        let json_str = serde_json::to_string(&req).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(
            parsed["params"]["arguments"]["options"]["encoding"],
            "utf-8"
        );
    }

    #[test]
    fn test_notification_with_params() {
        let notif = JsonRpcNotification::new(
            "notifications/progress",
            Some(json!({"token": "abc", "progress": 50})),
        );
        let json_str = serde_json::to_string(&notif).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["params"]["progress"], 50);
        assert!(parsed.get("id").is_none()); // notifications don't have ids
    }

    #[test]
    fn test_response_with_null_id() {
        let json_val = json!({
            "jsonrpc": "2.0",
            "id": null,
            "result": {}
        });
        let resp: JsonRpcResponse = serde_json::from_value(json_val).unwrap();
        assert!(resp.id.is_none());
    }

    #[test]
    fn test_tool_info_with_empty_description() {
        let json_val = json!({
            "name": "test",
            "description": ""
        });
        let info: McpToolInfo = serde_json::from_value(json_val).unwrap();
        assert_eq!(info.description, Some("".to_string()));
    }

    #[test]
    fn test_tool_call_result_multiple_content() {
        let json_val = json!({
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
                {"type": "image", "text": null}
            ],
            "is_error": false
        });
        let result: McpToolCallResult = serde_json::from_value(json_val).unwrap();
        assert_eq!(result.content.len(), 3);
        assert_eq!(result.content[0].text, Some("first".to_string()));
        assert_eq!(result.content[1].text, Some("second".to_string()));
        assert!(result.content[2].text.is_none());
    }

    #[test]
    fn test_tool_call_result_default_is_error() {
        // When is_error is absent, should default to false
        let json_val = json!({
            "content": [{"type": "text", "text": "ok"}]
        });
        let result: McpToolCallResult = serde_json::from_value(json_val).unwrap();
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_call_result_camel_case_is_error() {
        // Real MCP servers use camelCase "isError"
        let json_val = json!({
            "content": [{"type": "text", "text": "access denied"}],
            "isError": true
        });
        let result: McpToolCallResult = serde_json::from_value(json_val).unwrap();
        assert!(result.is_error);
    }

    #[test]
    fn test_tool_info_list_parsing() {
        // Simulate tools/list response parsing
        let tools_json = json!({
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}
                },
                {
                    "name": "write_file",
                    "description": "Write a file",
                    "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}
                }
            ]
        });

        let tools: Vec<McpToolInfo> = serde_json::from_value(tools_json["tools"].clone()).unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "read_file");
        assert_eq!(tools[1].name, "write_file");
    }

    #[test]
    fn test_json_rpc_error_negative_code() {
        let json_val = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"}
        });
        let resp: JsonRpcResponse = serde_json::from_value(json_val).unwrap();
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
        assert_eq!(err.message, "Method not found");
    }
}
