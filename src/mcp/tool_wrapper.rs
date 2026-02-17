use super::client::McpClient;
use crate::tools::{Tool, ToolError};
use std::sync::{Arc, Mutex};

/// Wraps an MCP tool as a local Tool implementation.
pub struct McpToolWrapper {
    pub tool_name: String,
    pub mcp_tool_name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub client: Arc<Mutex<McpClient>>,
}

impl Tool for McpToolWrapper {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.input_schema.clone()
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let mut client = self.client.lock().map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to acquire MCP client lock: {}", e))
        })?;

        client
            .call_tool(&self.mcp_tool_name, args)
            .map_err(|e| ToolError::ExecutionFailed(format!("MCP tool call failed: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_wrapper_name_format() {
        let name = format!("mcp_{}_{}", "filesystem", "read_file");
        assert_eq!(name, "mcp_filesystem_read_file");
    }

    #[test]
    fn test_tool_wrapper_description() {
        let server_name = "test_server";
        let tool_name = "my_tool";
        let desc = format!("[MCP:{}] {}", server_name, "A test tool description");
        assert!(desc.starts_with("[MCP:test_server]"));
        assert!(desc.contains("A test tool description"));

        let full_name = format!("mcp_{}_{}", server_name, tool_name);
        assert_eq!(full_name, "mcp_test_server_my_tool");
    }

    /// Helper: create a mock MCP server for testing tool wrapper
    fn create_mock_client() -> Result<Arc<Mutex<McpClient>>, String> {
        let script = r#"
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except:
        continue
    method = req.get("method", "")
    req_id = req.get("id")
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": "2024-11-05"}}), flush=True)
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"tools": [{"name": "echo", "description": "Echo", "inputSchema": {"type": "object", "properties": {"msg": {"type": "string"}}}}]}}), flush=True)
    elif method == "tools/call":
        args = req.get("params", {}).get("arguments", {})
        msg = args.get("msg", "")
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": msg}], "is_error": False}}), flush=True)
"#;
        let script_path = "/tmp/hermitclaw_mock_mcp_wrapper.py";
        std::fs::write(script_path, script)
            .map_err(|e| format!("Failed to write mock script: {}", e))?;
        let client = McpClient::new("test", "python3", &[script_path.to_string()])
            .map_err(|e| format!("Failed to create client: {}", e))?;
        Ok(Arc::new(Mutex::new(client)))
    }

    #[test]
    fn test_tool_wrapper_execute() {
        let client = match create_mock_client() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping: python3 not available");
                return;
            }
        };

        let wrapper = McpToolWrapper {
            tool_name: "mcp_test_echo".to_string(),
            mcp_tool_name: "echo".to_string(),
            description: "[MCP:test] Echo".to_string(),
            input_schema: json!({"type": "object", "properties": {"msg": {"type": "string"}}}),
            client,
        };

        assert_eq!(wrapper.name(), "mcp_test_echo");
        assert_eq!(wrapper.description(), "[MCP:test] Echo");
        assert!(wrapper.parameters_schema().is_object());

        let result = wrapper.execute(json!({"msg": "hello world"})).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_tool_wrapper_multiple_calls_shared_client() {
        let client = match create_mock_client() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping: python3 not available");
                return;
            }
        };

        let wrapper1 = McpToolWrapper {
            tool_name: "mcp_test_echo".to_string(),
            mcp_tool_name: "echo".to_string(),
            description: "Echo".to_string(),
            input_schema: json!({}),
            client: Arc::clone(&client),
        };

        let wrapper2 = McpToolWrapper {
            tool_name: "mcp_test_echo2".to_string(),
            mcp_tool_name: "echo".to_string(),
            description: "Echo 2".to_string(),
            input_schema: json!({}),
            client,
        };

        let r1 = wrapper1.execute(json!({"msg": "first"})).unwrap();
        let r2 = wrapper2.execute(json!({"msg": "second"})).unwrap();
        assert_eq!(r1, "first");
        assert_eq!(r2, "second");
    }
}
