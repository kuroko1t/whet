use super::protocol::{
    JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpToolCallResult, McpToolInfo,
};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

pub struct McpClient {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
    writer: std::process::ChildStdin,
    next_id: u64,
    pub server_name: String,
}

#[derive(Debug)]
pub enum McpError {
    SpawnFailed(String),
    IoError(String),
    ProtocolError(String),
    ServerError(String),
}

impl std::fmt::Display for McpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpError::SpawnFailed(msg) => write!(f, "MCP spawn failed: {}", msg),
            McpError::IoError(msg) => write!(f, "MCP I/O error: {}", msg),
            McpError::ProtocolError(msg) => write!(f, "MCP protocol error: {}", msg),
            McpError::ServerError(msg) => write!(f, "MCP server error: {}", msg),
        }
    }
}

impl McpClient {
    /// Spawn an MCP server process and perform the initialize handshake.
    pub fn new(name: &str, command: &str, args: &[String]) -> Result<Self, McpError> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| McpError::SpawnFailed(format!("Failed to spawn '{}': {}", command, e)))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::SpawnFailed("Failed to capture stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::SpawnFailed("Failed to capture stdout".to_string()))?;

        let reader = BufReader::new(stdout);

        let mut client = Self {
            child,
            reader,
            writer: stdin,
            next_id: 1,
            server_name: name.to_string(),
        };

        // Perform initialize handshake
        client.initialize()?;

        Ok(client)
    }

    fn initialize(&mut self) -> Result<(), McpError> {
        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "hermitclaw",
                "version": "0.1.0"
            }
        });

        let response = self.send_request("initialize", Some(params))?;

        if let Some(error) = response.error {
            return Err(McpError::ServerError(format!(
                "Initialize failed: {}",
                error.message
            )));
        }

        // Send initialized notification
        self.send_notification("notifications/initialized", None)?;

        Ok(())
    }

    fn send_request(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<JsonRpcResponse, McpError> {
        let id = self.next_id;
        self.next_id += 1;

        let request = JsonRpcRequest::new(id, method, params);
        let mut json_str =
            serde_json::to_string(&request).map_err(|e| McpError::IoError(e.to_string()))?;
        json_str.push('\n');

        self.writer
            .write_all(json_str.as_bytes())
            .map_err(|e| McpError::IoError(format!("Write failed: {}", e)))?;
        self.writer
            .flush()
            .map_err(|e| McpError::IoError(format!("Flush failed: {}", e)))?;

        // Read response lines until we get one with matching id
        loop {
            let mut line = String::new();
            let bytes_read = self
                .reader
                .read_line(&mut line)
                .map_err(|e| McpError::IoError(format!("Read failed: {}", e)))?;

            if bytes_read == 0 {
                return Err(McpError::IoError(
                    "Server closed connection".to_string(),
                ));
            }

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Try to parse as a JSON-RPC response
            if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(line) {
                if response.id == Some(id) {
                    return Ok(response);
                }
                // Skip responses with different ids (notifications, etc.)
            }
            // Skip non-response lines (notifications from server)
        }
    }

    fn send_notification(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), McpError> {
        let notification = JsonRpcNotification::new(method, params);
        let mut json_str =
            serde_json::to_string(&notification).map_err(|e| McpError::IoError(e.to_string()))?;
        json_str.push('\n');

        self.writer
            .write_all(json_str.as_bytes())
            .map_err(|e| McpError::IoError(format!("Write failed: {}", e)))?;
        self.writer
            .flush()
            .map_err(|e| McpError::IoError(format!("Flush failed: {}", e)))?;

        Ok(())
    }

    /// Discover available tools from the MCP server.
    pub fn list_tools(&mut self) -> Result<Vec<McpToolInfo>, McpError> {
        let response = self.send_request("tools/list", None)?;

        if let Some(error) = response.error {
            return Err(McpError::ServerError(format!(
                "tools/list failed: {}",
                error.message
            )));
        }

        let result = response
            .result
            .ok_or_else(|| McpError::ProtocolError("No result in tools/list response".to_string()))?;

        let tools_value = result
            .get("tools")
            .ok_or_else(|| McpError::ProtocolError("No 'tools' field in result".to_string()))?;

        let tools: Vec<McpToolInfo> = serde_json::from_value(tools_value.clone())
            .map_err(|e| McpError::ProtocolError(format!("Failed to parse tools: {}", e)))?;

        Ok(tools)
    }

    /// Call a tool on the MCP server.
    pub fn call_tool(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, McpError> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let response = self.send_request("tools/call", Some(params))?;

        if let Some(error) = response.error {
            return Err(McpError::ServerError(format!(
                "tools/call failed: {}",
                error.message
            )));
        }

        let result = response
            .result
            .ok_or_else(|| McpError::ProtocolError("No result in tools/call response".to_string()))?;

        let call_result: McpToolCallResult = serde_json::from_value(result)
            .map_err(|e| McpError::ProtocolError(format!("Failed to parse call result: {}", e)))?;

        let text = call_result
            .content
            .iter()
            .filter_map(|c| c.text.as_deref())
            .collect::<Vec<_>>()
            .join("\n");

        if call_result.is_error {
            Err(McpError::ServerError(text))
        } else {
            Ok(text)
        }
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_error_display() {
        let err = McpError::SpawnFailed("test".to_string());
        assert!(err.to_string().contains("spawn failed"));

        let err = McpError::IoError("test".to_string());
        assert!(err.to_string().contains("I/O error"));

        let err = McpError::ProtocolError("test".to_string());
        assert!(err.to_string().contains("protocol error"));

        let err = McpError::ServerError("test".to_string());
        assert!(err.to_string().contains("server error"));
    }

    #[test]
    fn test_spawn_nonexistent_command() {
        let result = McpClient::new("test", "/nonexistent_command_xyz", &[]);
        assert!(result.is_err());
        match result {
            Err(McpError::SpawnFailed(msg)) => assert!(msg.contains("/nonexistent_command_xyz")),
            Err(other) => panic!("Expected SpawnFailed, got: {}", other),
            Ok(_) => panic!("Expected error"),
        }
    }

    /// Helper to create a mock MCP server script that handles initialize + tools/list + tools/call
    fn mock_mcp_script() -> String {
        // This script reads JSON-RPC requests from stdin and responds appropriately.
        // It handles: initialize, tools/list, tools/call
        r#"
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
        resp = {"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": "2024-11-05", "capabilities": {}}}
        print(json.dumps(resp), flush=True)
    elif method == "notifications/initialized":
        pass  # notification, no response needed
    elif method == "tools/list":
        resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": [
            {"name": "echo_tool", "description": "Echo back the input", "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}
        ]}}
        print(json.dumps(resp), flush=True)
    elif method == "tools/call":
        name = req.get("params", {}).get("name", "")
        args = req.get("params", {}).get("arguments", {})
        if name == "echo_tool":
            msg = args.get("message", "")
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": "echo: " + msg}], "is_error": False}}
        else:
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": "unknown tool"}], "is_error": True}}
        print(json.dumps(resp), flush=True)
    else:
        resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}
        print(json.dumps(resp), flush=True)
"#.to_string()
    }

    fn create_mock_mcp_server() -> Result<McpClient, McpError> {
        let script = mock_mcp_script();
        let script_path = "/tmp/hermitclaw_mock_mcp.py";
        std::fs::write(script_path, &script).map_err(|e| {
            McpError::SpawnFailed(format!("Failed to write mock script: {}", e))
        })?;
        McpClient::new("mock_server", "python3", &[script_path.to_string()])
    }

    #[test]
    fn test_mcp_initialize_handshake() {
        let client = create_mock_mcp_server();
        match client {
            Ok(_) => {
                // Successfully initialized - the handshake works
            }
            Err(McpError::SpawnFailed(msg)) if msg.contains("python3") => {
                // python3 not available, skip test
                eprintln!("Skipping: python3 not available");
                return;
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_mcp_list_tools() {
        let mut client = match create_mock_mcp_server() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping: mock MCP server unavailable");
                return;
            }
        };

        let tools = client.list_tools().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo_tool");
        assert_eq!(tools[0].description, Some("Echo back the input".to_string()));
        assert!(tools[0].input_schema.is_some());
    }

    #[test]
    fn test_mcp_call_tool() {
        let mut client = match create_mock_mcp_server() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping: mock MCP server unavailable");
                return;
            }
        };

        let result = client
            .call_tool("echo_tool", serde_json::json!({"message": "hello"}))
            .unwrap();
        assert_eq!(result, "echo: hello");
    }

    #[test]
    fn test_mcp_call_unknown_tool() {
        let mut client = match create_mock_mcp_server() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping: mock MCP server unavailable");
                return;
            }
        };

        let result = client.call_tool("nonexistent", serde_json::json!({}));
        assert!(result.is_err());
        match result.unwrap_err() {
            McpError::ServerError(msg) => assert!(msg.contains("unknown tool")),
            other => panic!("Expected ServerError, got: {:?}", other),
        }
    }

    #[test]
    fn test_mcp_multiple_tool_calls() {
        let mut client = match create_mock_mcp_server() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping: mock MCP server unavailable");
                return;
            }
        };

        let r1 = client
            .call_tool("echo_tool", serde_json::json!({"message": "first"}))
            .unwrap();
        assert_eq!(r1, "echo: first");

        let r2 = client
            .call_tool("echo_tool", serde_json::json!({"message": "second"}))
            .unwrap();
        assert_eq!(r2, "echo: second");
    }

    #[test]
    fn test_mcp_client_drop_kills_process() {
        let client = match create_mock_mcp_server() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping: mock MCP server unavailable");
                return;
            }
        };

        // Drop the client - should kill the child process
        drop(client);
        // If we get here without hanging, the Drop impl works
    }

    #[test]
    fn test_mcp_server_closes_connection() {
        // Use a command that exits immediately
        let result = McpClient::new("dying", "echo", &["{}".to_string()]);
        // Should fail during initialization since echo just prints and exits
        assert!(result.is_err());
    }
}
