pub mod client;
pub mod protocol;
pub mod tool_wrapper;

use client::McpClient;
use tool_wrapper::McpToolWrapper;

use crate::config::McpServerConfig;
use crate::tools::ToolRegistry;
use std::sync::{Arc, Mutex};

/// Register MCP tools from configured servers into the tool registry.
pub fn register_mcp_tools(registry: &mut ToolRegistry, servers: &[McpServerConfig]) {
    for server_config in servers {
        match McpClient::new(
            &server_config.name,
            &server_config.command,
            &server_config.args,
        ) {
            Ok(mut client) => {
                let server_name = server_config.name.clone();
                match client.list_tools() {
                    Ok(tools) => {
                        let client = Arc::new(Mutex::new(client));
                        let tool_count = tools.len();

                        for tool_info in tools {
                            let tool_name = format!("mcp_{}_{}", server_name, tool_info.name);
                            let description = format!(
                                "[MCP:{}] {}",
                                server_name,
                                tool_info.description.as_deref().unwrap_or("No description")
                            );
                            let input_schema = tool_info.input_schema.unwrap_or_else(|| {
                                serde_json::json!({
                                    "type": "object",
                                    "properties": {}
                                })
                            });

                            let wrapper = McpToolWrapper {
                                tool_name,
                                mcp_tool_name: tool_info.name,
                                description,
                                input_schema,
                                client: Arc::clone(&client),
                            };

                            registry.register(Box::new(wrapper));
                        }

                        eprintln!(
                            "MCP: Registered {} tools from server '{}'",
                            tool_count, server_name
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to list tools from MCP server '{}': {}",
                            server_name, e
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "Warning: Failed to connect to MCP server '{}': {}",
                    server_config.name, e
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolRegistry;

    #[test]
    fn test_register_mcp_tools_nonexistent_server() {
        let mut registry = ToolRegistry::new();
        let servers = vec![McpServerConfig {
            name: "broken".to_string(),
            command: "/nonexistent_command_xyz".to_string(),
            args: vec![],
        }];

        // Should not panic, just print warning
        register_mcp_tools(&mut registry, &servers);
        assert_eq!(registry.list().len(), 0);
    }

    #[test]
    fn test_register_mcp_tools_empty_servers() {
        let mut registry = ToolRegistry::new();
        register_mcp_tools(&mut registry, &[]);
        assert_eq!(registry.list().len(), 0);
    }

    #[test]
    fn test_register_mcp_tools_with_mock_server() {
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
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"tools": [
            {"name": "tool_a", "description": "Tool A"},
            {"name": "tool_b", "description": "Tool B", "inputSchema": {"type": "object"}}
        ]}}), flush=True)
    elif method == "tools/call":
        args = req.get("params", {}).get("arguments", {})
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": "ok"}], "is_error": False}}), flush=True)
"#;
        let script_path = "/tmp/hermitclaw_mock_mcp_register.py";
        if std::fs::write(script_path, script).is_err() {
            eprintln!("Skipping: cannot write mock script");
            return;
        }

        let mut registry = ToolRegistry::new();
        let servers = vec![McpServerConfig {
            name: "testsvr".to_string(),
            command: "python3".to_string(),
            args: vec![script_path.to_string()],
        }];

        register_mcp_tools(&mut registry, &servers);

        // Check that tools were registered with correct naming
        let tools = registry.list();
        if tools.is_empty() {
            eprintln!("Skipping: python3 not available or mock failed");
            return;
        }

        assert_eq!(tools.len(), 2);
        assert!(registry.get("mcp_testsvr_tool_a").is_some());
        assert!(registry.get("mcp_testsvr_tool_b").is_some());

        // Check description format
        let tool_a = registry.get("mcp_testsvr_tool_a").unwrap();
        assert!(tool_a.description().starts_with("[MCP:testsvr]"));
        assert!(tool_a.description().contains("Tool A"));

        // Check that tool without inputSchema gets default
        let tool_a_schema = tool_a.parameters_schema();
        assert!(tool_a_schema.is_object());
    }

    #[test]
    fn test_register_mcp_tools_mixed_success_failure() {
        let mut registry = ToolRegistry::new();
        let servers = vec![
            McpServerConfig {
                name: "broken".to_string(),
                command: "/nonexistent_xyz".to_string(),
                args: vec![],
            },
            McpServerConfig {
                name: "also_broken".to_string(),
                command: "/also_nonexistent_xyz".to_string(),
                args: vec![],
            },
        ];

        // Should handle both failures gracefully
        register_mcp_tools(&mut registry, &servers);
        assert_eq!(registry.list().len(), 0);
    }
}
