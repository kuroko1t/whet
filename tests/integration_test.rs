use whet::llm::ollama::OllamaClient;
use whet::llm::{LlmProvider, Message};

#[test]
#[ignore] // Requires Ollama to be running
fn test_ollama_chat_simple() {
    let client = OllamaClient::new("http://localhost:11434", "qwen2.5:7b");
    let messages = vec![Message::user("Say hello in one word.")];
    let result = client.chat(&messages, &[]);
    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.content.is_some());
}

#[test]
fn test_ollama_connection_error() {
    let client = OllamaClient::new("http://localhost:99999", "qwen2.5:7b");
    let messages = vec![Message::user("Hello")];
    let result = client.chat(&messages, &[]);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// MCP E2E tests — requires npx + @modelcontextprotocol/server-filesystem
// ---------------------------------------------------------------------------
mod mcp_e2e {
    use std::process::Command;
    use whet::config::McpServerConfig;
    use whet::mcp;
    use whet::tools::ToolRegistry;

    /// Check if npx is available
    fn has_npx() -> bool {
        Command::new("npx")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Get canonicalized path from TempDir.
    /// On macOS, /var → /private/var symlink causes MCP server path mismatch
    /// unless we canonicalize first.
    fn canonical_temp_dir(dir: &tempfile::TempDir) -> String {
        std::fs::canonicalize(dir.path())
            .unwrap_or_else(|_| dir.path().to_path_buf())
            .to_string_lossy()
            .to_string()
    }

    #[test]
    fn test_mcp_filesystem_server_connect_and_list_tools() {
        if !has_npx() {
            eprintln!("Skipping: npx not available");
            return;
        }

        let test_dir = tempfile::TempDir::new().unwrap();
        let test_dir_path = canonical_temp_dir(&test_dir);

        let mut registry = ToolRegistry::new();
        let servers = vec![McpServerConfig {
            name: "fs".to_string(),
            command: "npx".to_string(),
            args: vec![
                "-y".to_string(),
                "@modelcontextprotocol/server-filesystem".to_string(),
                test_dir_path,
            ],
        }];

        mcp::register_mcp_tools(&mut registry, &servers);

        let tools = registry.list();
        assert!(
            !tools.is_empty(),
            "MCP filesystem server should register at least one tool"
        );

        // Check essential tools are registered with correct naming
        let tool_names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(
            tool_names.contains(&"mcp_fs_read_file"),
            "Should have mcp_fs_read_file, got: {:?}",
            tool_names
        );
        assert!(
            tool_names.contains(&"mcp_fs_write_file"),
            "Should have mcp_fs_write_file, got: {:?}",
            tool_names
        );
        assert!(
            tool_names.contains(&"mcp_fs_list_directory"),
            "Should have mcp_fs_list_directory, got: {:?}",
            tool_names
        );

        // Verify description format
        let read_tool = registry.get("mcp_fs_read_file").unwrap();
        assert!(
            read_tool.description().starts_with("[MCP:fs]"),
            "Description should start with [MCP:fs], got: {}",
            read_tool.description()
        );

        // Verify schema is valid
        let schema = read_tool.parameters_schema();
        assert!(schema.is_object(), "Schema should be an object");

        eprintln!(
            "MCP filesystem server registered {} tools: {:?}",
            tools.len(),
            tool_names
        );
    }

    #[test]
    fn test_mcp_filesystem_write_and_read() {
        if !has_npx() {
            eprintln!("Skipping: npx not available");
            return;
        }

        let test_dir = tempfile::TempDir::new().unwrap();
        let test_dir_path = canonical_temp_dir(&test_dir);

        let mut registry = ToolRegistry::new();
        let servers = vec![McpServerConfig {
            name: "fs".to_string(),
            command: "npx".to_string(),
            args: vec![
                "-y".to_string(),
                "@modelcontextprotocol/server-filesystem".to_string(),
                test_dir_path.clone(),
            ],
        }];

        mcp::register_mcp_tools(&mut registry, &servers);

        if registry.list().is_empty() {
            eprintln!("Skipping: MCP server registration failed");
            return;
        }

        // Write a file via MCP
        let test_file = format!("{}/mcp_e2e_test.txt", test_dir_path);
        let write_tool = registry.get("mcp_fs_write_file").unwrap();
        let write_result = write_tool.execute(serde_json::json!({
            "path": test_file,
            "content": "Hello from whet MCP E2E test!\nLine 2"
        }));
        assert!(
            write_result.is_ok(),
            "write_file should succeed: {:?}",
            write_result.err()
        );
        eprintln!("Write result: {}", write_result.unwrap());

        // Verify the file exists on disk
        let disk_content = std::fs::read_to_string(&test_file).unwrap();
        assert_eq!(disk_content, "Hello from whet MCP E2E test!\nLine 2");

        // Read the file back via MCP
        let read_tool = registry.get("mcp_fs_read_file").unwrap();
        let read_result = read_tool.execute(serde_json::json!({
            "path": test_file
        }));
        assert!(
            read_result.is_ok(),
            "read_file should succeed: {:?}",
            read_result.err()
        );
        let content = read_result.unwrap();
        assert!(
            content.contains("Hello from whet MCP E2E test!"),
            "Read content should contain written text, got: {}",
            content
        );
        eprintln!("Read result: {}", content);
    }

    #[test]
    fn test_mcp_filesystem_list_directory() {
        if !has_npx() {
            eprintln!("Skipping: npx not available");
            return;
        }

        let test_dir = tempfile::TempDir::new().unwrap();
        let test_dir_path = canonical_temp_dir(&test_dir);

        // Create test files
        std::fs::write(format!("{}/file_a.txt", test_dir_path), "content a").unwrap();
        std::fs::write(format!("{}/file_b.txt", test_dir_path), "content b").unwrap();

        let mut registry = ToolRegistry::new();
        let servers = vec![McpServerConfig {
            name: "fs".to_string(),
            command: "npx".to_string(),
            args: vec![
                "-y".to_string(),
                "@modelcontextprotocol/server-filesystem".to_string(),
                test_dir_path.clone(),
            ],
        }];

        mcp::register_mcp_tools(&mut registry, &servers);

        if registry.get("mcp_fs_list_directory").is_none() {
            eprintln!("Skipping: list_directory tool not registered");
            return;
        }

        let list_tool = registry.get("mcp_fs_list_directory").unwrap();
        let result = list_tool.execute(serde_json::json!({
            "path": test_dir_path
        }));
        assert!(
            result.is_ok(),
            "list_directory should succeed: {:?}",
            result.err()
        );
        let listing = result.unwrap();
        assert!(
            listing.contains("file_a.txt"),
            "Listing should contain file_a.txt, got: {}",
            listing
        );
        assert!(
            listing.contains("file_b.txt"),
            "Listing should contain file_b.txt, got: {}",
            listing
        );
        eprintln!("Directory listing: {}", listing);
    }

    #[test]
    fn test_mcp_filesystem_security_blocks_outside_dir() {
        if !has_npx() {
            eprintln!("Skipping: npx not available");
            return;
        }

        let test_dir = tempfile::TempDir::new().unwrap();
        let test_dir_path = canonical_temp_dir(&test_dir);

        let mut registry = ToolRegistry::new();
        let servers = vec![McpServerConfig {
            name: "fs".to_string(),
            command: "npx".to_string(),
            args: vec![
                "-y".to_string(),
                "@modelcontextprotocol/server-filesystem".to_string(),
                test_dir_path,
            ],
        }];

        mcp::register_mcp_tools(&mut registry, &servers);

        if registry.get("mcp_fs_read_file").is_none() {
            eprintln!("Skipping: read_file tool not registered");
            return;
        }

        // Try to read outside allowed directory — server should block this
        let read_tool = registry.get("mcp_fs_read_file").unwrap();
        let result = read_tool.execute(serde_json::json!({
            "path": "/etc/passwd"
        }));
        assert!(
            result.is_err(),
            "Reading /etc/passwd should be blocked by MCP server"
        );
        eprintln!(
            "Security test passed: reading /etc/passwd blocked with: {:?}",
            result.err()
        );
    }
}
