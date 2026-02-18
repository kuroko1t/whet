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
