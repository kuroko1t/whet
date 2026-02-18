use super::{Tool, ToolError};
use serde_json::json;
use std::sync::OnceLock;

/// Shared HTTP client for web fetch operations — created once, reused across all calls.
fn http_client() -> &'static reqwest::blocking::Client {
    static CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new())
    })
}

pub struct WebFetchTool;

impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch the contents of a URL. Returns the text content of the page. Only works when online."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                }
            },
            "required": ["url"]
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let url = args["url"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'url' argument".to_string()))?;

        // Validate URL
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(ToolError::InvalidArguments(
                "URL must start with http:// or https://".to_string(),
            ));
        }

        let response = http_client()
            .get(url)
            .header("User-Agent", "hermitclaw/0.1.0")
            .send()
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to fetch URL '{}': {}", url, e))
            })?;

        let status = response.status();
        if !status.is_success() {
            return Err(ToolError::ExecutionFailed(format!(
                "HTTP error {}: {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("Unknown")
            )));
        }

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let body = response
            .text()
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read response: {}", e)))?;

        // Simple HTML to text conversion
        let text = if content_type.contains("text/html") {
            html_to_text(&body)
        } else {
            body
        };

        // Truncate if too large
        let max_len = 50_000;
        if text.len() > max_len {
            let mut end = max_len;
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            Ok(format!(
                "{}\n\n... (truncated, {} total chars)",
                &text[..end],
                text.len()
            ))
        } else {
            Ok(text)
        }
    }
}

/// Simple HTML to text extraction — strips tags and decodes basic entities.
/// Uses char_indices() for safe UTF-8 handling.
fn html_to_text(html: &str) -> String {
    let mut text = String::with_capacity(html.len() / 3);
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut tag_name = String::with_capacity(16);

    let mut chars = html.char_indices().peekable();

    while let Some((i, ch)) = chars.next() {
        if ch == '<' {
            in_tag = true;
            tag_name.clear();
            continue;
        }

        if in_tag {
            if ch == '>' {
                in_tag = false;
                let lower = tag_name.to_lowercase();
                match lower.as_str() {
                    "script" => in_script = true,
                    "/script" => in_script = false,
                    "style" => in_style = true,
                    "/style" => in_style = false,
                    "br" | "br/" | "p" | "/p" | "div" | "/div" | "h1" | "/h1" | "h2" | "/h2"
                    | "h3" | "/h3" | "li" | "tr" | "/tr" => {
                        text.push('\n');
                    }
                    s if s.starts_with("br ") => {
                        text.push('\n');
                    }
                    _ => {}
                }
            } else {
                tag_name.push(ch);
            }
            continue;
        }

        if in_script || in_style {
            continue;
        }

        // Handle HTML entities
        if ch == '&' {
            let remaining = &html[i..];
            if remaining.starts_with("&amp;") {
                text.push('&');
                // Skip the remaining 4 chars of "&amp;"
                for _ in 0..4 { chars.next(); }
            } else if remaining.starts_with("&lt;") {
                text.push('<');
                for _ in 0..3 { chars.next(); }
            } else if remaining.starts_with("&gt;") {
                text.push('>');
                for _ in 0..3 { chars.next(); }
            } else if remaining.starts_with("&quot;") {
                text.push('"');
                for _ in 0..5 { chars.next(); }
            } else if remaining.starts_with("&#39;") {
                text.push('\'');
                for _ in 0..4 { chars.next(); }
            } else if remaining.starts_with("&apos;") {
                text.push('\'');
                for _ in 0..5 { chars.next(); }
            } else if remaining.starts_with("&nbsp;") {
                text.push(' ');
                for _ in 0..5 { chars.next(); }
            } else {
                text.push('&');
            }
            continue;
        }

        text.push(ch);
    }

    // Clean up multiple blank lines — single pass
    let mut result = String::with_capacity(text.len());
    let mut prev_blank = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !prev_blank {
                result.push('\n');
                prev_blank = true;
            }
        } else {
            result.push_str(trimmed);
            result.push('\n');
            prev_blank = false;
        }
    }

    result.truncate(result.trim_end().len());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_to_text_basic() {
        let html = "<html><body><p>Hello <b>world</b></p></body></html>";
        let text = html_to_text(html);
        assert!(text.contains("Hello world"));
    }

    #[test]
    fn test_html_to_text_strips_script() {
        let html = "<p>Before</p><script>var x = 1;</script><p>After</p>";
        let text = html_to_text(html);
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
        assert!(!text.contains("var x"));
    }

    #[test]
    fn test_html_to_text_strips_style() {
        let html = "<p>Content</p><style>.foo { color: red; }</style>";
        let text = html_to_text(html);
        assert!(text.contains("Content"));
        assert!(!text.contains("color"));
    }

    #[test]
    fn test_html_to_text_entities() {
        let html = "&amp; &lt;tag&gt; &quot;hello&quot;";
        let text = html_to_text(html);
        assert_eq!(text, "& <tag> \"hello\"");
    }

    #[test]
    fn test_html_to_text_line_breaks() {
        let html = "<p>Para 1</p><p>Para 2</p>";
        let text = html_to_text(html);
        assert!(text.contains("Para 1"));
        assert!(text.contains("Para 2"));
    }

    #[test]
    fn test_web_fetch_invalid_url() {
        let tool = WebFetchTool;
        let result = tool.execute(json!({"url": "not-a-url"}));
        assert!(matches!(
            result.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[test]
    fn test_web_fetch_missing_url() {
        let tool = WebFetchTool;
        let result = tool.execute(json!({}));
        assert!(matches!(
            result.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[test]
    fn test_web_fetch_connection_refused() {
        let tool = WebFetchTool;
        // Use a port that should not be listening
        let result = tool.execute(json!({"url": "http://127.0.0.1:19999/"}));
        assert!(matches!(result.unwrap_err(), ToolError::ExecutionFailed(_)));
    }
}
