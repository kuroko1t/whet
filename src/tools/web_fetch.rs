use super::{Tool, ToolError};
use serde_json::json;

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

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to create HTTP client: {}", e)))?;

        let response = client
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
            Ok(format!(
                "{}\n\n... (truncated, {} total chars)",
                &text[..max_len],
                text.len()
            ))
        } else {
            Ok(text)
        }
    }
}

/// Simple HTML to text extraction — strips tags and decodes basic entities.
fn html_to_text(html: &str) -> String {
    let mut text = String::new();
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut tag_name = String::new();

    let chars: Vec<char> = html.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        if ch == '<' {
            in_tag = true;
            tag_name.clear();
            i += 1;
            continue;
        }

        if in_tag {
            if ch == '>' {
                in_tag = false;
                let lower = tag_name.to_lowercase();
                if lower == "script" {
                    in_script = true;
                } else if lower == "/script" {
                    in_script = false;
                } else if lower == "style" {
                    in_style = true;
                } else if lower == "/style" {
                    in_style = false;
                } else if lower == "br" || lower == "br/" || lower.starts_with("br ") {
                    text.push('\n');
                } else if lower == "p"
                    || lower == "/p"
                    || lower == "div"
                    || lower == "/div"
                    || lower == "h1"
                    || lower == "/h1"
                    || lower == "h2"
                    || lower == "/h2"
                    || lower == "h3"
                    || lower == "/h3"
                    || lower == "li"
                    || lower == "tr"
                    || lower == "/tr"
                {
                    text.push('\n');
                }
            } else {
                tag_name.push(ch);
            }
            i += 1;
            continue;
        }

        if in_script || in_style {
            i += 1;
            continue;
        }

        // Handle HTML entities
        if ch == '&' {
            let rest: String = chars[i..].iter().take(10).collect();
            if rest.starts_with("&amp;") {
                text.push('&');
                i += 5;
            } else if rest.starts_with("&lt;") {
                text.push('<');
                i += 4;
            } else if rest.starts_with("&gt;") {
                text.push('>');
                i += 4;
            } else if rest.starts_with("&quot;") {
                text.push('"');
                i += 6;
            } else if rest.starts_with("&#39;") || rest.starts_with("&apos;") {
                text.push('\'');
                i += if rest.starts_with("&#39;") { 5 } else { 6 };
            } else if rest.starts_with("&nbsp;") {
                text.push(' ');
                i += 6;
            } else {
                text.push('&');
                i += 1;
            }
            continue;
        }

        text.push(ch);
        i += 1;
    }

    // Clean up multiple blank lines
    let mut result = String::new();
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

    result.trim().to_string()
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
        assert!(matches!(result.unwrap_err(), ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_web_fetch_missing_url() {
        let tool = WebFetchTool;
        let result = tool.execute(json!({}));
        assert!(matches!(result.unwrap_err(), ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_web_fetch_connection_refused() {
        let tool = WebFetchTool;
        // Use a port that should not be listening
        let result = tool.execute(json!({"url": "http://127.0.0.1:19999/"}));
        assert!(matches!(result.unwrap_err(), ToolError::ExecutionFailed(_)));
    }
}
