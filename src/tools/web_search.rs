use super::{Tool, ToolError};
use serde_json::json;

pub struct WebSearchTool;

impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web using DuckDuckGo. Returns search results with titles, URLs, and snippets. Only works when online."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5, max: 10)"
                }
            },
            "required": ["query"]
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'query' argument".to_string()))?;
        let max_results = args["max_results"]
            .as_u64()
            .unwrap_or(5)
            .min(10) as usize;

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to create HTTP client: {}", e)))?;

        // Use DuckDuckGo HTML search (no API key required)
        let url = format!("https://html.duckduckgo.com/html/?q={}", urlencoding(query));

        let response = client
            .get(&url)
            .header("User-Agent", "hermitclaw/0.1.0")
            .send()
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Search failed: {}", e))
            })?;

        if !response.status().is_success() {
            return Err(ToolError::ExecutionFailed(format!(
                "Search returned HTTP {}",
                response.status().as_u16()
            )));
        }

        let body = response
            .text()
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read response: {}", e)))?;

        let results = parse_ddg_results(&body, max_results);

        if results.is_empty() {
            return Ok(format!("No results found for: {}", query));
        }

        let mut output = format!("Search results for: {}\n\n", query);
        for (i, result) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. {}\n   {}\n   {}\n\n",
                i + 1,
                result.title,
                result.url,
                result.snippet
            ));
        }

        Ok(output)
    }
}

struct SearchResult {
    title: String,
    url: String,
    snippet: String,
}

/// Simple URL encoding for the query
fn urlencoding(s: &str) -> String {
    let mut result = String::new();
    for ch in s.chars() {
        match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => result.push(ch),
            ' ' => result.push('+'),
            _ => {
                for byte in ch.to_string().as_bytes() {
                    result.push_str(&format!("%{:02X}", byte));
                }
            }
        }
    }
    result
}

/// Parse DuckDuckGo HTML search results
fn parse_ddg_results(html: &str, max: usize) -> Vec<SearchResult> {
    let mut results = Vec::new();

    // DuckDuckGo HTML results have <a class="result__a" href="..."> for links
    // and <a class="result__snippet" ...> for snippets
    // Simple extraction without a full HTML parser

    let mut pos = 0;
    while results.len() < max {
        // Find result link
        let link_marker = "class=\"result__a\"";
        let link_pos = match html[pos..].find(link_marker) {
            Some(p) => pos + p,
            None => break,
        };

        // Extract href from the <a> tag — search backwards for href="
        let tag_start = html[..link_pos].rfind('<').unwrap_or(link_pos);
        let href = extract_attr(&html[tag_start..], "href").unwrap_or_default();

        // Extract title (text between > and </a>)
        let title_start = match html[link_pos..].find('>') {
            Some(p) => link_pos + p + 1,
            None => {
                pos = link_pos + link_marker.len();
                continue;
            }
        };
        let title_end = match html[title_start..].find("</a>") {
            Some(p) => title_start + p,
            None => {
                pos = title_start;
                continue;
            }
        };
        let title = strip_tags(&html[title_start..title_end]);

        // Extract snippet
        let snippet_marker = "class=\"result__snippet\"";
        let snippet = if let Some(sp) = html[title_end..].find(snippet_marker) {
            let snippet_pos = title_end + sp;
            let snippet_start = match html[snippet_pos..].find('>') {
                Some(p) => snippet_pos + p + 1,
                None => snippet_pos,
            };
            let snippet_end_tag = if let Some(p) = html[snippet_start..].find("</a>") {
                snippet_start + p
            } else if let Some(p) = html[snippet_start..].find("</td>") {
                snippet_start + p
            } else {
                (snippet_start + 500).min(html.len())
            };
            strip_tags(&html[snippet_start..snippet_end_tag])
        } else {
            String::new()
        };

        // Clean up the URL (DuckDuckGo wraps URLs in redirect)
        let clean_url = if let Some(uddg_pos) = href.find("uddg=") {
            let url_start = uddg_pos + 5;
            let url_end = href[url_start..].find('&').map(|p| url_start + p).unwrap_or(href.len());
            url_decode(&href[url_start..url_end])
        } else {
            href
        };

        if !title.is_empty() && !clean_url.is_empty() {
            results.push(SearchResult {
                title: decode_entities(&title),
                url: clean_url,
                snippet: decode_entities(&snippet),
            });
        }

        pos = title_end + 1;
    }

    results
}

fn extract_attr(tag_html: &str, attr: &str) -> Option<String> {
    let pattern = format!("{}=\"", attr);
    let start = tag_html.find(&pattern)? + pattern.len();
    let end = tag_html[start..].find('"')? + start;
    Some(tag_html[start..end].to_string())
}

fn strip_tags(html: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    for ch in html.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(ch);
        }
    }
    result.trim().to_string()
}

fn decode_entities(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

fn url_decode(s: &str) -> String {
    let mut result = String::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(
                &String::from_utf8_lossy(&bytes[i + 1..i + 3]),
                16,
            ) {
                result.push(byte as char);
                i += 3;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_urlencoding() {
        assert_eq!(urlencoding("hello world"), "hello+world");
        assert_eq!(urlencoding("rust lang"), "rust+lang");
        assert_eq!(urlencoding("a&b"), "a%26b");
    }

    #[test]
    fn test_strip_tags() {
        assert_eq!(strip_tags("<b>bold</b> text"), "bold text");
        assert_eq!(strip_tags("no tags"), "no tags");
        assert_eq!(strip_tags("<a href=\"url\">link</a>"), "link");
    }

    #[test]
    fn test_decode_entities() {
        assert_eq!(decode_entities("&amp; &lt;"), "& <");
        assert_eq!(decode_entities("&quot;hi&quot;"), "\"hi\"");
    }

    #[test]
    fn test_url_decode() {
        assert_eq!(url_decode("hello%20world"), "hello world");
        assert_eq!(url_decode("a%26b"), "a&b");
    }

    #[test]
    fn test_extract_attr() {
        assert_eq!(
            extract_attr(r#"<a href="https://example.com" class="x">"#, "href"),
            Some("https://example.com".to_string())
        );
        assert_eq!(extract_attr("<a>", "href"), None);
    }

    #[test]
    fn test_web_search_missing_query() {
        let tool = WebSearchTool;
        let result = tool.execute(json!({}));
        assert!(matches!(result.unwrap_err(), ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_parse_ddg_results_empty_html() {
        let results = parse_ddg_results("", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_ddg_results_no_results() {
        let html = "<html><body>No results</body></html>";
        let results = parse_ddg_results(html, 5);
        assert!(results.is_empty());
    }
}
