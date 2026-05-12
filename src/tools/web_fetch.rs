use super::{Tool, ToolError};
use serde_json::json;
use std::net::{IpAddr, ToSocketAddrs};
use std::sync::OnceLock;

/// Hard cap on the markdown payload we return to the agent (and feed
/// into the extraction LLM). 32 KB ≈ 6–8 K tokens depending on the
/// language. Tight under q3's 8 K context window but necessary
/// because chrome-heavy sites (Wikipedia, GitHub) easily burn 16 KB
/// on navigation/sidebars before reaching the article body.
pub const MAX_MARKDOWN_CHARS: usize = 32_000;

/// Shared HTTP client. Configured with:
///   - 30 s overall timeout.
///   - Limited redirects (5) with per-hop SSRF re-check via the
///     redirect policy callback.
///   - Generic User-Agent and no automatic Referer.
fn http_client() -> &'static reqwest::blocking::Client {
    static CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt.previous().len() >= 5 {
                    return attempt.error("too many redirects");
                }
                // Re-validate every redirect target against the SSRF
                // rules; same-policy as the initial URL gate.
                match validate_url(attempt.url().as_str()) {
                    Ok(()) => attempt.follow(),
                    Err(e) => attempt.error(format!("redirect blocked: {}", e)),
                }
            }))
            .user_agent("whet")
            .referer(false)
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
        "Fetch a URL and return its content as markdown. When the optional `prompt` argument is provided, the page is additionally passed through a focused-extraction step (an internal LLM call) and the answer to your prompt is returned instead of the raw page. Use `prompt` for \"find X in this page\" queries; omit it when you want the markdown body for further inspection. Only http(s); private/loopback hosts are blocked."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch. Must start with http:// or https://. Private, loopback, and link-local hosts are rejected."
                },
                "prompt": {
                    "type": "string",
                    "description": "Optional. If present and non-empty, the fetched page is summarised by a focused LLM-extraction pass keyed off this prompt, and the answer is returned instead of the raw page. Examples: \"Who designed the language?\", \"What's the latest stable version?\"."
                }
            },
            "required": ["url"]
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        // NOTE: the `prompt` argument is intentionally NOT consumed here.
        // When present, the agent loop intercepts the tool call BEFORE
        // dispatching to this `execute`, routes the fetch through
        // `fetch_and_convert`, performs the LLM extraction itself, and
        // returns the answer. If we ever land here with a prompt set
        // (e.g. someone instantiates this tool standalone), we just
        // return the markdown body — same as the no-prompt path —
        // because we have no LLM handle from inside a Tool.
        let url = args["url"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'url' argument".to_string()))?;
        fetch_and_convert(url)
    }
}

/// Fetch a URL, validate it against SSRF rules, convert the HTML body
/// to markdown via `htmd`, and truncate to `MAX_MARKDOWN_CHARS`.
/// Exposed `pub` so the agent loop can call it from the
/// `web_fetch + prompt` special case without duplicating the HTTP and
/// conversion plumbing.
pub fn fetch_and_convert(url: &str) -> Result<String, ToolError> {
    validate_url(url)?;

    let response = http_client()
        .get(url)
        .send()
        .map_err(|e| ToolError::ExecutionFailed(format!("Failed to fetch URL '{}': {}", url, e)))?;

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

    let markdown = if content_type.contains("text/html") || content_type.is_empty() {
        // Empty content-type defaults to HTML conversion — many servers
        // return text without an explicit content-type and the page is
        // still HTML-shaped.
        htmd::convert(&body).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to convert HTML to markdown: {}", e))
        })?
    } else {
        body
    };

    Ok(truncate_to_chars(markdown, MAX_MARKDOWN_CHARS))
}

/// Validate that the URL is http(s) and points to a non-private,
/// non-loopback, non-link-local host. Resolves the hostname to a
/// concrete IP and re-checks — DNS rebinding still possible at the
/// TCP layer but is out of scope for v1.
fn validate_url(url: &str) -> Result<(), ToolError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| ToolError::InvalidArguments(format!("Invalid URL '{}': {}", url, e)))?;

    let scheme = parsed.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(ToolError::InvalidArguments(format!(
            "URL must use http:// or https:// (got '{}')",
            scheme
        )));
    }

    let host_str = parsed
        .host_str()
        .ok_or_else(|| ToolError::InvalidArguments(format!("URL has no host: '{}'", url)))?;

    // `host_str()` wraps IPv6 literals in `[...]`. Strip the brackets
    // before attempting to parse as an `IpAddr`, otherwise the parse
    // fails and we fall through to DNS resolution of the literal
    // string — which never succeeds.
    let candidate_ip = host_str
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(host_str);

    if let Ok(ip) = candidate_ip.parse::<IpAddr>() {
        check_ip_allowed(&ip)?;
    } else {
        let port = parsed.port_or_known_default().unwrap_or(80);
        let addrs = (host_str, port).to_socket_addrs().map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to resolve host '{}': {}", host_str, e))
        })?;
        let mut any_resolved = false;
        for addr in addrs {
            any_resolved = true;
            check_ip_allowed(&addr.ip())?;
        }
        if !any_resolved {
            return Err(ToolError::ExecutionFailed(format!(
                "Host '{}' did not resolve to any address",
                host_str
            )));
        }
    }

    Ok(())
}

/// Reject IPs that should never be fetched from a coding agent:
/// loopback, link-local, private RFC1918 + ULA, multicast,
/// unspecified, and the IPv4-mapped/translated v6 ranges.
fn check_ip_allowed(ip: &IpAddr) -> Result<(), ToolError> {
    let blocked_reason = match ip {
        IpAddr::V4(v4) => {
            if v4.is_loopback() {
                Some("loopback")
            } else if v4.is_private() {
                Some("private (RFC1918)")
            } else if v4.is_link_local() {
                // also covers 169.254.169.254 (cloud metadata)
                Some("link-local")
            } else if v4.is_multicast() {
                Some("multicast")
            } else if v4.is_unspecified() {
                Some("0.0.0.0")
            } else if v4.octets()[0] == 0 {
                Some("0.0.0.0/8")
            } else if v4.is_broadcast() {
                Some("broadcast")
            } else {
                None
            }
        }
        IpAddr::V6(v6) => {
            if v6.is_loopback() {
                Some("loopback")
            } else if v6.is_unspecified() {
                Some("::")
            } else if v6.is_multicast() {
                Some("multicast")
            } else if (v6.segments()[0] & 0xfe00) == 0xfc00 {
                Some("ULA (fc00::/7)")
            } else if (v6.segments()[0] & 0xffc0) == 0xfe80 {
                Some("link-local (fe80::/10)")
            } else if let Some(mapped) = v6.to_ipv4_mapped() {
                return check_ip_allowed(&IpAddr::V4(mapped));
            } else {
                None
            }
        }
    };
    if let Some(reason) = blocked_reason {
        return Err(ToolError::PermissionDenied(format!(
            "Refusing to fetch {} address (reason: {})",
            ip, reason
        )));
    }
    Ok(())
}

/// Truncate at a UTF-8 char boundary, appending a marker that records
/// the pre-truncation length so the agent (and the extraction LLM)
/// knows content was elided.
fn truncate_to_chars(s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = String::with_capacity(end + 64);
    out.push_str(&s[..end]);
    out.push_str(&format!(
        "\n\n... (truncated; original was {} chars)",
        s.len()
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- URL validation / SSRF ---

    #[test]
    fn test_validate_url_rejects_ftp() {
        let err = validate_url("ftp://example.com/").unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_validate_url_rejects_schemeless() {
        let err = validate_url("//example.com").unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_validate_url_rejects_file_scheme() {
        let err = validate_url("file:///etc/passwd").unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_validate_url_rejects_loopback_v4() {
        let err = validate_url("http://127.0.0.1/admin").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("loopback"), "got: {msg}");
    }

    #[test]
    fn test_validate_url_rejects_loopback_v6() {
        let err = validate_url("http://[::1]/").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
    }

    #[test]
    fn test_validate_url_rejects_private_rfc1918() {
        let err = validate_url("http://10.0.0.1/").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("private"), "got: {msg}");
    }

    #[test]
    fn test_validate_url_rejects_link_local_metadata() {
        // 169.254.169.254 — AWS / GCP / Azure metadata endpoint.
        let err = validate_url("http://169.254.169.254/latest/meta-data/").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
    }

    #[test]
    fn test_validate_url_rejects_ula_v6() {
        let err = validate_url("http://[fd00::1]/").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("ULA"), "got: {msg}");
    }

    #[test]
    fn test_validate_url_rejects_unspecified_v4() {
        let err = validate_url("http://0.0.0.0/").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
    }

    // --- markdown conversion (via htmd) ---

    #[test]
    fn test_htmd_preserves_headings() {
        let html = "<h1>Title</h1><h2>Section</h2><p>Body</p>";
        let md = htmd::convert(html).unwrap();
        assert!(md.contains("# Title"));
        assert!(md.contains("## Section"));
    }

    #[test]
    fn test_htmd_preserves_lists() {
        let html = "<ul><li>one</li><li>two</li></ul>";
        let md = htmd::convert(html).unwrap();
        assert!(md.contains("one"));
        assert!(md.contains("two"));
        // htmd uses one of `*` / `-` followed by one or more spaces.
        // Accept any reasonable bullet marker without pinning the
        // exact whitespace (htmd 0.5.x emits `*   one`).
        let trimmed = md.trim_start();
        assert!(
            trimmed.starts_with("* ") || trimmed.starts_with("- ") || trimmed.starts_with("*\t"),
            "expected a bullet marker at the start, got: {md:?}"
        );
    }

    #[test]
    fn test_htmd_preserves_code_blocks() {
        let html = "<pre><code>fn main() {}</code></pre>";
        let md = htmd::convert(html).unwrap();
        assert!(md.contains("fn main"));
        assert!(md.contains("```"));
    }

    // --- truncation ---

    #[test]
    fn test_truncate_to_chars_short_unchanged() {
        let s = "hello".to_string();
        assert_eq!(truncate_to_chars(s.clone(), 100), s);
    }

    #[test]
    fn test_truncate_to_chars_long_truncated_with_marker() {
        let s = "x".repeat(40_000);
        let out = truncate_to_chars(s, 32_000);
        assert!(out.starts_with(&"x".repeat(32_000)));
        assert!(out.contains("(truncated; original was 40000 chars)"));
    }

    #[test]
    fn test_truncate_to_chars_utf8_boundary_safe() {
        // 日本語 = 3 bytes each. Force a cap mid-character.
        let s = "日本語".repeat(20_000);
        let out = truncate_to_chars(s, 32_000);
        // Must not panic and must end on a valid UTF-8 boundary —
        // walking the resulting string by chars proves it.
        assert!(out.chars().count() > 0);
    }

    // --- WebFetchTool::execute integration of the no-prompt path ---

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
    fn test_web_fetch_blocks_loopback_before_connecting() {
        // The SSRF check fires BEFORE any HTTP call, so this returns
        // PermissionDenied (not ExecutionFailed from a refused
        // connection like the old test_web_fetch_connection_refused).
        let tool = WebFetchTool;
        let result = tool.execute(json!({"url": "http://127.0.0.1:19999/"}));
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    // --- schema introspection ---

    #[test]
    fn test_schema_advertises_optional_prompt() {
        let schema = WebFetchTool.parameters_schema();
        let props = &schema["properties"];
        assert!(props["prompt"].is_object(), "prompt property missing");
        let required = schema["required"].as_array().unwrap();
        // Only `url` is required; `prompt` is optional.
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].as_str(), Some("url"));
    }

    #[test]
    fn test_description_mentions_extraction_and_ssrf() {
        let desc = WebFetchTool.description();
        assert!(
            desc.contains("prompt"),
            "description should mention the optional prompt arg"
        );
        assert!(
            desc.contains("private/loopback") || desc.contains("private") || desc.contains("loopback"),
            "description should mention the SSRF guard so the model doesn't try to fetch internal hosts"
        );
    }
}
