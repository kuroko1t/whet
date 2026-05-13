use super::{Tool, ToolError};
use serde_json::json;
use std::io::Read;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::sync::OnceLock;

/// Hard cap on the markdown payload we return to the agent (and feed
/// into the extraction LLM). 32 KB ≈ 6–8 K tokens depending on the
/// language. Tight under q3's 8 K context window but necessary
/// because chrome-heavy sites (Wikipedia, GitHub) easily burn 16 KB
/// on navigation/sidebars before reaching the article body.
const MAX_MARKDOWN_CHARS: usize = 32_000;

/// Hard cap on the raw response body we'll read from a server before
/// HTML→markdown conversion. 5 MB is generous for any real page and
/// fails a slow-stream attacker quickly. Enforced both via
/// `Content-Length` (if advertised) and via a bounded `Read::take`.
const MAX_RESPONSE_BYTES: u64 = 5 * 1024 * 1024;

/// Shared HTML→markdown converter pre-configured to drop common
/// chrome elements (`<nav>`, `<header>`, `<footer>`, `<aside>`,
/// `<form>`, `<script>`, `<style>`) before the conversion proper.
/// Built once; htmd's converters are cheap to clone but cheaper to
/// reuse. Stripping chrome is what gets the article body of sites
/// like Wikipedia / GitHub / MDN into the 32 KB cap on real-world
/// pages — the raw markdown of those sites burns most of its first
/// 30 KB on side navigation, header menus, and footers.
fn html_converter() -> &'static htmd::HtmlToMarkdown {
    static CONVERTER: OnceLock<htmd::HtmlToMarkdown> = OnceLock::new();
    CONVERTER.get_or_init(|| {
        htmd::HtmlToMarkdown::builder()
            .skip_tags(vec![
                "nav", "header", "footer", "aside", "form", "script", "style",
            ])
            .build()
    })
}

/// User-Agent advertised by every outbound fetch. Versioned so server
/// logs can correlate behaviour to a specific whet release; a bare
/// `whet` token gets 403'd by some hardened endpoints.
fn user_agent() -> String {
    format!("whet/{}", env!("CARGO_PKG_VERSION"))
}

/// Build a per-fetch `reqwest::blocking::Client` with the resolved
/// addresses pinned for `host`, so the actual TCP connect cannot
/// re-resolve to a different IP between our SSRF validation and
/// reqwest's connect. This closes the DNS-rebinding TOCTOU that
/// affected the previous static-client design.
///
/// We deliberately `.expect()` on `.build()` rather than fall back to
/// a default client: a default `Client::new()` would silently drop
/// the SSRF redirect policy and the DNS pin, which is strictly
/// less safe than refusing to fetch at all. If reqwest fails to
/// construct a client at all, panic is the correct outcome —
/// continuing would mean fetching with no SSRF guards.
fn build_pinned_client(
    host: &str,
    addrs: &[SocketAddr],
) -> Result<reqwest::blocking::Client, ToolError> {
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::custom(redirect_policy))
        .user_agent(user_agent())
        .referer(false)
        .resolve_to_addrs(host, addrs)
        .build()
        .map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "Failed to build HTTP client (SSRF policy could not be applied): {}",
                e
            ))
        })
}

/// Redirect-policy callback factored out so it can be unit-tested
/// without a real `reqwest::redirect::Attempt`. We reject cross-host
/// redirects unconditionally (matches Claude Code's behaviour and
/// avoids re-doing DNS pin for an arbitrary new host mid-flight),
/// re-validate every target against the SSRF rules, and cap the
/// chain at 5 hops.
fn redirect_policy(attempt: reqwest::redirect::Attempt) -> reqwest::redirect::Action {
    if attempt.previous().len() >= 5 {
        return attempt.error("too many redirects");
    }
    let target = attempt.url();
    let previous = attempt.previous().last();
    if let Some(prev) = previous {
        if target.host_str() != prev.host_str() {
            return attempt
                .error("cross-host redirects are not followed (refetch the new URL explicitly)");
        }
    }
    match validate_redirect_target(target.as_str()) {
        Ok(()) => attempt.follow(),
        Err(e) => attempt.error(format!("redirect blocked: {}", e)),
    }
}

/// SSRF check used by the redirect-policy callback. Identical to the
/// initial-URL gate but returns the rejection reason as a `String`
/// (reqwest's redirect policy wants a `Box<dyn StdError + Send + Sync>`
/// constructed from a `&str`, which String coerces to).
fn validate_redirect_target(url: &str) -> Result<(), String> {
    validate_url(url).map(|_| ()).map_err(|e| format!("{}", e))
}

/// Structured result of a successful `fetch_and_convert` call. Lets
/// the agent special-case pass the final-after-redirect URL into the
/// extraction footer (so the user sees where the answer actually
/// came from when a same-host redirect rewrote the path) and lets
/// the bare-fetch path attach a truncation notice without parsing
/// magic substrings out of the body. Replaces the earlier
/// `fetch_and_convert -> String` shape flagged in the code review.
///
/// `content_type` and `status` are populated but not yet consumed by
/// any caller; they are part of the documented internal contract so
/// future additions (a `/web` debug command, content-type-based
/// dispatch, structured failure logs) don't need a second refactor
/// pass. `#[allow(dead_code)]` until then.
#[derive(Debug, Clone)]
pub(crate) struct FetchResult {
    pub markdown: String,
    pub final_url: String,
    #[allow(dead_code)]
    pub content_type: String,
    #[allow(dead_code)]
    pub status: u16,
    pub truncated: bool,
    pub original_chars: usize,
}

pub struct WebFetchTool;

impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a URL and return content as markdown (up to 32 K chars). \
         STRONGLY PREFER passing the optional `prompt` argument: with it, an internal LLM call extracts a focused answer to your question from the fetched page and returns just that answer (typically 1–3 K tokens) instead of the full page (up to 32 K). \
         Omit `prompt` only when you genuinely need to read or quote the whole page body. \
         Use `prompt` for: looking up a fact, summarising a section, checking whether a topic is covered. \
         Only http(s); private/loopback/CGNAT hosts are blocked. \
         Cross-host redirects are refused — refetch the new URL explicitly if you see one."
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
        let result = fetch_and_convert(url)?;
        // Bare-fetch path: emit the markdown plus a tail notice when
        // the body was truncated. The structured FetchResult.truncated
        // / original_chars flags carry the signal explicitly (no
        // magic substrings) and we format the user-facing notice
        // here so the bare-fetch text is self-describing.
        if result.truncated {
            Ok(format!(
                "{}\n\n_(truncated to {} chars; original was {} chars)_",
                result.markdown, MAX_MARKDOWN_CHARS, result.original_chars
            ))
        } else {
            Ok(result.markdown)
        }
    }
}

/// Fetch a URL, validate it against SSRF rules, convert the HTML body
/// to markdown via `htmd`, and truncate to `MAX_MARKDOWN_CHARS`.
/// Returns a structured `FetchResult` so callers can see status,
/// content-type, the post-redirect final URL, and whether truncation
/// fired — instead of having to grep magic substrings out of a bare
/// string body. Exposed `pub(crate)` so the agent loop's
/// `web_fetch + prompt` special case can use it directly.
pub(crate) fn fetch_and_convert(url: &str) -> Result<FetchResult, ToolError> {
    // Resolve + validate the URL's host once. The returned addresses
    // are then pinned on the per-request client so reqwest's connect
    // step uses the same IPs we just validated (closes DNS rebinding).
    let resolved = validate_url(url)?;

    // The host string we pin against is whatever reqwest's URL parser
    // produces — including the `[...]` brackets for IPv6 literals.
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| ToolError::InvalidArguments(format!("Invalid URL '{}': {}", url, e)))?;
    let host_for_pin = parsed
        .host_str()
        .ok_or_else(|| ToolError::InvalidArguments(format!("URL has no host: '{}'", url)))?
        .to_string();

    let client = build_pinned_client(&host_for_pin, &resolved)?;

    let response = client
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

    // Capture the URL reqwest actually settled on after any same-host
    // redirect, so callers can report THIS to the user instead of the
    // input string (which may have moved). The redirect policy
    // refuses cross-host hops, so this is always within the original
    // pinned-IP host.
    let final_url = response.url().to_string();
    let status_code = status.as_u16();

    // Defensive cap on response size BEFORE we read it into memory.
    // Catches both honest-server-too-large and slow-stream attackers.
    if let Some(declared) = response.content_length() {
        if declared > MAX_RESPONSE_BYTES {
            return Err(ToolError::ExecutionFailed(format!(
                "Response body too large: server declared {} bytes (cap {})",
                declared, MAX_RESPONSE_BYTES
            )));
        }
    }

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    // Read at most `MAX_RESPONSE_BYTES + 1` to detect oversized
    // bodies whose Content-Length was missing or a lie. The +1 is
    // the trip-wire: if we actually fill it, the body was over the
    // cap and we reject. Note: `Read::take` consumes via the
    // blocking response's `Read` impl, which the connection-level
    // timeout (30 s) also bounds.
    let mut buf = Vec::with_capacity(8 * 1024);
    response
        .take(MAX_RESPONSE_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read response body: {}", e)))?;
    if buf.len() as u64 > MAX_RESPONSE_BYTES {
        return Err(ToolError::ExecutionFailed(format!(
            "Response body exceeded {} bytes during streaming read",
            MAX_RESPONSE_BYTES
        )));
    }

    // Lossy UTF-8: matches the previous `response.text()` behaviour
    // for non-UTF-8 pages. A future improvement could detect charset
    // from Content-Type, but most modern pages are UTF-8 anyway.
    let body = String::from_utf8_lossy(&buf).into_owned();

    let raw_markdown = if content_type.contains("text/html") || content_type.is_empty() {
        // Empty content-type defaults to HTML conversion — many servers
        // return text without an explicit content-type and the page is
        // still HTML-shaped.
        html_converter().convert(&body).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to convert HTML to markdown: {}", e))
        })?
    } else {
        body
    };

    let original_chars = raw_markdown.chars().count();
    let truncated = raw_markdown.len() > MAX_MARKDOWN_CHARS;
    let markdown = truncate_to_chars(raw_markdown, MAX_MARKDOWN_CHARS);

    Ok(FetchResult {
        markdown,
        final_url,
        content_type,
        status: status_code,
        truncated,
        original_chars,
    })
}

/// Validate that the URL is http(s) and points to a non-private,
/// non-loopback, non-link-local host. Returns the SocketAddrs that
/// the host resolves to — caller pins these onto the request so the
/// TCP connect doesn't re-resolve (DNS-rebinding mitigation).
fn validate_url(url: &str) -> Result<Vec<SocketAddr>, ToolError> {
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

    let port = parsed.port_or_known_default().unwrap_or(80);

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
        return Ok(vec![SocketAddr::new(ip, port)]);
    }

    let resolved: Vec<SocketAddr> = (host_str, port)
        .to_socket_addrs()
        .map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to resolve host '{}': {}", host_str, e))
        })?
        .collect();
    if resolved.is_empty() {
        return Err(ToolError::ExecutionFailed(format!(
            "Host '{}' did not resolve to any address",
            host_str
        )));
    }
    for addr in &resolved {
        check_ip_allowed(&addr.ip())?;
    }
    Ok(resolved)
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
            } else if v4.octets()[0] == 100 && (v4.octets()[1] & 0xc0) == 0x40 {
                // CGNAT 100.64.0.0/10 — NOT covered by `is_private()`
                // (which only matches 10/8, 172.16/12, 192.168/16).
                Some("CGNAT (100.64.0.0/10)")
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

/// Truncate at a UTF-8 char boundary. The caller decides whether to
/// surface a "truncated" notice — `FetchResult.truncated` and
/// `original_chars` carry the signal so we don't have to embed magic
/// substrings inside the returned body.
fn truncate_to_chars(s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- URL validation / SSRF ---

    #[test]
    fn test_validate_url_rejects_ftp() {
        let err = validate_url("ftp://example.com/").unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
        // Pin the message so a future change that drops to a different
        // error path (e.g. parse-reject vs scheme-reject) gets caught.
        let msg = format!("{}", err);
        assert!(
            msg.contains("http://") && msg.contains("https://"),
            "got: {msg}"
        );
    }

    #[test]
    fn test_validate_url_rejects_schemeless() {
        let err = validate_url("//example.com").unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
        let msg = format!("{}", err);
        assert!(
            msg.contains("Invalid URL")
                || msg.contains("relative URL without a base")
                || msg.contains("missing scheme"),
            "got: {msg}"
        );
    }

    #[test]
    fn test_validate_url_rejects_file_scheme() {
        let err = validate_url("file:///etc/passwd").unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
        let msg = format!("{}", err);
        assert!(
            msg.contains("http://") && msg.contains("https://"),
            "scheme-reject message must explain why: got: {msg}"
        );
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
        let msg = format!("{}", err);
        assert!(
            msg.contains("loopback"),
            "v6 loopback message must mention loopback (recursion path used): {msg}"
        );
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
        let msg = format!("{}", err);
        assert!(
            msg.contains("link-local"),
            "metadata endpoint must be rejected via link-local rule (more general than IP-literal pin): {msg}"
        );
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
        let msg = format!("{}", err);
        assert!(msg.contains("0.0.0.0"), "got: {msg}");
    }

    #[test]
    fn test_validate_url_rejects_cgnat() {
        // CGNAT 100.64.0.0/10. `Ipv4Addr::is_private()` returns
        // false for this range, so without a dedicated check it
        // would be reachable and could route to a hostile network.
        let err = validate_url("http://100.64.0.1/").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("CGNAT"), "got: {msg}");
        // Boundary: 100.127.255.255 still inside /10.
        let err = validate_url("http://100.127.255.255/").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        // 100.128.x.x is OUTSIDE /10 and should NOT be CGNAT-blocked.
        // (It's still a public address, no other rule rejects it
        // — DNS resolution may still error in a sandbox, but the
        // is_path_safe-style v4 rules should pass it through.)
        let result = validate_url("http://100.128.0.1/");
        // If the response is an error, it must NOT be the CGNAT
        // PermissionDenied — only an upstream resolution error is OK.
        if let Err(ToolError::PermissionDenied(msg)) = &result {
            assert!(
                !msg.contains("CGNAT"),
                "100.128.0.1 must NOT be classified as CGNAT, got: {msg}"
            );
        }
    }

    #[test]
    fn test_validate_url_rejects_ipv6_mapped_loopback() {
        // IPv4-mapped-into-IPv6: `::ffff:127.0.0.1` is the v6 spelling
        // of the v4 loopback. Without the to_ipv4_mapped() recursion
        // in check_ip_allowed, this would bypass the loopback rule.
        let err = validate_url("http://[::ffff:127.0.0.1]/").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("loopback"), "got: {msg}");
    }

    #[test]
    fn test_validate_url_rejects_ipv6_link_local() {
        let err = validate_url("http://[fe80::1]/").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("link-local"), "got: {msg}");
    }

    #[test]
    fn test_validate_url_rejects_ipv6_multicast() {
        let err = validate_url("http://[ff02::1]/").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("multicast"), "got: {msg}");
    }

    #[test]
    fn test_validate_url_rejects_zero_slash_eight() {
        // 0.0.0.0/8 — `is_unspecified()` only catches the exact
        // 0.0.0.0, not 0.1.2.3 etc.
        let err = validate_url("http://0.1.2.3/").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("0.0.0.0/8"), "got: {msg}");
    }

    #[test]
    fn test_validate_url_rejects_broadcast() {
        let err = validate_url("http://255.255.255.255/").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("broadcast"), "got: {msg}");
    }

    #[test]
    fn test_validate_url_returns_pinned_addrs_for_ip_literal() {
        // validate_url's contract: for an IP literal it returns
        // exactly one SocketAddr (the host's IP + port), and the
        // port matches the URL's scheme default.
        let addrs = validate_url("http://8.8.8.8:8080/").unwrap();
        assert_eq!(addrs.len(), 1);
        assert_eq!(addrs[0].port(), 8080);
        assert_eq!(addrs[0].ip().to_string(), "8.8.8.8");
        let addrs = validate_url("https://8.8.8.8/").unwrap();
        assert_eq!(addrs[0].port(), 443);
    }

    // --- redirect policy ---

    #[test]
    fn test_validate_redirect_target_blocks_loopback() {
        // The redirect-policy callback delegates to this for the
        // SSRF re-check on every hop. If a 302 points to a private
        // address, this must reject.
        let err = validate_redirect_target("http://127.0.0.1/").unwrap_err();
        assert!(err.contains("loopback"), "got: {err}");
    }

    #[test]
    fn test_validate_redirect_target_blocks_cloud_metadata() {
        // The classic SSRF target: AWS / GCP / Azure metadata.
        let err = validate_redirect_target("http://169.254.169.254/latest/").unwrap_err();
        assert!(err.contains("link-local"), "got: {err}");
    }

    #[test]
    fn test_validate_redirect_target_allows_public() {
        // Sanity: a clearly-public IP should pass the redirect gate
        // (it can still fail at connect time but that's downstream).
        // We accept either Ok or a non-SSRF Err (e.g. DNS resolution
        // in a sandbox); SSRF rejection is what we explicitly
        // forbid here.
        let result = validate_redirect_target("http://8.8.8.8/");
        if let Err(msg) = result {
            assert!(
                !msg.contains("loopback")
                    && !msg.contains("private")
                    && !msg.contains("link-local"),
                "8.8.8.8 must not be SSRF-rejected: {msg}"
            );
        }
    }

    // --- response-size cap ---

    #[test]
    fn test_max_response_bytes_is_reasonable() {
        // 5 MB is the documented cap. Anything substantially
        // smaller risks rejecting common pages; substantially
        // larger defeats the DoS guard.
        assert_eq!(MAX_RESPONSE_BYTES, 5 * 1024 * 1024);
    }

    // --- User-Agent ---

    #[test]
    fn test_user_agent_includes_version() {
        let ua = user_agent();
        assert!(ua.starts_with("whet/"), "got: {ua}");
        // Either a semver-ish number or "0.0.0"; ensures we're not
        // sending a bare "whet" string that some servers 403.
        assert!(
            ua.split('/').nth(1).is_some_and(|v| !v.is_empty()),
            "got: {ua}"
        );
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

    // --- chrome stripping via html_converter() ---

    #[test]
    fn test_converter_strips_nav_and_header_and_footer() {
        // Regression guard for the Wikipedia-chrome problem: real
        // pages waste tens of KB on <nav>/<header>/<footer>/<aside>
        // before the article body, which then never fits under the
        // 32 KB cap. Our shared converter drops those wholesale.
        let html = r#"
            <html><body>
                <header><nav><a href="/home">Home</a> <a href="/about">About</a></nav></header>
                <main><h1>Real Title</h1><p>Real body.</p></main>
                <aside><a>related link</a></aside>
                <footer>© 2026 site</footer>
            </body></html>
        "#;
        let md = html_converter().convert(html).unwrap();
        assert!(
            md.contains("Real Title"),
            "main heading must survive: {md:?}"
        );
        assert!(md.contains("Real body"), "main body must survive: {md:?}");
        assert!(
            !md.contains("Home") && !md.contains("About"),
            "<nav> contents must be dropped: {md:?}"
        );
        assert!(
            !md.contains("related link"),
            "<aside> contents must be dropped: {md:?}"
        );
        assert!(
            !md.contains("© 2026"),
            "<footer> contents must be dropped: {md:?}"
        );
    }

    #[test]
    fn test_converter_strips_script_and_style() {
        // We rely on htmd to strip <script>/<style> contents on top
        // of our nav-class stripping. Confirm both paths kill them.
        let html = r#"
            <html><head><style>body { color: red }</style></head>
            <body>
                <script>alert('xss')</script>
                <p>Visible text</p>
            </body></html>
        "#;
        let md = html_converter().convert(html).unwrap();
        assert!(md.contains("Visible text"), "got: {md:?}");
        assert!(!md.contains("color: red"), "got: {md:?}");
        assert!(!md.contains("alert"), "got: {md:?}");
    }

    #[test]
    fn test_converter_strips_forms() {
        // <form> blocks (search bars, login forms) are common page
        // chrome and very rarely contain user-relevant content.
        let html = r#"
            <html><body>
                <form><input name="q"><button>Search</button></form>
                <article><p>Article prose.</p></article>
            </body></html>
        "#;
        let md = html_converter().convert(html).unwrap();
        assert!(md.contains("Article prose"));
        assert!(!md.contains("Search"));
    }

    #[test]
    fn test_converter_preserves_article_and_main() {
        // The shared converter must NOT drop semantic main-content
        // tags. If it did, every well-marked-up page would lose its
        // body to the chrome-stripping pass.
        let html = r#"<article><h2>Section</h2><p>Body in article.</p></article>
                      <main><p>Body in main.</p></main>"#;
        let md = html_converter().convert(html).unwrap();
        assert!(md.contains("Section"));
        assert!(md.contains("Body in article"));
        assert!(md.contains("Body in main"));
    }

    // --- truncation ---

    #[test]
    fn test_truncate_to_chars_short_unchanged() {
        let s = "hello".to_string();
        assert_eq!(truncate_to_chars(s.clone(), 100), s);
    }

    #[test]
    fn test_truncate_to_chars_long_truncated_clean() {
        // truncate_to_chars now returns the prefix only — the
        // FetchResult.truncated + original_chars flags signal
        // truncation, no inline magic substring is embedded.
        let s = "x".repeat(40_000);
        let out = truncate_to_chars(s, 32_000);
        assert_eq!(out.len(), 32_000);
        assert!(out.chars().all(|c| c == 'x'));
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
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
        // Pin the message: parsing failed at the URL parser, not at
        // the scheme/host gate. If a future refactor pushes the
        // rejection elsewhere we want to know.
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid URL"), "got: {msg}");
    }

    #[test]
    fn test_web_fetch_missing_url() {
        let tool = WebFetchTool;
        let result = tool.execute(json!({}));
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("missing 'url'"), "got: {msg}");
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

    #[test]
    fn test_description_steers_model_to_prefer_prompt() {
        // Regression guard for the dog-food failure mode where the
        // model would call web_fetch(url) by default and pay the
        // full markdown context tax — because the description didn't
        // tell it `prompt` was the preferred form.
        let desc = WebFetchTool.description();
        assert!(
            desc.contains("STRONGLY PREFER") || desc.contains("PREFER passing"),
            "description must actively steer the model toward `prompt` for fact lookup: {desc}"
        );
        assert!(
            desc.contains("32 K") || desc.contains("32K") || desc.contains("32,000"),
            "description should mention the size budget so the model understands the cost of bare fetch: {desc}"
        );
    }
}
