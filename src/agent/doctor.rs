//! `/doctor` diagnostics — UX.6.
//!
//! A small set of checks that surface the most common reasons Whet
//! "doesn't work" for a first-time user: ollama not running, configured
//! model not pulled, config file syntax error, MCP server can't spawn,
//! `~/.whet` not writable.
//!
//! Each check is pure (or near-pure: takes its IO dependencies as
//! callbacks) so it can be unit-tested without network or filesystem.
//! `run_all` is the aggregator the slash command calls.

use crate::config::{Config, McpServerConfig};
use std::path::{Path, PathBuf};

/// Verdict for a single check.
#[derive(Debug, PartialEq, Eq)]
pub enum DiagnosticStatus {
    Pass,
    Fail,
    Warn,
}

/// One row of the `/doctor` report.
#[derive(Debug)]
pub struct Diagnostic {
    pub name: &'static str,
    pub status: DiagnosticStatus,
    pub detail: String,
}

impl Diagnostic {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            status: DiagnosticStatus::Pass,
            detail: detail.into(),
        }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            status: DiagnosticStatus::Fail,
            detail: detail.into(),
        }
    }
    fn warn(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            status: DiagnosticStatus::Warn,
            detail: detail.into(),
        }
    }
}

/// Render one row Claude-Code-style: `✓ Name — detail` / `✗ ...` / `⚠ ...`.
/// No ANSI here — caller adds colour if desired.
pub fn format_row(d: &Diagnostic) -> String {
    let mark = match d.status {
        DiagnosticStatus::Pass => "✓",
        DiagnosticStatus::Fail => "✗",
        DiagnosticStatus::Warn => "⚠",
    };
    format!("{} {} — {}", mark, d.name, d.detail)
}

/// Overall exit code: 0 if no Fail, 1 otherwise. Warns don't fail.
pub fn overall_exit_code(rows: &[Diagnostic]) -> i32 {
    if rows.iter().any(|d| d.status == DiagnosticStatus::Fail) {
        1
    } else {
        0
    }
}

// --- Individual checks ---

/// Verify ollama answers `/api/tags`. The `fetch` closure is the IO
/// dependency so tests can mock the response.
pub fn check_ollama_reachable<F>(base_url: &str, fetch: F) -> Diagnostic
where
    F: FnOnce(&str) -> Result<String, String>,
{
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    match fetch(&url) {
        Ok(body) if body.contains("\"models\"") => {
            Diagnostic::pass("ollama reachable", format!("{} responded", base_url))
        }
        Ok(body) => Diagnostic::warn(
            "ollama reachable",
            format!(
                "{} replied but body lacks `\"models\"` key (truncated: {:?})",
                base_url,
                &body.chars().take(60).collect::<String>()
            ),
        ),
        Err(e) => Diagnostic::fail(
            "ollama reachable",
            format!("cannot reach {}: {}", base_url, e),
        ),
    }
}

/// Look for the configured model name in the JSON `body` returned by
/// `/api/tags`. Pure parser — no IO.
///
/// Handles ollama's implicit `:latest` tag: if the user configures
/// `qwen3.6-q3` (no tag), ollama actually stores it as `qwen3.6-q3:latest`.
/// We accept either exact match or the `:latest`-tagged variant.
pub fn check_model_present(model: &str, ollama_tags_json: &str) -> Diagnostic {
    let exact = format!("\"name\":\"{}\"", model);
    let with_latest = if model.contains(':') {
        // User specified a tag explicitly — only accept exact match.
        String::new()
    } else {
        format!("\"name\":\"{}:latest\"", model)
    };
    let found = ollama_tags_json.contains(&exact)
        || (!with_latest.is_empty() && ollama_tags_json.contains(&with_latest));

    if found {
        Diagnostic::pass("model available", format!("`{}` is pulled", model))
    } else {
        Diagnostic::fail(
            "model available",
            format!(
                "`{}` not found in `ollama list`; pull with: ollama pull {}",
                model, model
            ),
        )
    }
}

/// Try to parse `<home>/.whet/config.toml`.
pub fn check_config_parses(home: &Path) -> Diagnostic {
    let path = home.join(".whet").join("config.toml");
    if !path.exists() {
        return Diagnostic::warn(
            "config parses",
            format!("{} not found — using defaults", path.display()),
        );
    }
    match std::fs::read_to_string(&path) {
        Err(e) => Diagnostic::fail(
            "config parses",
            format!("cannot read {}: {}", path.display(), e),
        ),
        Ok(contents) => match toml::from_str::<Config>(&contents) {
            Ok(_) => Diagnostic::pass("config parses", format!("{} OK", path.display())),
            Err(e) => Diagnostic::fail(
                "config parses",
                format!("{} has TOML syntax error: {}", path.display(), e),
            ),
        },
    }
}

/// Check that `<home>/.whet` exists and is writable. Creates the dir if
/// missing.
pub fn check_whet_dir_writable(home: &Path) -> Diagnostic {
    let dir = home.join(".whet");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return Diagnostic::fail(
            "~/.whet writable",
            format!("cannot create {}: {}", dir.display(), e),
        );
    }
    let probe = dir.join(".whet_doctor_probe");
    match std::fs::write(&probe, b"ok") {
        Err(e) => Diagnostic::fail(
            "~/.whet writable",
            format!("cannot write to {}: {}", dir.display(), e),
        ),
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            Diagnostic::pass("~/.whet writable", format!("{} OK", dir.display()))
        }
    }
}

/// Check that each configured MCP server's binary is on `PATH`. We don't
/// spawn it here — that's expensive and might modify state — but we look
/// for the binary so the most common error ("typo in server name", "tool
/// not installed") surfaces.
pub fn check_mcp_binaries(servers: &[McpServerConfig], which: impl Fn(&str) -> bool) -> Diagnostic {
    if servers.is_empty() {
        return Diagnostic::pass("MCP servers", "no servers configured (skipped)");
    }
    let missing: Vec<&str> = servers
        .iter()
        .filter(|s| !which(&s.command))
        .map(|s| s.command.as_str())
        .collect();
    if missing.is_empty() {
        Diagnostic::pass(
            "MCP servers",
            format!("all {} server binaries on PATH", servers.len()),
        )
    } else {
        Diagnostic::fail(
            "MCP servers",
            format!("binaries not on PATH: {}", missing.join(", ")),
        )
    }
}

// --- Aggregator ---

/// Run all checks against a real environment. The `fetch` argument
/// allows tests to inject a mock HTTP getter; production callers pass a
/// reqwest-backed implementation.
pub fn run_all<F>(cfg: &Config, home: &Path, fetch: F) -> Vec<Diagnostic>
where
    F: Fn(&str) -> Result<String, String>,
{
    let mut rows = Vec::new();

    // Ollama-specific checks only run if provider is ollama.
    let is_ollama = cfg.llm.provider == "ollama";
    let ollama_body = if is_ollama {
        let url = format!("{}/api/tags", cfg.llm.base_url.trim_end_matches('/'));
        let body = fetch(&url).ok();
        rows.push(check_ollama_reachable(&cfg.llm.base_url, |u| fetch(u)));
        body
    } else {
        rows.push(Diagnostic::pass(
            "ollama reachable",
            format!("provider is `{}` (skipped)", cfg.llm.provider),
        ));
        None
    };

    if is_ollama {
        if let Some(body) = ollama_body {
            rows.push(check_model_present(&cfg.llm.model, &body));
        } else {
            rows.push(Diagnostic::fail(
                "model available",
                "ollama unreachable (see above)",
            ));
        }
    }

    rows.push(check_config_parses(home));
    rows.push(check_whet_dir_writable(home));

    rows.push(check_mcp_binaries(&cfg.mcp.servers, |cmd| {
        which_in_path(cmd)
    }));

    rows
}

/// `which`-style check: is `cmd` present somewhere on `$PATH`?
/// Trivial implementation — testable indirectly via `check_mcp_binaries`.
fn which_in_path(cmd: &str) -> bool {
    if cmd.contains('/') {
        return PathBuf::from(cmd).exists();
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            if dir.join(cmd).exists() {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AgentConfig, LlmConfig, LlmOptions, McpConfig, McpServerConfig, MemoryConfig,
        PermissionMode,
    };

    fn fake_cfg() -> Config {
        Config {
            llm: LlmConfig {
                provider: "ollama".to_string(),
                model: "qwen3.5:9b".to_string(),
                base_url: "http://localhost:11434".to_string(),
                api_key: None,
                streaming: false,
                options: LlmOptions::default(),
            },
            agent: AgentConfig {
                max_iterations: 10,
                permission_mode: PermissionMode::Default,
                context_compression: true,
                skills_dir: "~/.whet/skills".to_string(),
                web_enabled: false,
            },
            memory: MemoryConfig {
                database_path: ":memory:".to_string(),
            },
            mcp: McpConfig { servers: vec![] },
        }
    }

    // --- check_ollama_reachable ---

    #[test]
    fn ollama_reachable_pass_on_models_key() {
        let d = check_ollama_reachable("http://x", |_url| {
            Ok(r#"{"models":[{"name":"qwen3.5:9b"}]}"#.to_string())
        });
        assert_eq!(d.status, DiagnosticStatus::Pass);
    }

    #[test]
    fn ollama_reachable_warn_on_unexpected_body() {
        let d = check_ollama_reachable("http://x", |_url| Ok("hello world".to_string()));
        assert_eq!(d.status, DiagnosticStatus::Warn);
    }

    #[test]
    fn ollama_reachable_fail_on_network_error() {
        let d = check_ollama_reachable("http://x", |_url| Err("connection refused".to_string()));
        assert_eq!(d.status, DiagnosticStatus::Fail);
        assert!(d.detail.contains("connection refused"));
    }

    // --- check_model_present ---

    #[test]
    fn model_present_pass() {
        let body = r#"{"models":[{"name":"qwen3.5:9b"},{"name":"qwen3.6-q3:latest"}]}"#;
        let d = check_model_present("qwen3.5:9b", body);
        assert_eq!(d.status, DiagnosticStatus::Pass);
    }

    #[test]
    fn model_present_fail_when_missing() {
        let body = r#"{"models":[{"name":"other:1b"}]}"#;
        let d = check_model_present("qwen3.5:9b", body);
        assert_eq!(d.status, DiagnosticStatus::Fail);
        assert!(d.detail.contains("ollama pull qwen3.5:9b"));
    }

    #[test]
    fn model_present_does_not_match_substring_of_other_model_name() {
        // "qwen" must NOT match "qwen3.5:9b"
        let body = r#"{"models":[{"name":"qwen3.5:9b"}]}"#;
        let d = check_model_present("qwen", body);
        assert_eq!(d.status, DiagnosticStatus::Fail);
    }

    #[test]
    fn model_present_pass_with_implicit_latest_tag() {
        // ollama returns "name":"foo:latest" but user configured "foo" without a tag
        let body = r#"{"models":[{"name":"qwen3.6-q3:latest"}]}"#;
        let d = check_model_present("qwen3.6-q3", body);
        assert_eq!(d.status, DiagnosticStatus::Pass);
    }

    #[test]
    fn model_present_fail_when_explicit_tag_does_not_match_latest() {
        // user asked for foo:7b but only foo:latest is pulled — should fail
        let body = r#"{"models":[{"name":"qwen3.6-q3:latest"}]}"#;
        let d = check_model_present("qwen3.6-q3:7b", body);
        assert_eq!(d.status, DiagnosticStatus::Fail);
    }

    // --- check_config_parses ---

    #[test]
    fn config_parses_warn_when_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        let d = check_config_parses(dir.path());
        assert_eq!(d.status, DiagnosticStatus::Warn);
        assert!(d.detail.contains("not found"));
    }

    #[test]
    fn config_parses_pass_on_valid_toml() {
        let dir = tempfile::TempDir::new().unwrap();
        let whet = dir.path().join(".whet");
        std::fs::create_dir(&whet).unwrap();
        std::fs::write(
            whet.join("config.toml"),
            r#"
[llm]
provider = "ollama"
model = "x"
base_url = "http://localhost:11434"

[agent]
max_iterations = 7

[memory]
database_path = "test.db"
"#,
        )
        .unwrap();
        let d = check_config_parses(dir.path());
        assert_eq!(d.status, DiagnosticStatus::Pass);
    }

    #[test]
    fn config_parses_fail_on_broken_toml() {
        let dir = tempfile::TempDir::new().unwrap();
        let whet = dir.path().join(".whet");
        std::fs::create_dir(&whet).unwrap();
        std::fs::write(whet.join("config.toml"), "this is not = = toml").unwrap();
        let d = check_config_parses(dir.path());
        assert_eq!(d.status, DiagnosticStatus::Fail);
        assert!(d.detail.contains("TOML syntax error"));
    }

    // --- check_whet_dir_writable ---

    #[test]
    fn whet_dir_writable_pass_in_tempdir() {
        let dir = tempfile::TempDir::new().unwrap();
        let d = check_whet_dir_writable(dir.path());
        assert_eq!(d.status, DiagnosticStatus::Pass);
        // The probe must NOT linger.
        assert!(!dir.path().join(".whet").join(".whet_doctor_probe").exists());
    }

    // --- check_mcp_binaries ---

    #[test]
    fn mcp_binaries_pass_when_no_servers() {
        let d = check_mcp_binaries(&[], |_| true);
        assert_eq!(d.status, DiagnosticStatus::Pass);
        assert!(d.detail.contains("no servers"));
    }

    #[test]
    fn mcp_binaries_fail_when_one_missing() {
        let servers = vec![
            McpServerConfig {
                name: "good".to_string(),
                command: "python3".to_string(),
                args: vec![],
            },
            McpServerConfig {
                name: "bad".to_string(),
                command: "definitely_not_installed_xyz".to_string(),
                args: vec![],
            },
        ];
        let d = check_mcp_binaries(&servers, |cmd| cmd == "python3");
        assert_eq!(d.status, DiagnosticStatus::Fail);
        assert!(d.detail.contains("definitely_not_installed_xyz"));
    }

    // --- format_row + overall_exit_code ---

    #[test]
    fn format_row_uses_correct_marks() {
        let p = Diagnostic::pass("a", "ok");
        let f = Diagnostic::fail("b", "broken");
        let w = Diagnostic::warn("c", "maybe");
        assert!(format_row(&p).starts_with("✓ a — "));
        assert!(format_row(&f).starts_with("✗ b — "));
        assert!(format_row(&w).starts_with("⚠ c — "));
    }

    #[test]
    fn overall_exit_code_is_zero_when_only_pass_or_warn() {
        let rows = vec![Diagnostic::pass("a", "ok"), Diagnostic::warn("b", "meh")];
        assert_eq!(overall_exit_code(&rows), 0);
    }

    #[test]
    fn overall_exit_code_is_one_when_any_fail() {
        let rows = vec![
            Diagnostic::pass("a", "ok"),
            Diagnostic::fail("b", "broken"),
            Diagnostic::warn("c", "meh"),
        ];
        assert_eq!(overall_exit_code(&rows), 1);
    }

    // --- run_all integration ---

    #[test]
    fn run_all_succeeds_with_mocked_ollama_and_tempdir() {
        let dir = tempfile::TempDir::new().unwrap();
        let cfg = fake_cfg();
        let rows = run_all(&cfg, dir.path(), |_url| {
            Ok(r#"{"models":[{"name":"qwen3.5:9b"}]}"#.to_string())
        });
        // ollama reachable + model available + config parses (warn, no file)
        // + ~/.whet writable + MCP servers (pass, none).
        let names: Vec<&str> = rows.iter().map(|d| d.name).collect();
        assert!(names.contains(&"ollama reachable"));
        assert!(names.contains(&"model available"));
        assert!(names.contains(&"config parses"));
        assert!(names.contains(&"~/.whet writable"));
        assert!(names.contains(&"MCP servers"));
        // No Fail entries.
        assert_eq!(overall_exit_code(&rows), 0);
    }

    #[test]
    fn run_all_marks_ollama_unreachable_as_fail_and_chains_model_check() {
        let dir = tempfile::TempDir::new().unwrap();
        let cfg = fake_cfg();
        let rows = run_all(&cfg, dir.path(), |_url| Err("connect refused".to_string()));
        let ollama_row = rows.iter().find(|d| d.name == "ollama reachable").unwrap();
        let model_row = rows.iter().find(|d| d.name == "model available").unwrap();
        assert_eq!(ollama_row.status, DiagnosticStatus::Fail);
        assert_eq!(model_row.status, DiagnosticStatus::Fail);
        assert!(model_row.detail.contains("ollama unreachable"));
        assert_eq!(overall_exit_code(&rows), 1);
    }
}
