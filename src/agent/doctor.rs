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

/// Pure classifier for an `/api/tags` response. The caller fetches the
/// URL (so `run_all` can reuse the body for the model-present check
/// without a second HTTP round-trip) and hands the result here.
pub fn classify_ollama_response(base_url: &str, response: Result<String, String>) -> Diagnostic {
    match response {
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
///
/// `active_model` is the model the user actually invoked (`-m` flag or
/// `/model` slash override). It can differ from `cfg.llm.model` (the
/// config-file default), and `/doctor` should reflect the live state, so
/// we check the active model rather than the static config.
pub fn run_all<F>(cfg: &Config, active_model: &str, home: &Path, fetch: F) -> Vec<Diagnostic>
where
    F: Fn(&str) -> Result<String, String>,
{
    let mut rows = Vec::new();

    // Ollama-specific checks only run if provider is ollama.
    let is_ollama = cfg.llm.provider == "ollama";
    let ollama_body = if is_ollama {
        // Fetch /api/tags ONCE and reuse for both the reachable check and
        // the model-present check. Avoids a second 5-second timeout when
        // ollama is unreachable.
        let url = format!("{}/api/tags", cfg.llm.base_url.trim_end_matches('/'));
        let response = fetch(&url);
        let body = response.as_ref().ok().cloned();
        rows.push(classify_ollama_response(&cfg.llm.base_url, response));
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
            rows.push(check_model_present(active_model, &body));
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
                compaction_token_threshold: 5000,
                skills_dir: "~/.whet/skills".to_string(),
                web_enabled: false,
            },
            memory: MemoryConfig {
                database_path: ":memory:".to_string(),
                max_inject_memories: 50,
            },
            mcp: McpConfig { servers: vec![] },
        }
    }

    // (classify_ollama_response is covered by classify_ollama_response_pass_warn_fail below.)

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
        let rows = run_all(&cfg, "qwen3.5:9b", dir.path(), |_url| {
            Ok(r#"{"models":[{"name":"qwen3.5:9b"}]}"#.to_string())
        });
        let names: Vec<&str> = rows.iter().map(|d| d.name).collect();
        assert!(names.contains(&"ollama reachable"));
        assert!(names.contains(&"model available"));
        assert!(names.contains(&"config parses"));
        assert!(names.contains(&"~/.whet writable"));
        assert!(names.contains(&"MCP servers"));
        assert_eq!(overall_exit_code(&rows), 0);
    }

    #[test]
    fn run_all_marks_ollama_unreachable_as_fail_and_chains_model_check() {
        let dir = tempfile::TempDir::new().unwrap();
        let cfg = fake_cfg();
        let rows = run_all(&cfg, "qwen3.5:9b", dir.path(), |_url| {
            Err("connect refused".to_string())
        });
        let ollama_row = rows.iter().find(|d| d.name == "ollama reachable").unwrap();
        let model_row = rows.iter().find(|d| d.name == "model available").unwrap();
        assert_eq!(ollama_row.status, DiagnosticStatus::Fail);
        assert_eq!(model_row.status, DiagnosticStatus::Fail);
        assert!(model_row.detail.contains("ollama unreachable"));
        assert_eq!(overall_exit_code(&rows), 1);
    }

    #[test]
    fn run_all_checks_active_model_not_config_default() {
        // The user invoked `whet -m runtime-model`, but the config default is
        // `config-model`. /doctor must check the runtime model, not the config
        // (otherwise it gives bogus PASS when the user typo'd -m).
        let dir = tempfile::TempDir::new().unwrap();
        let mut cfg = fake_cfg();
        cfg.llm.model = "config-model".to_string();
        let rows = run_all(&cfg, "runtime-model", dir.path(), |_url| {
            // ollama has runtime-model but NOT config-model
            Ok(r#"{"models":[{"name":"runtime-model:latest"}]}"#.to_string())
        });
        let model_row = rows.iter().find(|d| d.name == "model available").unwrap();
        assert_eq!(model_row.status, DiagnosticStatus::Pass);
        assert!(
            model_row.detail.contains("runtime-model"),
            "model row should refer to the runtime model: {:?}",
            model_row.detail
        );
    }

    #[test]
    fn run_all_fetches_ollama_tags_only_once() {
        // Regression guard: previously /doctor double-fetched /api/tags
        // (once for reachable, once for body), causing a 10s hang on
        // unreachable ollama. The aggregator must call fetch exactly once.
        use std::cell::Cell;
        let dir = tempfile::TempDir::new().unwrap();
        let cfg = fake_cfg();
        let calls = Cell::new(0u32);
        let _rows = run_all(&cfg, "qwen3.5:9b", dir.path(), |_url| {
            calls.set(calls.get() + 1);
            Ok(r#"{"models":[{"name":"qwen3.5:9b"}]}"#.to_string())
        });
        assert_eq!(calls.get(), 1, "expected exactly one /api/tags fetch");
    }

    #[test]
    fn classify_ollama_response_pass_warn_fail() {
        let pass =
            classify_ollama_response("http://x", Ok(r#"{"models":[{"name":"a"}]}"#.to_string()));
        assert_eq!(pass.status, DiagnosticStatus::Pass);
        let warn = classify_ollama_response("http://x", Ok("{}".to_string()));
        assert_eq!(warn.status, DiagnosticStatus::Warn);
        let fail = classify_ollama_response("http://x", Err("nope".to_string()));
        assert_eq!(fail.status, DiagnosticStatus::Fail);
    }
}
