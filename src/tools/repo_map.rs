use super::{Tool, ToolError};
use crate::security::path::is_path_safe;
use serde_json::json;
use std::fs;
use std::path::Path;

pub struct RepoMapTool;

const MAX_FILES: usize = 200;
const MAX_OUTPUT_LINES: usize = 5000;

const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
];

const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "go", "java", "c", "cpp", "h", "tsx", "jsx",
    "rb", "kt", "kts",
];

impl Tool for RepoMapTool {
    fn name(&self) -> &str {
        "repo_map"
    }

    fn description(&self) -> &str {
        "Show project structure with function/class/type definitions"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The root directory to map (default: current directory)"
                }
            },
            "required": []
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let path = args["path"].as_str().unwrap_or(".");

        if !is_path_safe(path) {
            return Err(ToolError::PermissionDenied(format!(
                "Access to '{}' is blocked for security",
                path
            )));
        }

        let root = Path::new(path);
        if !root.is_dir() {
            return Err(ToolError::ExecutionFailed(format!(
                "'{}' is not a directory",
                path
            )));
        }

        let mut files = Vec::new();
        collect_source_files(root, &mut files);
        files.sort();

        if files.len() > MAX_FILES {
            files.truncate(MAX_FILES);
        }

        let mut output_lines = Vec::new();

        for file_path in &files {
            if output_lines.len() >= MAX_OUTPUT_LINES {
                output_lines.push("(output truncated)".to_string());
                break;
            }

            let relative = file_path
                .strip_prefix(root)
                .unwrap_or(file_path)
                .display()
                .to_string();

            let symbols = extract_symbols(file_path);

            if symbols.is_empty() {
                output_lines.push(relative);
            } else {
                output_lines.push(relative);
                for symbol in symbols {
                    if output_lines.len() >= MAX_OUTPUT_LINES {
                        break;
                    }
                    output_lines.push(format!("  {}", symbol));
                }
            }
        }

        if output_lines.is_empty() {
            Ok("No source files found.".to_string())
        } else {
            Ok(output_lines.join("\n"))
        }
    }
}

fn collect_source_files(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
    if files.len() >= MAX_FILES {
        return;
    }

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut sorted_entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    sorted_entries.sort_by_key(|e| e.file_name());

    for entry in sorted_entries {
        if files.len() >= MAX_FILES {
            break;
        }
        let path = entry.path();

        if path.is_dir() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if SKIP_DIRS.contains(&name) {
                    continue;
                }
            }
            collect_source_files(&path, files);
        } else if is_source_file(&path) {
            files.push(path);
        }
    }
}

fn is_source_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| SOURCE_EXTENSIONS.contains(&ext))
        .unwrap_or(false)
}

fn extract_symbols(path: &Path) -> Vec<String> {
    let ext = match path.extension().and_then(|e| e.to_str()) {
        Some(e) => e,
        None => return vec![],
    };

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };

    let mut symbols = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        let indent = line.len() - line.trim_start().len();
        let symbol = match ext {
            "rs" => extract_rust_symbol(trimmed),
            "py" => extract_python_symbol(trimmed),
            "js" | "jsx" | "ts" | "tsx" => extract_js_ts_symbol(trimmed),
            "go" => extract_go_symbol(trimmed),
            "java" => extract_java_symbol(trimmed),
            "c" | "cpp" | "h" => extract_c_symbol(trimmed),
            "rb" => extract_ruby_symbol(trimmed),
            "kt" | "kts" => extract_kotlin_symbol(trimmed),
            _ => None,
        };
        if let Some(s) = symbol {
            // Add line number and indentation hint for nested symbols
            let prefix = if indent > 0 { "  " } else { "" };
            symbols.push(format!("{}L{}: {}", prefix, line_num + 1, s));
        }
    }

    symbols
}

fn extract_rust_symbol(line: &str) -> Option<String> {
    let prefixes = [
        "pub fn ", "fn ", "pub(crate) fn ", "pub(super) fn ",
        "pub async fn ", "async fn ", "pub const fn ", "const fn ",
        "pub unsafe fn ", "unsafe fn ",
        "pub struct ", "struct ", "pub(crate) struct ",
        "pub enum ", "enum ", "pub(crate) enum ",
        "pub trait ", "trait ", "pub(crate) trait ",
        "impl ", "pub mod ", "mod ", "pub(crate) mod ",
        "pub type ", "type ", "pub(crate) type ",
        "pub const ", "const ",
        "pub static ", "static ",
        "macro_rules! ",
    ];
    for prefix in &prefixes {
        if line.starts_with(prefix) {
            return Some(extract_signature(line));
        }
    }
    None
}

fn extract_python_symbol(line: &str) -> Option<String> {
    if line.starts_with("class ") || line.starts_with("def ") || line.starts_with("async def ") {
        return Some(extract_signature(line));
    }
    None
}

fn extract_js_ts_symbol(line: &str) -> Option<String> {
    let prefixes = [
        "export function ",
        "export class ",
        "export interface ",
        "export type ",
        "function ",
        "class ",
        "interface ",
        "type ",
    ];
    for prefix in &prefixes {
        if line.starts_with(prefix) {
            return Some(extract_signature(line));
        }
    }
    None
}

fn extract_go_symbol(line: &str) -> Option<String> {
    if line.starts_with("func ") || line.starts_with("type ") {
        return Some(extract_signature(line));
    }
    None
}

fn extract_java_symbol(line: &str) -> Option<String> {
    // Simple heuristic for Java
    let keywords = [
        "public class ",
        "public interface ",
        "public enum ",
        "private class ",
        "protected class ",
        "class ",
        "interface ",
        "enum ",
    ];
    for kw in &keywords {
        if line.starts_with(kw) {
            return Some(extract_signature(line));
        }
    }
    // Method detection: starts with access modifier + return type + name(
    if (line.starts_with("public ") || line.starts_with("private ") || line.starts_with("protected "))
        && line.contains('(')
        && !line.contains("class ")
        && !line.contains("interface ")
        && !line.contains("enum ")
    {
        return Some(extract_signature(line));
    }
    None
}

fn extract_c_symbol(line: &str) -> Option<String> {
    // Simple heuristic: lines with parentheses that look like function declarations
    if line.contains('(')
        && !line.starts_with(' ')
        && !line.starts_with('\t')
        && !line.starts_with("//")
        && !line.starts_with('#')
        && !line.starts_with("/*")
    {
        // Likely a function definition
        return Some(extract_signature(line));
    }
    // struct/enum/typedef
    if line.starts_with("struct ") || line.starts_with("enum ") || line.starts_with("typedef ") {
        return Some(extract_signature(line));
    }
    None
}

fn extract_ruby_symbol(line: &str) -> Option<String> {
    let prefixes = [
        "class ", "module ", "def ", "attr_reader ", "attr_writer ", "attr_accessor ",
    ];
    for prefix in &prefixes {
        if line.starts_with(prefix) {
            return Some(extract_signature(line));
        }
    }
    None
}

fn extract_kotlin_symbol(line: &str) -> Option<String> {
    let prefixes = [
        "fun ", "class ", "data class ", "sealed class ", "object ", "interface ",
        "enum class ", "abstract class ", "open class ",
        "val ", "var ",
        "suspend fun ", "inline fun ", "private fun ", "internal fun ",
        "override fun ", "protected fun ",
    ];
    for prefix in &prefixes {
        if line.starts_with(prefix) {
            return Some(extract_signature(line));
        }
    }
    None
}

/// Extract a clean signature from a line (up to the opening brace or end of line)
fn extract_signature(line: &str) -> String {
    // Truncate at opening brace
    let sig = if let Some(pos) = line.find('{') {
        line[..pos].trim()
    } else {
        line.trim_end_matches(';').trim_end_matches(':').trim()
    };

    // Limit length
    if sig.len() > 120 {
        format!("{}...", &sig[..117])
    } else {
        sig.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repo_map_current_dir() {
        let tool = RepoMapTool;
        let result = tool.execute(json!({})).unwrap();
        // Should find Rust source files
        assert!(result.contains("main.rs") || result.contains("lib.rs"));
    }

    #[test]
    fn test_repo_map_src_dir() {
        let tool = RepoMapTool;
        let result = tool.execute(json!({"path": "src"})).unwrap();
        assert!(result.contains("main.rs"));
        // Should contain some Rust symbols
        assert!(result.contains("fn ") || result.contains("pub fn ") || result.contains("struct "));
    }

    #[test]
    fn test_repo_map_nonexistent_dir() {
        let tool = RepoMapTool;
        let result = tool.execute(json!({"path": "/nonexistent_dir_xyz"}));
        assert!(result.is_err());
    }

    #[test]
    fn test_repo_map_file_instead_of_dir() {
        let tool = RepoMapTool;
        let result = tool.execute(json!({"path": "Cargo.toml"}));
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_rust_symbols() {
        assert_eq!(
            extract_rust_symbol("pub fn hello()"),
            Some("pub fn hello()".to_string())
        );
        assert_eq!(
            extract_rust_symbol("struct Foo {"),
            Some("struct Foo".to_string())
        );
        assert_eq!(
            extract_rust_symbol("impl Tool for GrepTool {"),
            Some("impl Tool for GrepTool".to_string())
        );
        assert_eq!(extract_rust_symbol("    let x = 5;"), None);
    }

    #[test]
    fn test_extract_python_symbols() {
        assert_eq!(
            extract_python_symbol("class MyClass:"),
            Some("class MyClass".to_string())
        );
        assert_eq!(
            extract_python_symbol("def hello(self):"),
            Some("def hello(self)".to_string())
        );
        assert_eq!(
            extract_python_symbol("async def fetch():"),
            Some("async def fetch()".to_string())
        );
        assert_eq!(extract_python_symbol("    x = 5"), None);
    }

    #[test]
    fn test_extract_js_ts_symbols() {
        assert_eq!(
            extract_js_ts_symbol("function hello() {"),
            Some("function hello()".to_string())
        );
        assert_eq!(
            extract_js_ts_symbol("export class Foo {"),
            Some("export class Foo".to_string())
        );
        assert_eq!(
            extract_js_ts_symbol("interface Bar {"),
            Some("interface Bar".to_string())
        );
        assert_eq!(extract_js_ts_symbol("  const x = 5;"), None);
    }

    #[test]
    fn test_extract_go_symbols() {
        assert_eq!(
            extract_go_symbol("func main() {"),
            Some("func main()".to_string())
        );
        assert_eq!(
            extract_go_symbol("type Config struct {"),
            Some("type Config struct".to_string())
        );
        assert_eq!(extract_go_symbol("  var x int"), None);
    }

    #[test]
    fn test_is_source_file() {
        assert!(is_source_file(Path::new("main.rs")));
        assert!(is_source_file(Path::new("app.py")));
        assert!(is_source_file(Path::new("index.ts")));
        assert!(is_source_file(Path::new("main.go")));
        assert!(!is_source_file(Path::new("README.md")));
        assert!(!is_source_file(Path::new("Cargo.toml")));
    }

    #[test]
    fn test_extract_signature_truncation() {
        let long_line = format!("pub fn very_long_function_name(arg1: String, arg2: String, arg3: String, arg4: String, arg5: String, arg6: String) -> Result<String, Error> {{");
        let sig = extract_signature(&long_line);
        assert!(sig.len() <= 120);
        assert!(sig.ends_with("..."));
    }

    #[test]
    fn test_extract_java_symbols() {
        assert_eq!(
            extract_java_symbol("public class MyApp {"),
            Some("public class MyApp".to_string())
        );
        assert_eq!(
            extract_java_symbol("public interface Runnable {"),
            Some("public interface Runnable".to_string())
        );
        assert_eq!(
            extract_java_symbol("public static void main(String[] args) {"),
            Some("public static void main(String[] args)".to_string())
        );
        assert_eq!(
            extract_java_symbol("private int getValue() {"),
            Some("private int getValue()".to_string())
        );
        assert_eq!(extract_java_symbol("    int x = 5;"), None);
    }

    #[test]
    fn test_extract_c_symbols() {
        assert_eq!(
            extract_c_symbol("int main(int argc, char *argv[]) {"),
            Some("int main(int argc, char *argv[])".to_string())
        );
        assert_eq!(
            extract_c_symbol("struct Point {"),
            Some("struct Point".to_string())
        );
        assert_eq!(
            extract_c_symbol("typedef unsigned long size_t;"),
            Some("typedef unsigned long size_t".to_string())
        );
        // Should skip comments and preprocessor directives
        assert_eq!(extract_c_symbol("// comment with parens()"), None);
        assert_eq!(extract_c_symbol("#include <stdio.h>"), None);
        assert_eq!(extract_c_symbol("/* block comment */"), None);
        // Indented lines should not match
        assert_eq!(extract_c_symbol("    int x = func();"), None);
    }

    #[test]
    fn test_repo_map_no_source_files() {
        let dir = "/tmp/hermitclaw_test_repo_map_nosrc";
        std::fs::create_dir_all(dir).ok();
        std::fs::write(format!("{}/readme.md", dir), "# Hello").unwrap();
        std::fs::write(format!("{}/config.toml", dir), "[section]").unwrap();

        let tool = RepoMapTool;
        let result = tool.execute(json!({"path": dir})).unwrap();
        assert_eq!(result, "No source files found.");

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_repo_map_blocked_path() {
        let tool = RepoMapTool;
        let result = tool.execute(json!({"path": "/etc/shadow"}));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::PermissionDenied(_)));
    }

    #[test]
    fn test_repo_map_skips_node_modules() {
        let dir = "/tmp/hermitclaw_test_repo_map_skip";
        std::fs::create_dir_all(format!("{}/node_modules/pkg", dir)).ok();
        std::fs::write(format!("{}/app.js", dir), "function main() {}").unwrap();
        std::fs::write(
            format!("{}/node_modules/pkg/index.js", dir),
            "function hidden() {}",
        )
        .unwrap();

        let tool = RepoMapTool;
        let result = tool.execute(json!({"path": dir})).unwrap();
        assert!(result.contains("app.js"));
        assert!(!result.contains("node_modules"));

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_extract_signature_no_brace() {
        // Line without opening brace (e.g., Python)
        assert_eq!(
            extract_signature("def hello(self):"),
            "def hello(self)".to_string()
        );
    }

    #[test]
    fn test_extract_signature_with_semicolon() {
        assert_eq!(
            extract_signature("typedef int myint;"),
            "typedef int myint".to_string()
        );
    }

    #[test]
    fn test_is_source_file_edge_cases() {
        assert!(is_source_file(Path::new("component.tsx")));
        assert!(is_source_file(Path::new("component.jsx")));
        assert!(is_source_file(Path::new("main.c")));
        assert!(is_source_file(Path::new("main.cpp")));
        assert!(is_source_file(Path::new("header.h")));
        assert!(is_source_file(Path::new("App.java")));
        assert!(is_source_file(Path::new("server.rb")));
        assert!(is_source_file(Path::new("app.kt")));
        assert!(is_source_file(Path::new("build.gradle.kts")));
        assert!(!is_source_file(Path::new("no_extension")));
        assert!(!is_source_file(Path::new(".hidden")));
    }

    #[test]
    fn test_extract_rust_advanced_symbols() {
        // pub(crate)
        assert!(extract_rust_symbol("pub(crate) fn helper()").is_some());
        assert!(extract_rust_symbol("pub(super) fn parent()").is_some());
        // async/const/unsafe fn
        assert!(extract_rust_symbol("pub async fn fetch()").is_some());
        assert!(extract_rust_symbol("async fn do_stuff()").is_some());
        assert!(extract_rust_symbol("pub const fn size()").is_some());
        assert!(extract_rust_symbol("const fn zero()").is_some());
        assert!(extract_rust_symbol("pub unsafe fn raw_ptr()").is_some());
        assert!(extract_rust_symbol("unsafe fn danger()").is_some());
        // type alias, const, static
        assert!(extract_rust_symbol("pub type Result<T> = std::result::Result<T, Error>;").is_some());
        assert!(extract_rust_symbol("pub const MAX: usize = 100;").is_some());
        assert!(extract_rust_symbol("pub static INSTANCE: Lazy<Foo> = Lazy::new(|| Foo);").is_some());
        // macro_rules
        assert!(extract_rust_symbol("macro_rules! my_macro {").is_some());
    }

    #[test]
    fn test_extract_ruby_symbols() {
        assert!(extract_ruby_symbol("class MyClass").is_some());
        assert!(extract_ruby_symbol("module MyModule").is_some());
        assert!(extract_ruby_symbol("def initialize(name)").is_some());
        assert!(extract_ruby_symbol("attr_reader :name").is_some());
        assert_eq!(extract_ruby_symbol("  x = 5"), None);
    }

    #[test]
    fn test_extract_kotlin_symbols() {
        assert!(extract_kotlin_symbol("fun main(args: Array<String>) {").is_some());
        assert!(extract_kotlin_symbol("class MyClass {").is_some());
        assert!(extract_kotlin_symbol("data class User(val name: String)").is_some());
        assert!(extract_kotlin_symbol("sealed class Result {").is_some());
        assert!(extract_kotlin_symbol("object Singleton {").is_some());
        assert!(extract_kotlin_symbol("interface Callback {").is_some());
        assert!(extract_kotlin_symbol("enum class Color {").is_some());
        assert!(extract_kotlin_symbol("suspend fun fetch()").is_some());
        assert_eq!(extract_kotlin_symbol("    val x = 5"), None);
    }

    #[test]
    fn test_symbols_include_line_numbers() {
        let dir = "/tmp/hermitclaw_test_repo_map_lineno";
        std::fs::create_dir_all(dir).ok();
        std::fs::write(
            format!("{}/test.rs", dir),
            "// comment\nfn hello() {\n}\n\npub struct Foo {\n}\n",
        )
        .unwrap();

        let tool = RepoMapTool;
        let result = tool.execute(json!({"path": dir})).unwrap();
        assert!(result.contains("L2:"), "Should contain line number for fn hello");
        assert!(result.contains("L5:"), "Should contain line number for struct Foo");

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_extract_rust_enum_and_trait() {
        assert_eq!(
            extract_rust_symbol("pub enum Color {"),
            Some("pub enum Color".to_string())
        );
        assert_eq!(
            extract_rust_symbol("pub trait Display {"),
            Some("pub trait Display".to_string())
        );
        assert_eq!(
            extract_rust_symbol("mod tests {"),
            Some("mod tests".to_string())
        );
    }

    #[test]
    fn test_extract_js_ts_type_and_interface() {
        assert_eq!(
            extract_js_ts_symbol("type Props = {"),
            Some("type Props =".to_string())
        );
        assert_eq!(
            extract_js_ts_symbol("export interface Config {"),
            Some("export interface Config".to_string())
        );
        assert_eq!(
            extract_js_ts_symbol("export type Result = {"),
            Some("export type Result =".to_string())
        );
    }
}
