//! Player-editable config files under `config/` (gitignored), one per concern:
//! `graphics.toml` (§13, see `graphics`) and `controls.toml` (§14, see
//! `controls`).
//!
//! Both are a flat `key = value` subset of TOML, parsed by hand; values are
//! scalars or an array of strings. Keeping each concern in its own file is
//! what keeps them flat, so tables (and a TOML crate) aren't needed. Two rules
//! shape everything here:
//!
//! - **Never fatal.** A missing file is created with defaults; a bad line is a
//!   warning and leaves that key at its default.
//! - **Never destructive.** An unreadable file is left untouched, not replaced,
//!   and saving (graphics only) edits one value in place.

pub mod controls;
pub mod graphics;

use std::fs;
use std::io;
use std::path::Path;

/// Byte index where a `#` comment starts (outside a quoted string), if any.
fn comment_start(line: &str) -> Option<usize> {
    let mut quoted = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '#' if !quoted => return Some(i),
            _ => {}
        }
    }
    None
}

/// The `key = value` part of a line: comment stripped, trimmed.
fn code(line: &str) -> &str {
    line[..comment_start(line).unwrap_or(line.len())].trim()
}

/// One `key = value` line.
struct Entry<'a> {
    line: usize, // 1-based
    key: &'a str,
    value: &'a str,
}

/// A warning about line `line`, in the one format both files use.
fn warning(line: usize, msg: impl std::fmt::Display) -> String {
    format!("line {line}: {msg}; ignored")
}

/// Every `key = value` line of `text` (keys and values trimmed, comments
/// dropped), plus a warning for each non-blank line that isn't one.
fn entries(text: &str) -> (Vec<Entry<'_>>, Vec<String>) {
    let mut out = Vec::new();
    let mut warnings = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = code(raw);
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            warnings.push(warning(
                i + 1,
                format!("tables are not supported (`{line}`)"),
            ));
            continue;
        }
        match line.split_once('=') {
            Some((k, v)) => out.push(Entry {
                line: i + 1,
                key: k.trim(),
                value: v.trim(),
            }),
            None => warnings.push(warning(
                i + 1,
                format!("expected `key = value`, got `{line}`"),
            )),
        }
    }
    (out, warnings)
}

/// The contents of a `"quoted"` string (no escapes: none of our values need them).
fn string(v: &str) -> Option<&str> {
    let s = v.strip_prefix('"')?.strip_suffix('"')?;
    (!s.contains('"') && !s.contains('\\')).then_some(s)
}

/// `["a", "b"]` (trailing comma allowed, `[]` is empty) or a lone `"a"`.
fn string_list(v: &str) -> Option<Vec<&str>> {
    if let Some(inner) = v.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
        let inner = inner.trim();
        let inner = inner.strip_suffix(',').unwrap_or(inner);
        if inner.trim().is_empty() {
            return Some(Vec::new());
        }
        inner.split(',').map(|item| string(item.trim())).collect()
    } else {
        string(v).map(|s| vec![s])
    }
}

/// Write via a temp file + rename, so a crash mid-write never leaves a
/// truncated file (the same pattern as the texture bake, §17).
fn write_atomic(path: &Path, text: &str) -> io::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("toml.tmp");
    fs::write(&tmp, text)?;
    fs::rename(&tmp, path)
}

/// The text of `path`, and whether it had to be created from `default` first.
/// An error means it could neither be read nor created: the caller then runs
/// on defaults and must not write to it.
fn open_or_create(path: &Path, default: impl FnOnce() -> String) -> io::Result<(String, bool)> {
    match fs::read_to_string(path) {
        Ok(text) => Ok((text, false)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let text = default();
            write_atomic(path, &text)?;
            Ok((text, true))
        }
        Err(e) => Err(e),
    }
}

/// A fresh, empty directory per test under the system temp dir.
#[cfg(test)]
fn test_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("feather-config-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_inside_quotes_is_not_a_comment() {
        assert_eq!(comment_start("a = \"x#y\" # c"), Some(10));
        assert_eq!(code("  # only a comment"), "");
    }

    #[test]
    fn string_lists() {
        assert_eq!(string_list("[\"W\", \"Up\"]"), Some(vec!["W", "Up"]));
        assert_eq!(string_list("[ \"W\", ]"), Some(vec!["W"]));
        assert_eq!(string_list("[]"), Some(vec![]));
        assert_eq!(string_list("\"W\""), Some(vec!["W"]));
        for bad in [
            "W",
            "[W]",
            "[\"W\" \"S\"]",
            "[\"W\"",
            "\"a\\b\"",
            "[\"\"\"]",
        ] {
            assert_eq!(string_list(bad), None, "`{bad}` should not parse");
        }
    }

    #[test]
    fn entries_split_lines_and_warn_on_junk() {
        let (e, w) = entries("# c\n a = 1 # x\n[t]\nb=\"#\"\njunk\n");
        let got: Vec<_> = e.iter().map(|e| (e.line, e.key, e.value)).collect();
        assert_eq!(got, vec![(2, "a", "1"), (4, "b", "\"#\"")]);
        assert_eq!(w.len(), 2, "{w:?}");
    }
}
