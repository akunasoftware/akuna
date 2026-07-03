//! Shared helpers for akuna-core parity integration tests.
//!
//! Provides fixture/script path resolution, generic `uv` reference-script
//! runners, and string-comparison helpers reused across the embedding,
//! reranking, and OCR parity suites.
//!
//! Included via `#[path = "common.rs"] mod common;` into multiple test
//! binaries; not every binary uses every helper, so dead code is allowed.

#![allow(dead_code)]

use std::fmt::{self, Display};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use anyhow::{Context, Result};
use regex::Regex;
use serde::Serialize;
use serde::de::DeserializeOwned;

const HF_REPO_TEST_CORPUS: &str = "akunasoftware/test-corpus";
const HF_REPO_CONTENT_PREFIX: &str = "content/";

/// A struct representing a test fixture file stored on huggingface
#[derive(Debug)]
pub struct CorpusFixture {
    /// String representing relative file path within HF repo
    repo_file_path: String,
}

impl CorpusFixture {
    /// Initialise a fixture by literal repo file path
    pub fn new(repo_file_path: &str) -> Self {
        Self {
            repo_file_path: repo_file_path.to_string(),
        }
    }

    /// Provides a local path to a downloaded/cached test fixture
    pub fn get(&self) -> anyhow::Result<PathBuf> {
        Ok(get_corpus_repo()?.download(&self.repo_file_path)?)
    }
}

impl Display for CorpusFixture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.repo_file_path)
    }
}

/// Resolves a reference script at `<manifest>/../../scripts/<name>`.
pub fn script_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts")
        .join(name)
}

/// Spawns `uv run <script_rel> <args>`, writes `input` as JSON on stdin, and
/// returns parsed JSON from stdout.
///
/// `script_rel` is resolved relative to `akuna/scripts/` via [`script_path`].
pub fn run_uv_script_json<I, O>(
    script_rel: &str,
    args: &[&str],
    input: &I,
) -> Result<O>
where
    I: Serialize,
    O: DeserializeOwned,
{
    let script = script_path(script_rel);
    let script_str = script.to_str().context("non-utf8 script path")?;
    let mut child = Command::new("uv")
        .args(["run", script_str])
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to spawn uv script {script_rel}"))?;

    {
        let mut stdin = child
            .stdin
            .take()
            .context("failed to open reference script stdin")?;
        let payload = serde_json::to_vec(input)
            .context("failed to serialize stdin input")?;
        stdin
            .write_all(&payload)
            .context("failed to write stdin input")?;
    }

    let output = child.wait_with_output().with_context(|| {
        format!("failed to wait for uv script {script_rel}")
    })?;

    parse_script_output(script_rel, output)
}

/// Spawns `uv run <script_rel> <args>` with no stdin and returns parsed JSON
/// from stdout.
///
/// `script_rel` is resolved relative to `akuna/scripts/` via [`script_path`].
pub fn run_uv_script_args<O>(script_rel: &str, args: &[&str]) -> Result<O>
where
    O: DeserializeOwned,
{
    let script = script_path(script_rel);
    let script_str = script.to_str().context("non-utf8 script path")?;
    let output = Command::new("uv")
        .args(["run", script_str])
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to spawn uv script {script_rel}"))?;

    parse_script_output(script_rel, output)
}

fn parse_script_output<O: DeserializeOwned>(
    script_rel: &str,
    output: Output,
) -> Result<O> {
    if !output.status.success() {
        anyhow::bail!(
            "uv script {script_rel} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("failed to parse {script_rel} output"))
}

/// Lowercases and collapses all whitespace to single spaces.
pub fn normalise_text(text: &str) -> String {
    text.chars()
        .filter_map(|c| c.to_lowercase().next())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Joins block texts and normalises the combined string.
pub fn aggregated_text<T: AsRef<str>>(blocks: &[T]) -> String {
    let combined = blocks
        .iter()
        .map(|b| b.as_ref().to_owned())
        .collect::<Vec<_>>()
        .join(" ");
    normalise_text(&combined)
}

/// Levenshtein edit distance between two strings.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev = (0..=b.len()).collect::<Vec<_>>();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, &ac) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &bc) in b.iter().enumerate() {
            let cost = if ac == bc { 0 } else { 1 };
            curr[j + 1] =
                (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Similarity ratio between two strings: `1 - levenshtein / max_len`.
///
/// Returns `1.0` when both strings are empty.
pub fn similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let max_len = a.len().max(b.len());
    if max_len == 0 {
        return 1.0;
    }
    1.0 - (levenshtein(a, b) as f64 / max_len as f64)
}

fn get_corpus_repo() -> Result<hf_hub::api::sync::ApiRepo> {
    let client = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(false)
        .build()?;
    Ok(client.dataset(HF_REPO_TEST_CORPUS.into()))
}

/// Get a list of all available corpus fixture files, with optional path filtering
pub fn get_corpus_files(
    regex_filter: Option<&str>,
) -> anyhow::Result<Vec<CorpusFixture>> {
    let repo = get_corpus_repo()?;
    let info = repo.info()?;

    let pattern = match regex_filter {
        Some(path) => format!("^{HF_REPO_CONTENT_PREFIX}{path}"),
        None => format!("^{HF_REPO_CONTENT_PREFIX}"),
    };
    let pattern = Regex::new(&pattern)?;

    let fixtures: Vec<_> = info
        .siblings
        .into_iter()
        .map(|sibling| sibling.rfilename)
        .filter(|filename| pattern.is_match(filename))
        .map(|file_name| CorpusFixture {
            repo_file_path: file_name,
        })
        .collect();

    Ok(fixtures)
}
