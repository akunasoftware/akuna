use std::path::PathBuf;

/// Returns the path to the shared test corpus directory, optionally joined with a subpath.
pub fn get_corpus_path() -> PathBuf {
    PathBuf::from("/Users/smissingham/Projects/akunasoftware/test-corpus/")
}

/// Returns an owned pathbuf for a named fixture from test corpus fixtures
pub fn get_extraction_fixture(file_path: &str) -> PathBuf {
    get_corpus_path()
        .join("content")
        .join("fixtures")
        .join(file_path)
}

/// Returns an owned pathbuf root dir for all embedding bench/test fixtures
pub fn get_embedding_corpus_root() -> PathBuf {
    get_corpus_path().join("content")
}
