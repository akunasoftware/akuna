//! Parity tests comparing akuna-core detection output against the rust
//! `magika` crate as a reference implementation.
//!
//! These tests require fixture files at `<workspace>/target/fixtures/` and
//! are therefore marked `#[ignore]` by default. Run with:
//!
//! ```sh
//! cargo test -p akuna-core --test magika_parity -- --ignored --nocapture
//! ```
#![cfg(feature = "detection")]

use std::fs;

#[path = "common.rs"]
mod common;

use akuna_core::detection::Session;

#[ignore = "requires fixture files at target/fixtures/"]
#[test]
fn parity_against_rust_magika_on_repo_fixtures() {
    // `new` picks wgpu when a GPU is available, ndarray CPU otherwise.
    let session = Session::new().expect("build session");

    assert_parity_against_rust_magika(&session);
}

fn assert_parity_against_rust_magika(session: &Session) {
    let fixture_files = common::fixture_files();

    let mut rust_magika =
        magika::Session::new().expect("build rust magika session");
    let expected = fixture_files
        .iter()
        .map(|path| {
            let detection =
                rust_magika.identify_file_sync(path).unwrap_or_else(|err| {
                    panic!("rust magika failed for {path:?}: {err}")
                });
            (
                path.clone(),
                detection.info().label.to_string(),
                detection.info().mime_type.to_string(),
            )
        })
        .collect::<Vec<_>>();

    let actual = fixture_files
        .iter()
        .map(|path| {
            let bytes = fs::read(path)
                .unwrap_or_else(|err| panic!("failed to read {path:?}: {err}"));
            let info = session
                .identify_content_sync(&bytes)
                .expect("classify fixture")
                .info();
            (info.label.to_string(), info.mime_type.to_string())
        })
        .collect::<Vec<_>>();

    let mismatches = expected
        .into_iter()
        .zip(actual)
        .filter_map(|((path, rust_label, rust_mime_type), ours)| {
            if ours.0 == rust_label && ours.1 == rust_mime_type {
                return None;
            }

            Some((path, rust_label, rust_mime_type, ours.0, ours.1))
        })
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "found {} parity mismatches across fixtures, first few: {:#?}",
        mismatches.len(),
        mismatches.into_iter().take(10).collect::<Vec<_>>()
    );
}
