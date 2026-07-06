//! Python binding generator.

use std::env;
use std::io::{Error, ErrorKind};

use uniffi_bindgen::bindings::GenerateOptions;
use uniffi_bindgen::bindings::TargetLanguage;
use uniffi_bindgen::bindings::generate;

/// Generates Python bindings.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        let program = args
            .first()
            .map_or("uniffi-bindgen", std::string::String::as_str);
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("usage: {program} <cdylib-path> <out-dir>"),
        )
        .into());
    }

    generate(GenerateOptions {
        languages: vec![TargetLanguage::Python],
        source: args[1].clone().into(),
        out_dir: args[2].clone().into(),
        config_override: None,
        format: true,
        crate_filter: None,
        metadata_no_deps: false,
    })?;

    Ok(())
}
