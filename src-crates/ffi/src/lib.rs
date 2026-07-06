//! UniFFI bindings crate.

pub mod detection;
pub mod embedding;
pub mod extraction;
pub mod layout;
pub mod ocr;
pub mod reranking;
mod stack;

uniffi::setup_scaffolding!("akuna_core");
