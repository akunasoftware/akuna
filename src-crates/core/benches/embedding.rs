//! Benchmarks embedding throughput over the configured embedding corpus.

use std::{hint::black_box, path::PathBuf};

use akuna_core::{
    embedding,
    extraction::{self, FileExtractionError},
    testing::get_embedding_corpus_root,
};
use criterion::{Criterion, criterion_group, criterion_main};
use walkdir::WalkDir;

fn embedding_bench(c: &mut Criterion) {
    let runtime =
        tokio::runtime::Runtime::new().expect("failed to create tokio runtime");

    let file_paths: Vec<PathBuf> = WalkDir::new(get_embedding_corpus_root())
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();

    let contents = runtime.block_on(async {
        let mut contents = Vec::new();

        for file_path in &file_paths {
            let extraction = match extraction::extract_file(
                file_path,
                &akuna_core::ExtractionConfig {
                    return_content: true,
                    ..akuna_core::ExtractionConfig::default()
                },
            )
            .await
            {
                Ok(extraction) => extraction,
                Err(FileExtractionError::UnsupportedFileType { .. }) => {
                    continue;
                }
                Err(e) => panic!("Unexpected text extraction error: {:?}", e),
            };

            if let Some(content) = extraction.content
                && let Some(text) = content.text
            {
                contents.push(text);
            }
        }
        contents
    });

    let mut contents_expanded = Vec::with_capacity(contents.len() * 10);
    for _ in 0..10 {
        contents_expanded.extend_from_slice(&contents);
    }

    let model = runtime
        .block_on(async { embedding::model().await })
        .expect("failed loading embedding model");

    c.bench_function("embed_files_in_dir", |b| {
        b.iter(|| {
            let embedding = model
                .embed_batch(black_box(&contents_expanded), None)
                .expect("failed generating embedding");

            black_box(embedding);
        });
    });
}

criterion_group!(embedding, embedding_bench);
criterion_main!(embedding);
