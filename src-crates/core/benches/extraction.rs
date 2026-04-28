//! Benchmarks text extraction throughput across supported fixture file types.

use std::hint::black_box;

use akuna_core::{extraction::extract_file, testing::get_extraction_fixture};
use criterion::{criterion_group, criterion_main};

macro_rules! extraction_bench {
    ($bench_name:ident, $file_name:expr) => {
        fn $bench_name(criterion: &mut criterion::Criterion) {
            let runtime = tokio::runtime::Runtime::new()
                .expect("failed to create Criterion tokio runtime");

            let file_path = get_extraction_fixture($file_name);

            criterion.bench_function($file_name, |bencher| {
                bencher.to_async(&runtime).iter(|| async {
                    black_box(
                        extract_file(
                            &file_path,
                            &akuna_core::ExtractionConfig {
                                return_content: true,
                                ..akuna_core::ExtractionConfig::default()
                            },
                        )
                        .await
                        .expect("extraction benchmark fixture should load"),
                    )
                })
            });
        }
    };
}

extraction_bench!(bench_text_txt, "text.txt");
extraction_bench!(bench_text_md, "text.md");
extraction_bench!(bench_text_pdf, "text.pdf");
extraction_bench!(bench_text_odt, "text.odt");
extraction_bench!(bench_text_docx, "text.docx");
extraction_bench!(bench_text_html, "text.html");
extraction_bench!(bench_text_rtf, "text.rtf");

extraction_bench!(bench_code_js, "code.js");
extraction_bench!(bench_code_ts, "code.ts");
extraction_bench!(bench_code_py, "code.py");
extraction_bench!(bench_code_rs, "code.rs");
extraction_bench!(bench_code_go, "code.go");
extraction_bench!(bench_code_java, "code.java");
extraction_bench!(bench_code_c, "code.c");
extraction_bench!(bench_code_cpp, "code.cpp");
extraction_bench!(bench_code_cs, "code.cs");
extraction_bench!(bench_code_php, "code.php");
extraction_bench!(bench_code_rb, "code.rb");
extraction_bench!(bench_code_sh, "code.sh");
extraction_bench!(bench_code_sql, "code.sql");
extraction_bench!(bench_code_css, "code.css");
extraction_bench!(bench_code_html, "code.html");
extraction_bench!(bench_code_yaml, "code.yaml");
extraction_bench!(bench_code_toml, "code.toml");

criterion_group!(
    extraction_benches,
    bench_text_txt,
    bench_text_md,
    bench_text_pdf,
    bench_text_odt,
    bench_text_docx,
    bench_text_html,
    bench_text_rtf,
    bench_code_js,
    bench_code_ts,
    bench_code_py,
    bench_code_rs,
    bench_code_go,
    bench_code_java,
    bench_code_c,
    bench_code_cpp,
    bench_code_cs,
    bench_code_php,
    bench_code_rb,
    bench_code_sh,
    bench_code_sql,
    bench_code_css,
    bench_code_html,
    bench_code_yaml,
    bench_code_toml,
);
criterion_main!(extraction_benches);
