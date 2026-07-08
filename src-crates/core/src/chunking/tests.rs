use crate::chunking::code::segment_code;
use crate::chunking::packer::{Chunk, pack};
use crate::chunking::prose::segment_prose;
use crate::chunking::{ChunkingOptions, PartKind, Segment};

fn texts(chunks: Vec<Chunk>) -> Vec<String> {
    chunks.into_iter().map(|chunk| chunk.text).collect()
}

fn test_segment(text: &str) -> Vec<Segment<'_>> {
    segment_prose(text)
}

#[test]
fn api_options_default_enabled() {
    assert_eq!(
        ChunkingOptions::default(),
        ChunkingOptions {
            enabled: true,
            max_chars: 1600,
            overlap_chars: 200,
        }
    );
}

#[test]
fn prose_splits_blank_lines() {
    let text = " first line\ncontinues \n\n second \r\n\r\nthird";
    let segments = segment_prose(text);

    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].text.trim(), "first line\ncontinues");
    assert_eq!(segments[0].kind, PartKind::Paragraph);
    assert_eq!(segments[0].byte_range, 0..22);
    assert_eq!(segments[1].text.trim(), "second");
    assert_eq!(segments[2].text.trim(), "third");
}

#[test]
fn code_unsupported_extension_returns_none() {
    assert!(segment_code("let x = 1;", Some("txt")).is_none());
}

#[test]
fn code_segments_named_leaves() {
    let Some(segments) =
        segment_code("fn main() {\n    println!(\"hi\");\n}\n", Some("rs"))
    else {
        panic!("rust source should segment");
    };

    assert!(segments.len() > 1);
    assert!(
        segments
            .iter()
            .all(|segment| segment.kind == PartKind::Code)
    );
    assert!(
        segments
            .iter()
            .all(|segment| !segment.text.trim().is_empty())
    );
}

#[test]
fn code_extension_is_case_insensitive() {
    assert!(segment_code("fn main() {}", Some("RS")).is_some());
}

#[test]
fn pack_empty_returns_zero() {
    let text = " \n\n\t ";
    let chunks = pack(text, &test_segment(text), &ChunkingOptions::default());

    assert!(chunks.is_empty());
}

#[test]
fn pack_disabled_preserves_record_text() {
    let text = "  alpha\n\n beta  ";
    let chunks = pack(
        text,
        &test_segment(text),
        &ChunkingOptions {
            enabled: false,
            ..Default::default()
        },
    );

    assert_eq!(texts(chunks), vec![text]);
}

#[test]
fn pack_segments_greedy_without_overlap() {
    let text = "alpha\n\nbeta\n\ngamma";
    let chunks = pack(
        text,
        &test_segment(text),
        &ChunkingOptions {
            max_chars: 13,
            overlap_chars: 4,
            ..Default::default()
        },
    );

    assert_eq!(texts(chunks), vec!["alpha\n\nbeta", "gamma"]);
}

#[test]
fn pack_long_segment_prefers_sentence() {
    let text = "One. Two. Three.";
    let chunks = pack(
        text,
        &test_segment(text),
        &ChunkingOptions {
            max_chars: 10,
            overlap_chars: 0,
            ..Default::default()
        },
    );

    assert_eq!(texts(chunks), vec!["One. Two.", "Three."]);
}

#[test]
fn pack_long_segment_prefers_whitespace() {
    let text = "alpha beta gamma";
    let chunks = pack(
        text,
        &test_segment(text),
        &ChunkingOptions {
            max_chars: 12,
            overlap_chars: 0,
            ..Default::default()
        },
    );

    assert_eq!(texts(chunks), vec!["alpha beta", "gamma"]);
}

#[test]
fn pack_long_segment_hard_cuts_unbroken() {
    let text = "abcdefghijkl";
    let chunks = pack(
        text,
        &test_segment(text),
        &ChunkingOptions {
            max_chars: 5,
            overlap_chars: 0,
            ..Default::default()
        },
    );

    assert_eq!(texts(chunks), vec!["abcde", "fghij", "kl"]);
}

#[test]
fn pack_long_segment_overlaps() {
    let text = "0123456789abcdefghij";
    let chunks = pack(
        text,
        &test_segment(text),
        &ChunkingOptions {
            max_chars: 10,
            overlap_chars: 3,
            ..Default::default()
        },
    );

    assert_eq!(texts(chunks), vec!["0123456789", "789abcdefg", "efghij"]);
}

#[test]
fn pack_segments_do_not_overlap() {
    let text = "abcdef\n\nghijkl";
    let chunks = pack(
        text,
        &test_segment(text),
        &ChunkingOptions {
            max_chars: 8,
            overlap_chars: 5,
            ..Default::default()
        },
    );

    assert_eq!(texts(chunks), vec!["abcdef", "ghijkl"]);
}

#[test]
fn pack_unicode_hard_cuts_safely() {
    let text = "a😀b😀c";
    let chunks = pack(
        text,
        &test_segment(text),
        &ChunkingOptions {
            max_chars: 2,
            overlap_chars: 0,
            ..Default::default()
        },
    );

    assert_eq!(texts(chunks), vec!["a😀", "b😀", "c"]);
}

#[test]
fn options_reject_invalid_limits() {
    assert_eq!(
        ChunkingOptions {
            max_chars: 0,
            ..Default::default()
        }
        .validate(),
        Err(crate::chunking::ChunkingError::ZeroMaxChars)
    );
    assert_eq!(
        ChunkingOptions {
            max_chars: 8,
            overlap_chars: 8,
            ..Default::default()
        }
        .validate(),
        Err(crate::chunking::ChunkingError::OverlapTooLarge {
            overlap_chars: 8,
            max_chars: 8,
        })
    );
}

#[test]
#[should_panic(expected = "invalid chunking options")]
fn pack_rejects_invalid_limits() {
    let text = "abc";
    let _ = pack(
        text,
        &test_segment(text),
        &ChunkingOptions {
            max_chars: 0,
            ..Default::default()
        },
    );
}

#[test]
fn pack_deterministic() {
    let text = "alpha beta gamma delta";
    let options = ChunkingOptions {
        max_chars: 11,
        overlap_chars: 2,
        ..Default::default()
    };

    assert_eq!(
        pack(text, &test_segment(text), &options),
        pack(text, &test_segment(text), &options)
    );
}
