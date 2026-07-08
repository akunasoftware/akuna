use crate::chunking::{ChunkingOptions, Segment};

/// Retrieval chunk text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Chunk {
    pub(crate) text: String,
}

/// Pack source segments into retrieval chunks.
pub(crate) fn pack(
    text: &str,
    segments: &[Segment<'_>],
    options: &ChunkingOptions,
) -> Vec<Chunk> {
    if let Err(error) = options.validate() {
        panic!("invalid chunking options: {error}");
    }
    if !options.enabled {
        return whole_text_chunk(text);
    }

    let max_chars = options.max_chars;
    let overlap_chars = options.overlap_chars;
    let mut chunks = Vec::new();
    let mut current = String::new();

    for segment in segments {
        let segment_text = segment.text.trim();
        if segment_text.is_empty() {
            continue;
        }

        if char_count(segment_text) > max_chars {
            push_current(&mut chunks, &mut current);
            append_split_pieces(
                &mut chunks,
                &mut current,
                split_segment(segment_text, max_chars, overlap_chars),
            );
            continue;
        }

        push_segment(&mut chunks, &mut current, segment_text, max_chars);
    }

    push_current(&mut chunks, &mut current);
    chunks
}

/// Build a single disabled-mode chunk.
fn whole_text_chunk(text: &str) -> Vec<Chunk> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    vec![Chunk {
        text: text.to_owned(),
    }]
}

/// Add a whole segment to the current chunk.
fn push_segment(
    chunks: &mut Vec<Chunk>,
    current: &mut String,
    segment_text: &str,
    max_chars: usize,
) {
    let separator_chars = if current.is_empty() { 0 } else { 2 };
    let needed_chars = separator_chars + char_count(segment_text);

    if char_count(current) + needed_chars > max_chars {
        push_current(chunks, current);
    }

    if !current.is_empty() {
        current.push_str("\n\n");
    }
    current.push_str(segment_text);
}

/// Add finished split pieces to chunks.
fn append_split_pieces(
    chunks: &mut Vec<Chunk>,
    current: &mut String,
    pieces: Vec<String>,
) {
    let mut pieces = pieces.into_iter().peekable();
    while let Some(piece) = pieces.next() {
        if pieces.peek().is_some() {
            chunks.push(Chunk { text: piece });
            continue;
        }

        current.push_str(&piece);
    }
}

/// Finish the current chunk.
fn push_current(chunks: &mut Vec<Chunk>, current: &mut String) {
    if current.is_empty() {
        return;
    }

    chunks.push(Chunk {
        text: std::mem::take(current),
    });
}

/// Split one oversized segment.
fn split_segment(
    text: &str,
    max_chars: usize,
    overlap_chars: usize,
) -> Vec<String> {
    let chars = text.chars().collect::<Vec<_>>();
    let boundaries = char_boundaries(text);
    let mut pieces = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        let limit = (start + max_chars).min(chars.len());
        let end = if limit == chars.len() {
            chars.len()
        } else {
            choose_end(&chars, start, limit, overlap_chars)
        };
        let piece = text[boundaries[start]..boundaries[end]].trim();
        if !piece.is_empty() {
            pieces.push(piece.to_owned());
        }
        if end == chars.len() {
            break;
        }

        start = end.saturating_sub(overlap_chars);
    }

    pieces
}

/// Choose a split boundary inside the allowed window.
fn choose_end(
    chars: &[char],
    start: usize,
    limit: usize,
    overlap_chars: usize,
) -> usize {
    let min_end = start + overlap_chars + 1;
    if let Some(end) = sentence_boundary(chars, start, limit, min_end) {
        return end;
    }
    if let Some(end) = whitespace_boundary(chars, start, limit, min_end) {
        return end;
    }

    limit
}

/// Find the last sentence boundary in a window.
fn sentence_boundary(
    chars: &[char],
    start: usize,
    limit: usize,
    min_end: usize,
) -> Option<usize> {
    let mut boundary = None;
    for (index, character) in chars.iter().enumerate().take(limit).skip(start) {
        if !matches!(*character, '.' | '!' | '?') {
            continue;
        }

        let end = index + 1;
        if end < min_end {
            continue;
        }
        if end < chars.len() && !chars[end].is_whitespace() {
            continue;
        }

        boundary = Some(end);
    }

    boundary
}

/// Find the last word boundary in a window.
fn whitespace_boundary(
    chars: &[char],
    start: usize,
    limit: usize,
    min_end: usize,
) -> Option<usize> {
    let mut boundary = None;
    for (index, character) in
        chars.iter().enumerate().take(limit).skip(start + 1)
    {
        if index >= min_end && character.is_whitespace() {
            boundary = Some(index);
        }
    }

    boundary
}

/// Return byte boundaries for character indexes.
fn char_boundaries(text: &str) -> Vec<usize> {
    let mut boundaries = text
        .char_indices()
        .map(|(index, _character)| index)
        .collect::<Vec<_>>();
    boundaries.push(text.len());
    boundaries
}

/// Count Unicode scalar values.
fn char_count(text: &str) -> usize {
    text.chars().count()
}
