use crate::chunking::{PartKind, Segment};

/// Split prose into paragraph segments.
pub(crate) fn segment_prose(text: &str) -> Vec<Segment<'_>> {
    let mut segments = Vec::new();
    let mut paragraph_start = None;
    let mut paragraph_end = 0;
    let mut line_start = 0;

    for line in text.split_inclusive('\n') {
        let line_end = line_start + line.len();
        if line.trim().is_empty() {
            push_paragraph(
                text,
                paragraph_start.take(),
                paragraph_end,
                &mut segments,
            );
            line_start = line_end;
            continue;
        }

        if paragraph_start.is_none() {
            paragraph_start = Some(line_start);
        }
        paragraph_end = content_end(line_start, line);
        line_start = line_end;
    }

    push_paragraph(text, paragraph_start, paragraph_end, &mut segments);
    segments
}

/// Add a paragraph segment when text is present.
fn push_paragraph<'a>(
    source: &'a str,
    start: Option<usize>,
    end: usize,
    segments: &mut Vec<Segment<'a>>,
) {
    let Some(start) = start else {
        return;
    };
    let Some(text) = source.get(start..end) else {
        return;
    };
    if text.trim().is_empty() {
        return;
    }

    segments.push(Segment {
        text,
        kind: PartKind::Paragraph,
        byte_range: start..end,
    });
}

/// Return the byte end before a line terminator.
fn content_end(line_start: usize, line: &str) -> usize {
    let mut end = line_start + line.len();
    if line.ends_with('\n') {
        end -= 1;
    }
    if end > line_start
        && line.as_bytes().get(end - line_start - 1) == Some(&b'\r')
    {
        end -= 1;
    }

    end
}
