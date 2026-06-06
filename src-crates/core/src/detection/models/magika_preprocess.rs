use crate::detection::{config::ModelConfig, vendor::content::ContentType};

/// Pre-classified result for an input that bypasses model inference.
pub(crate) enum PreparedInput {
    /// Feature vector ready for model inference.
    Features(Vec<i32>),
    /// Rule-resolved content type, skipping inference.
    Ruled(ContentType),
}

/// Preprocesses `input` into either model features or a rule-based content type.
pub(crate) fn prepare_input(
    input: &[u8],
    config: &ModelConfig,
) -> PreparedInput {
    if input.is_empty() {
        return PreparedInput::Ruled(ContentType::Empty);
    }

    let first_block_len = config.block_size.min(input.len());
    let first_block = &input[..first_block_len];
    let buffer_size = config.block_size.min(input.len());
    let beg = strip_prefix(&input[..buffer_size]);
    let end = strip_suffix(&input[input.len() - buffer_size..]);

    let mut features = vec![config.padding_token; config.features_size()];
    let (beg_features, end_features) = features.split_at_mut(config.beg_size);
    copy_features(beg_features, beg, 0);
    copy_features(end_features, end, 1);

    if features[config.min_file_size_for_dl - 1] != config.padding_token {
        return PreparedInput::Features(features);
    }

    if std::str::from_utf8(first_block).is_ok() {
        PreparedInput::Ruled(ContentType::Txt)
    } else {
        PreparedInput::Ruled(ContentType::Unknown)
    }
}

fn copy_features(dst: &mut [i32], src: &[u8], align: usize) {
    let dst_len = dst.len();
    let len = dst_len.min(src.len());
    let dst_start = (dst_len - len) * align;
    let src_start = (src.len() - len) * align;

    for index in 0..len {
        dst[dst_start + index] = src[src_start + index] as i32;
    }
}

fn strip_prefix(xs: &[u8]) -> &[u8] {
    strip(xs, |slice| slice.split_first())
}

fn strip_suffix(xs: &[u8]) -> &[u8] {
    strip(xs, |slice| slice.split_last())
}

fn strip<'a>(
    mut xs: &'a [u8],
    mut split: impl FnMut(&'a [u8]) -> Option<(&'a u8, &'a [u8])>,
) -> &'a [u8] {
    while let Some((&x, ys)) = split(xs) {
        if !is_whitespace(x) {
            break;
        }
        xs = ys;
    }

    xs
}

fn is_whitespace(x: u8) -> bool {
    x.is_ascii_whitespace() || x == 0x0b
}

#[cfg(test)]
mod tests {
    use crate::detection::vendor::{
        content::ContentType, model as vendor_model,
    };

    use super::{PreparedInput, prepare_input};

    #[test]
    fn short_utf8_input_is_ruled_as_text() {
        match prepare_input(b"hello".as_slice(), &vendor_model::CONFIG) {
            PreparedInput::Ruled(ContentType::Txt) => {}
            _ => panic!("expected ruled text"),
        }
    }
}
