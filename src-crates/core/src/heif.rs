//! HEIC/HEIF byte helpers.

/// Returns whether the leading bytes declare a HEIF/HEIC brand.
pub(crate) fn has_heif_brand(bytes: &[u8]) -> bool {
    bytes.len() >= 12
        && matches!(
            &bytes[4..12],
            b"ftypheic"
                | b"ftypheix"
                | b"ftyphevc"
                | b"ftyphevx"
                | b"ftypheif"
                | b"ftypheim"
                | b"ftypheis"
                | b"ftyphevm"
                | b"ftyphevs"
                | b"ftypmif1"
                | b"ftypmsf1"
        )
}
