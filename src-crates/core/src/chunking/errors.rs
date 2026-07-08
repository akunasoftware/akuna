/// Errors raised by invalid chunking options.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ChunkingError {
    /// Chunks need a positive character limit.
    #[error("max_chars must be greater than zero")]
    ZeroMaxChars,

    /// Overlap must leave at least one new character in every chunk.
    #[error(
        "overlap_chars ({overlap_chars}) must be less than max_chars ({max_chars})"
    )]
    OverlapTooLarge {
        /// Configured character overlap.
        overlap_chars: usize,
        /// Configured character limit.
        max_chars: usize,
    },
}
