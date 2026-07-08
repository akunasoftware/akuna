/// Errors returned by [`super::Index`].
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    /// Caller-provided index input was invalid.
    #[error("{message}")]
    InvalidInput {
        /// Description of the invalid input.
        message: String,
    },

    /// Index storage or model construction failed.
    #[error("Index initialization failed: {source}")]
    Open {
        /// Underlying initialization failure.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// An index write or rollback failed.
    #[error("Index write failed: {source}")]
    Write {
        /// Underlying write failure.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// An index read failed.
    #[error("Index read failed: {source}")]
    Read {
        /// Underlying read failure.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// An index search failed.
    #[error("Index search failed: {source}")]
    Search {
        /// Underlying search failure.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl IndexError {
    /// Wraps initialization failures at the index boundary.
    pub(super) fn open(source: anyhow::Error) -> Self {
        Self::Open {
            source: source.into_boxed_dyn_error(),
        }
    }

    /// Wraps write failures at the index boundary.
    pub(super) fn write(source: anyhow::Error) -> Self {
        Self::Write {
            source: source.into_boxed_dyn_error(),
        }
    }

    /// Wraps read failures at the index boundary.
    pub(super) fn read(source: anyhow::Error) -> Self {
        Self::Read {
            source: source.into_boxed_dyn_error(),
        }
    }

    /// Wraps search failures at the index boundary.
    pub(super) fn search(source: anyhow::Error) -> Self {
        Self::Search {
            source: source.into_boxed_dyn_error(),
        }
    }
}
