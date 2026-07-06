//! Shared extraction pipeline helpers.

use std::collections::HashMap;

use crate::extraction::{ExtractionPipelineStep, ExtractionPipelineStepKind};

/// Build pipeline step with output counts.
pub(in crate::extraction) fn step(
    step: ExtractionPipelineStepKind,
    engine: impl Into<String>,
    duration_ms: u64,
    outputs: HashMap<String, u64>,
) -> ExtractionPipelineStep {
    ExtractionPipelineStep {
        step,
        engine: engine.into(),
        duration_ms,
        outputs,
    }
}
