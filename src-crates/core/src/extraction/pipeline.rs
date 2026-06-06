//! Shared extraction pipeline helpers.

use std::collections::HashMap;

use crate::extraction::ExtractionPipelineStep;

/// Build pipeline step with output counts.
pub(in crate::extraction) fn step(
    step: &str,
    engine: impl Into<String>,
    duration_ms: u64,
    outputs: HashMap<String, usize>,
) -> ExtractionPipelineStep {
    ExtractionPipelineStep {
        step: step.to_owned(),
        engine: engine.into(),
        duration_ms,
        outputs,
    }
}
