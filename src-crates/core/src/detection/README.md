# File-Type Detection

File-type inference using the [Google Magika ML model](https://github.com/google/magika)
with native Burn backends.

## Usage

```rust
use akuna_core::detection::Session;
use burn_wgpu::{Wgpu, WgpuDevice};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::default();
    let mut session = Session::<Wgpu>::new(&device)?;

    let detected = session.identify_file_sync(Path::new("target/fixtures/text.pdf"))?;
    assert_eq!(detected.info().label, "pdf");
    assert_eq!(detected.info().mime_type, "application/pdf");

    Ok(())
}
```

## Features

- Magika-compatible preprocessing and output post-processing.
- Generic `Session<B>` and `MagikaModel<B>` built on Burn's `Backend` abstraction.
- Vendored `standard_v3_3` model from `src-crates/core/src/detection/vendor/assets/models/standard_v3_3/model.onnx`.
- Tested parity against the Rust `magika` crate on local test fixtures.
