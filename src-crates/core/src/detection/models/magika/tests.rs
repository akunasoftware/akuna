use anyhow::{anyhow, Result};

use crate::detection::{
    models::magika::MagikaModel, vendor::model as vendor_model, DetectionOrigin,
};
use crate::ml::backend::{
    active_device, cpu_device, gpu_available, gpu_device, Backend,
};
use crate::testkit::run_with_model_stack;

// Existing Python detection parity tolerance, not a measured batch tolerance.
const CONFIDENCE_TOLERANCE: f32 = 5e-4;

/// Asserts element-wise finiteness and closeness under a per-element tolerance.
fn assert_close(
    actual: &[f32],
    expected: &[f32],
    tolerance: impl Fn(f32) -> f32,
    context: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let delta = (actual - expected).abs();
        let tolerance = tolerance(expected.abs());
        assert!(
            actual.is_finite() && expected.is_finite() && delta <= tolerance,
            "{context}[{index}] actual={actual} expected={expected} \
             delta={delta} tolerance={tolerance}"
        );
    }
}

const MODEL_INPUTS: [&[u8]; 4] = [
    br#"<!DOCTYPE html><html><head><title>Sample</title></head><body><h1>Hello</h1><p>A small HTML document.</p></body></html>"#,
    br#"{"name":"sample","enabled":true,"items":[{"id":1,"value":"first"},{"id":2,"value":"second"}],"count":2}"#,
    b"#!/usr/bin/env python3\nimport json\n\ndef greet(name):\n    return f'Hello {name}'\n\nif __name__ == '__main__':\n    print(json.dumps({'message': greet('world')}))\n",
    b"<?xml version=\"1.0\" encoding=\"UTF-8\"?><catalog><book id=\"1\"><title>Example</title><author>Alice</author></book><book id=\"2\"><title>Another</title></book></catalog>",
];

#[test]
fn gpu_convolution_matches_native() -> Result<()> {
    use burn::tensor::{module::conv1d, ops::ConvOptions, Tensor, TensorData};

    use crate::detection::models::magika::{
        CHANNELS_PER_TOKEN, CONV_KERNEL, CONV_OUT_CHANNELS, TOKENS_PER_BLOCK,
    };

    if !gpu_available() {
        return Ok(());
    }

    run_with_model_stack(|| {
        let device = gpu_device();
        let model = MagikaModel::<Backend>::from_embedded(&device)?;
        let batch = 2;
        let values = (0..batch * TOKENS_PER_BLOCK * CHANNELS_PER_TOKEN)
            .map(|i| ((i * 73 + i / 257) % 1021) as f32 / 510.0 - 1.0)
            .collect::<Vec<_>>();
        // Match the non-contiguous layout entering production convolution.
        let x = Tensor::<Backend, 3>::from_data(
            TensorData::new(
                values,
                [batch, TOKENS_PER_BLOCK, CHANNELS_PER_TOKEN],
            ),
            &device,
        )
        .permute([0, 2, 1]);
        let expected = conv1d(
            x.clone(),
            model.conv_weight.clone(),
            Some(model.conv_bias.clone()),
            ConvOptions::new([1], [0], [1], 1),
        );
        let actual = model.convolve(x);
        let shape =
            [batch, CONV_OUT_CHANNELS, TOKENS_PER_BLOCK - CONV_KERNEL + 1];
        assert_eq!(actual.dims(), shape);
        assert_eq!(expected.dims(), shape);
        let actual = actual
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| anyhow!("{error:?}"))?;
        let expected = expected
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| anyhow!("{error:?}"))?;
        assert_close(
            &actual,
            &expected,
            |expected| 1e-4 + 1e-4 * expected,
            "conv",
        );
        Ok(())
    })
}

#[test]
fn batch_matches_single() -> Result<()> {
    run_with_model_stack(|| {
        let model = MagikaModel::<Backend>::from_embedded(&active_device())?;
        assert!(model.detect_content_type_batch(Vec::new())?.is_empty());

        let inputs = [
            b"".as_slice(),
            MODEL_INPUTS[2],
            b"hi",
            MODEL_INPUTS[0],
            b"\xff",
            MODEL_INPUTS[3],
            MODEL_INPUTS[1],
            b"",
        ];
        let expected = inputs
            .iter()
            .map(|input| model.identify_bytes(input))
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            expected.iter().map(|row| row.origin()).collect::<Vec<_>>(),
            [
                DetectionOrigin::Rule,
                DetectionOrigin::Model,
                DetectionOrigin::Rule,
                DetectionOrigin::Model,
                DetectionOrigin::Rule,
                DetectionOrigin::Model,
                DetectionOrigin::Model,
                DetectionOrigin::Rule,
            ]
        );

        let actual = model.detect_content_type_batch(inputs.to_vec())?;
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in
            actual.iter().zip(&expected).enumerate()
        {
            assert_eq!(actual.info(), expected.info(), "index={index}");
            assert_eq!(actual.origin(), expected.origin(), "index={index}");
            let delta = (actual.confidence() - expected.confidence()).abs();
            assert!(
                delta <= CONFIDENCE_TOLERANCE,
                "index={index} confidence delta={delta}"
            );
        }
        Ok(())
    })
}

#[test]
#[ignore = "manual cached ORT parity for normalization, convolution and embedding table"]
fn parity_official() -> Result<()> {
    use anyhow::Context;

    run_with_model_stack(|| {
        let dylib = std::env::var("ORT_DYLIB_PATH").map_err(|_| {
            anyhow!(
                "ORT_DYLIB_PATH must name an API-24-compatible cached runtime"
            )
        })?;
        assert!(
            std::path::Path::new(&dylib).is_absolute(),
            "ORT_DYLIB_PATH must be absolute"
        );
        assert!(
            ort::init_from(&dylib)?
                .with_execution_providers([ort::ep::CPU::default()
                    .with_arena_allocator(true)
                    .build()
                    .error_on_failure(),])
                .commit(),
            "run parity_official alone: ORT already configured"
        );
        assert_eq!(magika::MODEL_NAME, "standard_v3_3");
        let mut reference = magika::Session::builder()
            .with_inter_threads(1)
            .with_intra_threads(1)
            .build()?;
        let mut inputs: Vec<_> =
            MODEL_INPUTS.iter().map(|input| input.to_vec()).collect();
        // Constant tokens stress near-zero variance and cancellation in norm0.
        for byte in [0, 1, 65, 127, 128, 255] {
            for len in [64, 4096] {
                inputs.push(vec![byte; len]);
            }
        }
        let mut state = 0x1234_5678_u32;
        inputs.push(
            (0..4096)
                .map(|_| {
                    state = state
                        .wrapping_mul(1_664_525)
                        .wrapping_add(1_013_904_223);
                    (state >> 24) as u8
                })
                .collect(),
        );
        inputs.push(
            "The caf\u{e9} serves tea while we discuss na\u{ef}ve ideas and \
             write a r\u{e9}sum\u{e9} of our journey through the city.\n"
                .repeat(128)
                .into_bytes(),
        );
        for signature in [b"%PDF-".as_slice(), b"PK\x03\x04".as_slice()] {
            inputs.push(signature.to_vec());
            let mut minimum = signature.to_vec();
            minimum.resize(vendor_model::CONFIG.min_file_size_for_dl, 0);
            inputs.push(minimum);
        }
        assert_eq!(inputs.len(), 22);
        let expected = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                reference
                    .identify_content_sync(input.as_slice())
                    .with_context(|| format!("path=official index={index}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let devices = std::iter::once(cpu_device())
            .chain(gpu_available().then(active_device));
        for device in devices {
            let model = MagikaModel::<Backend>::from_embedded(&device)?;
            for batch in [1, inputs.len()] {
                let mut max_delta = 0.0_f32;
                for (chunk, inputs) in inputs.chunks(batch).enumerate() {
                    let start = chunk * batch;
                    let rows = model
                        .detect_content_type_batch(
                            inputs.iter().map(Vec::as_slice).collect(),
                        )
                        .with_context(|| {
                            format!(
                                "path={device:?} batch={batch} index={start}"
                            )
                        })?;
                    assert_eq!(
                        rows.len(),
                        inputs.len(),
                        "path={device:?} batch={batch} index={start}"
                    );
                    for (offset, actual) in rows.iter().enumerate() {
                        let index = start + offset;
                        let expected = &expected[index];
                        let origin = match expected {
                            magika::FileType::Inferred(_) => {
                                DetectionOrigin::Model
                            }
                            _ => DetectionOrigin::Rule,
                        };
                        let delta =
                            (actual.confidence() - expected.score()).abs();
                        anyhow::ensure!(
                            actual.info().label == expected.info().label
                                && actual.origin() == origin
                                && delta <= CONFIDENCE_TOLERANCE,
                            "path={device:?} batch={batch} index={index} \
                             actual={:?}/{:?} confidence={} \
                             expected={:?}/{origin:?} confidence={} \
                             delta={delta} tolerance={CONFIDENCE_TOLERANCE}",
                            actual.info().label,
                            actual.origin(),
                            actual.confidence(),
                            expected.info().label,
                            expected.score(),
                        );
                        max_delta = max_delta.max(delta);
                    }
                }
                eprintln!(
                    "path={device:?} batch={batch} cases={} max_confidence_delta={max_delta:.8}",
                    inputs.len()
                );
            }
        }
        Ok(())
    })
}

#[test]
fn heic_override_requires_brand() {
    use crate::detection::vendor::content::ContentType;

    let heic_bytes = b"\x00\x00\x00\x18ftypheic....";
    assert_eq!(
        super::misdetection_override(ContentType::Mp4, heic_bytes)
            .map(|info| info.mime_type),
        Some("image/heic")
    );
    let mp4_bytes = b"\x00\x00\x00\x18ftypisom....";
    assert_eq!(
        super::misdetection_override(ContentType::Mp4, mp4_bytes)
            .map(|info| info.mime_type),
        None
    );
    assert!(crate::heif::has_heif_brand(heic_bytes));
    assert!(!crate::heif::has_heif_brand(mp4_bytes));
    assert!(!crate::heif::has_heif_brand(b"short"));
}
