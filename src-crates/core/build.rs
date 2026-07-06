//! Build support for bundled detection assets.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use onnx_ir::ModelProto;
use protobuf::Message;
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};

// Output safetensors file name.
const MAGIKA_OUT: &str = "magika.safetensors";
// Committed upstream ONNX weights, relative to `CARGO_MANIFEST_DIR`.
const MAGIKA_ONNX: &str =
    "src/detection/vendor/assets/models/standard_v3_3/model.onnx";
// ONNX `TensorProto` `data_type` for IEEE-754 single-precision float.
const ONNX_DTYPE_FLOAT: i32 = 1;

fn main() {
    if env::var_os("CARGO_FEATURE_DETECTION").is_none() {
        return;
    }

    let manifest = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set"),
    );
    let onnx_path = manifest.join(MAGIKA_ONNX);
    println!("cargo:rerun-if-changed={}", onnx_path.display());

    let onnx_bytes = fs::read(&onnx_path).unwrap_or_else(|e| {
        panic!("read magika onnx {}: {e}", onnx_path.display())
    });

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set"));
    let dest = out_dir.join(MAGIKA_OUT);
    convert_onnx_to_safetensors(&onnx_bytes, &dest);
}

// Converts detection weights to safetensors.
fn convert_onnx_to_safetensors(onnx_bytes: &[u8], dest: &Path) {
    let model = ModelProto::parse_from_bytes(onnx_bytes)
        .unwrap_or_else(|e| panic!("parse magika onnx: {e}"));
    let graph = model.graph.as_ref().expect("magika onnx graph missing");

    // Own the (name, shape, bytes) tuples so the borrowing `TensorView`s below
    // stay valid until serialization completes.
    let owned: Vec<(String, Vec<usize>, &[u8])> = graph
        .initializer
        .iter()
        .filter(|tensor| tensor.data_type == ONNX_DTYPE_FLOAT)
        .filter(|tensor| !tensor.raw_data.is_empty())
        .map(|tensor| {
            let shape = tensor.dims.iter().map(|dim| *dim as usize).collect();
            (tensor.name.clone(), shape, &tensor.raw_data[..])
        })
        .collect();

    let views = owned.iter().map(|(name, shape, data)| {
        let view = TensorView::new(Dtype::F32, shape.clone(), data)
            .unwrap_or_else(|e| panic!("build view for {name}: {e}"));
        (name.clone(), view)
    });

    serialize_to_file(views, None, dest)
        .unwrap_or_else(|e| panic!("write {}: {e}", dest.display()));
}
