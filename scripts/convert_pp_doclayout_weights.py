#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy",
#   "paddlepaddle>=3.0",
#   "safetensors",
# ]
# ///

"""Convert official PP-DocLayoutV3 PIR params to semantic safetensors."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import paddle
from safetensors import safe_open
from safetensors.numpy import save_file
from paddle.base import core, framework
from paddle.jit.pir_translated_layer import _load_pir_program
from paddle.static.pir_io import get_pir_parameters


def load_base_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Read base semantic safetensors into owned NumPy arrays."""
    with safe_open(path, framework="np") as tensors:
        return {name: np.array(tensors.get_tensor(name)) for name in tensors.keys()}


def load_official_tensors(model_dir: Path) -> dict[str, np.ndarray]:
    """Load official PIR tensors from inference.json and pdiparams."""
    program, _trainable = _load_pir_program(str(model_dir / "inference.json"))
    variables = sum(get_pir_parameters(program), [])
    by_name = {variable.name: variable for variable in variables}
    names = sorted(by_name)
    loaded = {}
    dense_tensors = []

    for name in names:
        variable = by_name[name]
        parameter = framework.EagerParamBase(
            shape=variable.shape,
            dtype=core.VarDesc.VarType.FP32,
            name=name,
            persistable=True,
        )
        loaded[name] = parameter
        dense_tensors.append(parameter.get_tensor())

    core.load_combine_func(
        str(model_dir / "inference.pdiparams"),
        names,
        dense_tensors,
        False,
        framework._current_expected_place(),
    )

    return {
        name: np.asarray(tensor).astype(np.float32, copy=False)
        for name, tensor in loaded.items()
    }


def walk_json(value: object) -> Iterator[dict[str, object]]:
    """Yield every object from nested inference.json."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json(child)
        return
    if isinstance(value, list):
        for child in value:
            yield from walk_json(child)


def load_struct_names(model_dir: Path) -> dict[str, str]:
    """Map PIR parameter names to Paddle semantic struct names."""
    data = json.loads((model_dir / "inference.json").read_text())
    mapping: dict[str, str] = {}
    for item in walk_json(data):
        name = string_field(item, ("name", "parameter_name", "var_name"))
        struct = string_field(
            item,
            ("struct_name", "structured_name", "persistable_name"),
        )
        if name and struct:
            mapping[name] = struct
    return mapping


def string_field(item: dict[str, object], names: tuple[str, ...]) -> str | None:
    """Return first string field present in an object."""
    for name in names:
        value = item.get(name)
        if isinstance(value, str):
            return value
    return None


def official_by_struct(
    official: dict[str, np.ndarray],
    structs: dict[str, str],
) -> dict[str, np.ndarray]:
    """Index official tensors by semantic struct name where known."""
    by_struct = {}
    for name, tensor in official.items():
        by_struct[name] = tensor
        struct = structs.get(name)
        if struct:
            by_struct[struct] = tensor
    return by_struct


def require_tensor(tensors: dict[str, np.ndarray], name: str) -> np.ndarray:
    """Fetch tensor or fail with useful context."""
    if name not in tensors:
        raise KeyError(f"official tensor not found: {name}")
    return tensors[name]


def require_one(tensors: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """Fetch single tensor whose name starts with prefix."""
    matches = [name for name in tensors if name.startswith(prefix)]
    if len(matches) != 1:
        raise KeyError(f"expected one tensor for {prefix}, got {matches}")
    return tensors[matches[0]]


def put_linear_arrays(
    out: dict[str, np.ndarray],
    weight: np.ndarray,
    bias: np.ndarray,
    target_prefix: str,
) -> int:
    """Write Paddle Linear arrays to semantic transposed layout."""
    out[f"{target_prefix}.weight"] = np.ascontiguousarray(weight.T)
    out[f"{target_prefix}.bias"] = np.ascontiguousarray(bias)
    return 2


def put_qkv_arrays(
    out: dict[str, np.ndarray],
    weight: np.ndarray,
    bias: np.ndarray,
    target_prefix: str,
) -> int:
    """Split fused qkv arrays into semantic q/k/v tensors."""
    weights = np.split(weight, 3, axis=1)
    biases = np.split(bias, 3, axis=0)
    count = 0
    for name, part_weight, part_bias in zip(
        ("q_proj", "k_proj", "v_proj"), weights, biases, strict=True
    ):
        count += put_linear_arrays(
            out, part_weight, part_bias, f"{target_prefix}.{name}"
        )
    return count


def put_linear(
    out: dict[str, np.ndarray],
    by_struct: dict[str, np.ndarray],
    source_prefix: str,
    target_prefix: str,
) -> int:
    """Copy Paddle Linear tensors, transposing weight to semantic shape."""
    weight = require_tensor(by_struct, f"{source_prefix}.weight")
    bias = require_tensor(by_struct, f"{source_prefix}.bias")
    out[f"{target_prefix}.weight"] = np.ascontiguousarray(weight.T)
    out[f"{target_prefix}.bias"] = np.ascontiguousarray(bias)
    return 2


def put_norm(
    out: dict[str, np.ndarray],
    by_struct: dict[str, np.ndarray],
    source_prefix: str,
    target_prefix: str,
) -> int:
    """Copy LayerNorm tensors."""
    out[f"{target_prefix}.weight"] = require_tensor(
        by_struct, f"{source_prefix}.weight"
    )
    out[f"{target_prefix}.bias"] = require_tensor(by_struct, f"{source_prefix}.bias")
    return 2


def put_qkv_linear(
    out: dict[str, np.ndarray],
    by_struct: dict[str, np.ndarray],
    source_prefix: str,
    target_prefix: str,
) -> int:
    """Split fused Paddle qkv Linear tensors into q, k, v linears."""
    weight = require_tensor(by_struct, f"{source_prefix}.qkv.weight")
    bias = require_tensor(by_struct, f"{source_prefix}.qkv.bias")
    weights = np.split(weight, 3, axis=1)
    biases = np.split(bias, 3, axis=0)
    count = 0
    for name, part_weight, part_bias in zip(
        ("q_proj", "k_proj", "v_proj"), weights, biases, strict=True
    ):
        out[f"{target_prefix}.{name}.weight"] = np.ascontiguousarray(part_weight.T)
        out[f"{target_prefix}.{name}.bias"] = np.ascontiguousarray(part_bias)
        count += 2
    count += put_linear(
        out, by_struct, f"{source_prefix}.out_proj", f"{target_prefix}.out_proj"
    )
    return count


def put_conv_bn(
    out: dict[str, np.ndarray],
    by_struct: dict[str, np.ndarray],
    source_prefix: str,
    target_prefix: str,
) -> int:
    """Copy Conv2D plus BatchNorm tensors."""
    names = (
        ("conv.weight", "weight"),
        ("bn.weight", "weight"),
        ("bn.bias", "bias"),
        ("bn._mean", "running_mean"),
        ("bn._variance", "running_var"),
    )
    for source_suffix, target_suffix in names:
        out[f"{target_prefix}.{target_suffix}"] = require_tensor(
            by_struct, f"{source_prefix}.{source_suffix}"
        )
    return len(names)


def put_fused_conv_branch(
    out: dict[str, np.ndarray],
    weight: np.ndarray,
    bias: np.ndarray,
    target_prefix: str,
) -> int:
    """Store fused conv as conv+identity batch norm branch."""
    channels = weight.shape[0]
    out[f"{target_prefix}.conv.weight"] = np.ascontiguousarray(weight)
    out[f"{target_prefix}.norm.weight"] = np.ones(channels, dtype=np.float32)
    out[f"{target_prefix}.norm.bias"] = np.ascontiguousarray(bias)
    out[f"{target_prefix}.norm.running_mean"] = np.zeros(channels, dtype=np.float32)
    out[f"{target_prefix}.norm.running_var"] = np.full(
        channels, 1.0 - 1.0e-5, dtype=np.float32
    )
    return 5


def put_zero_conv_branch(
    out: dict[str, np.ndarray], target_prefix: str, channels: int
) -> int:
    """Store zero 1x1 branch for fused RepVGG static weights."""
    out[f"{target_prefix}.conv.weight"] = np.zeros(
        (channels, channels, 1, 1), dtype=np.float32
    )
    out[f"{target_prefix}.norm.weight"] = np.ones(channels, dtype=np.float32)
    out[f"{target_prefix}.norm.bias"] = np.zeros(channels, dtype=np.float32)
    out[f"{target_prefix}.norm.running_mean"] = np.zeros(channels, dtype=np.float32)
    out[f"{target_prefix}.norm.running_var"] = np.full(
        channels, 1.0 - 1.0e-5, dtype=np.float32
    )
    return 5


def fuse_conv_bn(
    weight: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold Conv2D and BatchNorm into one conv weight and bias."""
    scale = gamma / np.sqrt(var + 1.0e-5)
    fused_weight = weight * scale.reshape((-1, 1, 1, 1))
    fused_bias = beta - mean * scale
    return np.ascontiguousarray(fused_weight), np.ascontiguousarray(fused_bias)


def identity_kernel(channels: int) -> np.ndarray:
    """Build 3x3 identity convolution kernel."""
    kernel = np.zeros((channels, channels, 3, 3), dtype=np.float32)
    for channel in range(channels):
        kernel[channel, channel, 1, 1] = 1.0
    return kernel


def synthesize_repvgg(
    out: dict[str, np.ndarray],
    by_struct: dict[str, np.ndarray],
    source_prefix: str,
    target_prefix: str,
) -> int:
    """Synthesize fused RepVGG conv+identity BN for semantic conv1."""
    conv_weight = require_tensor(by_struct, f"{source_prefix}.rbr_dense.0.weight")
    dense_weight, dense_bias = fuse_conv_bn(
        conv_weight,
        require_tensor(by_struct, f"{source_prefix}.rbr_dense.1.weight"),
        require_tensor(by_struct, f"{source_prefix}.rbr_dense.1.bias"),
        require_tensor(by_struct, f"{source_prefix}.rbr_dense.1._mean"),
        require_tensor(by_struct, f"{source_prefix}.rbr_dense.1._variance"),
    )
    identity_weight, identity_bias = fuse_conv_bn(
        identity_kernel(conv_weight.shape[0]),
        require_tensor(by_struct, f"{source_prefix}.rbr_identity.weight"),
        require_tensor(by_struct, f"{source_prefix}.rbr_identity.bias"),
        require_tensor(by_struct, f"{source_prefix}.rbr_identity._mean"),
        require_tensor(by_struct, f"{source_prefix}.rbr_identity._variance"),
    )
    out[f"{target_prefix}.weight"] = dense_weight + identity_weight
    out[f"{target_prefix}.bias"] = dense_bias + identity_bias
    return 2


def convert_encoder(out: dict[str, np.ndarray], official: dict[str, np.ndarray]) -> int:
    """Convert encoder AIFI linears and norms."""
    count = put_qkv_arrays(
        out,
        require_one(official, "multi_head_attention_0.w_0"),
        require_one(official, "multi_head_attention_0.b_0"),
        "model.encoder.encoder.0.layers.0.self_attn",
    )
    count += put_linear_arrays(
        out,
        require_one(official, "linear_0.w_0"),
        require_one(official, "linear_0.b_0"),
        "model.encoder.encoder.0.layers.0.self_attn.out_proj",
    )
    count += put_linear_arrays(
        out,
        require_one(official, "linear_1.w_0"),
        require_one(official, "linear_1.b_0"),
        "model.encoder.encoder.0.layers.0.fc1",
    )
    count += put_linear_arrays(
        out,
        require_one(official, "linear_2.w_0"),
        require_one(official, "linear_2.b_0"),
        "model.encoder.encoder.0.layers.0.fc2",
    )
    out["model.encoder.encoder.0.layers.0.position_embedding"] = require_one(
        official, "eager_tmp_0"
    )
    count += 1
    return count


def convert_decoder(out: dict[str, np.ndarray], official: dict[str, np.ndarray]) -> int:
    """Convert decoder attention, MLP, and head linears."""
    count = 0
    suffixes = [12, 34, 56, 78, 100, 122]
    for index in range(6):
        target = f"model.decoder.layers.{index}"
        suffix = suffixes[index]
        count += put_qkv_arrays(
            out,
            require_one(official, f"multi_head_attention_1.w_0_deepcopy_{suffix}_"),
            require_one(official, f"multi_head_attention_1.b_0_deepcopy_{suffix + 1}_"),
            f"{target}.self_attn",
        )
        count += put_linear_arrays(
            out,
            require_one(official, f"linear_3.w_0_deepcopy_{suffix + 2}_"),
            require_one(official, f"linear_3.b_0_deepcopy_{suffix + 3}_"),
            f"{target}.self_attn.out_proj",
        )
        cross_suffix = suffix + 6
        for linear_index, name, output_features in (
            (4, "sampling_offsets", 192),
            (5, "attention_weights", 96),
            (6, "value_proj", 256),
            (7, "output_proj", 256),
        ):
            count += put_linear_arrays(
                out,
                require_one(
                    official, f"linear_{linear_index}.w_0_deepcopy_{cross_suffix}_"
                ),
                require_one(
                    official,
                    f"linear_{linear_index}.b_0_deepcopy_{cross_suffix + 1}_",
                ),
                f"{target}.encoder_attn.{name}",
            )
            cross_suffix += 2
        count += put_linear_arrays(
            out,
            require_one(official, f"linear_8.w_0_deepcopy_{suffix + 16}_"),
            require_one(official, f"linear_8.b_0_deepcopy_{suffix + 17}_"),
            f"{target}.fc1",
        )
        count += put_linear_arrays(
            out,
            require_one(official, f"linear_9.w_0_deepcopy_{suffix + 18}_"),
            require_one(official, f"linear_9.b_0_deepcopy_{suffix + 19}_"),
            f"{target}.fc2",
        )
    return count


def convert_heads(out: dict[str, np.ndarray], official: dict[str, np.ndarray]) -> int:
    """Convert detection, mask, order, and pointer heads."""
    count = 0
    count += put_linear_arrays(
        out,
        require_one(official, "linear_15.w_0"),
        require_one(official, "linear_15.b_0"),
        "model.enc_output.0",
    )
    count += put_linear_arrays(
        out,
        require_one(official, "linear_16.w_0"),
        require_one(official, "linear_16.b_0"),
        "model.enc_score_head",
    )
    count += put_linear_arrays(
        out,
        require_one(official, "linear_26.w_0"),
        require_one(official, "linear_26.b_0"),
        "model.decoder_global_pointer.dense",
    )
    for index, linear_index in enumerate(range(20, 26)):
        count += put_linear_arrays(
            out,
            require_one(official, f"linear_{linear_index}.w_0"),
            require_one(official, f"linear_{linear_index}.b_0"),
            f"model.decoder_order_head.{index}",
        )
    for layer, linear_index, output_features in (
        (0, 17, 256),
        (1, 18, 256),
        (2, 19, 4),
    ):
        count += put_linear_arrays(
            out,
            require_one(official, f"linear_{linear_index}.w_0"),
            require_one(official, f"linear_{linear_index}.b_0"),
            f"model.enc_bbox_head.layers.{layer}",
        )
    for layer, linear_index, output_features in ((0, 10, 512), (1, 11, 256)):
        count += put_linear_arrays(
            out,
            require_one(official, f"linear_{linear_index}.w_0"),
            require_one(official, f"linear_{linear_index}.b_0"),
            f"model.decoder.query_pos_head.layers.{layer}",
        )
    for layer, linear_index, output_features in (
        (0, 12, 256),
        (1, 13, 256),
        (2, 14, 32),
    ):
        count += put_linear_arrays(
            out,
            require_one(official, f"linear_{linear_index}.w_0"),
            require_one(official, f"linear_{linear_index}.b_0"),
            f"model.mask_query_head.layers.{layer}",
        )
    return count


def convert_repvgg(out: dict[str, np.ndarray], official: dict[str, np.ndarray]) -> int:
    """Convert CSPRepLayer RepVGG bottleneck convs."""
    count = 0
    blocks = (
        "model.encoder.fpn_blocks.0",
        "model.encoder.fpn_blocks.1",
        "model.encoder.pan_blocks.0",
        "model.encoder.pan_blocks.1",
    )
    conv_index = 130
    for target in blocks:
        for index in range(3):
            block_target = f"{target}.bottlenecks.{index}"
            weight = require_tensor(official, f"conv2d_{conv_index}.w_0")
            bias = require_tensor(official, f"conv2d_{conv_index}.b_0")
            count += put_fused_conv_branch(out, weight, bias, f"{block_target}.conv1")
            count += put_zero_conv_branch(out, f"{block_target}.conv2", 256)
            conv_index += 1
    return count


def main() -> None:
    """Run converter CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--base-safetensors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    converted = load_base_safetensors(args.base_safetensors)
    official = load_official_tensors(args.model_dir)

    changed = 0
    changed += convert_encoder(converted, official)
    changed += convert_decoder(converted, official)
    changed += convert_heads(converted, official)
    changed += convert_repvgg(converted, official)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, args.output)
    print(f"wrote {args.output}")
    print(f"official tensors: {len(official)}")
    print(f"semantic tensors replaced: {changed}")


if __name__ == "__main__":
    main()
