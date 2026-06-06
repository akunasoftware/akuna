# /// script
# dependencies = [
#   "numpy",
#   "paddlepaddle>=3.0",
#   "safetensors",
# ]
# ///

"""Compare official PP-DocLayoutV3 PIR weights with Rust safetensors."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import paddle
from safetensors import safe_open


def tensor_signature(value: np.ndarray) -> tuple[tuple[int, ...], str]:
    """Return shape and byte hash for exact float32 tensor matching."""
    array = np.asarray(value).astype(np.float32, copy=False)
    return tuple(array.shape), hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    """Load official inference prefix and report value matches in safetensors."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference-prefix",
        required=True,
        help="Path prefix without extension, e.g. /path/to/inference",
    )
    parser.add_argument("--safetensors", required=True)
    args = parser.parse_args()

    model = paddle.jit.load(args.inference_prefix)
    official_state = model.state_dict()

    with safe_open(args.safetensors, framework="np") as tensors:
        safe_by_signature: dict[tuple[tuple[int, ...], str], list[str]] = {}
        for name in tensors.keys():
            signature = tensor_signature(tensors.get_tensor(name))
            safe_by_signature.setdefault(signature, []).append(name)

        matched = []
        missing = []
        for official_name, official_tensor in official_state.items():
            array = np.asarray(official_tensor)
            names = safe_by_signature.get(tensor_signature(array), [])
            if names:
                matched.append((official_name, names[0], tuple(array.shape)))
                continue
            missing.append((official_name, tuple(array.shape)))

    print(f"official params: {len(official_state)}")
    print(f"matched in safetensors: {len(matched)}")
    print(f"missing from safetensors: {len(missing)}")
    if missing:
        print("first missing:")
        for name, shape in missing[:20]:
            print(f"  {name}: {shape}")


if __name__ == "__main__":
    main()
