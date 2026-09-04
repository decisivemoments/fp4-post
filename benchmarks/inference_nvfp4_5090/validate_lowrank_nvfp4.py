#!/usr/bin/env python3
"""Validate full native-linear error from NVFP4 low-rank inference.

Both branches share the same FP4 residual/V path. The only difference is the
low-rank U @ scaled_v computation: BF16 reference versus separately quantized
NVFP4 U and scaled_v.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Metis.Metis.native_nvfp4 import NativeFullNVFP4Linear, require_native_nvfp4  # noqa: E402


MODEL_IDS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--projection", choices=PROJECTIONS, default="up_proj")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(1, 16, 64, 128))
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def projection_for(model: torch.nn.Module, layer_index: int, target: str) -> torch.nn.Linear:
    for name, module in model.model.layers[layer_index].named_modules():
        if name.rsplit(".", 1)[-1] == target and isinstance(module, torch.nn.Linear):
            return module
    raise RuntimeError(f"Could not find {target} in layer {layer_index}")


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual - expected).float()
    expected_float = expected.float()
    return {
        "relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected_float)),
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                actual.float().flatten(), expected_float.flatten(), dim=0
            )
        ),
    }


def main() -> None:
    args = parse_args()
    require_native_nvfp4()
    source = args.model_path or MODEL_IDS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device).eval()
    source_linear = projection_for(model, args.layer_index, args.projection)
    reference = NativeFullNVFP4Linear.from_linear(
        source_linear,
        rank=args.rank,
        stochastic_rounding=False,
        lowrank_compute="bf16",
    ).eval()
    candidate = NativeFullNVFP4Linear.from_linear(
        source_linear,
        rank=args.rank,
        stochastic_rounding=False,
        lowrank_compute="nvfp4",
    ).eval()

    records = []
    with torch.inference_mode():
        for batch in args.batch_sizes:
            x = torch.randn(
                batch,
                args.seq_length,
                source_linear.in_features,
                device=args.device,
                dtype=torch.bfloat16,
            )
            expected = reference(x)
            actual = candidate(x)
            records.append({"batch_size": batch, **metrics(actual, expected)})

    payload = {
        "validation": "full_native_linear_lowrank_nvfp4_vs_bf16",
        "model": args.model,
        "model_source": source,
        "projection": args.projection,
        "layer_index": args.layer_index,
        "rank": args.rank,
        "seq_length": args.seq_length,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
