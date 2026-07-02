#!/usr/bin/env python3
"""Convert Metis BitLinear checkpoints back to standard Linear weights.

The converter accepts either a single checkpoint file or a HuggingFace model
directory.  For BitLinear modules it reconstructs:

    weight = ulinear.weight @ diag(s) @ vlinear.weight + warmup_linear.weight

and writes regular ``*.weight`` / ``*.bias`` tensors.  Directory conversion also
copies tokenizer/config files so the output can be used as a normal
``model_name_or_path``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file


WEIGHT_FILE_NAMES = {
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}


def dtype_from_name(name: str | None) -> torch.dtype | None:
    if name is None or name == "keep":
        return None
    choices = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in choices:
        raise ValueError(f"Unsupported dtype: {name}")
    return choices[name]


def parse_size(size: str) -> int:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGTP]?B?)", size.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid shard size: {size}")
    value = float(match.group(1))
    unit = match.group(2).upper().rstrip("B")
    scale = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
    }[unit]
    return int(value * scale)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def convert_tensor_dtype(tensor: torch.Tensor, target_dtype: torch.dtype | None) -> torch.Tensor:
    if target_dtype is not None and torch.is_floating_point(tensor):
        return tensor.to(target_dtype)
    return tensor


def reconstruct_weight(
    ulinear_weight: torch.Tensor,
    s: torch.Tensor,
    vlinear_weight: torch.Tensor,
    warmup_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    compute_dtype = torch.float32
    lowrank = (
        ulinear_weight.to(compute_dtype)
        @ torch.diag(s.to(compute_dtype))
        @ vlinear_weight.to(compute_dtype)
    )
    if warmup_weight is not None:
        lowrank = lowrank + warmup_weight.to(compute_dtype)
        return lowrank.to(warmup_weight.dtype)
    return lowrank.to(ulinear_weight.dtype)


def convert_state_dict(
    state_dict: OrderedDict[str, torch.Tensor],
    *,
    target_dtype: torch.dtype | None,
    verbose: bool,
) -> OrderedDict[str, torch.Tensor]:
    bitlinear_prefixes = set()
    for key in state_dict:
        match = re.match(r"(.+)\.(ulinear|vlinear|warmup_linear)\.", key)
        if match:
            bitlinear_prefixes.add(match.group(1))

    print(f"Found {len(bitlinear_prefixes)} BitLinear modules")

    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    processed_keys = set()

    for prefix in sorted(bitlinear_prefixes):
        ulinear_weight = state_dict.get(f"{prefix}.ulinear.weight")
        vlinear_weight = state_dict.get(f"{prefix}.vlinear.weight")
        s = state_dict.get(f"{prefix}.s")
        warmup_weight = state_dict.get(f"{prefix}.warmup_linear.weight")
        warmup_bias = state_dict.get(f"{prefix}.warmup_linear.bias")
        ulinear_bias = state_dict.get(f"{prefix}.ulinear.bias")

        for suffix in (
            "ulinear.weight",
            "ulinear.bias",
            "vlinear.weight",
            "vlinear.bias",
            "s",
            "warmup_linear.weight",
            "warmup_linear.bias",
        ):
            key = f"{prefix}.{suffix}"
            if key in state_dict:
                processed_keys.add(key)

        if ulinear_weight is not None and vlinear_weight is not None and s is not None:
            weight = reconstruct_weight(ulinear_weight, s, vlinear_weight, warmup_weight)
            converted[f"{prefix}.weight"] = convert_tensor_dtype(weight, target_dtype)
            if warmup_bias is not None:
                converted[f"{prefix}.bias"] = convert_tensor_dtype(warmup_bias, target_dtype)
            elif ulinear_bias is not None:
                converted[f"{prefix}.bias"] = convert_tensor_dtype(ulinear_bias, target_dtype)

            if verbose:
                residual = " + residual" if warmup_weight is not None else ""
                print(f"  merged {prefix}: U diag(S) V{residual} -> {tuple(weight.shape)}")
        elif warmup_weight is not None:
            converted[f"{prefix}.weight"] = convert_tensor_dtype(warmup_weight, target_dtype)
            if warmup_bias is not None:
                converted[f"{prefix}.bias"] = convert_tensor_dtype(warmup_bias, target_dtype)
            if verbose:
                print(f"  copied {prefix}.warmup_linear -> {tuple(warmup_weight.shape)}")
        else:
            missing = [
                name
                for name, value in (
                    ("ulinear.weight", ulinear_weight),
                    ("s", s),
                    ("vlinear.weight", vlinear_weight),
                    ("warmup_linear.weight", warmup_weight),
                )
                if value is None
            ]
            print(f"  skipped {prefix}: missing {missing}")

    for key, value in state_dict.items():
        if key not in processed_keys:
            converted[key] = convert_tensor_dtype(value, target_dtype)

    print(f"Original tensors: {len(state_dict)}")
    print(f"Converted tensors: {len(converted)}")
    return converted


def load_checkpoint_file(path: Path) -> OrderedDict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return OrderedDict(load_file(str(path), device="cpu"))
    checkpoint = torch.load(path, map_location="cpu")
    return OrderedDict(checkpoint.get("model_state_dict", checkpoint))


def save_checkpoint_file(state_dict: OrderedDict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".safetensors":
        save_file(state_dict, str(path), metadata={"format": "pt"})
    else:
        torch.save({"model_state_dict": state_dict}, path)


def find_weight_files(model_dir: Path) -> list[Path]:
    safetensor_index = model_dir / "model.safetensors.index.json"
    if safetensor_index.exists():
        with safetensor_index.open() as f:
            index = json.load(f)
        return sorted({model_dir / name for name in index["weight_map"].values()})

    bin_index = model_dir / "pytorch_model.bin.index.json"
    if bin_index.exists():
        with bin_index.open() as f:
            index = json.load(f)
        return sorted({model_dir / name for name in index["weight_map"].values()})

    for name in ("model.safetensors", "pytorch_model.bin"):
        path = model_dir / name
        if path.exists():
            return [path]

    raise FileNotFoundError(f"No model weights found in {model_dir}")


def load_model_dir(model_dir: Path) -> OrderedDict[str, torch.Tensor]:
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    weight_files = find_weight_files(model_dir)
    print(f"Loading {len(weight_files)} weight file(s) from {model_dir}")
    for weight_file in weight_files:
        shard = load_checkpoint_file(weight_file)
        overlap = set(state_dict).intersection(shard)
        if overlap:
            raise ValueError(f"Duplicate tensor keys across shards, e.g. {sorted(overlap)[:3]}")
        state_dict.update(shard)
    return state_dict


def copy_model_side_files(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in input_dir.iterdir():
        if src.name in WEIGHT_FILE_NAMES:
            continue
        if re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", src.name):
            continue
        if re.fullmatch(r"pytorch_model-\d{5}-of-\d{5}\.bin", src.name):
            continue
        dst = output_dir / src.name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def cleanup_model_weight_files(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.iterdir():
        if path.name in WEIGHT_FILE_NAMES:
            path.unlink()
        elif re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", path.name):
            path.unlink()
        elif re.fullmatch(r"pytorch_model-\d{5}-of-\d{5}\.bin", path.name):
            path.unlink()


def shard_state_dict(
    state_dict: OrderedDict[str, torch.Tensor],
    max_shard_size: int,
) -> list[OrderedDict[str, torch.Tensor]]:
    shards: list[OrderedDict[str, torch.Tensor]] = []
    current: OrderedDict[str, torch.Tensor] = OrderedDict()
    current_size = 0

    for key, tensor in state_dict.items():
        size = tensor_nbytes(tensor)
        if current and current_size + size > max_shard_size:
            shards.append(current)
            current = OrderedDict()
            current_size = 0
        current[key] = tensor
        current_size += size

    if current:
        shards.append(current)
    return shards


def save_model_dir(
    state_dict: OrderedDict[str, torch.Tensor],
    output_dir: Path,
    *,
    max_shard_size: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_model_weight_files(output_dir)
    max_bytes = parse_size(max_shard_size)
    total_size = sum(tensor_nbytes(tensor) for tensor in state_dict.values())
    shards = shard_state_dict(state_dict, max_bytes)

    if len(shards) == 1:
        save_file(shards[0], str(output_dir / "model.safetensors"), metadata={"format": "pt"})
        return

    weight_map = {}
    shard_count = len(shards)
    width = max(5, int(math.log10(shard_count)) + 1)
    for index, shard in enumerate(shards, start=1):
        filename = f"model-{index:0{width}d}-of-{shard_count:0{width}d}.safetensors"
        for key in shard:
            weight_map[key] = filename
        save_file(shard, str(output_dir / filename), metadata={"format": "pt"})

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with (output_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")


def convert_path(
    input_path: Path,
    output_path: Path,
    *,
    target_dtype: torch.dtype | None,
    max_shard_size: str,
    verbose: bool,
) -> None:
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    if input_path.is_dir():
        state_dict = load_model_dir(input_path)
        converted = convert_state_dict(state_dict, target_dtype=target_dtype, verbose=verbose)
        copy_model_side_files(input_path, output_path)
        save_model_dir(converted, output_path, max_shard_size=max_shard_size)
    else:
        state_dict = load_checkpoint_file(input_path)
        converted = convert_state_dict(state_dict, target_dtype=target_dtype, verbose=verbose)
        save_checkpoint_file(converted, output_path)

    print("Conversion completed.")


def verify_no_bitlinear_keys(path: Path) -> None:
    if path.is_dir():
        state_dict = load_model_dir(path)
    else:
        state_dict = load_checkpoint_file(path)
    leftovers = [
        key
        for key in state_dict
        if ".ulinear." in key or ".vlinear." in key or ".warmup_linear." in key
    ]
    if leftovers:
        raise RuntimeError(f"Found leftover BitLinear keys, e.g. {leftovers[:5]}")
    print("Verification passed: no BitLinear split keys remain.")


def default_output_path(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path.parent / f"{input_path.name}-merged"
    return input_path.with_name(f"{input_path.stem}_merged{input_path.suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge Metis BitLinear SVD weights into standard Linear weights."
    )
    parser.add_argument("input", type=Path, help="Input checkpoint file or HF model directory.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output checkpoint file or HF model directory. Defaults to '<input>-merged'.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Backward-compatible alias for the output path.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["keep", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
        help="Floating tensor dtype for the converted output.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum safetensors shard size when output is a directory.",
    )
    parser.add_argument("--verify", action="store_true", help="Verify no BitLinear split keys remain.")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-layer logging.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = args.input
    output_path = args.output_file or args.output or default_output_path(input_path)
    convert_path(
        input_path,
        output_path,
        target_dtype=dtype_from_name(args.dtype),
        max_shard_size=args.max_shard_size,
        verbose=not args.quiet,
    )
    if args.verify:
        verify_no_bitlinear_keys(output_path)


if __name__ == "__main__":
    main()
