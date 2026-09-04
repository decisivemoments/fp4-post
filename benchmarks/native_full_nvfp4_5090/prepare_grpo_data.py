#!/usr/bin/env python3
"""Prepare a deterministic DeepMath subset for the GRPO timing benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--min-prompt-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if not 0 < args.min_prompt_tokens <= args.max_prompt_tokens:
        raise ValueError("Invalid prompt token bounds")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    dataset = load_dataset(str(args.dataset_dir), split=args.split)
    required = {"prompt", "solution"}
    if not required.issubset(dataset.column_names):
        raise ValueError(
            f"Dataset columns {dataset.column_names} do not contain {required}"
        )

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    selected: list[dict] = []
    prompt_lengths: list[int] = []
    for index in indices:
        example = dataset[index]
        prompt = example["prompt"]
        if not isinstance(prompt, list) or not prompt:
            continue
        token_ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt_length = len(token_ids)
        if not args.min_prompt_tokens <= prompt_length <= args.max_prompt_tokens:
            continue
        selected.append(
            {
                "prompt": prompt,
                "solution": example["solution"],
                "source_index": index,
                "prompt_tokens": prompt_length,
            }
        )
        prompt_lengths.append(prompt_length)
        if len(selected) == args.num_samples:
            break

    if len(selected) != args.num_samples:
        raise RuntimeError(
            f"Only found {len(selected)} eligible examples; "
            f"requested {args.num_samples}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for example in selected:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    metadata = {
        "schema_version": 1,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "model_path": str(args.model_path.resolve()),
        "split": args.split,
        "num_samples": len(selected),
        "seed": args.seed,
        "min_prompt_tokens": min(prompt_lengths),
        "max_prompt_tokens": max(prompt_lengths),
        "mean_prompt_tokens": sum(prompt_lengths) / len(prompt_lengths),
        "configured_prompt_token_bounds": [
            args.min_prompt_tokens,
            args.max_prompt_tokens,
        ],
        "source_indices": [example["source_index"] for example in selected],
        "jsonl_sha256": sha256(args.output),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        f"Prepared {len(selected)} samples at {args.output}; "
        f"prompt tokens min/mean/max="
        f"{min(prompt_lengths)}/{metadata['mean_prompt_tokens']:.1f}/"
        f"{max(prompt_lengths)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
