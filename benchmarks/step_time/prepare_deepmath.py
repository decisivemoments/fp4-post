#!/usr/bin/env python3
"""Prepare a deterministic DeepMath token cache for step-time benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_messages(prompt: Any, solution: Any) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = str(item.get("content", ""))
                messages.append({"role": role, "content": content})
    elif prompt is not None:
        messages.append({"role": "user", "content": str(prompt)})

    messages.append({"role": "assistant", "content": str(solution or "")})
    return messages


def encode_messages(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        return list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )

    text = "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )
    return list(tokenizer(text, add_special_tokens=True)["input_ids"])


def main() -> None:
    args = parse_args()
    if args.seq_length <= 1:
        raise ValueError("--seq-length must be greater than 1")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() and metadata_path.exists() and not args.force:
        print(f"Token cache already exists: {args.output}")
        return

    parquet_files = sorted(
        (args.dataset_dir / "data").glob(f"{args.split}-*.parquet")
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"No {args.split}-*.parquet files under {args.dataset_dir / 'data'}"
        )

    table = pq.read_table(parquet_files, columns=["prompt", "solution"])
    if table.num_rows < args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} samples, dataset has {table.num_rows}"
        )

    rng = random.Random(args.seed)
    indices = rng.sample(range(table.num_rows), args.num_samples)
    selected = table.take(indices).to_pylist()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

    all_input_ids: list[list[int]] = []
    all_attention_masks: list[list[int]] = []
    all_labels: list[list[int]] = []
    untruncated_lengths: list[int] = []

    for row in selected:
        messages = normalize_messages(row["prompt"], row["solution"])
        token_ids = encode_messages(tokenizer, messages)
        untruncated_lengths.append(len(token_ids))
        token_ids = token_ids[: args.seq_length]
        valid_length = len(token_ids)
        padding = args.seq_length - valid_length

        all_input_ids.append(token_ids + [pad_token_id] * padding)
        all_attention_masks.append([1] * valid_length + [0] * padding)
        all_labels.append(token_ids + [-100] * padding)

    cache = {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)

    metadata = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "parquet_files": [
            {
                "path": str(path.resolve()),
                "sha256": sha256sum(path),
            }
            for path in parquet_files
        ],
        "model_path": str(args.model_path.resolve()),
        "split": args.split,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "sample_indices": indices,
        "pad_token_id": pad_token_id,
        "chat_template_used": bool(getattr(tokenizer, "chat_template", None)),
        "untruncated_length_min": min(untruncated_lengths),
        "untruncated_length_max": max(untruncated_lengths),
        "untruncated_length_mean": (
            sum(untruncated_lengths) / len(untruncated_lengths)
        ),
        "nonpadding_tokens": int(cache["attention_mask"].sum().item()),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {args.num_samples} samples at seq_length={args.seq_length}: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
