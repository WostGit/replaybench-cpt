#!/usr/bin/env python3
"""Minimal CPT benchmark entrypoint built on Transformers + Datasets."""

from __future__ import annotations

import argparse
import random
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List

import yaml
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact CPT benchmark sweeps")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--model", type=str, default=None, help="Optional model override")
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=None,
        help="Optional single replay ratio override",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _take_iterable(ds: IterableDataset, n: int) -> Dataset:
    return Dataset.from_list(list(islice(ds, n)))


def maybe_streaming_to_dataset(ds, limit: int):
    if isinstance(ds, IterableDataset):
        return _take_iterable(ds, limit)
    return ds.select(range(min(limit, len(ds))))


def load_or_mock_corpora(cfg: Dict):
    data_cfg = cfg["data"]
    text_col = data_cfg["text_column"]

    if data_cfg["replay_dataset"] and data_cfg["domain_dataset"]:
        replay = load_dataset(data_cfg["replay_dataset"], split="train", streaming=data_cfg["streaming"])
        domain = load_dataset(data_cfg["domain_dataset"], split="train", streaming=data_cfg["streaming"])
        replay = maybe_streaming_to_dataset(replay, data_cfg["max_train_samples"])
        domain = maybe_streaming_to_dataset(domain, data_cfg["max_train_samples"])
        return replay, domain, text_col

    # Fallback mock data to keep local/GHA smoke runs deterministic.
    replay_text = [
        "The benchmark replays old-domain text for stability.",
        "Replay data reduces catastrophic forgetting in small runs.",
    ] * 2048
    domain_text = [
        "New-domain adaptation is measured across replay ratio sweeps.",
        "Compact models should remain reliable on constrained CI runners.",
    ] * 2048

    replay = Dataset.from_dict({text_col: replay_text[: data_cfg["max_train_samples"]]})
    domain = Dataset.from_dict({text_col: domain_text[: data_cfg["max_train_samples"]]})
    return replay, domain, text_col


def mix_by_replay_ratio(replay: Dataset, domain: Dataset, replay_ratio: float) -> Dataset:
    total = min(len(replay), len(domain))
    replay_n = int(total * replay_ratio)
    domain_n = total - replay_n

    replay_slice = replay.select(range(replay_n)) if replay_n else Dataset.from_dict({k: [] for k in replay.column_names})
    domain_slice = domain.select(range(domain_n)) if domain_n else Dataset.from_dict({k: [] for k in domain.column_names})

    mixed_rows: List[Dict] = replay_slice.to_list() + domain_slice.to_list()
    random.shuffle(mixed_rows)
    return Dataset.from_list(mixed_rows)


def tokenize_and_group(dataset: Dataset, tokenizer, text_column: str, block_size: int, num_proc: int) -> Dataset:
    tokenized = dataset.map(
        lambda x: tokenizer(x[text_column]),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )

    def group_texts(examples: Dict[str, List[List[int]]]):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized.map(group_texts, batched=True, num_proc=num_proc)


def build_eval_split(dataset: Dataset, eval_size: int) -> DatasetDict:
    eval_size = min(eval_size, max(1, len(dataset) // 10))
    split = dataset.train_test_split(test_size=eval_size, shuffle=True, seed=42)
    return DatasetDict(train=split["train"], eval=split["test"])


def run_single(cfg: Dict, model_name: str, replay_ratio: float):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    replay, domain, text_col = load_or_mock_corpora(cfg)
    mixed = mix_by_replay_ratio(replay, domain, replay_ratio)
    split = build_eval_split(mixed, data_cfg["max_eval_samples"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train = tokenize_and_group(
        split["train"],
        tokenizer,
        text_col,
        block_size=data_cfg["block_size"],
        num_proc=data_cfg["num_proc"],
    )
    tokenized_eval = tokenize_and_group(
        split["eval"],
        tokenizer,
        text_col,
        block_size=data_cfg["block_size"],
        num_proc=data_cfg["num_proc"],
    )

    model = AutoModelForCausalLM.from_pretrained(model_name)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = Path(train_cfg["output_dir"]) / model_name.replace("/", "_") / f"replay_{replay_ratio:.2f}"
    args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        report_to=[],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print({"model": model_name, "replay_ratio": replay_ratio, **metrics})


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    model_name = args.model or cfg["model"]["default"]
    replay_ratios = [args.replay_ratio] if args.replay_ratio is not None else cfg["benchmark"]["replay_ratios"]

    for replay_ratio in replay_ratios:
        run_single(cfg, model_name=model_name, replay_ratio=float(replay_ratio))


if __name__ == "__main__":
    main()
