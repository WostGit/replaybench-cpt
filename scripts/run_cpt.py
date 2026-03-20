#!/usr/bin/env python3
"""Minimal CPT benchmark runner with replay-ratio sweeps."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact CPT replay sweeps.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    return parser.parse_args()


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_mixed_dataset(replay_ds: Dataset, new_ds: Dataset, replay_ratio: float, seed: int) -> Dataset:
    if replay_ratio <= 0:
        return new_ds

    new_count = len(new_ds)
    replay_take = int((replay_ratio / max(1e-8, 1.0 - replay_ratio)) * new_count)
    replay_take = min(replay_take, len(replay_ds))

    replay_sample = replay_ds.shuffle(seed=seed).select(range(replay_take))
    mixed = concatenate_datasets([new_ds, replay_sample]).shuffle(seed=seed)
    return mixed


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)

    seed = cfg.get("seed", 42)
    set_seed(seed)
    random.seed(seed)

    model_name = args.model_name or cfg["models"]["default"]
    text_column = cfg.get("text_column", "text")
    max_length = int(cfg.get("max_length", 512))
    num_proc = int(cfg.get("num_proc", 2))

    replay_cfg = cfg["datasets"]["replay"]
    new_cfg = cfg["datasets"]["new_domain"]

    replay_ds = load_dataset(replay_cfg["path"], data_files=replay_cfg["data_files"], split="train")
    new_ds = load_dataset(new_cfg["path"], data_files=new_cfg["data_files"], split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    tcfg = cfg["training"]
    base_output = Path(tcfg["output_dir"])

    for ratio in cfg["replay"]["ratios"]:
        mixed = build_mixed_dataset(replay_ds, new_ds, float(ratio), seed)
        tokenized = mixed.map(
            tokenize_batch,
            batched=True,
            remove_columns=mixed.column_names,
            num_proc=num_proc,
            desc=f"Tokenizing ratio={ratio}",
        )
        split = tokenized.train_test_split(test_size=0.02, seed=seed)

        run_dir = base_output / f"ratio_{str(ratio).replace('.', '_')}"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        training_args = TrainingArguments(
            output_dir=str(run_dir),
            max_steps=int(tcfg["max_steps"]),
            per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(tcfg["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
            learning_rate=float(tcfg["learning_rate"]),
            weight_decay=float(tcfg["weight_decay"]),
            warmup_ratio=float(tcfg["warmup_ratio"]),
            logging_steps=int(tcfg["logging_steps"]),
            eval_steps=int(tcfg["eval_steps"]),
            save_steps=int(tcfg["save_steps"]),
            bf16=bool(tcfg.get("bf16", False)),
            fp16=bool(tcfg.get("fp16", False)),
            evaluation_strategy="steps",
            save_strategy="steps",
            report_to=[],
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            data_collator=collator,
            tokenizer=tokenizer,
        )
        trainer.train()
        metrics = trainer.evaluate()
        print(f"ratio={ratio} eval={metrics}")


if __name__ == "__main__":
    main()
