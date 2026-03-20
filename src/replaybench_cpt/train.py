"""Minimal CPT training entrypoint using Hugging Face Trainer.

This script mixes replay and new-domain text using a replay ratio and then
trains/evaluates a causal language model with short, CI-friendly settings.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


SUPPORTED_MODELS = {
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "distilgpt2",
}


@dataclass
class Corpora:
    replay: Dataset
    new_domain: Dataset


def read_text_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"No non-empty lines found in {path}")
    return lines


def load_corpora(replay_file: str, new_file: str) -> Corpora:
    replay_lines = read_text_file(replay_file)
    new_lines = read_text_file(new_file)
    return Corpora(
        replay=Dataset.from_dict({"text": replay_lines}),
        new_domain=Dataset.from_dict({"text": new_lines}),
    )


def build_mixed_dataset(corpora: Corpora, replay_ratio: float, seed: int) -> Dataset:
    if not 0.0 <= replay_ratio <= 1.0:
        raise ValueError("replay_ratio must be in [0.0, 1.0]")

    total = len(corpora.replay) + len(corpora.new_domain)
    replay_target = int(total * replay_ratio)
    new_target = max(total - replay_target, 1)

    replay_sampled = corpora.replay.shuffle(seed=seed).select(
        range(min(replay_target, len(corpora.replay)))
    )
    new_sampled = corpora.new_domain.shuffle(seed=seed).select(
        range(min(new_target, len(corpora.new_domain)))
    )

    mixed = concatenate_datasets([replay_sampled, new_sampled])
    return mixed.shuffle(seed=seed)


def tokenize_and_group(dataset: Dataset, tokenizer, block_size: int) -> Dataset:
    def tokenize_batch(batch):
        return tokenizer(batch["text"])

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized.map(group_texts, batched=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiny CPT benchmark training.")
    parser.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--replay_file", required=True)
    parser.add_argument("--new_file", required=True)
    parser.add_argument("--output_dir", default="runs/default")
    parser.add_argument("--replay_ratio", type=float, default=0.3)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {args.model_name}")

    set_seed(args.seed)
    random.seed(args.seed)

    corpora = load_corpora(args.replay_file, args.new_file)
    mixed = build_mixed_dataset(corpora, replay_ratio=args.replay_ratio, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm_dataset = tokenize_and_group(mixed, tokenizer=tokenizer, block_size=args.block_size)
    split = lm_dataset.train_test_split(test_size=0.1, seed=args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=max(1, args.max_steps // 2),
        save_steps=max(1, args.max_steps // 2),
        logging_steps=max(1, args.max_steps // 10),
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Final eval metrics:", metrics)


if __name__ == "__main__":
    main()
