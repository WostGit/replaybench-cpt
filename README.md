# replaybench-cpt

A minimal, reproducible benchmark harness for **continual pre-training (CPT)** under constrained CI resources.

## Design goals

- Keep the stack boring, small, and reliable.
- Optimize for **CPU-only GitHub Actions** runners.
- Make replay-ratio studies easy to reproduce.

## Recommended core stack

- **Transformers**: training/evaluation layer for causal language modeling and short `Trainer` loops.
- **Datasets**: streaming/sharding for replay and new-domain corpora.
- **Tokenizers**: fast CPU-side tokenization and tight integration with Transformers.
- **SentencePiece** (optional): only for custom vocabulary experiments.

This stack avoids bespoke training code and keeps tokenization/data loading from becoming hidden bottlenecks on modest CI runners.

## Model tiers

- **Default**: `HuggingFaceTB/SmolLM2-135M`
- **Stress tier**: `HuggingFaceTB/SmolLM2-360M` (optional)
- **Legacy control**: `distilgpt2` (baseline only)

Use **base language models** (not instruction-tuned variants) for cleaner CPT signal.

## Benchmark protocol

1. Build replay/new-domain mixes via replay-ratio sweeps.
2. Tokenize once, cache tokenized shards, and reuse between runs.
3. Run short, fixed-budget `Trainer` jobs for each sweep point.
4. Track perplexity/loss deltas against the same held-out slices.

## Why this shape works in GitHub Actions

Standard hosted runners have limited CPU/RAM/SSD, so the benchmark should:

- keep model sizes compact,
- minimize custom pipeline complexity,
- prioritize cacheable preprocessing,
- and use stable, widely supported libraries.

That keeps turnaround predictable while still yielding meaningful CPT comparisons.
