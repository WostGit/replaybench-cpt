# replaybench-cpt

A tiny, reproducible continual pre-training (CPT) benchmark scaffold focused on **boring, reliable defaults**.

## Recommended stack

- **Transformers** for training/eval loops (`Trainer` + causal LM objective).
- **Datasets** for loading, streaming, and sharding replay/new-domain corpora.
- **Tokenizers** (fast Rust-backed) for CPU-efficient preprocessing.
- **SentencePiece** only for optional tokenizer-retraining experiments.

This stack is intentionally chosen to behave well on standard GitHub Actions hosted runners where CPU, RAM, and I/O are limited.

## Model tiers

- **Default:** `HuggingFaceTB/SmolLM2-135M`
- **Stress tier:** `HuggingFaceTB/SmolLM2-360M`
- **Legacy control (optional):** `distilgpt2`

Use base (non-instruction-tuned) models for cleaner CPT signal.

## Benchmark shape

1. Build mixed corpora with replay ratio sweeps.
2. Cache tokenized shards to avoid repeated CPU bottlenecks.
3. Run short `Trainer` loops for quick comparison.
4. Track loss/perplexity by replay ratio and model tier.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m replaybench_cpt.train \
  --model_name HuggingFaceTB/SmolLM2-135M \
  --replay_file data/replay.txt \
  --new_file data/new_domain.txt \
  --replay_ratio 0.3 \
  --output_dir runs/smollm2_135m_r03
```

## Replay-ratio sweeps

Run multiple short jobs (for example 0.1, 0.3, 0.5, 0.7) and compare validation loss/perplexity. Keep `max_steps` small on CI (for example 20-100 steps) to stay inside runner budgets.

## Repo layout

- `src/replaybench_cpt/train.py`: minimal Trainer-based CPT entrypoint.
- `configs/benchmark_defaults.yaml`: stable defaults for CI-friendly runs.
- `requirements.txt`: intentionally small dependency surface.
