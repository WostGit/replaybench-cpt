# replaybench-cpt

A deliberately small, reproducible continual pre-training (CPT) benchmark focused on replay-ratio sweeps for compact causal language models.

## Design goals

- **Boring and reliable over exotic**: use mature libraries with predictable behavior.
- **GitHub Actions friendly**: keep CPU/RAM/disk pressure low.
- **Comparable runs**: fixed short training loops and repeatable replay-ratio grids.

## Recommended stack

- **Training**: Hugging Face `transformers` (`Trainer` + causal LM workflow).
- **Data**: Hugging Face `datasets` (streaming/sharding support for replay + new-domain corpora).
- **Tokenization**: Hugging Face `tokenizers` for fast CPU-side tokenization and compatibility with Transformers.
- **Tokenizer experiments (optional)**: `sentencepiece` for custom vocab retraining comparisons.

## Model tiers

- **Default benchmark target**: `HuggingFaceTB/SmolLM2-135M`
- **Stress-test tier**: `HuggingFaceTB/SmolLM2-360M`
- **Legacy control (optional)**: `distilgpt2`

Use **base** (non instruction-tuned) models for cleaner CPT signal.

## Benchmark protocol (default)

1. Build mixed corpora using replay ratio sweeps (e.g., `0.0, 0.1, 0.25, 0.5`).
2. Pre-tokenize and cache shards.
3. Run short Trainer loops with fixed `max_steps` per sweep point.
4. Track loss/perplexity on held-out slices.

See `configs/benchmark.yaml` for the default settings and `scripts/run_cpt.py` for the training entrypoint.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_cpt.py \
  --config configs/benchmark.yaml \
  --model-name HuggingFaceTB/SmolLM2-135M
```

## Notes for CI

- Keep tokenizer worker count conservative.
- Reuse cached tokenized shards between jobs when possible.
- Prefer tiny eval sets for smoke runs, full eval only in scheduled jobs.
