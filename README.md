# replaybench-cpt

A compact continual pre-training (CPT) benchmark scaffold designed for **reproducible CPU-first micro-runs** in GitHub Actions.

## Recommended stack (small, boring, reliable)

- **Transformers** for causal language modeling and `Trainer`-based train/eval loops.
- **Datasets** for replay/new-domain corpus loading with optional streaming and sharding.
- **Tokenizers** for fast CPU-side tokenization and compatibility with Transformers.
- **SentencePiece** as an optional tokenizer-vocabulary experiment path.

This keeps the implementation straightforward and avoids bespoke training code.

## Model tiers

- **Default benchmark tier:** `HuggingFaceTB/SmolLM2-135M`
- **Stress-test tier:** `HuggingFaceTB/SmolLM2-360M`
- **Legacy control (optional):** `distilgpt2`

The benchmark defaults to base language models (not instruction-tuned) so CPT signal quality is easier to interpret.

## Benchmark shape

- Replay-ratio sweeps (for example: `0.1, 0.25, 0.5, 0.75`)
- Short, repeatable `Trainer` runs
- Cached tokenized shards to reduce repeated CPU work

See:

- `configs/benchmark.yaml` for baseline settings.
- `scripts/run_cpt.py` for the minimal training entrypoint.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_cpt.py --config configs/benchmark.yaml
```

## GitHub Actions note

GitHub-hosted runners are CPU/RAM constrained for ML workloads. Keep runs intentionally small (few steps, tiny eval set, cached shards) to preserve signal while staying reliable.
