# REPO

Label-free reinforcement learning with an EMA teacher and evidence-weighted student updates, built on veRL.

## Setup

Use Linux, Python 3.10–3.12, and CUDA GPUs. Install CUDA-compatible PyTorch 2.6.0 first, then:

```bash
python -m pip install -c constraints-gpu.txt -e '.[vllm,gpu]'
```

Provide local model weights and the prepared parquet datasets listed in
[`repo.yaml`](verl/trainer/config/repo.yaml). To import an existing dataset copy
matching the bundled checksums:

```bash
python scripts/import_data.py --source /path/to/prepared/data
```

## Training

Run from the repository root:

```bash
export REPO_MODEL_PATH=/path/to/Qwen3-4B-Base
bash scripts/train_repo.sh --dry-run
bash scripts/train_repo.sh
```

For Qwen3-8B-Base, set its model path and use `REPO_CONFIG=repo_8b`.
Hydra overrides are supported, for example:

```bash
bash scripts/train_repo.sh trainer.total_training_steps=300 trainer.test_freq=5
```

## Defaults

- One node with 8 GPUs; 300 training steps; validation every 5 steps, without step-0 validation.
- Consensus threshold **0.8**; 8 teacher judgments per candidate; acceptance fraction strictly above **0.6**.
- EMA retention increases from **0.99** to **0.9999** over 100 steps, then stays fixed.
- 32 evidence resamples; weighting strength **0.25**.
- Evaluation: 32 responses per question; pass@1, pass@16, and pass⁴.

## Acknowledgments

This implementation builds on [veRL](https://github.com/volcengine/verl),
[TTRL](https://github.com/prime-rl/ttrl), and [CoCoV](https://github.com/shanjf666/CoCoV).
See [LICENSE](LICENSE). The local EMA/prompt implementation has CPU checks;
full multi-GPU integration validation is still pending.
