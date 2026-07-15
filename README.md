# BARS: budget-aware reasoning routing

This repository contains **BARS** (Budget-Aware Reasoning Scheduler):
choose whether a multimodal model should Solve, Verify, Replan, or Stop under a
finite *reasoning-token* budget. The main setting is **Qwen3-VL-4B + ChartQA**;
ScienceQA is used for transfer.

## BARS phase-1 baseline

Copy `config/bars.paths.example.yaml` to `config/bars.paths.yaml`, filling in
absolute paths on the offline GPU server. All model calls force Hugging Face
offline mode.

Install a CUDA-compatible PyTorch wheel first, then run
`python -m pip install -r requirements.txt` from the local offline wheelhouse.

```bash
python scripts/check_bars_environment.py --config config/bars.paths.yaml
python scripts/prepare_manifests.py --config config/bars.paths.yaml
python scripts/bars_fixed_budget.py \
  --config config/bars.paths.yaml \
  --manifest /absolute/path/to/manifests/chartqa_val_human.jsonl \
  --output /absolute/path/to/bars_outputs/fixed_budget/chartqa_val_human.jsonl \
  --max-samples 50
```

`bars_fixed_budget.py` is the first quality-cost curve: it evaluates identical
examples at fixed `max_new_tokens` budgets (128/256/512/1024 by default), saves
per-example generated-token counts, latency, memory, raw output, final answer,
ChartQA relaxed correctness, final-answer completion, and budget exhaustion.
Use ChartQA test only for the final report; controller tuning belongs to train/val.

The baseline uses Qwen3-VL's thinking-enabled chat template and requires a
step-by-step trace ending in `Final answer: ...`. It deliberately passes the
raw manifest `question`, not its legacy concise-answer prompt. If the mean
generated tokens are nearly identical across budget caps, inspect the saved
`prediction` fields before interpreting the curve: the model may be ending
before it consumes the available budget.

```bash
python scripts/summarize_bars_fixed_budget.py \
  --input /absolute/path/to/bars_outputs/fixed_budget/chartqa_val_human.jsonl \
  --output /absolute/path/to/bars_outputs/fixed_budget/chartqa_val_human.md
```

The next implementation stage is the four BARS actions and the rule controller;
do not treat this fixed-budget script as a dynamic router.
