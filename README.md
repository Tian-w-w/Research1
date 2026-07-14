# Dynamic visual-token budget routing: phase 1

This package is the first, deliberately small stage of the experiment:

1. Convert the local ScienceQA and ChartQA folders to shared JSONL manifests.
2. Load the offline Hugging Face LLaVA-1.5-7B checkpoint.
3. Confirm an image produces exactly 576 projected visual tokens.
4. Run one deterministic generation before implementing token pruning.

## Server setup

Copy `config/paths.example.yaml` to `config/paths.yaml`. The supplied example
already contains the paths provided for this server.

Install a PyTorch build compatible with the server CUDA driver, then install:

```bash
python -m pip install -r requirements.txt
```

The checkpoint path is expected to be a complete Hugging Face LLaVA checkpoint
(not a LoRA adapter). The full HF LLaVA checkpoint already contains its vision
tower. Keep the separately uploaded CLIP directory because it is useful for a
later custom LLaVA loader, but do not load it twice in this smoke test.

## Build manifests

```bash
python scripts/prepare_manifests.py --config config/paths.yaml
```

For a quick filesystem check first:

```bash
python scripts/prepare_manifests.py --config config/paths.yaml --max-per-split 2
```

Expected output is under:

```text
/home/wangzhengrui/wzr_research_optimize/outputs/manifests/
```

## Offline smoke test

```bash
python scripts/smoke_test.py \
  --config config/paths.yaml \
  --manifest /home/wangzhengrui/wzr_research_optimize/outputs/manifests/scienceqa_val.jsonl \
  --index 0
```

The required checkpoint check is:

```text
Image features shape: (1, 576, 4096)
Verified: 576 visual tokens.
```

Do not begin relevance-label generation or dynamic-budget training until this
command succeeds for one ScienceQA image and one ChartQA image.

## Fixed-budget baseline

After the smoke test, run a small 50-sample benchmark before creating any
explainability labels. It uniformly keeps 72, 144, 288, 432, or 576 image
tokens, so it is a sanity baseline rather than the proposed router.

```bash
python scripts/fixed_budget_benchmark.py \
  --config config/paths.yaml \
  --manifest /home/wangzhengrui/wzr_research_optimize/outputs/manifests/scienceqa_val.jsonl \
  --output /home/wangzhengrui/wzr_research_optimize/outputs/fixed_budget/scienceqa_val_50.jsonl \
  --max-samples 50
```

Repeat the same command with `chartqa_val_human.jsonl`. The output JSONL holds
one record per `(sample, budget)` pair, including prefill and generation time.
The reported ChartQA `quick_exact_correct` is diagnostic only; use ChartQA's
official numeric-tolerance metric for final paper results.

## Important compatibility note

This code uses `transformers.LlavaForConditionalGeneration`, so it requires a
Transformers version with LLaVA support. If the uploaded checkpoint is instead
the original `liuhaotian/LLaVA` format rather than the Hugging Face converted
format, report the exact `config.json` error instead of changing the checkpoint.
The loader will then be adapted to that format.
