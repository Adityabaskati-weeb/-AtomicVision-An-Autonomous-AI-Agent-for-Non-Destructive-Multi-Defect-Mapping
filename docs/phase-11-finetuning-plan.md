# Phase 11 Fine-Tuning Plan

## Purpose

Phase 11 starts actual agent fine-tuning. The goal is to train an LLM policy that uses AtomicVision tools to beat the `prior_submit` baseline while keeping scan cost low.

## Current Baseline To Beat

From `docs/reward-comparison-report.md`, the strongest practical baseline is `prior_submit`:

- Easy: `3.464`
- Medium: `4.620`
- Hard: `4.641`

The oracle ceiling is `7.900`.

## Training Approach

Use a two-stage plan:

1. Supervised fine-tuning on exact AtomicVision tool-copy traces.
2. Hugging Face TRL `GRPOTrainer` with `environment_factory`.

Why SFT first:

- Direct LoRA rollouts learned the low-cost `ask_prior -> submit_defect_map`
  policy but sometimes changed prior defect species or concentrations.
- A short SFT stage directly teaches exact JSON/tool argument copying.
- The resulting adapter is reusable for Kaggle now and HF credit-backed runs later.

Generate the SFT JSONL locally or in Kaggle:

```bash
python training/generate_atomicvision_sft_data.py \
  --episodes-per-difficulty 512 \
  --difficulties easy medium hard \
  --output-jsonl outputs/sft/atomicvision_tool_copy_sft.jsonl
```

The generated rows contain chat `messages`, target tool-call metadata, and the
exact prior copied into `submit_defect_map`.

Why GRPO after SFT:

- TRL's OpenEnv integration supports stateful multi-turn environments.
- Tool methods are exposed directly to the model.
- Reward functions can read final reward from environment instances.

## Implemented Scaffold

- `training/train_grpo_atomicvision.py`
- `training/generate_atomicvision_sft_data.py`
- `training/requirements-grpo.txt`
- `AtomicVisionToolEnv`
- Tools exposed to the model:
  - `request_scan`
  - `zoom_band`
  - `compare_reference`
  - `ask_prior`
  - `submit_defect_map`
- `reward_func(environments, **kwargs)`
- `--dry-run` mode for endpoint and wrapper validation.
- LoRA switches for memory-friendly Colab/Kaggle runs.
- Trackio reporting switch for visible reward curves.
- Optional Hub push switches for saving trained adapters.
- Strict tool guidance in the prompt and type hints:
  - exact scan modes only
  - exact resolution values only
  - valid frequency range `0.0` to `20.0`
  - default strategy of `ask_prior` before submission

## Local Smoke Commands

Install the training-only dependencies in a GPU notebook or training runtime:

```bash
git clone https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv AtomicVision
cd AtomicVision
pip install -r training/requirements-grpo.txt
```

Important: do not `pip install torch` on Colab/Kaggle unless the runtime's
preinstalled GPU Torch is broken. The training requirements intentionally leave
Torch to the runtime so pip does not replace the CUDA-matched build.

If Colab already imported an older Transformers version and still raises
`environment_factory requires transformers version 5.2.0 or higher`, restart the
runtime after installation and rerun from `cd /content/AtomicVision`.

Validate the environment wrapper without downloading an LLM:

```bash
python training/train_grpo_atomicvision.py --dry-run --difficulty easy
```

Run the smallest GRPO smoke experiment:

```bash
python training/train_grpo_atomicvision.py --preset smoke --report-to trackio
```

This first run is intentionally low-concurrency. `2 × 1` gives a
`generation_batch_size` of 2, which is divisible by `num_generations=2` and
opens only two rollout environments. The hosted Space accepts 32 concurrent
WebSocket sessions, so this stays far below the limit. After the smoke run
succeeds, increase `--num-generations` first, then batch/accumulation.

Save adapters to Hugging Face Hub only when a write token is available:

```bash
python training/train_grpo_atomicvision.py \
  --model Qwen/Qwen3-0.6B \
  --samples 128 \
  --max-steps 100 \
  --use-peft \
  --push-to-hub \
  --hub-model-id prodigyhuh/atomicvision-qwen3-0.6b-grpo-lora
```

Set `ATOMICVISION_ENV_URL` or pass `--env-url` if training against a duplicated Space or local Docker server.

After the smoke run reaches `100% 5/5`, run:

```bash
python training/train_grpo_atomicvision.py --preset colab-20 --report-to trackio
```

For the larger-parameter path, move next to `Qwen/Qwen3-1.7B`, which is about
2.03B parameters, using the commands in `docs/training-runtime-runbook.md`.

## Required User Choices Before Long Training

1. Runtime:
   - Kaggle GPU
   - RunPod/Lambda paid GPU
   - Hugging Face Jobs after credits are available
   - local CPU smoke only

2. Model size:
   - Starter: `Qwen/Qwen3-0.6B`
   - Better: `Qwen/Qwen3-1.7B`
   - Stronger but heavier: larger Qwen/Llama/Gemma-compatible model

3. Training budget:
   - SFT copy stage: 1k-5k generated rows
   - Smoke: 5-20 GRPO steps
   - Demo: 100-300 GRPO steps
   - Stronger run: 500+ GRPO steps

4. Persistence:
   - no Hub push for first smoke test
   - push LoRA adapters to Hub for the final judged run

5. Monitoring:
   - Trackio reward curve for the judging demo
   - local CSV/JSON backup if Trackio is unavailable

## Phase 11 Completion Gate

Phase 11 is complete only when:

- GRPO scaffold dry-run passes against the deployed Space.
- Colab/Kaggle notebook includes the GRPO scaffold.
- A short training run produces reward logs.
- Reward curve compares against `prior_submit`.
- The result is documented for judges.

## Kaggle SFT-Copy Result

The direct GRPO-only LoRA rollout revealed a copy-accuracy bottleneck, so a
Kaggle SFT-copy stage was run on generated AtomicVision tool-copy traces.

Result:

- Model: `prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora`
- Training loss: `0.370648` after `120` SFT steps
- Medium direct rollout: reward `4.458`, F1 `0.790`, scan cost `1.55`
- Tool failure rate: `0%`
- Done rate: `100%`

This now slightly exceeds the 32-episode `prior_submit` baseline
(`4.366` reward, `0.773` F1). Full aggregate details are in
[`sft-copy-lora-results.md`](sft-copy-lora-results.md).

## GRPO Continuation From SFT-Copy Adapter

The next training run should initialize from the SFT-copy adapter rather than a
fresh LoRA:

```bash
python training/train_grpo_atomicvision.py \
  --model Qwen/Qwen3-1.7B \
  --adapter-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora \
  --samples 128 \
  --max-steps 50 \
  --num-generations 2 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --max-completion-length 768 \
  --learning-rate 5e-6
```

The continuation target is to preserve exact tool-copy reliability while
learning when a weak prior deserves one extra scan.
