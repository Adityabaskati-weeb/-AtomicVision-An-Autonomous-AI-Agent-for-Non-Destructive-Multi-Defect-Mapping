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
  - system tool-contract prompt by default
  - exact scan modes only
  - exact resolution values only
  - valid frequency range `0.0` to `20.0`
  - default strategy of `ask_prior` before submission
- Small reward shaping terms:
  - valid single `<tool_call>...</tool_call>` JSON format
  - exact confident-prior copying into `submit_defect_map`

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

This first run is still conservative, but it now uses a comparison group large
enough to catch zero-variance failures. `1 x 4` gives a `generation_batch_size`
of 4, which is divisible by `num_generations=4`. If `reward_std=0`,
`frac_reward_zero_std=1`, and `grad_norm=0`, stop and increase sampling
diversity before extending the run.

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

## Kaggle Cost-Aware Assistant-Masked SFT Result

The cost-aware assistant-masked SFT run is now the best promoted checkpoint.
It used `512` medium-difficulty generated examples:

- `submit_prior`: `435`
- `submit_after_reference`: `51`
- `ask_prior`: `26`

Promoted checkpoint:

`/kaggle/working/atomicvision-cost-aware-masked-sft-lora/checkpoint-40`

Medium direct rollout:

- Reward: `4.47530128125`
- F1: `0.79107153125`
- MAE: `0.0288233125`
- Mean steps: `2.0`
- Mean scan cost: `1.5`
- Tool failure rate: `0.0`
- Done rate: `1.0`

This beats both the prior-submit baseline and the previous SFT-copy adapter.
Full aggregate details are in
[`cost-aware-masked-sft-results.md`](cost-aware-masked-sft-results.md).

## GRPO Continuation From SFT-Copy Adapter

The first Kaggle continuation smoke proved that the training path runs and
pushes a LoRA, but the resulting adapter was not promoted:

- Model: `prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-grpo-smoke-lora`
- With tool-system prompt: reward `4.366`, F1 `0.773`, failures `0%`
- Without tool-system prompt: malformed rollout behavior, failures `100%`
- Decision: keep `prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora` as best

The next training run should initialize from the SFT-copy adapter and use the
variance-aware scaffold. Run a short variance probe first:

```bash
python training/train_grpo_atomicvision.py \
  --preset variance-probe \
  --adapter-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora \
  --report-to none \
  --run-name atomicvision-sft-copy-grpo-variance-probe
```

Only continue if the probe logs nonzero `reward_std`, nonzero `grad_norm`, and
`frac_reward_zero_std < 1`. Then run the longer continuation:

```bash
python training/train_grpo_atomicvision.py \
  --model Qwen/Qwen3-1.7B \
  --adapter-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora \
  --samples 128 \
  --max-steps 50 \
  --num-generations 8 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-completion-length 768 \
  --temperature 1.2 \
  --top-p 0.95 \
  --top-k 50 \
  --learning-rate 1e-6 \
  --report-to trackio \
  --push-to-hub \
  --hub-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-grpo-variance-lora
```

The continuation target is to preserve exact tool-copy reliability while
learning when a borderline prior deserves one extra scan.

## Curated Reference SFT Bridge

The variance-aware smoke can produce train-time learning signal, but the latest
medium eval still collapsed to the prior-submit behavior:

- Adapter: `/kaggle/working/atomicvision-mid-explore-smoke-lora`
- Reward: `3.7664260625` in eval-sum style
- F1: `0.773214375`
- MAE: `0.0317914375`
- Mean scan cost: `1.5`
- Tool failures: `0.0`

The exact reason is now visible in the code path: old SFT taught only
prior-copy, and the model-facing observation formatter did not expose the
actual spectrum or reference deltas. Extra scans were valid but mostly
unusable. The formatter now emits compact spectral summaries and the SFT
generator can create curated `submit_after_reference` examples.

The first assistant-masked mixed SFT run improved tool-call reliability, but it
over-used expensive tools:

- Reward: `3.83183534375`
- F1: `0.722023875`
- MAE: `0.03924984375`
- Mean steps: `3.0`
- Mean scan cost: `2.0`
- Tool failure rate: `0.09375`

The next bridge should be cost-aware rather than reference-heavy. Generate the
recommended dataset:

```bash
python training/generate_atomicvision_sft_data.py \
  --profile cost_aware \
  --episodes-per-difficulty 512 \
  --difficulties medium \
  --submit-prior-ratio 0.85 \
  --reference-ratio 0.10 \
  --min-scan-improvement 0.25 \
  --output-jsonl outputs/sft/atomicvision_cost_aware_masked_sft.jsonl
```

Expected 512-row mix:

- `submit_prior`: about 435
- `submit_after_reference`: about 51
- `ask_prior`: about 26

The older reference-heavy ablation command is:

```bash
python training/generate_atomicvision_sft_data.py \
  --episodes-per-difficulty 256 \
  --difficulties medium \
  --sample-types submit_prior submit_after_reference \
  --min-scan-improvement 0.25 \
  --output-jsonl outputs/sft/atomicvision_mixed_copy_reference_sft.jsonl
```

The cost-aware run passed the promotion gate. Do not use GRPO as the default
next step; keep GRPO as a controlled ablation only. The demo checkpoint is
`checkpoint-40`.
