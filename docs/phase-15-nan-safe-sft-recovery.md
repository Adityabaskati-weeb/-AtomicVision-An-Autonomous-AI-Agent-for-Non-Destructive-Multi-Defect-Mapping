# Phase 15 NaN-Safe SFT Recovery

## Decision

Any SFT run that logs `loss nan` is invalid. Do not promote its checkpoints and
do not run GRPO from it.

This output must be treated as a hard failure:

```text
update 1/80 | loss nan
...
saved checkpoint: /kaggle/working/atomicvision-cost-aware-masked-sft-lora/checkpoint-40
...
STEP 3: Promote checkpoint-40
STEP 4: Run GRPO variance probe
```

Saving a checkpoint after NaN does not make the adapter usable. It only freezes
corrupted or undefined updates. The next training path must use
[`training/train_sft_atomicvision_safe.py`](../training/train_sft_atomicvision_safe.py).

## What Changed

The safe SFT trainer now adds these gates:

- JSONL rows must be valid objects.
- Every row must end with an assistant `<tool_call>...</tool_call>`.
- The final tool call must match `target_tool_name` when present.
- Assistant masking must leave at least one valid label token.
- The first forward loss must be finite.
- Every later loss must be finite.
- Gradient norm must be finite.
- Checkpoints are saved only after finite updates.

## Kaggle Recovery Sequence

Start from a clean Kaggle GPU runtime. If an old NaN adapter folder exists, do
not reuse it.

### 1. Generate The Dataset

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

### 2. Validate The JSONL Before Loading A Model

```bash
python training/train_sft_atomicvision_safe.py \
  --dataset-jsonl outputs/sft/atomicvision_cost_aware_masked_sft.jsonl \
  --validate-only
```

Expected output:

```text
DATASET VALIDATION PASSED
```

If this fails, fix the dataset. Do not train.

### 3. Run A Tiny Finite-Loss Sanity Train

```bash
python training/train_sft_atomicvision_safe.py \
  --dataset-jsonl outputs/sft/atomicvision_cost_aware_masked_sft.jsonl \
  --model Qwen/Qwen3-1.7B \
  --output-dir /kaggle/working/atomicvision-sft-sanity-lora \
  --max-examples 64 \
  --max-updates 5 \
  --checkpoint-steps 5 \
  --learning-rate 2e-5 \
  --max-grad-norm 1.0 \
  --overwrite-output-dir
```

Expected output:

```text
DATASET VALIDATION PASSED
ASSISTANT MASK VALIDATION PASSED
update 1/5 | loss <finite number>
...
TRAINING DONE
```

If any loss is `nan` or `inf`, the script will stop. Do not continue.

### 4. Full Safe SFT

Run this only after the 5-update sanity train passes:

```bash
python training/train_sft_atomicvision_safe.py \
  --dataset-jsonl outputs/sft/atomicvision_cost_aware_masked_sft.jsonl \
  --model Qwen/Qwen3-1.7B \
  --output-dir /kaggle/working/atomicvision-cost-aware-masked-sft-lora \
  --max-updates 80 \
  --grad-accum 8 \
  --batch-size 1 \
  --max-length 1536 \
  --learning-rate 2e-5 \
  --max-grad-norm 1.0 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --checkpoint-steps 40 60 80 \
  --overwrite-output-dir
```

## Promotion Gate

After safe SFT, promote nothing until checkpoint evaluation passes:

- loss must be finite for all updates
- `safe_sft_report.json` must show `"status": "success"`
- medium eval must beat or match prior-submit
- held-out medium eval must not collapse
- hard eval must not increase tool failures
- tool failure rate must stay near `0`
- scan cost should stay near `1.5`

Only then copy the best checkpoint to:

```text
/kaggle/working/atomicvision-best-cost-aware-masked-sft-lora
```

## GRPO Gate

GRPO remains blocked until the promoted adapter is finite and evaluated.

When SFT is healthy, the first GRPO step is still only:

```bash
python training/train_grpo_atomicvision.py \
  --preset cost-aware-variance-probe \
  --adapter-model-id /kaggle/working/atomicvision-best-cost-aware-masked-sft-lora \
  --report-to none
```

Continue only if:

- `reward_std > 0`
- `frac_reward_zero_std < 1`
- `grad_norm > 0`
- no tool-failure spike
- no post-terminal tool calls

## Why This Fixes The Failure

The earlier NaN run could still save folders, which made it look successful.
The new script makes NaN impossible to ignore. A non-finite loss raises a hard
exception before promotion, before zip packaging, and before GRPO.
