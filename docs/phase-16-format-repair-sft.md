# Phase 16 Format-Repair SFT

## Decision

The promoted cost-aware SFT adapter failed held-out evaluation before it ever
took a valid tool step. The problem is not GRPO yet. The model is emitting
placeholder or malformed tool text instead of one valid JSON tool call.

This failure must be fixed with a small supervised repair pass before any more
evaluation or GRPO.

## What Failed

Held-out evaluation on seeds `1000-1031` showed:

- `steps = 0`
- `scan_cost = 0`
- `tool_failure_rate = 1.0`
- repeated outputs like:
  - `<tool_call>...</tool_call>`
  - `ask_prior`
  - mixed text plus malformed JSON

That means the first action format collapsed.

## Root Cause

Two issues aligned:

1. The old tool-system prompt showed the placeholder
   `<tool_call>...</tool_call>`, which the model copied literally.
2. The SFT dataset was too skewed toward final submit rows:
   - `ask_prior`: `26`
   - `submit_after_reference`: `51`
   - `submit_prior`: `435`

That mix taught the final submission pattern much more strongly than the first
legal tool call.

## Repair Strategy

Use the corrected prompt in `training/train_grpo_atomicvision.py` and build a
repair dataset with many more `ask_prior` rows.

Recommended repair mix per difficulty:

- `ask_prior`: `50%`
- `submit_prior`: `40%`
- `submit_after_reference`: `10%`

Recommended repair seed pool:

- `seed_start=1000`
- `difficulties=medium hard`

Recommended dataset command:

```bash
python training/generate_atomicvision_sft_data.py \
  --profile two_step_curriculum \
  --episodes-per-difficulty 256 \
  --seed-start 1000 \
  --difficulties medium hard \
  --min-scan-improvement 0.25 \
  --output-jsonl outputs/sft/atomicvision_two_step_curriculum_sft.jsonl
```

This official curriculum now replaces the earlier notebook-only merge. It
concatenates:

1. `format_repair`: first-action schema repair
2. `submit_bridge`: second-turn submit schema repair

Recommended training gate:

1. validate-only
2. 5-update sanity train
3. 40-update full repair SFT
4. held-out evaluation with `training/evaluate_atomicvision_adapter.py`
5. GRPO only if held-out tool formatting recovers

## Why 40 Updates

The current failure is schema generalization, not undertraining on the original
512-row set. The first repair run should therefore stay short and controlled.
Use `40` updates as the first gate and only extend if held-out evaluation
improves.
