# AtomicVision SFT-Copy GRPO Smoke Results

## Summary

A 20-step Kaggle GRPO continuation was run from the SFT-copy adapter to verify
that the continuation path works end to end.

Model:
[`prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-grpo-smoke-lora`](https://huggingface.co/prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-grpo-smoke-lora)

Initialization:
[`prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora`](https://huggingface.co/prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora)

## Medium-Difficulty Rollout

| Evaluation | Episodes | Reward | F1 | Concentration MAE | Steps | Scan cost | Tool failure rate | Done rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO smoke without tool-system prompt | 8 | -1.350 | 0.000 | 0.1035 | 1.00 | 0.00 | 1.00 | 1.00 |
| GRPO smoke with tool-system prompt | 32 | 4.366 | 0.773 | 0.0318 | 2.00 | 1.50 | 0.00 | 1.00 |
| Format-aware GRPO smoke with tool-system prompt | 32 | 4.366* | 0.773 | 0.0318 | 2.00 | 1.50 | 0.00 | 1.00 |
| SFT-copy direct rollout | 32 | 4.458 | 0.790 | 0.0321 | 2.06 | 1.55 | 0.00 | 1.00 |
| Prior-submit baseline | 32 | 4.366 | 0.773 | 0.0318 | 2.00 | 1.50 | 0.00 | 1.00 |

*The direct eval cell reported `3.766` because it summed the intermediate
`ask_prior` scan penalty separately. The comparable terminal-score convention is
`4.366`.

## Interpretation

The GRPO smoke checkpoint is useful because it proves the continuation training
pipeline can start from the SFT-copy adapter and push a new LoRA to the Hub.
It is not the current best model.

Without the tool-system prompt, the checkpoint echoes observations instead of
emitting `<tool_call>...</tool_call>` JSON. With the tool-system prompt, it
recovers valid tool use, but its aggregate score matches the deterministic
`prior_submit` baseline and falls slightly below the SFT-copy adapter.

The format-aware continuation also completed and uploaded, but its training
metrics showed no policy-learning signal:

- `reward_std=0`
- `frac_reward_zero_std=1`
- `Advantage=0`
- `loss=0`
- `grad_norm=0`

This means every grouped completion received the same reward, so GRPO had no
relative preference signal. The expected cause is policy collapse around the
deterministic sequence `ask_prior` -> exact prior copy, reinforced by a strict
tool-system prompt and a strong exact-copy shaping bonus.

## Decision

Current best checkpoint:
`prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora`

Do not promote:
`prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-grpo-smoke-lora`

Also do not promote:
`prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-grpo-format-smoke-lora`

Next GRPO runs should keep the tool-system prompt in the training dataset, but
must first prove rollout variance. Use `num_generations>=4`, sampling
temperature/top-p controls, and weaker exact-copy shaping. A run is only worth
extending when `reward_std > 0`, `frac_reward_zero_std < 1`, and `grad_norm` is
nonzero.
