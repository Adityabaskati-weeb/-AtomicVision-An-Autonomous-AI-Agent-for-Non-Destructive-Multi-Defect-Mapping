# AtomicVision SFT-Copy LoRA Results

## Summary

The first direct Qwen3-1.7B GRPO LoRA learned the correct low-cost tool
strategy, but its final submissions sometimes changed the DefectNet-lite prior
instead of copying it exactly. A short Kaggle SFT stage was added to teach exact
tool-argument copying before future GRPO continuation.

Model:
[`prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora`](https://huggingface.co/prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora)

Training setup:

- Base model: `Qwen/Qwen3-1.7B`
- Method: Unsloth SFT LoRA
- Dataset: `3,072` generated AtomicVision tool-copy examples
- Trainable parameters: `17,432,576` of `1,738,007,552` (`1.00%`)
- Steps: `120`
- Final training loss: `0.370648`
- Runtime: `1009.246` seconds

## Medium-Difficulty Direct Rollout

The rollout evaluates the actual loaded model and LoRA by generating tool calls
against the local AtomicVision environment. It is not a hand-coded policy.

| Model or policy | Episodes | Reward | F1 | Concentration MAE | Steps | Scan cost | Tool failure rate | Done rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO-only direct rollout | 32 | 2.625 | 0.599 | 0.0783 | 2.03 | 1.55 | 0.00 | 1.00 |
| SFT-copy direct rollout | 32 | 4.458 | 0.790 | 0.0321 | 2.06 | 1.55 | 0.00 | 1.00 |
| Prior-submit baseline | 32 | 4.366 | 0.773 | 0.0318 | 2.00 | 1.50 | 0.00 | 1.00 |
| Oracle upper bound | 32 | 7.900 | 1.000 | 0.0000 | 1.00 | 0.00 | 0.00 | 1.00 |

## Interpretation

The SFT-copy LoRA improved direct rollout reward from `2.625` to `4.458` and
F1 from `0.599` to `0.790`, while keeping malformed tool calls at `0%` and
episode completion at `100%`.

The result confirms the diagnosis from the GRPO-only run: the agent had already
learned tool discipline and scan-cost control, but needed explicit training on
copying prior defect identities and concentrations into JSON tool arguments.

## Next Step

Use the SFT-copy LoRA as the initialization for a longer HF-credit-backed GRPO
continuation run. The target is to keep the same tool reliability while moving
closer to the oracle ceiling through better decisions when the prior is weak.
