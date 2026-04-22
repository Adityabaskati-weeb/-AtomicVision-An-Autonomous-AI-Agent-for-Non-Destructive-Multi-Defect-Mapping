# AtomicVision Cost-Aware Assistant-Masked SFT Results

## Summary

The reference-heavy assistant-masked SFT run fixed much of the tool-call
collapse, but it over-used expensive evidence tools. A cost-aware dataset was
therefore generated with mostly cheap `ask_prior -> submit_defect_map` traces
and a small curated slice of `compare_reference` traces.

Promoted checkpoint:

`/kaggle/working/atomicvision-cost-aware-masked-sft-lora/checkpoint-40`

Training setup:

- Base model: `Qwen/Qwen3-1.7B`
- Method: 4-bit QLoRA assistant-masked SFT
- Dataset: `512` generated AtomicVision cost-aware examples
- Sample mix: `435` `submit_prior`, `51` `submit_after_reference`, `26` `ask_prior`
- Trainable parameters: `17,432,576` of `1,738,007,552` (`1.0030%`)
- Learning rate: `5e-5`
- Gradient accumulation: `8`
- Max sequence length: `1536`
- Saved checkpoints: `40`, `60`, `80`

## Medium-Difficulty Direct Rollout

The rollout evaluates the loaded model and LoRA by generating tool calls against
the local AtomicVision environment. It is not a hand-coded policy.

| Model or policy | Episodes | Reward | F1 | Concentration MAE | Steps | Scan cost | Tool failure rate | Done rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cost-aware masked SFT checkpoint-40 | 32 | 4.475 | 0.791 | 0.0288 | 2.00 | 1.50 | 0.00 | 1.00 |
| Cost-aware masked SFT checkpoint-60 | 32 | 4.475 | 0.791 | 0.0288 | 2.00 | 1.50 | 0.00 | 1.00 |
| Cost-aware masked SFT checkpoint-80 | 32 | 4.475 | 0.791 | 0.0288 | 2.00 | 1.50 | 0.00 | 1.00 |
| Previous SFT-copy direct rollout | 32 | 4.458 | 0.790 | 0.0321 | 2.06 | 1.55 | 0.00 | 1.00 |
| Prior-submit baseline | 32 | 4.366 | 0.773 | 0.0318 | 2.00 | 1.50 | 0.00 | 1.00 |
| Oracle upper bound | 32 | 7.900 | 1.000 | 0.0000 | 1.00 | 0.00 | 0.00 | 1.00 |

## Interpretation

The cost-aware assistant-masked SFT adapter is the current best promoted
checkpoint. It improves reward and F1 over the prior-submit baseline while
preserving the same low scan cost, exact episode completion, and zero tool
failures.

Compared with the previous SFT-copy adapter, it slightly improves reward,
concentration MAE, step count, and scan cost. The key behavioral gain is that
the model keeps the cheap prior-submit policy instead of over-using reference or
scan tools.

## Decision

Promote `checkpoint-40` as the best adapter for the demo. Checkpoints `60`,
`80`, and the final adapter scored identically in the 32-episode eval, so the
earliest checkpoint is preferred as the least-trained stable checkpoint.

Do not run GRPO as the next default step. GRPO is now an ablation only, because
the current model already beats the baseline and GRPO previously risked
zero-variance or behavior regression.
