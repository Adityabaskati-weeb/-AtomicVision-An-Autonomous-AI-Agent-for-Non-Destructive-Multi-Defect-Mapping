# AtomicVision Reward Comparison

This report compares deterministic baseline policies for the AtomicVision OpenEnv lab.
The trained GRPO agent in the next phase must improve over these baselines, especially `prior_submit`.

Episodes per policy/difficulty: `6`

## Best Policy By Difficulty

| Difficulty | Best policy | Mean reward | Mean F1 | Mean scan cost |
| --- | --- | ---: | ---: | ---: |
| easy | oracle | 7.900 | 1.000 | 0.000 |
| medium | oracle | 7.900 | 1.000 | 0.000 |
| hard | oracle | 7.900 | 1.000 | 0.000 |

## Full Results

| Difficulty | Policy | Reward | F1 | Concentration MAE | Steps | Scan cost | Timeout rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | cheap_submit | -0.600 | 0.000 | 0.1806 | 1.00 | 0.00 | 0.00 |
| easy | random | -0.955 | 0.250 | 0.1313 | 2.83 | 5.00 | 0.17 |
| easy | scan_heavy | 3.130 | 0.778 | 0.0362 | 5.00 | 8.00 | 0.00 |
| easy | prior_submit | 3.464 | 0.611 | 0.0450 | 2.00 | 1.50 | 0.00 |
| easy | oracle | 7.900 | 1.000 | 0.0000 | 1.00 | 0.00 | 0.00 |
| medium | cheap_submit | -1.300 | 0.000 | 0.1087 | 1.00 | 0.00 | 0.00 |
| medium | random | -1.164 | 0.244 | 0.0998 | 2.83 | 5.00 | 0.17 |
| medium | scan_heavy | 3.100 | 0.800 | 0.0356 | 5.00 | 8.00 | 0.00 |
| medium | prior_submit | 4.620 | 0.778 | 0.0254 | 2.00 | 1.50 | 0.00 |
| medium | oracle | 7.900 | 1.000 | 0.0000 | 1.00 | 0.00 | 0.00 |
| hard | cheap_submit | -2.000 | 0.000 | 0.0754 | 1.00 | 0.00 | 0.00 |
| hard | random | -1.564 | 0.293 | 0.0719 | 2.83 | 5.00 | 0.17 |
| hard | scan_heavy | -6.000 | 0.000 | 0.0754 | 4.00 | 8.00 | 1.00 |
| hard | prior_submit | 4.641 | 0.813 | 0.0237 | 2.00 | 1.50 | 0.00 |
| hard | oracle | 7.900 | 1.000 | 0.0000 | 1.00 | 0.00 | 0.00 |

## Judge Narrative

- `cheap_submit` shows the environment punishes blind guessing.
- `scan_heavy` shows extra scans are useful only when they improve final accuracy enough to justify cost.
- `prior_submit` is the strongest non-training baseline for Phase 11.
- `oracle` is an upper-bound sanity check, not a deployable agent.

## Trained Model Rollout Update

After baseline comparison, the Kaggle SFT-copy LoRA
`prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora` was evaluated through direct
tool-call rollouts on 32 medium-difficulty episodes. It reached `4.458` mean
reward and `0.790` F1 with `0%` tool-call failures and `100%` completion,
slightly exceeding the 32-episode `prior_submit` baseline (`4.366` reward,
`0.773` F1). See [`sft-copy-lora-results.md`](sft-copy-lora-results.md).
