# Hard-Only GRPO Soft-Strict Probe

This page records the follow-up Hugging Face Jobs probe that softened the
strict-submit contract after the v5 regression.

## Goal

Keep the helpful parts of the terminal-repair work, but reduce the prompt and
reward pressure so the model can recover `ask_prior` behavior and useful reward
variance.

## Run Setup

- Base adapter:
  [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
- HF Jobs run:
  [69ec7d0fd2c8bd8662bcd59a](https://huggingface.co/jobs/prodigyhuh/69ec7d0fd2c8bd8662bcd59a)
- Hardware: `a10g-large`
- Branch: `codex-strict-submit-contract-probe`
- Commit under test: `14bd15eb0f3a116f8b6b71412572378a5fecd515`
- Prompt focus: `reference-improvement`
- Difficulty: `hard`
- Seed band: `4000-7999`
- Output mode: no push-to-hub, metrics persisted in the HF job logs

## Final Probe Metrics

| Metric | Value |
| --- | ---: |
| `reward_std` | `0.3490` |
| `frac_reward_zero_std` | `0.75` |
| `done_rate` | `0.25` |
| `ask_prior_tool_rate` | `0.25` |
| `normalized_tool_call_pass_rate` | `0.25` |
| `normalized_tool_call_repair_rate` | `0.25` |
| `submit_tool_rate` | `0.00` |
| `strict_tool_call_pass_rate` | `0.00` |
| `strict_submit_reward_mean` | `0.00` |
| `tools/failure_frequency` | `0.00` |
| `total_reward_mean` | `0.2911` |
| `train_loss` | `4.39e-08` |

## Interpretation

This patch fixed one important failure from v5:

- reward variance came back (`reward_std` > `0`)
- the rollout started calling `ask_prior` again

But it still did not get the probe over the line:

- `submit_tool_rate` stayed at `0.00`
- `strict_tool_call_pass_rate` stayed at `0.00`
- `done_rate` only recovered to `0.25`

So v6 was healthier than v5, but still not strong enough to justify a longer
hard-only GRPO continuation.

## Stored Artifact

- [hard-only-grpo-reference-probe-softstrict-v6-metrics.json](hard-only-grpo-reference-probe-softstrict-v6-metrics.json)
