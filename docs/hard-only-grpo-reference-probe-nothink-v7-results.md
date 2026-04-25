# Hard-Only GRPO No-Think Probe

This page records the follow-up Hugging Face Jobs probe that added the official
Qwen `/no_think` switch to both the system and user prompts.

## Goal

Cross-check the GRPO prompt path against the official Qwen guidance for
non-thinking mode and test whether that removes the stray `<think>` wrapper
behavior seen in earlier probes.

## Run Setup

- Base adapter:
  [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
- HF Jobs run:
  [69ec7e99d70108f37acde34b](https://huggingface.co/jobs/prodigyhuh/69ec7e99d70108f37acde34b)
- Hardware: `a10g-large`
- Branch: `codex-strict-submit-contract-probe`
- Commit under test: `85a0874053c956d7ed5d2109e60154432494baef`
- Prompt focus: `reference-improvement`
- Difficulty: `hard`
- Seed band: `4000-7999`
- Output mode: no push-to-hub, metrics persisted in the HF job logs

## Sanity Markers

The job confirmed the prompt patch was present:

- `PATCH_V7_SYSTEM_NO_THINK True`
- `PATCH_V7_USER_NO_THINK True`

## Final Probe Metrics

| Metric | Value |
| --- | ---: |
| `reward_std` | `0.4570` |
| `frac_reward_zero_std` | `0.75` |
| `done_rate` | `0.25` |
| `ask_prior_tool_rate` | `0.25` |
| `normalized_tool_call_pass_rate` | `0.25` |
| `normalized_tool_call_repair_rate` | `0.25` |
| `submit_tool_rate` | `0.00` |
| `strict_tool_call_pass_rate` | `0.00` |
| `strict_submit_reward_mean` | `0.00` |
| `tools/failure_frequency` | `0.00` |
| `total_reward_mean` | `0.1677` |
| `train_loss` | `3.95e-08` |

## Interpretation

The official Qwen control did help one part of the picture:

- `reward_std` improved from the v6 value of `0.3490` to `0.4570`

But it did **not** solve the practical blocker:

- the rendered sample log still showed an empty `<think>...</think>` wrapper
- `submit_tool_rate` remained `0.00`
- `strict_tool_call_pass_rate` remained `0.00`
- `done_rate` remained `0.25`

So the evidence points to a deeper issue than prompt wording alone. In this GRPO
path, `enable_thinking=False` plus `/no_think` is still not enough to suppress
the wrapper behavior in practice.

## Stored Artifact

- [hard-only-grpo-reference-probe-nothink-v7-metrics.json](hard-only-grpo-reference-probe-nothink-v7-metrics.json)
