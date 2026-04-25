# Hard-Only GRPO Reference Probe

This page records the first Hugging Face Jobs probe that used the cleaned seed
policy and the post-run metrics summary patch.

## Goal

Test whether a short hard-only GRPO run can improve frontier behavior without
breaking the strict two-step tool policy recovered by SFT.

## Run Setup

- Base adapter:
  [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
- HF Jobs run:
  [69ec694ad2c8bd8662bcd2d2](https://huggingface.co/jobs/prodigyhuh/69ec694ad2c8bd8662bcd2d2)
- Hardware: `a10g-large`
- Prompt focus: `reference-improvement`
- Difficulty: `hard`
- Seed band: `4000-7999`
- Output mode: no push-to-hub, metrics persisted in job logs and a JSON summary

## Final Probe Metrics

| Metric | Value |
| --- | ---: |
| `reward_std` | `0.5065` |
| `frac_reward_zero_std` | `0.75` |
| `done_rate` | `0.75` |
| `normalized_tool_call_pass_rate` | `1.00` |
| `submit_tool_rate` | `0.00` |
| `strict_tool_call_pass_rate` | `0.00` |
| `tools/failure_frequency` | `0.00` |
| `train_loss` | `6.76e-08` |
| `train_runtime_s` | `104.2181` |

## Interpretation

The probe answered an important question cleanly:

- good: reward variance is now real
- good: normalized tool parsing and tool failure rate stayed healthy
- bad: the policy still does not reliably land a strict final submit action
- bad: `submit_tool_rate` stayed at `0.00`
- bad: `strict_tool_call_pass_rate` stayed at `0.00`

So this run is useful as instrumentation proof, but it is **not** a green
light for a longer GRPO continuation.

## Promotion Decision

- Probe status: **completed**
- Probe quality: **informative**
- Promotion status: **not promoted**
- Next action: improve verifier / reward plumbing and prompt routing before
  spending more GPU time on a longer hard-only GRPO run

## Stored Artifact

The machine-readable summary for this run is committed at:

- [hard-only-grpo-reference-probe-metrics.json](hard-only-grpo-reference-probe-metrics.json)
