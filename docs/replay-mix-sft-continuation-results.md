# Replay-Mix SFT Continuation

This page records the short Hugging Face Jobs continuation that mixed the
previously helpful medium replay path with a smaller hard frontier slice.

## Goal

Test whether a small replay-mixed SFT continuation can preserve the current
best adapter's strict execution while improving held-out hard quality without
giving back medium performance.

## Run Setup

- Base adapter:
  [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
- HF Jobs run:
  [69ece95ad70108f37acde9ce](https://huggingface.co/jobs/prodigyhuh/69ece95ad70108f37acde9ce)
- Hardware: `a10g-large`
- Data mix:
  - medium replay from `cost_aware`
  - smaller hard slice from `hard_frontier_boost`
- Continuation updates: `6`
- Learning rate: `2e-6`
- Held-out eval band: `10000-10999`

## Final Comparison Vs Current Best

| Metric | Current best | Replay-mix run | Delta |
| --- | ---: | ---: | ---: |
| `medium_reward` | `4.5065` | `4.5065` | `0.0000` |
| `medium_f1` | `0.7891` | `0.7891` | `0.0000` |
| `hard_reward` | `4.6917` | `4.6575` | `-0.0343` |
| `hard_f1` | `0.8162` | `0.8132` | `-0.0030` |
| `strict_tool_call_pass_rate` | `1.00` | `1.00` | `0.00` |
| `normalized_tool_call_pass_rate` | `1.00` | `1.00` | `0.00` |
| `tool_failure_rate` | `0.00` | `0.00` | `0.00` |
| `done_rate` | `1.00` | `1.00` | `0.00` |

## Interpretation

The run answered the question cleanly:

- good: strict execution stayed perfect
- good: normalized execution stayed perfect
- good: medium quality did not regress
- bad: hard reward slipped slightly
- bad: hard F1 slipped slightly

So the replay-mix path is stable, but it is **not** an upgrade over the
current best adapter.

## Promotion Decision

- Run status: **completed**
- Run quality: **stable but not improved**
- Promotion status: **not promoted**
- Current best remains:
  [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)

## Stored Artifact

The machine-readable summary for this run is committed at:

- [replay-mix-sft-continuation-metrics.json](replay-mix-sft-continuation-metrics.json)
