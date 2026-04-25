# Hard Error Mining

This page records the first held-out hard error mining run against the current
best adapter.

## Goal

Move from vague "hard cases are weaker" intuition to a concrete diagnosis of
how the current best model loses on held-out hard seeds.

## Run Setup

- Base adapter:
  [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
- HF Jobs run:
  [69ed04d2d70108f37acdec35](https://huggingface.co/jobs/prodigyhuh/69ed04d2d70108f37acdec35)
- Hardware: `a10g-large`
- Difficulty: `hard`
- Episodes: `64`
- Seed band: `10000-10063`
- Verifier mode: `strict`

## Final Summary

| Metric | Value |
| --- | ---: |
| `mean_reward` | `4.6413` |
| `baseline_mean_reward` | `5.0404` |
| `mean_reward_delta_vs_baseline` | `-0.3991` |
| `worse_than_baseline_count` | `9` |

## Failure Buckets

Among the seeds where the adapter was worse than the prior-submit baseline:

- `missed_defects`: `9`

Action patterns across all hard episodes:

- `prior_then_submit`: `64`

## Interpretation

This run gave us the first clean answer about the remaining hard weakness:

- the problem is **not** formatting
- the problem is **not** tool failure
- the problem is **not** scan-loop instability
- the dominant hard regression is **missed defects**
- the model is taking the same simple path on every hard seed:
  `ask_prior -> submit_defect_map`

That means the next useful improvement path should focus on **hard-case recall**:

- cases where the prior under-covers the true defect set
- cases where `compare_reference` would recover a missed species
- cases where the model should avoid immediate submit on sparse priors

## Top Negative Deltas

| seed | delta vs baseline | issue bucket | action pattern |
| --- | ---: | --- | --- |
| `10048` | `-6.6555` | `missed_defects` | `prior_then_submit` |
| `10013` | `-4.6358` | `missed_defects` | `prior_then_submit` |
| `10056` | `-4.0046` | `missed_defects` | `prior_then_submit` |
| `10047` | `-3.8041` | `missed_defects` | `prior_then_submit` |
| `10004` | `-1.9997` | `missed_defects` | `prior_then_submit` |

## Promotion Decision

- Run status: **completed**
- Run quality: **high-value diagnostic**
- Promotion status: **not a model promotion run**
- Next action: build a targeted hard-recall data path from missed-defect seeds

## Stored Artifact

The machine-readable summary for this run is committed at:

- [hard-error-mining-metrics.json](hard-error-mining-metrics.json)
