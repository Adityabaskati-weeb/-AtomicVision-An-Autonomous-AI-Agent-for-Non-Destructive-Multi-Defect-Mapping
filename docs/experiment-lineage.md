# Experiment Lineage

This page is the short, judge-friendly map of which adapters matter, what they
were trying to fix, and which one is currently the best base for future work.

## Current Promotion Order

| Role | Adapter | Status | Why it matters |
| --- | --- | --- | --- |
| Best current checkpoint | [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora) | Promoted | Preserves perfect strict execution and improves the medium held-out slice |
| Stable fallback | [prodigyhuh/atomicvision-format-submit-merged-lora](https://huggingface.co/prodigyhuh/atomicvision-format-submit-merged-lora) | Preserved | Recovery-safe adapter with reliable two-step tool behavior |
| Hard-frontier SFT experiment | [prodigyhuh/atomicvision-hard-frontier-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-hard-frontier-boost-lora) | Not promoted | Stayed reliable but did not improve the hard slice |
| Hard-only GRPO probe | `atomicvision-hard-only-grpo-reference-probe` | In progress | Current method test for improving hard-case quality without breaking execution |

## What Each Stage Solved

### 1. Format-repair / two-step recovery

Goal:

- stop malformed tool calls
- stabilize `ask_prior -> submit_defect_map`
- recover from NaN-prone or schema-fragile SFT runs

Result:

- strict and normalized verifier columns became reliable enough to trust

Primary artifact:

- [prodigyhuh/atomicvision-format-submit-merged-lora](https://huggingface.co/prodigyhuh/atomicvision-format-submit-merged-lora)

### 2. Medium-fidelity booster

Goal:

- keep the recovered process policy
- improve outcome quality on medium cases

Result:

- medium held-out reward improved over the prior-submit baseline
- hard stayed unchanged relative to the stable recovery adapter

Primary artifact:

- [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)

### 3. Hard-frontier SFT booster

Goal:

- improve the remaining hard-case quality gap with more targeted hard-only SFT

Result:

- remained reliable
- did not improve the hard held-out slice

Primary artifact:

- [prodigyhuh/atomicvision-hard-frontier-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-hard-frontier-boost-lora)

### 4. Hard-only GRPO probe

Goal:

- test whether reward-based optimization can move the hard slice now that SFT
  has mostly stopped helping

Status:

- active investigation
- not a promoted checkpoint yet

## Promotion Rule

An adapter is promoted only if:

1. strict execution remains healthy,
2. normalized execution remains healthy,
3. held-out evaluation uses the official eval-only seed band,
4. hard reward improves materially,
5. medium does not regress beyond tolerance.

Until then, the best current base remains:

- [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
