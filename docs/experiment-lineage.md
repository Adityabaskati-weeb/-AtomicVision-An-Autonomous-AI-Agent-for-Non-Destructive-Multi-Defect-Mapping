# Experiment Lineage

This page is the short, judge-friendly map of which adapters matter, what they
were trying to fix, and which one is currently the best base for future work.

## Current Promotion Order

| Role | Adapter | Status | Why it matters |
| --- | --- | --- | --- |
| Best current checkpoint | [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora) | Promoted | Preserves perfect strict execution and improves the medium held-out slice |
| Stable fallback | [prodigyhuh/atomicvision-format-submit-merged-lora](https://huggingface.co/prodigyhuh/atomicvision-format-submit-merged-lora) | Preserved | Recovery-safe adapter with reliable two-step tool behavior |
| Hard-frontier SFT experiment | [prodigyhuh/atomicvision-hard-frontier-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-hard-frontier-boost-lora) | Not promoted | Stayed reliable but did not improve the hard slice |
| Hard-only GRPO probe | `atomicvision-hard-only-grpo-reference-probe` | Completed, not promoted | Produced real reward variance but still failed strict submit behavior |
| Replay-mix SFT continuation | `replay-mix-sft-continuation` | Completed, not promoted | Preserved perfect execution but did not beat the current best on held-out hard quality |
| Hard error mining diagnostic | `hard-error-mining` | Completed, informative | Showed that hard regressions are dominated by missed defects after `ask_prior -> submit` |

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

- completed on HF Jobs
- useful for instrumentation and gating
- not a promoted checkpoint yet

Primary artifacts:

- [hard-only-grpo-reference-probe-results.md](hard-only-grpo-reference-probe-results.md)
- [hard-only-grpo-reference-probe-metrics.json](hard-only-grpo-reference-probe-metrics.json)

### 5. Replay-mix SFT continuation

Goal:

- keep the strict execution recovered by earlier SFT
- reuse the only previously helpful medium replay path
- add a smaller hard slice without over-correcting into hard-only drift

Result:

- strict and normalized execution stayed perfect
- medium stayed flat
- hard regressed slightly on held-out evaluation

Primary artifacts:

- [replay-mix-sft-continuation-results.md](replay-mix-sft-continuation-results.md)
- [replay-mix-sft-continuation-metrics.json](replay-mix-sft-continuation-metrics.json)

### 6. Hard error mining

Goal:

- identify the real scientific error mode on held-out hard seeds
- separate hard semantic misses from formatting or execution failures

Result:

- the main remaining weakness is missed defects
- the dominant action path is still `ask_prior -> submit_defect_map`
- this points to a hard-recall bottleneck, not a verifier or XML bottleneck

Primary artifacts:

- [hard-error-mining-results.md](hard-error-mining-results.md)
- [hard-error-mining-metrics.json](hard-error-mining-metrics.json)

## Promotion Rule

An adapter is promoted only if:

1. strict execution remains healthy,
2. normalized execution remains healthy,
3. held-out evaluation uses the official eval-only seed band,
4. hard reward improves materially,
5. medium does not regress beyond tolerance.

Until then, the best current base remains:

- [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
