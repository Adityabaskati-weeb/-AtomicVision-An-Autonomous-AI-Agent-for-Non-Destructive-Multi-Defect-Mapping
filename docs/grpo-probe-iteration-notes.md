# GRPO Probe Iteration Notes

This note keeps the experimental branch results together without promoting them
into the main submission story too early.

## Branch

- `codex-strict-submit-contract-probe`

## Probe Comparison

| Probe | Job | Main change | reward_std | ask_prior | submit | strict | done | Read |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v5 strict-submit | [69ec79d0d2c8bd8662bcd53a](https://huggingface.co/jobs/prodigyhuh/69ec79d0d2c8bd8662bcd53a) | stronger strict-submit prompt/reward | `0.00` | `0.00` | `0.50` | `0.00` | `0.00` | over-corrected and collapsed the rollout |
| v6 soft-strict | [69ec7d0fd2c8bd8662bcd59a](https://huggingface.co/jobs/prodigyhuh/69ec7d0fd2c8bd8662bcd59a) | softer reward, no literal `<think>` mention | `0.3490` | `0.25` | `0.00` | `0.00` | `0.25` | healthier, but still not finishing |
| v7 no-think | [69ec7e99d70108f37acde34b](https://huggingface.co/jobs/prodigyhuh/69ec7e99d70108f37acde34b) | official Qwen `/no_think` switch | `0.4570` | `0.25` | `0.00` | `0.00` | `0.25` | prompt-level no-think helps variance a bit, but not the final tool-call gap |

## Main Takeaway

The prompt-only fixes have probably gone as far as they can.

The remaining issue now looks more like a generation / parsing mismatch:

- the GRPO logs still show an empty `<think>...</think>` wrapper
- strict final tool calls never land
- normalized repair only lands part of the time

## Best Next Step

The next engineering fix should target the wrapper behavior directly instead of
adding more prompt pressure:

1. inspect whether the TRL/Qwen generation path is still prepending an empty
   think block even with `enable_thinking=False` and `/no_think`
2. decide whether an empty leading think wrapper should be stripped before
   repair/normalization
3. rerun the same short hard-only probe before spending credits on a longer
   continuation
