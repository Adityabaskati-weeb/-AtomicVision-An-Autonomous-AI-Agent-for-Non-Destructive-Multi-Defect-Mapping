# Hackathon Submission Checklist

This file turns the OpenEnv hackathon requirements into a concrete AtomicVision
checklist.

## Theme Fit

- Primary fit: `Theme #3.1 - World Modeling / Professional Tasks`
- Why: AtomicVision is a partially observable scientific workflow environment
  with tool use, cost-aware decision making, and verifiable outcomes.

## Minimum Requirements

| Requirement | AtomicVision status | Notes |
| --- | --- | --- |
| Use OpenEnv latest release | Implemented | `openenv.yaml` plus `openenv-core==0.2.3` |
| OpenEnv environment hosted on HF Spaces | Implemented | `prodigyhuh/atomicvision-openenv` |
| Minimal training script using HF TRL or Unsloth | Implemented | `training/train_grpo_atomicvision.py` plus notebook |
| Evidence of real training | Implemented | SFT and GRPO artifacts, evaluator, charts |
| README with links and results | Implemented | README now links Space, notebook, best adapter, evaluator, and charts |
| Mini-blog / short video / slide deck | Draft ready | See `hackathon-mini-blog-draft.md`; publish externally before submission |

## Verifier Gates Before GRPO

- `strict_tool_call_pass_rate`
- `normalized_tool_call_pass_rate`
- `first_action_valid_rate`
- `first_action_ask_prior_rate`
- `submit_action_rate`
- `done_rate`
- `tool_failure_rate`

GRPO stays blocked until held-out evaluation is healthy enough to show non-zero
success on real seeds.

## Current Honest Status

- Environment quality: strong
- Deployment quality: strong
- SFT stability: strong
- Held-out strict execution: pass
- Official normalized evaluator: implemented
- Demo story: good, but final reward-improvement claim must use held-out data

## Current Promotion Status

- Best current checkpoint:
  [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
- Stable fallback:
  [prodigyhuh/atomicvision-format-submit-merged-lora](https://huggingface.co/prodigyhuh/atomicvision-format-submit-merged-lora)
- Hard frontier quality: still the main open gap

## Held-Out Seed Policy

- SFT data generation: `1000-3999`
- GRPO prompt selection: `4000-7999`
- held-out evaluation only: `10000-10999`

## Before Final Submission

1. Run `training/evaluate_atomicvision_adapter.py` on the latest candidate with
   held-out seeds only.
2. Freeze one final adapter and one final metrics table.
3. If a new run does not beat the current best honestly, keep
   `atomicvision-medium-fidelity-boost-lora` as the final promoted model.
4. Publish the mini-blog or a short video.
5. Add the external blog/video link to `README.md`.
