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
| README with links and results | In progress | README now links Space, notebook, and runbook; keep improving story |
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
- Held-out strict execution: still needs work
- Official normalized evaluator: implemented
- Demo story: good, but final reward-improvement claim must use held-out data

## Before Final Submission

1. Run `training/evaluate_atomicvision_adapter.py` on the latest adapter.
2. If strict held-out still fails, use normalized results only as diagnosis, not
   as the main claim.
3. Publish the mini-blog or a short video.
4. Add the external blog/video link to `README.md`.
5. Freeze one final adapter and one final metrics table.
