# Hugging Face Jobs Training Playbook

This is the short operational guide for future AtomicVision training on Hugging
Face Jobs.

## Why HF Jobs Is The Default

Compared with Kaggle or ad hoc Colab use, HF Jobs gives us:

- stable cloud GPU access
- direct artifact persistence strategy
- pinned commit reproducibility
- easier reruns for short probes and longer continuations

## Default Hardware Choice

Use the strongest practical GPU for speed-sensitive runs.

- short smoke checks: `a10g-small` is acceptable
- real GRPO or continuation runs: prefer `a10g-large` or `a100-large`
- if time matters most and credits allow it: bias toward `a100-large`

## Required Secrets

- `HF_TOKEN`

The token should have:

- model read/write access
- `job.write`

## Seed Policy

Keep the official split:

- SFT data generation: `1000-3999`
- GRPO prompt selection: `4000-7999`
- held-out evaluation only: `10000-10999`

Do not promote a model using overlapping eval seeds.

## First Training Gate

Before a long GRPO run, always run a short probe first.

Probe must show:

- nonzero `reward_std`
- `frac_reward_zero_std < 1`
- nonzero submit behavior
- healthy strict execution
- no post-terminal tool-call drift

If the probe fails those checks, fix verifier/reward plumbing or prompt
selection before spending more GPU time.

## Current Preferred Base

Use this checkpoint as the current starting point for new hard-focused work:

- [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)

Use this only as a fallback:

- [prodigyhuh/atomicvision-format-submit-merged-lora](https://huggingface.co/prodigyhuh/atomicvision-format-submit-merged-lora)

## Persistence Rule

Short probes may run without push-to-hub if the only question is whether the
method works.

Longer runs should persist artifacts by either:

- pushing to an existing model repo, or
- using a repo that is created ahead of time

Do not assume a job token can create new repos unless that has already been
verified.
