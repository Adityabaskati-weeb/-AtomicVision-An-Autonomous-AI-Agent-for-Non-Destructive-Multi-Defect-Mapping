# AtomicVision: Training An LLM To Do Cost-Aware Defect Mapping In OpenEnv

AtomicVision is our OpenEnv environment for non-destructive atomic defect
mapping. The agent sees compact spectral evidence, chooses characterization
tools, and submits a final defect map while paying scan costs along the way.

## Problem

Large language models are usually tested on static prompts, but real scientific
workflows are interactive. We wanted an environment where an agent must:

- reason step by step,
- choose whether to spend budget on more evidence,
- update its belief after tool outputs,
- and end with a verifiable final answer.

That makes AtomicVision a strong fit for OpenEnv and for RL with verifiable
rewards.

## Environment

AtomicVision is a partially observable lab environment with:

- `ask_prior`
- `compare_reference`
- `request_scan`
- `zoom_band`
- `submit_defect_map`

Each episode starts with compact spectral summaries. The agent must infer hidden
defects without destructive access to the underlying material.

## Reward Design

The reward combines:

- final defect-map quality,
- concentration accuracy,
- scan-cost pressure,
- terminal-action discipline,
- and tool-call format checks.

We also added explicit verifier columns for:

- strict tool-call pass rate,
- normalized tool-call pass rate,
- first-action validity,
- first-action `ask_prior` rate,
- submit-action rate,
- done rate,
- and tool-failure rate.

## Training Stack

Our stack is:

`OpenEnv environment -> verifier / reward logic -> TRL trainer -> LoRA adapters -> Hugging Face Space deployment`

We started with supervised warm-starting because the task is verifiable but the
policy needs formatting and action scaffolding before RL makes sense.

## What We Learned

The biggest failure mode was not raw scientific reasoning. It was interface
reliability: the model often had the right intent but emitted malformed tool
calls on held-out episodes.

To address that, we added:

- a NaN-safe SFT trainer,
- held-out evaluator scripts,
- strict and normalized verifier modes,
- and a two-step curriculum for first-action plus submit-schema repair.

## Why It Matters

AtomicVision is interesting because it is not a toy game or a single-step
classification prompt. It is an environment where a model must balance evidence
quality, cost, and final accuracy in a realistic scientific workflow.

That is exactly the kind of setting where OpenEnv-style training can push LLMs
beyond shallow text completion and toward reliable action.
