# AtomicVision Judge Writeup

AtomicVision is an OpenEnv environment for non-destructive multi-defect mapping.
The agent receives compact spectral evidence, chooses low-cost characterization
actions, and submits a final defect map under scan-budget pressure.

## Deliverables

- Hugging Face Space: [prodigyhuh/atomicvision-openenv](https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv)
- Judge reproducible notebook: [notebooks/AtomicVision_Judge_Repro_Colab.ipynb](../notebooks/AtomicVision_Judge_Repro_Colab.ipynb)
- NaN-safe SFT trainer: [training/train_sft_atomicvision_safe.py](../training/train_sft_atomicvision_safe.py)
- Held-out evaluator: [training/evaluate_atomicvision_adapter.py](../training/evaluate_atomicvision_adapter.py)
- Adapter publisher: [training/publish_adapter_to_hub.py](../training/publish_adapter_to_hub.py)
- Current best adapter: [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
- Stable fallback adapter: [prodigyhuh/atomicvision-format-submit-merged-lora](https://huggingface.co/prodigyhuh/atomicvision-format-submit-merged-lora)

## Why this project fits the hackathon

AtomicVision is not a static benchmark. It is a world-modeling task with:

- step-by-step tool use,
- verifiable outcomes,
- explicit cost tradeoffs,
- and a judge-visible deployment path.

The OpenEnv environment inherits the standard environment base class, exposes
real `reset` and `step` methods, and scores the final submission with layered
reward components for identity, concentration, confidence, and penalties.

## What we learned

The hardest failure mode was not scientific reasoning. It was interface
reliability. The model often chose the right action semantically, but failed to
serialize exact tool calls on held-out seeds.

That led to the current recovery path:

1. NaN-safe SFT validation,
2. schema-aware two-step curriculum,
3. strict plus normalized held-out evaluation,
4. only then GRPO continuation.

## Current Best Result

The current best published checkpoint is the medium-fidelity boost adapter:

- model repo: [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
- training style: small continuation SFT from the saved reliable adapter
- goal: improve outcome quality without disturbing perfect tool-call execution

Held-out result summary:

| Difficulty | Baseline reward | Boost reward | Strict pass | Done rate |
| --- | ---: | ---: | ---: | ---: |
| medium | 4.5115 | 4.5707 | 1.00 | 1.00 |
| hard | 4.8883 | 4.6466 | 1.00 | 1.00 |

That is the important shape of progress for this project: the interface layer
is now reliable, the medium slice improved, and the remaining gap is hard-case
quality rather than formatting.

## Held-Out Seed Policy

AtomicVision now uses a permanent seed split:

- SFT data generation: `1000-3999`
- GRPO prompt selection: `4000-7999`
- held-out evaluation only: `10000-10999`

This keeps promotion claims tied to unseen evaluation seeds rather than
overlapping rebuild data.

## Public Data Strategy

Public materials datasets are useful for AtomicVision, but not as raw policy
training rows.

They are used upstream for:

- prior improvement,
- reference retrieval,
- and simulator calibration.

Policy SFT and GRPO stay AtomicVision-native: chat messages, tool calls, and
environment-grounded rewards.

## Evidence

The repository now includes validator-facing training evidence as committed PNG
artifacts:

- [training-loss-curve.png](training-loss-curve.png)
- [training-reward-curve.png](training-reward-curve.png)

Those plots are also embedded in the main README so the automated pass can
reach them directly from the project root.

The repository also includes the latest Hugging Face Jobs GRPO probe summary as
committed evidence rather than leaving it only in cloud logs:

- [hard-only-grpo-reference-probe-results.md](hard-only-grpo-reference-probe-results.md)
- [hard-only-grpo-reference-probe-metrics.json](hard-only-grpo-reference-probe-metrics.json)
