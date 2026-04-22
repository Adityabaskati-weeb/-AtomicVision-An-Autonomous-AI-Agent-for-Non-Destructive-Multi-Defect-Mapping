---
title: AtomicVision OpenEnv
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: OpenEnv lab for non-destructive atomic defect mapping.
---

# AtomicVision

**AtomicVision: An Autonomous AI Agent for Non-Destructive Multi-Defect Mapping**

AtomicVision is a hackathon-focused OpenEnv project for AI-assisted materials
characterization. It frames atomic defect mapping as a partially observable
scientific lab environment: an agent receives non-invasive vibrational spectra,
chooses characterization actions, and submits a defect map while balancing
accuracy against scan cost.

The project is moving phase by phase. Each stage is implemented only after the
previous gate has been validated.

## Current Phase

- Phase 0: Scope Lock
- Phase 1: System Design
- Phase 2: Environment Contract
- Phase 3: Synthetic Materials World
- Phase 4: Reward Scoring And Metrics
- Phase 5: OpenEnv Wrapper
- Phase 6: Baselines And Evaluation
- Phase 7: DefectNet-Lite
- Phase 8: Model Prior Training
- Phase 9: Hugging Face Space Deployment
- Phase 10: Reward Comparison And Colab Bridge
- Phase 11: GRPO Fine-Tuning Scaffold
- Phase 12: SFT Copy LoRA Rollout
- Phase 13: Format-Aware GRPO Continuation
- Status: Phase 13 scaffold ready; SFT-copy adapter remains the best checkpoint
- Scope document: [docs/phase-0-scope-lock.md](docs/phase-0-scope-lock.md)
- System design: [docs/phase-1-system-design.md](docs/phase-1-system-design.md)
- Environment contract: [docs/phase-2-environment-contract.md](docs/phase-2-environment-contract.md)
- Synthetic world: [docs/phase-3-synthetic-world.md](docs/phase-3-synthetic-world.md)
- Rewards and metrics: [docs/phase-4-rewards-and-metrics.md](docs/phase-4-rewards-and-metrics.md)
- OpenEnv wrapper: [docs/phase-5-openenv-wrapper.md](docs/phase-5-openenv-wrapper.md)
- Baselines and evaluation: [docs/phase-6-baselines-and-evaluation.md](docs/phase-6-baselines-and-evaluation.md)
- DefectNet-lite: [docs/phase-7-defectnet-lite.md](docs/phase-7-defectnet-lite.md)
- Model prior training: [docs/phase-8-model-prior-training.md](docs/phase-8-model-prior-training.md)
- Hugging Face deployment: [docs/phase-9-huggingface-deployment.md](docs/phase-9-huggingface-deployment.md)
- Phase 10 notes: [docs/phase-10-reward-comparison-and-colab.md](docs/phase-10-reward-comparison-and-colab.md)
- Phase 11 notes: [docs/phase-11-finetuning-plan.md](docs/phase-11-finetuning-plan.md)
- Training runbook: [docs/training-runtime-runbook.md](docs/training-runtime-runbook.md)
- Reward comparison: [docs/reward-comparison-report.md](docs/reward-comparison-report.md)
- SFT-copy rollout result: [docs/sft-copy-lora-results.md](docs/sft-copy-lora-results.md)
- GRPO continuation smoke result: [docs/grpo-continuation-smoke-results.md](docs/grpo-continuation-smoke-results.md)
- Colab bridge: [notebooks/AtomicVision_GRPO_Colab.ipynb](notebooks/AtomicVision_GRPO_Colab.ipynb)

## Latest Result

Kaggle SFT-copy training produced a Qwen3-1.7B LoRA adapter that fixes the
main direct-rollout weakness from the GRPO-only run: exact copying of
DefectNet-lite prior tool outputs into the terminal `submit_defect_map` call.

Model: [`prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora`](https://huggingface.co/prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora)

| Evaluation | Episodes | Reward | F1 | MAE | Steps | Scan cost | Tool failures | Done rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO-only direct rollout | 32 | 2.625 | 0.599 | 0.0783 | 2.03 | 1.55 | 0.00 | 1.00 |
| SFT-copy direct rollout | 32 | 4.458 | 0.790 | 0.0321 | 2.06 | 1.55 | 0.00 | 1.00 |
| Prior-submit baseline | 32 | 4.366 | 0.773 | 0.0318 | 2.00 | 1.50 | 0.00 | 1.00 |

The SFT-copy adapter slightly outperforms the deterministic prior-submit
baseline while preserving 0% malformed tool calls and 100% episode completion.

The first 20-step GRPO continuation from this adapter completed successfully on
Kaggle, but it was not promoted: with the required tool-system prompt it matched
the prior-submit baseline (`4.366` reward, `0.773` F1) and remained below the
SFT-copy adapter. A follow-up format-aware smoke also produced valid tool calls
but logged `reward_std=0`, `frac_reward_zero_std=1`, `loss=0`, and
`grad_norm=0`, confirming that grouped rollouts had no relative reward signal.
The GRPO scaffold now keeps the tool-system prompt by default, exposes sampling
knobs, weakens exact-copy shaping, and adds a variance-probe preset before the
next promotion attempt.
