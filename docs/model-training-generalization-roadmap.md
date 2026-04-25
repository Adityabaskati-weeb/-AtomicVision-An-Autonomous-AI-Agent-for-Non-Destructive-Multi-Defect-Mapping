# Model Training And Generalization Roadmap

This document turns the current AtomicVision training lessons into an
implementation roadmap. The goal is not just to improve benchmark numbers once,
but to improve the model in a way that remains reproducible, explainable, and
honest on unseen seeds.

## Current Starting Point

AtomicVision already has three important assets:

- Best current checkpoint:
  [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
- Stable fallback checkpoint:
  [prodigyhuh/atomicvision-format-submit-merged-lora](https://huggingface.co/prodigyhuh/atomicvision-format-submit-merged-lora)
- Public OpenEnv deployment:
  [prodigyhuh/atomicvision-openenv](https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv)

What is already working:

- strict and normalized tool-call reliability
- NaN-safe SFT recovery path
- public judge notebook and reproducible rebuild flow
- medium-slice improvement over the earlier stable adapter

What is not solved yet:

- hard-slice quality still trails the baseline
- rebuild training and evaluation are not fully separated by seed
- public materials data has not yet been integrated into the prior/reference
  stack
- GRPO is not yet promoted as the main improvement path

## Design Rules

Use these rules for every future training decision:

1. Train the policy only on AtomicVision-native episodes.
2. Use public materials datasets to improve the prior, reference layer, and
   simulator realism before feeding anything into policy training.
3. Promote a model only on held-out environment evaluation, not training loss.
4. Keep a stable fallback adapter alive at all times.
5. Do not run a larger RL experiment until verifier and reward plumbing are
   confirmed healthy on a short probe.

## Phase 1: Lock The Evaluation Split

### Objective

Create a permanent seed split so future claims about generalization are honest.

### Why this matters

The rebuild path currently trains and evaluates on overlapping seed ranges. That
makes early progress look cleaner than it really is.

Relevant files:

- [evaluate_atomicvision_adapter.py](</C:/Users/baska/OneDrive/Documents/New project/AtomicVision/training/evaluate_atomicvision_adapter.py:36>)
- [training-runtime-runbook.md](</C:/Users/baska/OneDrive/Documents/New project/AtomicVision/docs/training-runtime-runbook.md:330>)
- [training-runtime-runbook.md](</C:/Users/baska/OneDrive/Documents/New project/AtomicVision/docs/training-runtime-runbook.md:419>)

### Implementation

- Reserve `1000-3999` for SFT data generation only.
- Reserve `4000-7999` for GRPO prompt selection only.
- Reserve `10000-10999` for held-out evaluation only.
- Update the runbook, notebook, and any training presets to reflect this split.
- Refuse promotion when a run evaluated on overlapping seeds.

### Acceptance gate

- Every notebook and runbook path uses the same split.
- Held-out eval JSON reports an eval-only seed range.
- README and judge writeup describe the held-out split clearly.

## Phase 2: Separate Policy Data From World-Model Data

### Objective

Prevent raw external datasets from poisoning the tool-use policy.

### Why this matters

AtomicVision policy training expects exact chat messages and exact XML-wrapped
tool calls, not raw spectra tables or materials-property rows.

Relevant files:

- [generate_atomicvision_sft_data.py](</C:/Users/baska/OneDrive/Documents/New project/AtomicVision/training/generate_atomicvision_sft_data.py:209>)
- [train_sft_atomicvision_safe.py](</C:/Users/baska/OneDrive/Documents/New project/AtomicVision/training/train_sft_atomicvision_safe.py:99>)
- [train_grpo_atomicvision.py](</C:/Users/baska/OneDrive/Documents/New project/AtomicVision/training/train_grpo_atomicvision.py:65>)

### Implementation

- Add a small data architecture note that defines three data buckets:
  - policy training episodes
  - prior/reference corpora
  - simulator calibration data
- Keep public datasets out of policy SFT unless first converted into valid
  AtomicVision episodes.
- Add validation checks that reject rows that do not match the current
  `messages -> final tool_call` format.

### Acceptance gate

- A new contributor can tell which datasets are safe for policy SFT and which
  are not.
- No raw public dataset path is wired directly into SFT or GRPO scripts.

## Phase 3: Improve The Prior And Reference Stack

### Objective

Use public data where it helps most: upstream of the policy.

### Candidate sources

- JARVIS for vacancy, DOS, and related first-principles materials data:
  [JARVIS docs](https://jarvis-tools.readthedocs.io/en/master/databases.html)
- Computational Raman Database for Raman references:
  [CRD](https://ramandb.oulu.fi/)
- Foundry semiconductor defect levels for defect-prior supervision:
  [Foundry defect levels](https://huggingface.co/datasets/foundry-ml/semiconductor_defectlevels_v1-1)

### Implementation

- Prototype a new offline prior-building path that consumes curated external
  data and produces AtomicVision-native priors.
- Improve `ask_prior` confidence calibration before changing the policy.
- Add a small reference lookup layer for Raman-like nearest neighbors or
  template support.
- Record provenance so we know which external source affected which prior or
  reference suggestion.

### Acceptance gate

- Prior accuracy improves on held-out eval seeds before policy retraining.
- Confidence values become better calibrated, not just larger.
- Reference-backed priors help medium and hard without changing tool-call
  reliability.

## Phase 4: Regenerate Better Synthetic Episodes

### Objective

Create stronger training examples from a better world model instead of stuffing
raw public data into the policy trainer.

### Implementation

- Use the improved prior/reference stack to regenerate:
  - two-step curriculum episodes
  - medium prior-fidelity episodes
  - hard frontier episodes
- Track sample mix explicitly:
  - `ask_prior`
  - `submit_prior`
  - `submit_after_reference`
- Increase hard examples that truly differ from the baseline prior.

### Acceptance gate

- New datasets still pass assistant-mask and final-tool validation.
- Hard frontier generation yields more informative, non-duplicative examples.
- Data generation reports clearly show where the new signal came from.

## Phase 5: Harden Reward And Verifier Plumbing

### Objective

Make GRPO safe enough to trust before using it for larger hard-only runs.

### Why this matters

GRPO only helps if reward variance is real and metrics reflect actual multi-turn
behavior.

### Implementation

- Keep format rewards, but never let them dominate semantic reward.
- Preserve multi-turn parsing correctness for `ask_prior -> submit_defect_map`
  trajectories.
- Log these metrics in every probe:
  - `reward_std`
  - `frac_reward_zero_std`
  - `submit_tool_rate`
  - `strict_tool_call_pass_rate`
  - `done_rate`
  - `post_terminal_tool_calls_mean`
- Add a small regression test around any reward parser or verifier fix.

### Acceptance gate

- Probe runs show nonzero reward variance.
- `submit_tool_rate` is nonzero on the intended prompt focus.
- Reward metrics match the actual sampled transcript behavior.

## Phase 6: Run Small Hard-Focused GRPO Probes

### Objective

Use RL only where SFT stopped helping: the hard slice.

### Why this matters

Medium improved under targeted SFT continuation, but hard did not. That is the
signal that frontier decision quality, not formatting, is the remaining gap.

### Implementation

- Start from the current best adapter:
  [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
- Run short hard-only probes on held-out prompt seeds.
- Keep the run small enough to inspect manually before scaling.
- Stop immediately if probes collapse into zero-variance or pure `ask_prior`
  behavior.

### Acceptance gate

- Probe is healthy on reward variance and submit behavior.
- Medium does not collapse while targeting hard.
- At least one probe justifies a longer continuation run.

## Phase 7: Promote Only On Honest Eval

### Objective

Avoid replacing a good stable model with a flashy but less reliable one.

### Implementation

- Evaluate every candidate on the eval-only seed band.
- Compare against:
  - baseline prior-submit policy
  - current best promoted adapter
  - stable fallback adapter
- Promote only if all of the following hold:
  - strict execution remains healthy
  - normalized execution remains healthy
  - hard reward improves materially
  - medium does not regress beyond an agreed tolerance

### Acceptance gate

- Promotion decision is documented in the repo.
- The promoted adapter is pushed to the Hub with eval artifacts.
- README and judge-facing docs point to the new best model only after promotion.

## Phase 8: Move The Runtime To Hugging Face Jobs

### Objective

Run future training on a more stable and reusable platform than Kaggle.

### Implementation

- Keep Hugging Face token scopes validated, including `job.write`.
- Use pinned repo commits for job submission.
- Save every adapter directly to the Hub from the training job.
- Save eval JSON and training reports alongside the adapter.

### Acceptance gate

- A failed cloud run never destroys the last good checkpoint.
- A rerun from the same commit and config is straightforward.
- Cloud logs are sufficient to debug reward collapse or verifier issues.

## Recommended Immediate Sequence

If we want the shortest path to real progress, do this next:

1. Lock the permanent seed split in code, runbook, and notebook.
2. Add a small architecture note for policy data vs prior/reference data.
3. Prototype prior/reference improvement from one external source, starting with
   JARVIS or CRD.
4. Regenerate hard frontier episodes from the improved upstream stack.
5. Run short hard-only GRPO probes from the current best adapter on Hugging Face
   Jobs.
6. Promote only after eval-only seeds confirm a real hard-slice gain.

## Stop Conditions

Pause and reassess if any of these happen:

- held-out seed discipline is broken again
- format metrics improve while semantic reward falls
- hard-only tuning regresses medium sharply
- external data improves priors offline but does not improve generated episodes
- GRPO probes return zero reward variance repeatedly

## Success Definition

This roadmap is successful when AtomicVision has:

- a clean held-out evaluation split
- a documented separation between policy data and public materials data
- a stronger prior/reference stack
- harder synthetic episodes with real frontier signal
- a hard-slice improvement that survives honest held-out evaluation
- a reproducible Hugging Face Jobs training path with preserved artifacts
