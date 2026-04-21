# Phase 2 Environment Contract

## Purpose

Phase 2 defines the exact OpenEnv environment contract for AtomicVision before implementation. This document locks the observation shape, action semantics, reward components, episode termination rules, and validation expectations.

No environment or model code is implemented in this phase.

## Environment Name

`atomicvision_env`

## Environment Type

Single-agent, partially observable, scientific characterization environment.

The hidden world state is an unknown material sample with simulated substitutional point defects. The agent sees only non-invasive spectral observations and metadata, then chooses lab actions until it submits a final defect map or runs out of budget.

## Episode Objective

Infer the hidden defect species and concentrations from indirect vibrational evidence while using as few expensive scan actions as possible.

## Observation Contract

The observation should be serializable as a typed Pydantic/OpenEnv model.

| Field | Type Intent | Description |
| --- | --- | --- |
| `episode_id` | string | Stable id for the current episode. |
| `material_id` | string | Synthetic material identifier, safe to expose. |
| `difficulty` | string | One of `easy`, `medium`, `hard`, `expert`. |
| `host_family` | string | Human-readable material family label. |
| `frequency_axis` | list of floats | Shared x-axis for spectrum values. |
| `current_spectrum` | list of floats | Latest observed PDoS-like signal. |
| `pristine_reference` | list of floats or null | Reference spectrum if available or requested. |
| `scan_history` | list of records | Prior scan actions and costs. |
| `candidate_defects` | list of strings | Candidate species shown to the agent. |
| `prior_prediction` | record or null | DefectNet-lite/heuristic prior when requested. |
| `budget_remaining` | float | Remaining scan budget. |
| `step_count` | integer | Current episode step count. |
| `max_steps` | integer | Episode step limit. |
| `last_reward` | float | Reward from the previous step. |
| `reward_breakdown` | record or null | Latest component-level reward explanation. |
| `done` | boolean | Whether the episode has ended. |
| `message` | string | Short human-readable environment response. |

Server-only hidden state:

- True defect species.
- True defect concentrations.
- Full pristine spectrum.
- Full high-resolution defective spectrum.
- Noise profile.
- Random seed.
- Reward history.

## Action Contract

The action model should use a clear `action_type` plus optional typed payload fields.

| Action | Required Payload | Effect |
| --- | --- | --- |
| `request_scan` | `scan_mode`, `resolution` | Returns a new full-spectrum observation with cost based on scan type and resolution. |
| `zoom_band` | `freq_min`, `freq_max`, `resolution` | Returns higher-detail spectrum values in a selected frequency band. |
| `compare_reference` | none | Reveals or refreshes pristine reference comparison signal at a small cost. |
| `ask_prior` | none | Requests DefectNet-lite or heuristic prior prediction at a cost. |
| `submit_defect_map` | `predicted_defects`, `predicted_concentrations`, `confidence` | Ends episode and scores final answer. |

## Action Validation Rules

- Unknown `action_type` returns a negative reward and keeps the episode active unless max steps are reached.
- `freq_min` must be less than `freq_max`.
- Frequency bounds must overlap the environment frequency axis.
- `resolution` must be one of `low`, `medium`, or `high`.
- `predicted_defects` and `predicted_concentrations` must have the same length.
- Submitted concentrations must be non-negative.
- Confidence must be between `0.0` and `1.0`.
- Submit action may include zero predicted defects if the agent believes the sample is pristine.

## Scan Modes

| Scan Mode | Cost | Signal Quality | Purpose |
| --- | ---: | --- | --- |
| `quick_pdos` | 1.0 | Noisy, low resolution | Cheap initial evidence. |
| `standard_pdos` | 2.0 | Moderate noise/resolution | Main workhorse scan. |
| `high_res_pdos` | 4.0 | Lower noise/high resolution | Expensive confirmation. |
| `raman_proxy` | 2.5 | Alternate signal transform | Stretch mode for multimodal storytelling. |

MVP may implement only PDoS modes first. `raman_proxy` is a stretch action unless time permits.

## Reward Contract

Final submission reward should be decomposed into transparent components:

```text
final_reward =
  identity_reward
  + concentration_reward
  + confidence_reward
  - false_positive_penalty
  - missed_defect_penalty
  - scan_cost_penalty
  - timeout_penalty
```

Recommended first-pass component ranges:

| Component | Range Intent | Description |
| --- | ---: | --- |
| `identity_reward` | 0 to 4 | Multi-label species match reward. |
| `concentration_reward` | 0 to 3 | Higher reward for lower concentration MAE. |
| `confidence_reward` | -1 to 1 | Rewards calibrated confidence, penalizes overconfidence. |
| `false_positive_penalty` | 0 to -2 | Penalizes extra species. |
| `missed_defect_penalty` | 0 to -2 | Penalizes missing true species. |
| `scan_cost_penalty` | 0 to -4 | Penalizes expensive evidence gathering. |
| `timeout_penalty` | 0 or -2 | Penalizes failing to submit. |

Intermediate action rewards:

- Valid information-gathering action: negative cost feedback scaled to discourage unnecessary scans.
- Invalid action: clear negative penalty.
- Repeated wasteful action: increasing penalty after the first repeat.
- `ask_prior`: cost penalty, no direct accuracy reward until submit.

## Termination Rules

Episode ends when:

- Agent calls `submit_defect_map`.
- `step_count >= max_steps`.
- `budget_remaining <= 0`.
- Environment catches an unrecoverable validation error.

Timeout scoring:

- If the agent never submits, score as missed submission with timeout penalty.
- Preserve final observation and reward breakdown for demo clarity.

## Difficulty Contract

| Difficulty | Defect Count | Concentration Range | Noise | Budget | Max Steps |
| --- | ---: | --- | --- | ---: | ---: |
| `easy` | 1 | 8% to 25% | Low | 10 | 5 |
| `medium` | 2-3 | 2% to 20% | Medium | 9 | 6 |
| `hard` | 4-6 | 0.5% to 15% | High | 8 | 7 |
| `expert` | 4-6 plus distractors | 0.2% to 12% | High | 6 | 7 |

The MVP should fully support `easy`, `medium`, and `hard`. `expert` can be enabled after training and demo stability.

## Candidate Defect Set

MVP should use a compact candidate set so training is feasible and demo output is readable.

Recommended candidate species:

- `B`
- `C`
- `N`
- `O`
- `Al`
- `Si`
- `P`
- `S`
- `Ge`
- `Ga`
- `Mg`
- `Zn`

This set is intentionally smaller than research-scale DefectNet. The README and pitch must state that this is a hackathon-scale simulation.

## Metrics Contract

Every evaluation run should report:

- Mean episode reward.
- Defect identity precision.
- Defect identity recall.
- Defect identity F1.
- Concentration MAE.
- Mean scan cost.
- Submit rate.
- Invalid action rate.

## Baseline Policies

| Policy | Expected Role |
| --- | --- |
| Random policy | Lower bound. Should perform poorly. |
| Cheap submit policy | Tests whether immediate guessing is punished. |
| Scan-heavy policy | Tests whether reward penalizes unnecessary scans. |
| Heuristic prior policy | Strong non-trained baseline. |
| Trained GRPO policy | Target improvement over baselines. |

## Phase 2 Validation Checklist

Before implementation begins, confirm:

- Observation fields are complete.
- Action set has meaningful choices.
- Invalid actions are specified.
- Reward components are transparent.
- Difficulty settings are feasible.
- Metrics are enough to prove reward improvement.
- Scope remains simulation-backed and honest.
- No implementation code has been added.

## Phase 2 Completion Gate

Phase 2 is complete only when:

- Observation contract is documented.
- Action contract is documented.
- Reward contract is documented.
- Termination rules are documented.
- Difficulty contract is documented.
- Metrics contract is documented.
- Baseline policies are documented.
- Validation checklist passes.
- Repo validation passes for Phase 2 documentation.

## Phase 3 Entry Criteria

Start Phase 3 only after this environment contract is accepted. Phase 3 will create the synthetic materials world and its tests, making it the first implementation phase.
