# Phase 4 Reward Scoring And Metrics

## Purpose

Phase 4 implements transparent reward scoring and aggregate evaluation metrics. This phase does not implement the OpenEnv server, agent training, dashboard, or PyTorch model.

## Implemented Scope

- Final submission reward scoring.
- Identity precision, recall, and F1.
- Concentration mean absolute error, gated by identity F1 so blind submissions cannot earn concentration credit.
- Confidence calibration reward, gated by identity F1 so low-confidence blind submissions do not beat real investigation.
- False-positive penalty.
- Missed-defect penalty.
- Scan-cost penalty.
- Timeout penalty.
- Aggregate metrics across evaluated episodes.

## Reward Design Framing

Two recent reward-engineering surveys are useful framing for AtomicVision:

- [Comprehensive Overview of Reward Engineering and Shaping in Advancing Reinforcement Learning Applications](https://arxiv.org/abs/2408.10215)
- [Reward Engineering for Reinforcement Learning in Software Tasks](https://arxiv.org/abs/2601.19100)

Their shared message is simple: strong RL systems usually need more than one
reward source, more than one evaluation granularity, and explicit monitoring
for reward hacking or proxy collapse.

AtomicVision should therefore be read as a multi-objective reward stack rather
than a single scalar score with a few extras.

## AtomicVision Reward Taxonomy

### 1. Outcome Rewards

These measure whether the final scientific answer is right:

- `identity_reward`
- `concentration_reward`
- `confidence_reward`

This is the task-success layer.

### 2. Cost And Safety Penalties

These discourage wasteful or invalid behavior:

- `false_positive_penalty`
- `missed_defect_penalty`
- `scan_cost_penalty`
- `timeout_penalty`

This is the efficiency-and-safety layer.

### 3. Process / Verifier Signals

These are not part of the final submission scorer in this phase, but they are
part of the actual training and evaluation stack in later phases:

- strict tool-call pass rate
- normalized tool-call pass rate
- normalized repair rate
- first action valid rate
- first action `ask_prior` rate
- submit action rate
- post-terminal tool-call penalties

This is the interface-reliability layer.

## Granularity

The reward literature for RL in software tasks emphasizes that reward signals
can arrive at different granularities. AtomicVision now spans three of them:

- Final-submission granularity:
  one reward breakdown for the submitted defect map.
- Action / verifier granularity:
  whether a tool call is valid, canonical, cheap, and terminal-safe.
- Episode granularity:
  done rate, tool-failure rate, mean scan cost, and held-out reward.

This matters because our recent failures were not caused by poor scientific
intent. They were caused by action-serialization failures before the policy
could collect outcome reward.

## Aggregation Strategy

The current scalar episode reward is still the final-submission score below.
That scalar is useful for optimization, but it should always be read together
with the component and verifier metrics described above.

## Reward Formula

```text
final_reward =
  identity_reward
  + concentration_reward
  + confidence_reward
  + false_positive_penalty
  + missed_defect_penalty
  + scan_cost_penalty
  + timeout_penalty
```

The implementation stores penalties as negative values in the reward breakdown. This makes logs easy to read: positive fields help the agent, negative fields hurt it.

But following the survey guidance, AtomicVision should be interpreted as:

- one scalar optimization target for RL,
- plus component-wise logs for diagnosis,
- plus verifier metrics that tell us whether the scalar reward is even
  reachable under strict execution.

This is important because a model can have the right first action semantically
and still fail completely if the wire format is invalid.

## What Phase 4 Covers Well

- Multiple independent outcome components, not just binary success.
- Explicit cost penalty instead of rewarding only correctness.
- Confidence shaping tied to actual answer quality.
- Transparent reward breakdown for debugging and ablations.

## What Phase 4 Does Not Cover By Itself

Phase 4 alone does not solve:

- tool-call schema compliance,
- step-level credit assignment,
- normalized-vs-strict verifier analysis,
- reward uncertainty or preference modeling,
- adaptive weighting between competing objectives.

Those pieces belong to later training and evaluation phases, and should be
treated as part of the full reward-engineering story for the project.

## Validation Gate

Phase 4 is complete only when:

- Correct submissions score higher than wrong submissions.
- Expensive scans reduce reward.
- Missing defects reduce recall and reward.
- Overconfident wrong answers are penalized.
- Timeouts receive a timeout penalty.
- Invalid submission shapes raise clear errors.
- Aggregate metrics report mean reward and timeout rate.
- Phase 4 tests pass locally.

## Phase 5 Entry Criteria

Start Phase 5 only after reward tests pass. Phase 5 will implement the OpenEnv-compatible environment wrapper around the synthetic world and reward scorer.
