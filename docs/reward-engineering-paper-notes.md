# Reward Engineering Paper Notes For AtomicVision

## Papers

- [Comprehensive Overview of Reward Engineering and Shaping in Advancing Reinforcement Learning Applications](https://arxiv.org/abs/2408.10215)
- [Reward Engineering for Reinforcement Learning in Software Tasks](https://arxiv.org/abs/2601.19100)

## Why These Matter

AtomicVision is not a plain single-shot classifier. It is a tool-using,
partially observable, cost-sensitive agent. That makes reward design just as
important as model choice.

Both papers point in the same direction:

- do not rely on one scalar proxy alone,
- separate outcome reward from process feedback,
- monitor reward components independently,
- expect proxy mismatch and reward hacking,
- and use granularity-aware feedback when the task has multiple steps.

## Key Takeaways

### 1. Reward engineering is broader than reward shaping

The 2024 survey distinguishes reward engineering from reward shaping. Reward
engineering is the broader design problem: defining the reward sources,
penalties, and constraints. Reward shaping is one technique inside that space.

AtomicVision implication:

- final submission scoring is only one part of reward engineering,
- tool-call validity and terminal behavior also belong in the reward design.

### 2. Multi-objective rewards are normal in complex tasks

The 2024 survey highlights the tension between scalar and vector reward views.
The 2026 software-task survey makes this concrete: real systems often combine
execution feedback, similarity proxies, coverage-like signals, and preference
signals.

AtomicVision implication:

- treat the project as multi-objective,
- keep the scalar reward for optimization,
- but always log the components separately.

### 3. Reward source matters

The 2026 survey organizes reward design partly by reward source. In software,
common sources are execution-based, reference-based, and preference/model-based.

AtomicVision mapping:

- execution-like:
  valid tool calls, terminal-safe behavior, successful completion
- reference-like:
  prior-copy behavior, comparison-against-reference decisions
- outcome-based:
  final identity and concentration accuracy
- cost-based:
  scan cost and timeout penalties

### 4. Reward granularity matters

The 2026 survey also emphasizes reward granularity: token, line, function,
program, or trajectory. The practical lesson is that sparse end-of-episode
reward is often not enough.

AtomicVision implication:

- final defect-map reward is necessary but insufficient,
- action-level verifier signals are needed because the current bottleneck is
  step-1 and step-2 schema failure.

### 5. Proxy mismatch is unavoidable, so monitor the proxies

Both papers warn about deceptive rewards, reward hacking, and reward designs
that optimize the wrong thing.

AtomicVision implication:

- a rising training reward is not enough,
- we must inspect strict tool-call pass rate, normalized pass rate,
  first-action validity, submit-action rate, and done rate.

### 6. RL should start only after nonzero success is reachable

This is consistent with the hackathon guide and with the papers' discussion of
sparse and delayed rewards. If the agent cannot reliably enter a rewarding
state, RL mostly burns compute.

AtomicVision implication:

- GRPO remains blocked until held-out execution succeeds often enough to give
  the optimizer real signal.

## What AtomicVision Already Does Well

- Uses multiple outcome components instead of a single binary reward.
- Penalizes scan cost explicitly.
- Penalizes missed defects and false positives separately.
- Tracks confidence calibration instead of ignoring confidence entirely.
- Separates strict and normalized evaluation paths locally.
- Logs verifier metrics that expose interface collapse.

## What Was Missing Before

These papers make the missing pieces easier to name:

- the docs did not clearly separate reward sources,
- the docs under-explained action-level versus episode-level feedback,
- the project story leaned too hard on final scalar reward,
- strict-vs-normalized verifier behavior was not framed as part of reward
  engineering,
- the held-out gate did not read like a multi-objective reward diagnosis.

## Current AtomicVision Risk

The main current failure is not that the model lacks scientific intent.
It is that the model often fails to serialize tool actions into a strict,
machine-executable form. That means:

- outcome reward exists,
- but the agent cannot consistently reach it under strict execution.

From a reward-engineering perspective, this is a process-feedback bottleneck.

## Concrete Policy For Our Next Runs

1. Treat strict tool-call success as a first-class gate.
2. Treat normalized evaluation as a diagnosis tool, not a promotion shortcut.
3. Keep logging component rewards and verifier columns together.
4. Do not run GRPO while strict held-out success is near zero.
5. When comparing adapters, compare:
   - total reward
   - done rate
   - tool-failure rate
   - strict parse rate
   - normalized parse rate
   - first-action `ask_prior` rate
   - submit-action rate
   - scan cost

## Short Bottom Line

These papers support the direction we have already been moving toward:

- multi-component rewards,
- explicit cost penalties,
- held-out gates before RL,
- and process-aware verifier metrics.

They also sharpen the diagnosis:

AtomicVision's blocker is currently not "the reward is too simple."
It is "the policy still fails to reliably convert good intent into valid
tool-executable actions under strict evaluation."
