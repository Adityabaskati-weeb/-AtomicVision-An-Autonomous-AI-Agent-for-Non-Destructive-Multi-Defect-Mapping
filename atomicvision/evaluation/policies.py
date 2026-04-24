"""Baseline policies and evaluation helpers for AtomicVision."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from atomicvision_env.models import AtomicVisionAction, AtomicVisionObservation
from atomicvision_env.server.environment import AtomicVisionEnvironment


POLICY_NAMES: tuple[str, ...] = (
    "cheap_submit",
    "random",
    "scan_heavy",
    "prior_submit",
    "oracle",
)


@dataclass(frozen=True)
class PolicyEpisodeResult:
    """Metrics from one evaluated episode."""

    policy_name: str
    seed: int
    difficulty: str
    total_reward: float
    steps: int
    total_scan_cost: float
    done: bool
    f1: float
    concentration_mae: float
    identity_reward: float
    concentration_reward: float
    confidence_reward: float
    outcome_reward_total: float
    penalty_total: float
    timeout: bool


@dataclass(frozen=True)
class PolicyEvaluationSummary:
    """Aggregate baseline metrics for one policy."""

    policy_name: str
    difficulty: str
    episodes: int
    mean_reward: float
    mean_steps: float
    mean_scan_cost: float
    mean_f1: float
    mean_concentration_mae: float
    mean_identity_reward: float
    mean_concentration_reward: float
    mean_confidence_reward: float
    mean_outcome_reward_total: float
    mean_penalty_total: float
    timeout_rate: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def evaluate_policy(
    policy_name: str,
    seeds: list[int] | tuple[int, ...],
    difficulty: str = "medium",
) -> PolicyEvaluationSummary:
    """Evaluate a named baseline policy on fixed seeds."""

    if not seeds:
        raise ValueError("at least one seed is required")
    results = [run_policy_episode(policy_name, seed, difficulty) for seed in seeds]
    return PolicyEvaluationSummary(
        policy_name=policy_name,
        difficulty=difficulty,
        episodes=len(results),
        mean_reward=round(_mean(result.total_reward for result in results), 6),
        mean_steps=round(_mean(result.steps for result in results), 6),
        mean_scan_cost=round(_mean(result.total_scan_cost for result in results), 6),
        mean_f1=round(_mean(result.f1 for result in results), 6),
        mean_concentration_mae=round(
            _mean(result.concentration_mae for result in results),
            6,
        ),
        mean_identity_reward=round(
            _mean(result.identity_reward for result in results),
            6,
        ),
        mean_concentration_reward=round(
            _mean(result.concentration_reward for result in results),
            6,
        ),
        mean_confidence_reward=round(
            _mean(result.confidence_reward for result in results),
            6,
        ),
        mean_outcome_reward_total=round(
            _mean(result.outcome_reward_total for result in results),
            6,
        ),
        mean_penalty_total=round(
            _mean(result.penalty_total for result in results),
            6,
        ),
        timeout_rate=round(
            sum(1 for result in results if result.timeout) / len(results),
            6,
        ),
    )


def run_policy_episode(
    policy_name: str,
    seed: int,
    difficulty: str = "medium",
) -> PolicyEpisodeResult:
    """Run one deterministic baseline policy episode."""

    if policy_name not in POLICY_NAMES:
        raise ValueError(f"Unknown policy: {policy_name}")

    env = AtomicVisionEnvironment(difficulty=difficulty)
    observation = env.reset(seed=seed)
    rng = random.Random(seed + 90_000)

    if policy_name == "cheap_submit":
        observation = _cheap_submit(env)
    elif policy_name == "random":
        observation = _random_policy(env, observation, rng)
    elif policy_name == "scan_heavy":
        observation = _scan_heavy_policy(env)
    elif policy_name == "prior_submit":
        observation = _prior_submit_policy(env)
    elif policy_name == "oracle":
        observation = _oracle_policy(env)

    breakdown = observation.reward_breakdown or {}
    return PolicyEpisodeResult(
        policy_name=policy_name,
        seed=seed,
        difficulty=difficulty,
        total_reward=float(observation.reward or 0.0),
        steps=observation.step_count,
        total_scan_cost=env.state.total_scan_cost,
        done=observation.done,
        f1=float(breakdown.get("f1", 0.0)),
        concentration_mae=float(breakdown.get("concentration_mae", 0.0)),
        identity_reward=float(breakdown.get("identity_reward", 0.0)),
        concentration_reward=float(breakdown.get("concentration_reward", 0.0)),
        confidence_reward=float(breakdown.get("confidence_reward", 0.0)),
        outcome_reward_total=float(
            breakdown.get("identity_reward", 0.0)
            + breakdown.get("concentration_reward", 0.0)
            + breakdown.get("confidence_reward", 0.0)
        ),
        penalty_total=float(
            breakdown.get("false_positive_penalty", 0.0)
            + breakdown.get("missed_defect_penalty", 0.0)
            + breakdown.get("scan_cost_penalty", 0.0)
            + breakdown.get("timeout_penalty", 0.0)
        ),
        timeout=bool(breakdown.get("timeout_penalty", 0.0) < 0.0),
    )


def _cheap_submit(env: AtomicVisionEnvironment) -> AtomicVisionObservation:
    return env.step(
        AtomicVisionAction(
            action_type="submit_defect_map",
            predicted_defects=[],
            predicted_concentrations=[],
            confidence=0.1,
        )
    )


def _random_policy(
    env: AtomicVisionEnvironment,
    observation: AtomicVisionObservation,
    rng: random.Random,
) -> AtomicVisionObservation:
    current = observation
    for _ in range(observation.max_steps):
        if rng.random() < 0.35 or current.budget_remaining <= 1.0:
            count = rng.randint(0, min(3, len(current.candidate_defects)))
            defects = rng.sample(current.candidate_defects, count) if count else []
            concentrations = [round(rng.uniform(0.002, 0.20), 5) for _ in defects]
            return env.step(
                AtomicVisionAction(
                    action_type="submit_defect_map",
                    predicted_defects=defects,
                    predicted_concentrations=concentrations,
                    confidence=round(rng.random(), 3),
                )
            )

        if rng.random() < 0.5:
            current = env.step(
                AtomicVisionAction(
                    action_type="request_scan",
                    scan_mode="standard_pdos",
                    resolution="medium",
                )
            )
        else:
            start = rng.uniform(1.0, 12.0)
            current = env.step(
                AtomicVisionAction(
                    action_type="zoom_band",
                    freq_min=round(start, 3),
                    freq_max=round(start + rng.uniform(2.0, 5.0), 3),
                )
            )
        if current.done:
            return current
    return current


def _scan_heavy_policy(env: AtomicVisionEnvironment) -> AtomicVisionObservation:
    env.step(
        AtomicVisionAction(
            action_type="request_scan",
            scan_mode="standard_pdos",
            resolution="medium",
        )
    )
    env.step(
        AtomicVisionAction(
            action_type="zoom_band",
            freq_min=4.0,
            freq_max=9.0,
            resolution="high",
        )
    )
    env.step(AtomicVisionAction(action_type="compare_reference"))
    observation = env.step(AtomicVisionAction(action_type="ask_prior"))
    prior = observation.prior_prediction
    return env.step(
        AtomicVisionAction(
            action_type="submit_defect_map",
            predicted_defects=[] if prior is None else prior.predicted_defects,
            predicted_concentrations=[] if prior is None else prior.predicted_concentrations,
            confidence=0.55 if prior is None else prior.confidence,
        )
    )


def _prior_submit_policy(env: AtomicVisionEnvironment) -> AtomicVisionObservation:
    observation = env.step(AtomicVisionAction(action_type="ask_prior"))
    prior = observation.prior_prediction
    return env.step(
        AtomicVisionAction(
            action_type="submit_defect_map",
            predicted_defects=[] if prior is None else prior.predicted_defects,
            predicted_concentrations=[] if prior is None else prior.predicted_concentrations,
            confidence=0.45 if prior is None else prior.confidence,
        )
    )


def _oracle_policy(env: AtomicVisionEnvironment) -> AtomicVisionObservation:
    case = env._require_case()
    return env.step(
        AtomicVisionAction(
            action_type="submit_defect_map",
            predicted_defects=[defect.species for defect in case.defects],
            predicted_concentrations=[defect.concentration for defect in case.defects],
            confidence=0.95,
        )
    )


def _mean(values) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)
