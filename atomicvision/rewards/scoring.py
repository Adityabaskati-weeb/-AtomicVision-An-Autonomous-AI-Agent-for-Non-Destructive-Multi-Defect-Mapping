"""Reward scoring for AtomicVision defect-map submissions."""

from __future__ import annotations

from dataclasses import dataclass

from atomicvision.synthetic.types import MaterialCase


SCAN_COST_PENALTY_RATE = 0.4
SCAN_COST_PENALTY_CAP = 4.0


@dataclass(frozen=True)
class RewardBreakdown:
    """Transparent score components for one episode submission."""

    total_reward: float
    identity_reward: float
    concentration_reward: float
    confidence_reward: float
    false_positive_penalty: float
    missed_defect_penalty: float
    scan_cost_penalty: float
    timeout_penalty: float
    precision: float
    recall: float
    f1: float
    concentration_mae: float


def score_submission(
    case: MaterialCase,
    predicted_defects: list[str] | tuple[str, ...],
    predicted_concentrations: list[float] | tuple[float, ...],
    confidence: float,
    scan_cost: float = 0.0,
    timed_out: bool = False,
) -> RewardBreakdown:
    """Score an agent's final defect-map submission."""

    if len(predicted_defects) != len(predicted_concentrations):
        raise ValueError("predicted_defects and predicted_concentrations must match")
    if any(concentration < 0.0 for concentration in predicted_concentrations):
        raise ValueError("predicted concentrations must be non-negative")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    if scan_cost < 0.0:
        raise ValueError("scan_cost must be non-negative")

    truth = {defect.species: defect.concentration for defect in case.defects}
    predictions = _merge_predictions(predicted_defects, predicted_concentrations)

    true_species = set(truth)
    predicted_species = set(predictions)
    true_positive_count = len(true_species & predicted_species)
    false_positive_count = len(predicted_species - true_species)
    missed_count = len(true_species - predicted_species)

    precision = _safe_divide(true_positive_count, len(predicted_species))
    recall = _safe_divide(true_positive_count, len(true_species))
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    concentration_mae = _concentration_mae(truth, predictions)
    concentration_score = max(0.0, 1.0 - concentration_mae / 0.25)

    identity_reward = 4.0 * f1
    concentration_reward = 3.0 * concentration_score * f1
    accuracy_proxy = 0.5 * f1 + 0.5 * concentration_score
    raw_confidence_reward = max(
        -1.0,
        min(1.0, 1.0 - 2.0 * abs(confidence - accuracy_proxy)),
    )
    confidence_scale = f1 if predicted_species else 0.0
    confidence_reward = raw_confidence_reward * confidence_scale
    false_positive_penalty = -min(2.0, 0.5 * false_positive_count)
    missed_defect_penalty = -min(2.0, 0.6 * missed_count)
    scan_cost_penalty = scan_cost_penalty_for(scan_cost)
    timeout_penalty = -2.0 if timed_out else 0.0

    total = (
        identity_reward
        + concentration_reward
        + confidence_reward
        + false_positive_penalty
        + missed_defect_penalty
        + scan_cost_penalty
        + timeout_penalty
    )

    return RewardBreakdown(
        total_reward=round(total, 6),
        identity_reward=round(identity_reward, 6),
        concentration_reward=round(concentration_reward, 6),
        confidence_reward=round(confidence_reward, 6),
        false_positive_penalty=round(false_positive_penalty, 6),
        missed_defect_penalty=round(missed_defect_penalty, 6),
        scan_cost_penalty=round(scan_cost_penalty, 6),
        timeout_penalty=round(timeout_penalty, 6),
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        concentration_mae=round(concentration_mae, 6),
    )


def scan_cost_penalty_for(scan_cost: float) -> float:
    """Return the reward penalty for evidence-gathering cost."""

    if scan_cost < 0.0:
        raise ValueError("scan_cost must be non-negative")
    return -min(SCAN_COST_PENALTY_CAP, SCAN_COST_PENALTY_RATE * scan_cost)


def _merge_predictions(
    predicted_defects: list[str] | tuple[str, ...],
    predicted_concentrations: list[float] | tuple[float, ...],
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for species, concentration in zip(
        predicted_defects,
        predicted_concentrations,
        strict=True,
    ):
        merged[species] = max(merged.get(species, 0.0), concentration)
    return merged


def _concentration_mae(
    truth: dict[str, float],
    predictions: dict[str, float],
) -> float:
    species_union = sorted(set(truth) | set(predictions))
    if not species_union:
        return 0.0
    errors = [
        abs(predictions.get(species, 0.0) - truth.get(species, 0.0))
        for species in species_union
    ]
    return sum(errors) / len(errors)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
