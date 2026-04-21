from __future__ import annotations

import pytest

from atomicvision.evaluation import aggregate_rewards
from atomicvision.rewards import scan_cost_penalty_for, score_submission
from atomicvision.synthetic import generate_case


def _truth(case):
    return [defect.species for defect in case.defects], [
        defect.concentration for defect in case.defects
    ]


def test_correct_submission_scores_higher_than_wrong_submission() -> None:
    case = generate_case(seed=42, difficulty="medium")
    defects, concentrations = _truth(case)

    correct = score_submission(case, defects, concentrations, confidence=0.95)
    wrong = score_submission(case, ["Zn"], [0.01], confidence=0.95)

    assert correct.total_reward > wrong.total_reward
    assert correct.f1 == 1.0
    assert wrong.f1 < correct.f1


def test_scan_cost_reduces_reward() -> None:
    case = generate_case(seed=12, difficulty="easy")
    defects, concentrations = _truth(case)

    cheap = score_submission(case, defects, concentrations, confidence=0.95, scan_cost=1.0)
    expensive = score_submission(
        case,
        defects,
        concentrations,
        confidence=0.95,
        scan_cost=10.0,
    )

    assert expensive.total_reward < cheap.total_reward
    assert expensive.scan_cost_penalty < cheap.scan_cost_penalty


def test_scan_cost_penalty_strongly_discourages_extra_standard_scan() -> None:
    case = generate_case(seed=21, difficulty="medium")
    defects, concentrations = _truth(case)

    prior_only = score_submission(
        case,
        defects,
        concentrations,
        confidence=0.95,
        scan_cost=1.5,
    )
    prior_plus_standard_scan = score_submission(
        case,
        defects,
        concentrations,
        confidence=0.95,
        scan_cost=3.5,
    )

    assert prior_only.total_reward - prior_plus_standard_scan.total_reward == pytest.approx(0.8)
    assert scan_cost_penalty_for(3.5) == pytest.approx(-1.4)


def test_missing_defects_are_penalized() -> None:
    case = generate_case(seed=9, difficulty="hard")
    defects, concentrations = _truth(case)

    complete = score_submission(case, defects, concentrations, confidence=0.95)
    partial = score_submission(case, defects[:1], concentrations[:1], confidence=0.5)

    assert partial.total_reward < complete.total_reward
    assert partial.missed_defect_penalty < 0.0
    assert partial.recall < complete.recall


def test_empty_blind_submission_is_penalized() -> None:
    case = generate_case(seed=10, difficulty="easy")

    blind = score_submission(case, [], [], confidence=0.1)

    assert blind.f1 == 0.0
    assert blind.concentration_reward == 0.0
    assert blind.total_reward < 0.0


def test_overconfident_wrong_answer_has_lower_confidence_reward() -> None:
    case = generate_case(seed=17, difficulty="medium")

    cautious = score_submission(case, ["Zn"], [0.01], confidence=0.1)
    overconfident = score_submission(case, ["Zn"], [0.01], confidence=1.0)

    assert overconfident.confidence_reward < cautious.confidence_reward


def test_timeout_penalty_is_applied() -> None:
    case = generate_case(seed=31, difficulty="easy")

    timed_out = score_submission(
        case,
        [],
        [],
        confidence=0.0,
        scan_cost=case.budget,
        timed_out=True,
    )

    assert timed_out.timeout_penalty == -2.0
    assert timed_out.total_reward < 0.0


def test_invalid_submission_shapes_raise() -> None:
    case = generate_case(seed=3, difficulty="easy")

    with pytest.raises(ValueError, match="must match"):
        score_submission(case, ["B"], [], confidence=0.5)


def test_aggregate_rewards_reports_mean_metrics() -> None:
    case = generate_case(seed=42, difficulty="medium")
    defects, concentrations = _truth(case)
    first = score_submission(case, defects, concentrations, confidence=0.95)
    second = score_submission(case, [], [], confidence=0.0, timed_out=True)

    metrics = aggregate_rewards([first, second])

    assert metrics.episodes == 2
    assert metrics.mean_reward == round((first.total_reward + second.total_reward) / 2, 6)
    assert metrics.timeout_rate == 0.5
