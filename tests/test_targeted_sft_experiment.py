from __future__ import annotations

from training.run_targeted_sft_experiment import (
    candidate_summary,
    metric_value,
    qualifies_for_promotion,
    select_promotion_candidate,
    strict_stats,
)


def _report(
    *,
    medium_reward: float,
    hard_reward: float,
    medium_f1: float,
    hard_f1: float,
    medium_submit_key: str = "submit_action_rate",
    hard_submit_key: str = "submit_action_rate",
) -> dict:
    return {
        "results": {
            "medium": {
                "strict_adapter": {
                    "mean_reward": medium_reward,
                    "mean_f1": medium_f1,
                    "mean_mae": 0.1,
                    "strict_tool_call_pass_rate": 1.0,
                    "normalized_tool_call_pass_rate": 1.0,
                    "done_rate": 1.0,
                    "tool_failure_rate": 0.0,
                    medium_submit_key: 1.0,
                }
            },
            "hard": {
                "strict_adapter": {
                    "mean_reward": hard_reward,
                    "mean_f1": hard_f1,
                    "mean_mae": 0.2,
                    "strict_tool_call_pass_rate": 1.0,
                    "normalized_tool_call_pass_rate": 1.0,
                    "done_rate": 1.0,
                    "tool_failure_rate": 0.0,
                    hard_submit_key: 1.0,
                }
            },
        }
    }


def test_metric_value_uses_first_present_name() -> None:
    assert metric_value({"submit_action_rate": 0.75}, "submit_tool_rate", "submit_action_rate") == 0.75
    assert metric_value({}, "submit_tool_rate", default=0.5) == 0.5


def test_strict_stats_falls_back_to_submit_action_rate() -> None:
    report = _report(
        medium_reward=4.5,
        hard_reward=4.8,
        medium_f1=0.78,
        hard_f1=0.82,
    )

    medium = strict_stats(report, "medium")

    assert medium["reward"] == 4.5
    assert medium["submit"] == 1.0


def test_select_promotion_candidate_prefers_positive_hard_delta() -> None:
    base_report = _report(
        medium_reward=4.5,
        hard_reward=4.6,
        medium_f1=0.78,
        hard_f1=0.80,
    )
    base_medium = strict_stats(base_report, "medium")
    base_hard = strict_stats(base_report, "hard")

    weak = candidate_summary(
        "checkpoint-1",
        _report(
            medium_reward=4.49,
            hard_reward=4.61,
            medium_f1=0.78,
            hard_f1=0.801,
            medium_submit_key="submit_tool_rate",
            hard_submit_key="submit_tool_rate",
        ),
        base_medium,
        base_hard,
    )
    strong = candidate_summary(
        "checkpoint-2",
        _report(
            medium_reward=4.50,
            hard_reward=4.66,
            medium_f1=0.781,
            hard_f1=0.808,
            medium_submit_key="submit_tool_rate",
            hard_submit_key="submit_tool_rate",
        ),
        base_medium,
        base_hard,
    )

    candidates = {
        "base": candidate_summary("base", base_report, base_medium, base_hard),
        "checkpoint-1": weak,
        "checkpoint-2": strong,
    }

    assert qualifies_for_promotion(strong)
    assert select_promotion_candidate(candidates) == "checkpoint-2"

