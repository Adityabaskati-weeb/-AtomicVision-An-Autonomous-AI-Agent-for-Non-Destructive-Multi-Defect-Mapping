from __future__ import annotations

from training.evaluate_atomicvision_adapter import extract_tool_call, summarize


def test_extract_tool_call_distinguishes_strict_and_normalized_modes() -> None:
    strict_call, strict_text, strict_metrics = extract_tool_call("<tool_call> ask_prior", mode="strict")
    normalized_call, normalized_text, normalized_metrics = extract_tool_call(
        "<tool_call> ask_prior",
        mode="normalized",
    )

    assert strict_call is None
    assert strict_text == "<tool_call> ask_prior"
    assert strict_metrics == {
        "strict_parse": 0.0,
        "normalized_parse": 1.0,
        "normalized_repair": 1.0,
    }
    assert normalized_call == {"name": "ask_prior", "arguments": {}}
    assert normalized_text == '<tool_call>{"name":"ask_prior","arguments":{}}</tool_call>'
    assert normalized_metrics == strict_metrics


def test_summarize_reports_verifier_columns() -> None:
    summary = summarize(
        [
            {
                "reward": 4.0,
                "f1": 0.8,
                "mae": 0.03,
                "steps": 2,
                "scan_cost": 1.5,
                "done": True,
                "tool_failure": False,
                "repeated_tool_calls": 0,
                "strict_parse_rate": 0.5,
                "normalized_parse_rate": 1.0,
                "normalized_repair_rate": 0.5,
                "first_action_valid_rate": 1.0,
                "first_action_ask_prior_rate": 1.0,
                "submit_action_rate": 1.0,
                "identity_reward": 2.4,
                "concentration_reward": 1.2,
                "confidence_reward": 0.4,
                "false_positive_penalty": -0.1,
                "missed_defect_penalty": -0.2,
                "timeout_penalty": 0.0,
                "outcome_reward_total": 4.0,
                "penalty_total": -0.3,
            },
            {
                "reward": -1.0,
                "f1": 0.0,
                "mae": 1.0,
                "steps": 0,
                "scan_cost": 0.0,
                "done": False,
                "tool_failure": True,
                "repeated_tool_calls": 0,
                "strict_parse_rate": 0.0,
                "normalized_parse_rate": 0.0,
                "normalized_repair_rate": 0.0,
                "first_action_valid_rate": 0.0,
                "first_action_ask_prior_rate": 0.0,
                "submit_action_rate": 0.0,
                "identity_reward": 0.0,
                "concentration_reward": 0.0,
                "confidence_reward": 0.0,
                "false_positive_penalty": 0.0,
                "missed_defect_penalty": -0.5,
                "timeout_penalty": -0.4,
                "outcome_reward_total": 0.0,
                "penalty_total": -0.9,
            },
        ]
    )

    assert summary["episodes"] == 2
    assert summary["strict_tool_call_pass_rate"] == 0.25
    assert summary["normalized_tool_call_pass_rate"] == 0.5
    assert summary["normalized_tool_call_repair_rate"] == 0.25
    assert summary["first_action_valid_rate"] == 0.5
    assert summary["first_action_ask_prior_rate"] == 0.5
    assert summary["submit_action_rate"] == 0.5
    assert summary["mean_outcome_reward_total"] == 2.0
    assert summary["mean_penalty_total"] == -0.6
