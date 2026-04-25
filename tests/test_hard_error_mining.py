from __future__ import annotations

from training.mine_atomicvision_hard_failures import (
    _confidence_accuracy_gap,
    build_arg_parser,
    build_markdown_report,
    classify_action_pattern,
    classify_issue_bucket,
    summarize_mined_rows,
)


def test_hard_error_mining_cli_defaults_to_heldout_hard_scan() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(["--adapter-dir", "outputs/adapter"])

    assert args.difficulty == "hard"
    assert args.episodes == 64
    assert args.mode == "strict"
    assert "atomicvision_hard_error_mining.json" in args.output_json


def test_issue_bucket_prioritizes_structural_failures_first() -> None:
    assert (
        classify_issue_bucket(
            tool_failure=True,
            done=False,
            missing_species=[],
            extra_species=[],
            concentration_mae=0.0,
            confidence_gap=0.0,
        )
        == "tool_failure"
    )
    assert (
        classify_issue_bucket(
            tool_failure=False,
            done=False,
            missing_species=[],
            extra_species=[],
            concentration_mae=0.0,
            confidence_gap=0.0,
        )
        == "no_terminal_submit"
    )


def test_issue_bucket_distinguishes_species_concentration_and_confidence_errors() -> None:
    assert (
        classify_issue_bucket(
            tool_failure=False,
            done=True,
            missing_species=["Zn"],
            extra_species=["B"],
            concentration_mae=0.01,
            confidence_gap=0.10,
        )
        == "mixed_species_error"
    )
    assert (
        classify_issue_bucket(
            tool_failure=False,
            done=True,
            missing_species=["Zn"],
            extra_species=[],
            concentration_mae=0.01,
            confidence_gap=0.10,
        )
        == "missed_defects"
    )
    assert (
        classify_issue_bucket(
            tool_failure=False,
            done=True,
            missing_species=[],
            extra_species=["B"],
            concentration_mae=0.01,
            confidence_gap=0.10,
        )
        == "false_positive_defects"
    )
    assert (
        classify_issue_bucket(
            tool_failure=False,
            done=True,
            missing_species=[],
            extra_species=[],
            concentration_mae=0.05,
            confidence_gap=0.10,
        )
        == "concentration_error"
    )
    assert (
        classify_issue_bucket(
            tool_failure=False,
            done=True,
            missing_species=[],
            extra_species=[],
            concentration_mae=0.01,
            confidence_gap=0.35,
        )
        == "confidence_miscalibration"
    )
    assert (
        classify_issue_bucket(
            tool_failure=False,
            done=True,
            missing_species=[],
            extra_species=[],
            concentration_mae=0.01,
            confidence_gap=0.05,
        )
        == "clean_success"
    )


def test_action_pattern_flags_reference_submit_and_no_submit_paths() -> None:
    assert classify_action_pattern(["ask_prior", "submit_defect_map"], True, False) == "prior_then_submit"
    assert (
        classify_action_pattern(
            ["ask_prior", "compare_reference", "submit_defect_map"],
            True,
            False,
        )
        == "reference_then_submit"
    )
    assert (
        classify_action_pattern(
            ["ask_prior", "request_scan", "submit_defect_map"],
            True,
            False,
        )
        == "scan_then_submit"
    )
    assert classify_action_pattern(["ask_prior"], False, False) == "no_submit"


def test_confidence_accuracy_gap_tracks_miscalibration() -> None:
    assert _confidence_accuracy_gap(confidence=0.90, f1=1.0, concentration_mae=0.0) == 0.1
    assert _confidence_accuracy_gap(confidence=0.10, f1=1.0, concentration_mae=0.0) == 0.9


def test_summarize_rows_counts_regressions_and_buckets() -> None:
    rows = [
        {
            "seed": 10000,
            "reward": 4.0,
            "baseline_reward": 4.5,
            "reward_delta_vs_baseline": -0.5,
            "f1": 0.7,
            "concentration_mae": 0.04,
            "issue_bucket": "concentration_error",
            "action_pattern": "prior_then_submit",
        },
        {
            "seed": 10001,
            "reward": 4.6,
            "baseline_reward": 4.5,
            "reward_delta_vs_baseline": 0.1,
            "f1": 0.8,
            "concentration_mae": 0.01,
            "issue_bucket": "clean_success",
            "action_pattern": "reference_then_submit",
        },
    ]

    summary = summarize_mined_rows(rows)

    assert summary["episodes"] == 2
    assert summary["worse_than_baseline_count"] == 1
    assert summary["equal_or_better_than_baseline_count"] == 1
    assert summary["issue_bucket_counts"]["concentration_error"] == 1
    assert summary["issue_bucket_counts_when_worse_than_baseline"]["concentration_error"] == 1
    assert summary["action_pattern_counts"]["prior_then_submit"] == 1


def test_markdown_report_includes_top_regressions_table() -> None:
    report = {
        "adapter": "outputs/best",
        "base_model": "Qwen/Qwen3-1.7B",
        "difficulty": "hard",
        "episodes": 2,
        "seed_start": 10000,
        "mode": "strict",
        "summary": {
            "episodes": 2,
            "mean_reward": 4.3,
            "baseline_mean_reward": 4.5,
            "mean_reward_delta_vs_baseline": -0.2,
            "worse_than_baseline_count": 1,
            "issue_bucket_counts_when_worse_than_baseline": {
                "concentration_error": 1
            },
        },
        "episodes_by_regression": [
            {
                "seed": 10000,
                "reward_delta_vs_baseline": -0.5,
                "issue_bucket": "concentration_error",
                "action_pattern": "prior_then_submit",
                "missing_species": [],
                "extra_species": [],
                "concentration_mae": 0.04,
                "confidence_gap": 0.1,
            }
        ],
    }

    markdown = build_markdown_report(report, top_k=5)

    assert "# AtomicVision Hard Error Mining" in markdown
    assert "concentration_error" in markdown
    assert "| 10000 | -0.5000 | concentration_error |" in markdown
