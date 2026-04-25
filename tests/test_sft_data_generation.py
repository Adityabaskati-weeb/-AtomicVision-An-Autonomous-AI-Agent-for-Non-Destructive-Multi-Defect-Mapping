from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from training.generate_atomicvision_sft_data import (
    build_arg_parser,
    build_cost_aware_sft_examples,
    build_format_refresh_examples,
    build_hard_frontier_boost_examples,
    build_sft_examples,
)
from training.seed_ranges import SFT_TRAIN_SEED_START


def test_submit_prior_examples_copy_prior_exactly() -> None:
    examples = build_sft_examples(
        episodes_per_difficulty=4,
        difficulties=("medium",),
        sample_types=("submit_prior",),
    )

    assert len(examples) == 4
    for example in examples:
        prior = example["prior_prediction"]
        target = _parse_tool_call(example["target_tool_call"])
        args = target["arguments"]

        assert target["name"] == "submit_defect_map"
        assert args["predicted_defects"] == prior["predicted_defects"]
        assert args["predicted_concentrations"] == prior["predicted_concentrations"]
        assert args["confidence"] == prior["confidence"]
        assert example["messages"][-1]["content"] == example["target_tool_call"]


def test_sft_generator_cli_defaults_to_sft_training_band() -> None:
    parser = build_arg_parser()

    args = parser.parse_args([])

    assert args.seed_start == SFT_TRAIN_SEED_START


def test_sft_generator_cli_writes_jsonl() -> None:
    output_path = Path(f"outputs/test-sft-generator/atomicvision_sft_{os.getpid()}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [
            sys.executable,
            "training/generate_atomicvision_sft_data.py",
            "--episodes-per-difficulty",
            "2",
            "--difficulties",
            "easy",
            "--output-jsonl",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]

    assert "Wrote 4 examples" in completed.stdout
    assert len(rows) == 4
    assert {row["sample_type"] for row in rows} == {"ask_prior", "submit_prior"}
    assert all(row["messages"] for row in rows)


def test_scan_improvement_examples_use_reference_before_revised_submit() -> None:
    examples = build_sft_examples(
        episodes_per_difficulty=2,
        difficulties=("medium",),
        sample_types=("submit_after_reference",),
        max_scan_candidates_per_difficulty=16,
    )

    assert len(examples) == 2
    for example in examples:
        assistant_calls = [
            _parse_tool_call(message["content"])
            for message in example["messages"]
            if message["role"] == "assistant"
        ]

        assert [call["name"] for call in assistant_calls] == [
            "ask_prior",
            "compare_reference",
            "submit_defect_map",
        ]
        assert example["sample_type"] == "submit_after_reference"
        assert example["reward_improvement"] >= 0.25
        assert example["expected_scan_cost"] == 2.0
        assert "spectrum_delta_top_abs" in example["messages"][-2]["content"]
        assert example["oracle_defect_map"] == assistant_calls[-1]["arguments"]


def test_cost_aware_examples_use_cheap_prior_biased_mix() -> None:
    examples = build_cost_aware_sft_examples(
        examples_per_difficulty=20,
        difficulties=("medium",),
        max_scan_candidates_per_difficulty=24,
    )

    counts = _counts_by_type(examples)

    assert len(examples) == 20
    assert counts == {
        "ask_prior": 1,
        "submit_after_reference": 2,
        "submit_prior": 17,
    }
    for example in examples:
        assistant_calls = [
            _parse_tool_call(message["content"])
            for message in example["messages"]
            if message["role"] == "assistant"
        ]
        if example["sample_type"] == "submit_prior":
            assert [call["name"] for call in assistant_calls] == [
                "ask_prior",
                "submit_defect_map",
            ]
        elif example["sample_type"] == "submit_after_reference":
            assert [call["name"] for call in assistant_calls] == [
                "ask_prior",
                "compare_reference",
                "submit_defect_map",
            ]
        else:
            assert [call["name"] for call in assistant_calls] == ["ask_prior"]


def test_hard_frontier_boost_examples_use_hard_scan_search() -> None:
    examples = build_hard_frontier_boost_examples(
        examples_per_difficulty=10,
        difficulties=("hard",),
        max_scan_candidates_per_difficulty=256,
    )

    counts = _counts_by_type(examples)

    assert len(examples) == 10
    assert counts == {
        "ask_prior": 1,
        "submit_after_reference": 1,
        "submit_prior": 8,
    }
    assert {example["difficulty"] for example in examples} == {"hard"}
    assert any(example["sample_type"] == "submit_after_reference" for example in examples)


def test_format_refresh_examples_are_submit_heavy_and_reference_free() -> None:
    examples = build_format_refresh_examples(
        examples_per_difficulty=20,
        difficulties=("hard",),
    )

    counts = _counts_by_type(examples)

    assert len(examples) == 20
    assert counts == {
        "ask_prior": 2,
        "submit_prior": 18,
    }
    assert {example["difficulty"] for example in examples} == {"hard"}


def test_sft_generator_cli_writes_scan_improvement_jsonl() -> None:
    output_path = Path(f"outputs/test-sft-generator/atomicvision_scan_sft_{os.getpid()}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [
            sys.executable,
            "training/generate_atomicvision_sft_data.py",
            "--episodes-per-difficulty",
            "2",
            "--difficulties",
            "medium",
            "--sample-types",
            "submit_after_reference",
            "--output-jsonl",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert "Wrote 2 examples" in completed.stdout
    assert len(rows) == 2
    assert {row["sample_type"] for row in rows} == {"submit_after_reference"}


def test_sft_generator_cli_writes_cost_aware_jsonl() -> None:
    output_path = Path(f"outputs/test-sft-generator/atomicvision_cost_sft_{os.getpid()}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [
            sys.executable,
            "training/generate_atomicvision_sft_data.py",
            "--profile",
            "cost_aware",
            "--episodes-per-difficulty",
            "20",
            "--difficulties",
            "medium",
            "--max-scan-candidates-per-difficulty",
            "24",
            "--output-jsonl",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert "Wrote 20 examples" in completed.stdout
    assert _counts_by_type(rows) == {
        "ask_prior": 1,
        "submit_after_reference": 2,
        "submit_prior": 17,
    }


def test_sft_generator_cli_writes_hard_frontier_boost_jsonl() -> None:
    output_path = Path(f"outputs/test-sft-generator/atomicvision_hard_frontier_sft_{os.getpid()}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [
            sys.executable,
            "training/generate_atomicvision_sft_data.py",
            "--profile",
            "hard_frontier_boost",
            "--episodes-per-difficulty",
            "10",
            "--difficulties",
            "hard",
            "--max-scan-candidates-per-difficulty",
            "256",
            "--output-jsonl",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert "Wrote 10 examples" in completed.stdout
    assert _counts_by_type(rows) == {
        "ask_prior": 1,
        "submit_after_reference": 1,
        "submit_prior": 8,
    }
    assert {row["difficulty"] for row in rows} == {"hard"}


def test_sft_generator_cli_writes_format_refresh_jsonl() -> None:
    output_path = Path(f"outputs/test-sft-generator/atomicvision_format_refresh_{os.getpid()}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [
            sys.executable,
            "training/generate_atomicvision_sft_data.py",
            "--profile",
            "format_refresh",
            "--episodes-per-difficulty",
            "20",
            "--difficulties",
            "hard",
            "--output-jsonl",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert "Wrote 20 examples" in completed.stdout
    assert _counts_by_type(rows) == {
        "ask_prior": 2,
        "submit_prior": 18,
    }
    assert {row["difficulty"] for row in rows} == {"hard"}


def _parse_tool_call(text: str) -> dict:
    assert text.startswith("<tool_call>")
    assert text.endswith("</tool_call>")
    payload = text.removeprefix("<tool_call>").removesuffix("</tool_call>")
    return json.loads(payload)


def _counts_by_type(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["sample_type"]] = counts.get(row["sample_type"], 0) + 1
    return dict(sorted(counts.items()))
