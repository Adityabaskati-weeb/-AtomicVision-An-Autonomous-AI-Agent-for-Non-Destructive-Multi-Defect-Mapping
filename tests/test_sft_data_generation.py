from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from training.generate_atomicvision_sft_data import build_sft_examples


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


def _parse_tool_call(text: str) -> dict:
    assert text.startswith("<tool_call>")
    assert text.endswith("</tool_call>")
    payload = text.removeprefix("<tool_call>").removesuffix("</tool_call>")
    return json.loads(payload)
