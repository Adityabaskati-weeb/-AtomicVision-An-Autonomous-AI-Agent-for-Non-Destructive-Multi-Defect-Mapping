"""Generate AtomicVision supervised tool-copy data for SFT.

The generated JSONL is intentionally Kaggle/HF-Jobs friendly: each row contains
a `messages` chat transcript plus metadata. The key target behavior is exact
copying of the DefectNet-lite prior into the terminal `submit_defect_map` call.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomicvision_env.models import AtomicVisionAction  # noqa: E402
from atomicvision_env.server.environment import AtomicVisionEnvironment  # noqa: E402
from training.train_grpo_atomicvision import DEFAULT_PROMPT, _format_observation  # noqa: E402


TOOL_SYSTEM = (
    "You are using AtomicVision tools. Return exactly one tool call wrapped in "
    "<tool_call>...</tool_call>. Use ask_prior first. After the prior appears, "
    "copy its predicted_defects, predicted_concentrations, and confidence "
    "exactly into submit_defect_map. Do not invent species or concentrations."
)

ASK_PRIOR_CALL = {"name": "ask_prior", "arguments": {}}


def build_sft_examples(
    episodes_per_difficulty: int,
    difficulties: tuple[str, ...] = ("medium",),
    seed_start: int = 0,
    sample_types: tuple[str, ...] = ("ask_prior", "submit_prior"),
) -> list[dict[str, Any]]:
    """Build SFT examples from deterministic local AtomicVision episodes."""

    if episodes_per_difficulty <= 0:
        raise ValueError("episodes_per_difficulty must be positive")
    unknown_types = [kind for kind in sample_types if kind not in {"ask_prior", "submit_prior"}]
    if unknown_types:
        raise ValueError(f"Unknown sample types: {', '.join(unknown_types)}")

    examples: list[dict[str, Any]] = []
    for difficulty in difficulties:
        for seed in range(seed_start, seed_start + episodes_per_difficulty):
            examples.extend(
                build_episode_examples(
                    seed=seed,
                    difficulty=difficulty,
                    sample_types=sample_types,
                )
            )
    return examples


def build_episode_examples(
    seed: int,
    difficulty: str = "medium",
    sample_types: tuple[str, ...] = ("ask_prior", "submit_prior"),
) -> list[dict[str, Any]]:
    """Build ask-prior and submit-prior examples for one episode."""

    env = AtomicVisionEnvironment(difficulty=difficulty)
    initial_observation = env.reset(seed=seed)
    initial_text = _format_observation(initial_observation.model_dump())
    initial_user = _user_message(initial_text)
    ask_text = _tool_call_text(ASK_PRIOR_CALL)

    prior_observation = env.step(AtomicVisionAction(action_type="ask_prior"))
    prior_text = _format_observation(prior_observation.model_dump())
    prior = prior_observation.prior_prediction
    submit_args = _submit_args_from_prior(prior)
    submit_call = {"name": "submit_defect_map", "arguments": submit_args}
    submit_text = _tool_call_text(submit_call)

    final_observation = env.step(
        AtomicVisionAction(
            action_type="submit_defect_map",
            predicted_defects=submit_args["predicted_defects"],
            predicted_concentrations=submit_args["predicted_concentrations"],
            confidence=submit_args["confidence"],
        )
    )

    examples: list[dict[str, Any]] = []
    if "ask_prior" in sample_types:
        examples.append(
            {
                "sample_id": f"{difficulty}-{seed}-ask_prior",
                "sample_type": "ask_prior",
                "target_tool_name": "ask_prior",
                "target_tool_call": ask_text,
                "seed": seed,
                "difficulty": difficulty,
                "messages": [
                    {"role": "system", "content": TOOL_SYSTEM},
                    {"role": "user", "content": initial_user},
                    {"role": "assistant", "content": ask_text},
                ],
            }
        )
    if "submit_prior" in sample_types:
        examples.append(
            {
                "sample_id": f"{difficulty}-{seed}-submit_prior",
                "sample_type": "submit_prior",
                "target_tool_name": "submit_defect_map",
                "target_tool_call": submit_text,
                "seed": seed,
                "difficulty": difficulty,
                "prior_prediction": _model_dump(prior),
                "expected_reward": float(final_observation.reward or 0.0),
                "expected_reward_breakdown": final_observation.reward_breakdown or {},
                "expected_scan_cost": float(env.state.total_scan_cost),
                "messages": [
                    {"role": "system", "content": TOOL_SYSTEM},
                    {"role": "user", "content": initial_user},
                    {"role": "assistant", "content": ask_text},
                    {"role": "user", "content": _tool_response(prior_text)},
                    {"role": "assistant", "content": submit_text},
                ],
            }
        )
    return examples


def write_jsonl(examples: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write examples as UTF-8 JSONL."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")
    return path


def _submit_args_from_prior(prior: Any) -> dict[str, Any]:
    if prior is None:
        return {
            "predicted_defects": [],
            "predicted_concentrations": [],
            "confidence": 0.45,
        }
    return {
        "predicted_defects": list(prior.predicted_defects),
        "predicted_concentrations": list(prior.predicted_concentrations),
        "confidence": float(prior.confidence),
    }


def _model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value)


def _tool_call_text(call: dict[str, Any]) -> str:
    payload = json.dumps(call, separators=(",", ":"), ensure_ascii=True)
    return f"<tool_call>{payload}</tool_call>"


def _user_message(observation_text: str) -> str:
    return f"{DEFAULT_PROMPT}\n\nObservation:\n{observation_text}"


def _tool_response(observation_text: str) -> str:
    return f"<tool_response>\n{observation_text}\n</tool_response>"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AtomicVision SFT JSONL for exact prior-to-tool-call copying.",
    )
    parser.add_argument("--episodes-per-difficulty", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--difficulties", nargs="+", default=["medium"])
    parser.add_argument(
        "--sample-types",
        nargs="+",
        default=["ask_prior", "submit_prior"],
        choices=["ask_prior", "submit_prior"],
    )
    parser.add_argument(
        "--output-jsonl",
        default="outputs/sft/atomicvision_tool_copy_sft.jsonl",
    )
    args = parser.parse_args()

    examples = build_sft_examples(
        episodes_per_difficulty=args.episodes_per_difficulty,
        difficulties=tuple(args.difficulties),
        seed_start=args.seed_start,
        sample_types=tuple(args.sample_types),
    )
    output_path = write_jsonl(examples, args.output_jsonl)
    counts = Counter(example["sample_type"] for example in examples)
    print(f"Wrote {len(examples)} examples to {output_path}")
    print(f"Difficulties: {', '.join(args.difficulties)}")
    print(f"Sample counts: {dict(sorted(counts.items()))}")
    print("Kaggle next: load this JSONL with datasets.load_dataset('json', data_files=path).")


if __name__ == "__main__":
    main()
