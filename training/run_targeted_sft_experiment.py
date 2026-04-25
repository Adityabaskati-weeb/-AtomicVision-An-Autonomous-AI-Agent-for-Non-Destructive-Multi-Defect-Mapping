"""Run a small AtomicVision SFT experiment with robust checkpoint selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_EVAL_DIFFICULTIES = ("medium", "hard")
DEFAULT_TRAIN_DIFFICULTIES = ("hard",)
DEFAULT_CHECKPOINT_STEPS = (1, 2, 4)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a targeted AtomicVision SFT dataset, run a short continuation, "
            "evaluate checkpoints, and print a robust promotion summary."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("hard_recall_repair", "hard_recall_micro_repair"),
        default="hard_recall_micro_repair",
    )
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--init-adapter-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes-per-difficulty", type=int, default=16)
    parser.add_argument("--train-difficulties", nargs="+", default=list(DEFAULT_TRAIN_DIFFICULTIES))
    parser.add_argument("--eval-difficulties", nargs="+", default=list(DEFAULT_EVAL_DIFFICULTIES))
    parser.add_argument("--seed-start", type=int, default=3600)
    parser.add_argument("--eval-seed-start", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--max-scan-candidates-per-difficulty", type=int, default=2048)
    parser.add_argument("--max-updates", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, default=list(DEFAULT_CHECKPOINT_STEPS))
    parser.add_argument(
        "--output-json",
        default="summary.json",
        help="Summary filename relative to --output-root, or an absolute path.",
    )
    return parser


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_dataset_rows(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        counts[str(row.get("sample_type") or "unknown")] += 1
    return dict(sorted(counts.items()))


def metric_value(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return float(default)


def strict_stats(report: dict[str, Any], difficulty: str) -> dict[str, float]:
    row = report["results"][difficulty]["strict_adapter"]
    return {
        "reward": metric_value(row, "mean_reward"),
        "f1": metric_value(row, "mean_f1"),
        "mae": metric_value(row, "mean_mae"),
        "strict": metric_value(row, "strict_tool_call_pass_rate"),
        "normalized": metric_value(row, "normalized_tool_call_pass_rate"),
        "done": metric_value(row, "done_rate"),
        "fail": metric_value(row, "tool_failure_rate"),
        "submit": metric_value(row, "submit_tool_rate", "submit_action_rate"),
    }


def candidate_summary(
    label: str,
    report: dict[str, Any],
    base_medium: dict[str, float],
    base_hard: dict[str, float],
) -> dict[str, Any]:
    medium = strict_stats(report, "medium")
    hard = strict_stats(report, "hard")
    return {
        "label": label,
        "medium": medium,
        "hard": hard,
        "medium_reward_delta_vs_base": medium["reward"] - base_medium["reward"],
        "hard_reward_delta_vs_base": hard["reward"] - base_hard["reward"],
        "medium_f1_delta_vs_base": medium["f1"] - base_medium["f1"],
        "hard_f1_delta_vs_base": hard["f1"] - base_hard["f1"],
    }


def qualifies_for_promotion(candidate: dict[str, Any]) -> bool:
    medium = candidate["medium"]
    hard = candidate["hard"]
    return (
        medium["strict"] == 1.0
        and hard["strict"] == 1.0
        and medium["fail"] == 0.0
        and hard["fail"] == 0.0
        and candidate["medium_reward_delta_vs_base"] >= -0.01
        and candidate["hard_reward_delta_vs_base"] > 0.0
    )


def select_promotion_candidate(candidates: dict[str, dict[str, Any]]) -> str | None:
    winner: str | None = None
    winner_score: tuple[float, float, float] | None = None
    for label, candidate in candidates.items():
        if label == "base" or not qualifies_for_promotion(candidate):
            continue
        score = (
            candidate["hard_reward_delta_vs_base"],
            candidate["hard_f1_delta_vs_base"],
            candidate["medium_reward_delta_vs_base"],
        )
        if winner_score is None or score > winner_score:
            winner = label
            winner_score = score
    return winner


def resolve_output_json(output_root: Path, output_json: str) -> Path:
    output_path = Path(output_json)
    if not output_path.is_absolute():
        output_path = output_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def main() -> None:
    args = build_arg_parser().parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / f"{args.profile}.jsonl"
    train_output_dir = output_root / "train"
    base_eval_path = output_root / "base_eval.json"

    _run(
        [
            sys.executable,
            "training/generate_atomicvision_sft_data.py",
            "--profile",
            args.profile,
            "--episodes-per-difficulty",
            str(args.episodes_per_difficulty),
            "--difficulties",
            *args.train_difficulties,
            "--seed-start",
            str(args.seed_start),
            "--max-scan-candidates-per-difficulty",
            str(args.max_scan_candidates_per_difficulty),
            "--output-jsonl",
            str(dataset_path),
        ]
    )

    dataset_counts = _count_dataset_rows(dataset_path)

    _run(
        [
            sys.executable,
            "training/evaluate_atomicvision_adapter.py",
            "--adapter-dir",
            args.init_adapter_dir,
            "--difficulties",
            *args.eval_difficulties,
            "--episodes",
            str(args.eval_episodes),
            "--seed-start",
            str(args.eval_seed_start),
            "--modes",
            "strict",
            "--output-json",
            str(base_eval_path),
        ]
    )

    _run(
        [
            sys.executable,
            "training/train_sft_atomicvision_safe.py",
            "--dataset-jsonl",
            str(dataset_path),
            "--model",
            args.model,
            "--init-adapter-dir",
            args.init_adapter_dir,
            "--output-dir",
            str(train_output_dir),
            "--max-updates",
            str(args.max_updates),
            "--grad-accum",
            str(args.grad_accum),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--checkpoint-steps",
            *[str(step) for step in args.checkpoint_steps],
            "--overwrite-output-dir",
        ]
    )

    reports = {"base": _load_json(base_eval_path)}
    base_medium = strict_stats(reports["base"], "medium")
    base_hard = strict_stats(reports["base"], "hard")

    candidates: dict[str, dict[str, Any]] = {
        "base": candidate_summary("base", reports["base"], base_medium, base_hard)
    }
    for step in args.checkpoint_steps:
        label = f"checkpoint-{step}"
        adapter_dir = train_output_dir / label
        if not adapter_dir.exists():
            continue
        eval_path = output_root / f"{label}_eval.json"
        _run(
            [
                sys.executable,
                "training/evaluate_atomicvision_adapter.py",
                "--adapter-dir",
                str(adapter_dir),
                "--difficulties",
                *args.eval_difficulties,
                "--episodes",
                str(args.eval_episodes),
                "--seed-start",
                str(args.eval_seed_start),
                "--modes",
                "strict",
                "--output-json",
                str(eval_path),
            ]
        )
        reports[label] = _load_json(eval_path)
        candidates[label] = candidate_summary(label, reports[label], base_medium, base_hard)

    summary = {
        "profile": args.profile,
        "dataset_counts": dataset_counts,
        "train_difficulties": list(args.train_difficulties),
        "eval_difficulties": list(args.eval_difficulties),
        "seed_start": args.seed_start,
        "eval_seed_start": args.eval_seed_start,
        "episodes_per_difficulty": args.episodes_per_difficulty,
        "eval_episodes": args.eval_episodes,
        "max_updates": args.max_updates,
        "learning_rate": args.learning_rate,
        "checkpoint_steps": list(args.checkpoint_steps),
        "base_adapter": args.init_adapter_dir,
        "candidates": candidates,
        "promotion_candidate": select_promotion_candidate(candidates),
    }
    output_path = resolve_output_json(output_root, args.output_json)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("FINAL_TARGETED_SFT_SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
