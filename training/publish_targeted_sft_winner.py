"""Run a targeted AtomicVision SFT experiment and publish the winning checkpoint.

This wraps the short continuation workflow in ``run_targeted_sft_experiment.py``
and materializes only the promoted adapter checkpoint to the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download, upload_folder


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_BASE_ADAPTER = "prodigyhuh/atomicvision-medium-fidelity-boost-lora"
DEFAULT_TARGET_REPO = "prodigyhuh/atomicvision-hard-recall-micro-boost-lora"
EXCLUDED_CHECKPOINT_FILES = {
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "train.zip",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a targeted AtomicVision SFT continuation, select the promoted "
            "checkpoint, and upload it as a Hub adapter repo."
        )
    )
    parser.add_argument("--profile", default="hard_recall_micro_repair")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--init-adapter-dir", default=DEFAULT_BASE_ADAPTER)
    parser.add_argument("--target-repo", default=DEFAULT_TARGET_REPO)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes-per-difficulty", type=int, default=16)
    parser.add_argument("--train-difficulties", nargs="+", default=["hard"])
    parser.add_argument("--eval-difficulties", nargs="+", default=["medium", "hard"])
    parser.add_argument("--seed-start", type=int, default=3600)
    parser.add_argument("--eval-seed-start", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--max-scan-candidates-per-difficulty", type=int, default=2048)
    parser.add_argument("--max-updates", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, default=[1])
    parser.add_argument("--expected-promotion-candidate", default="checkpoint-1")
    parser.add_argument("--source-job-id", default="")
    parser.add_argument("--source-commit", default="")
    return parser


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return float(default)


def resolve_init_adapter_dir(adapter_source: str, output_root: Path) -> str:
    local_path = Path(adapter_source)
    if local_path.exists():
        return str(local_path)
    snapshot_dir = output_root / "base_adapter"
    snapshot_download(
        repo_id=adapter_source,
        local_dir=str(snapshot_dir),
        token=os.environ.get("HF_TOKEN"),
    )
    return str(snapshot_dir)


def should_publish_file(path: Path) -> bool:
    return path.is_file() and path.name not in EXCLUDED_CHECKPOINT_FILES


def copy_checkpoint_artifacts(source_dir: Path, publish_dir: Path) -> None:
    publish_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if should_publish_file(item):
            shutil.copy2(item, publish_dir / item.name)


def build_model_card(
    *,
    model: str,
    init_adapter_dir: str,
    target_repo: str,
    promotion_candidate: str,
    source_job_id: str,
    source_commit: str,
    medium: dict[str, Any],
    hard: dict[str, Any],
) -> str:
    source_job_line = (
        f"- Source HF job: [{source_job_id}](https://huggingface.co/jobs/prodigyhuh/{source_job_id})"
        if source_job_id
        else "- Source HF job: local publish rerun"
    )
    source_commit_line = f"- Source commit: `{source_commit}`" if source_commit else ""
    return f"""---
base_model: {model}
license: mit
library_name: peft
pipeline_tag: text-generation
---

# AtomicVision Hard Recall Micro Boost LoRA

This adapter materializes the promoted `{promotion_candidate}` checkpoint from the
targeted hard recall micro-repair continuation.

## Parent
- Base adapter: [{init_adapter_dir}](https://huggingface.co/{init_adapter_dir})
{source_job_line}
{source_commit_line}

## Held-out strict eval (`seed_start=10000`, `episodes=32`)

| Difficulty | Reward | F1 | MAE | Strict | Normalized | Done | Submit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium | {metric_value(medium, 'reward'):.4f} | {metric_value(medium, 'f1'):.4f} | {metric_value(medium, 'mae'):.5f} | {metric_value(medium, 'strict'):.2f} | {metric_value(medium, 'normalized'):.2f} | {metric_value(medium, 'done'):.2f} | {metric_value(medium, 'submit'):.2f} |
| hard | {metric_value(hard, 'reward'):.4f} | {metric_value(hard, 'f1'):.4f} | {metric_value(hard, 'mae'):.5f} | {metric_value(hard, 'strict'):.2f} | {metric_value(hard, 'normalized'):.2f} | {metric_value(hard, 'done'):.2f} | {metric_value(hard, 'submit'):.2f} |

This run preserves perfect strict execution and slightly improves the hard slice
over the previous best published adapter without regressing medium.
""".strip() + "\n"


def main() -> None:
    args = build_arg_parser().parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    resolved_init_adapter_dir = resolve_init_adapter_dir(args.init_adapter_dir, output_root)

    run_command(
        [
            sys.executable,
            "training/run_targeted_sft_experiment.py",
            "--profile",
            args.profile,
            "--model",
            args.model,
            "--init-adapter-dir",
            resolved_init_adapter_dir,
            "--output-root",
            str(output_root),
            "--episodes-per-difficulty",
            str(args.episodes_per_difficulty),
            "--train-difficulties",
            *args.train_difficulties,
            "--eval-difficulties",
            *args.eval_difficulties,
            "--seed-start",
            str(args.seed_start),
            "--eval-seed-start",
            str(args.eval_seed_start),
            "--eval-episodes",
            str(args.eval_episodes),
            "--max-scan-candidates-per-difficulty",
            str(args.max_scan_candidates_per_difficulty),
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
            "--output-json",
            str(summary_path),
        ]
    )

    summary = load_summary(summary_path)
    promotion_candidate = summary.get("promotion_candidate")
    if promotion_candidate != args.expected_promotion_candidate:
        raise RuntimeError(
            f"Expected promotion candidate {args.expected_promotion_candidate!r}, "
            f"got {promotion_candidate!r}"
        )

    candidates = summary["candidates"]
    winner = candidates[promotion_candidate]
    winner_dir = output_root / "train" / promotion_candidate
    publish_dir = output_root / "publish"
    copy_checkpoint_artifacts(winner_dir, publish_dir)

    model_card = build_model_card(
        model=args.model,
        init_adapter_dir=args.init_adapter_dir,
        target_repo=args.target_repo,
        promotion_candidate=promotion_candidate,
        source_job_id=args.source_job_id,
        source_commit=args.source_commit,
        medium=winner["medium"],
        hard=winner["hard"],
    )
    (publish_dir / "README.md").write_text(model_card, encoding="utf-8")
    shutil.copy2(summary_path, publish_dir / "promotion_summary.json")
    shutil.copy2(output_root / f"{promotion_candidate}_eval.json", publish_dir / "heldout_eval.json")
    shutil.copy2(output_root / f"{args.profile}.jsonl", publish_dir / "training_dataset.jsonl")

    upload_folder(
        repo_id=args.target_repo,
        repo_type="model",
        folder_path=str(publish_dir),
        commit_message="Upload promoted hard recall micro boost adapter",
    )

    print("PROMOTED_REPO", args.target_repo)
    print("PROMOTION_CANDIDATE", promotion_candidate)
    print("PROMOTION_EVAL", json.dumps({"medium": winner["medium"], "hard": winner["hard"]}, sort_keys=True))


if __name__ == "__main__":
    main()
