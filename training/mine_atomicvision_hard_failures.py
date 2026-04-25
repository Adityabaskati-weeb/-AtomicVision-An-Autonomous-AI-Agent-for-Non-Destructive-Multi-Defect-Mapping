"""Mine held-out hard-case failures for the current AtomicVision adapter."""

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

from atomicvision.synthetic import generate_case  # noqa: E402
from atomicvision_env.models import AtomicVisionAction  # noqa: E402
from atomicvision_env.server.environment import AtomicVisionEnvironment  # noqa: E402
from training.evaluate_atomicvision_adapter import (  # noqa: E402
    _load_model,
    _validate_heldout_seed_band,
    action_from_call,
    extract_tool_call,
    generate_tool_call,
    prior_submit_baseline,
    render_prompt,
    tool_response,
    user_message,
)
from training.seed_ranges import HELDOUT_EVAL_SEED_START, seed_policy_dict  # noqa: E402
from training.train_grpo_atomicvision import (  # noqa: E402
    TOOL_SYSTEM_PROMPT,
    _format_observation,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run held-out hard episodes, compare the current adapter against the "
            "prior-submit baseline, and bucket the real hard-case misses."
        )
    )
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=HELDOUT_EVAL_SEED_START)
    parser.add_argument("--max-tool-steps", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--mode", choices=("strict", "normalized"), default="strict")
    parser.add_argument(
        "--allow-non-heldout-seeds",
        action="store_true",
        help="Allow mining outside the official held-out seed band for debugging.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/error-mining/atomicvision_hard_error_mining.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/error-mining/atomicvision_hard_error_mining.md",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of worst regressions to include in the markdown report.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = mine_hard_errors(
        adapter_dir=Path(args.adapter_dir),
        base_model=args.base_model,
        difficulty=args.difficulty,
        episodes=args.episodes,
        seed_start=args.seed_start,
        max_tool_steps=args.max_tool_steps,
        max_new_tokens=args.max_new_tokens,
        mode=args.mode,
        allow_non_heldout_seeds=args.allow_non_heldout_seeds,
        top_k=args.top_k,
    )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown_report(report, top_k=args.top_k), encoding="utf-8")

    print(
        "Mined "
        f"{report['summary']['episodes']} episodes, "
        f"{report['summary']['worse_than_baseline_count']} worse than baseline, "
        f"mean reward delta {report['summary']['mean_reward_delta_vs_baseline']:.4f}."
    )
    print(f"Wrote JSON report to {output_json}")
    print(f"Wrote markdown report to {output_md}")


def mine_hard_errors(
    adapter_dir: Path,
    base_model: str,
    difficulty: str,
    episodes: int,
    seed_start: int,
    max_tool_steps: int,
    max_new_tokens: int,
    mode: str,
    allow_non_heldout_seeds: bool,
    top_k: int,
) -> dict[str, Any]:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    _validate_heldout_seed_band(
        seed_start=seed_start,
        episodes=episodes,
        allow_non_heldout_seeds=allow_non_heldout_seeds,
    )
    torch, tokenizer, model = _load_model(adapter_dir=adapter_dir, base_model=base_model)

    seeds = list(range(seed_start, seed_start + episodes))
    rows = [
        run_detailed_episode(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            seed=seed,
            difficulty=difficulty,
            max_tool_steps=max_tool_steps,
            max_new_tokens=max_new_tokens,
            mode=mode,
        )
        for seed in seeds
    ]
    rows.sort(key=lambda row: (row["reward_delta_vs_baseline"], row["seed"]))

    report = {
        "base_model": base_model,
        "adapter": str(adapter_dir),
        "difficulty": difficulty,
        "episodes": episodes,
        "seed_start": seed_start,
        "seed_policy": seed_policy_dict(),
        "heldout_seed_enforced": not allow_non_heldout_seeds,
        "mode": mode,
        "max_tool_steps": max_tool_steps,
        "max_new_tokens": max_new_tokens,
        "summary": summarize_mined_rows(rows),
        "episodes_by_regression": rows,
    }
    return report


def run_detailed_episode(
    torch: Any,
    tokenizer: Any,
    model: Any,
    seed: int,
    difficulty: str,
    max_tool_steps: int,
    max_new_tokens: int,
    mode: str,
) -> dict[str, Any]:
    case = generate_case(seed=seed, difficulty=difficulty)
    env = AtomicVisionEnvironment(difficulty=difficulty)
    obs = env.reset(seed=seed)
    obs_text = _format_observation(obs.model_dump())

    messages = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": user_message(obs_text)},
    ]

    actions: list[str] = []
    raw_outputs: list[str] = []
    normalized_outputs: list[str] = []
    strict_parse_count = 0
    normalized_parse_count = 0
    normalized_repair_count = 0
    repeated = 0
    tool_failure = False
    error = ""
    final_obs = obs
    final_submit: dict[str, Any] | None = None

    for _ in range(max_tool_steps):
        raw_text = generate_tool_call(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            messages=messages,
            max_new_tokens=max_new_tokens,
        )
        call, normalized_text, parse_metrics = extract_tool_call(raw_text, mode=mode)
        raw_outputs.append(raw_text)
        normalized_outputs.append(normalized_text)
        strict_parse_count += int(parse_metrics["strict_parse"])
        normalized_parse_count += int(parse_metrics["normalized_parse"])
        normalized_repair_count += int(parse_metrics["normalized_repair"])

        if call is None:
            tool_failure = True
            error = f"no valid tool_call after {mode} verification | raw={raw_text[:700]}"
            break

        name = call["name"]
        if actions and actions[-1] == name and name != "submit_defect_map":
            repeated += 1
        actions.append(name)

        if name == "submit_defect_map":
            final_submit = {
                "predicted_defects": list(call["arguments"].get("predicted_defects") or []),
                "predicted_concentrations": [
                    float(item)
                    for item in (call["arguments"].get("predicted_concentrations") or [])
                ],
                "confidence": float(call["arguments"].get("confidence") or 0.0),
            }

        try:
            action = action_from_call(call)
            final_obs = env.step(action)
        except Exception as exc:  # pragma: no cover - depends on live model output
            tool_failure = True
            error = f"action failed: {exc} | call={call}"
            break

        messages.append({"role": "assistant", "content": normalized_text})
        messages.append({"role": "user", "content": tool_response(_format_observation(final_obs.model_dump()))})

        if env.state.done:
            break

    reward_breakdown = final_obs.reward_breakdown or {}
    truth_map = {
        defect.species: round(float(defect.concentration), 6)
        for defect in case.defects
    }
    predicted_map = _prediction_map(final_submit)
    missing_species = sorted(set(truth_map) - set(predicted_map))
    extra_species = sorted(set(predicted_map) - set(truth_map))
    concentration_mae = float(reward_breakdown.get("concentration_mae", 1.0)) if reward_breakdown else 1.0
    f1 = float(reward_breakdown.get("f1", 0.0)) if reward_breakdown else 0.0
    confidence = float(final_submit["confidence"]) if final_submit is not None else 0.0
    confidence_gap = _confidence_accuracy_gap(
        confidence=confidence,
        f1=f1,
        concentration_mae=concentration_mae,
    )

    model_reward = float(final_obs.reward or 0.0) if env.state.done and not tool_failure else -1.0
    baseline = prior_submit_baseline(seed, difficulty)
    reward_delta = round(model_reward - float(baseline["reward"]), 6)

    row = {
        "seed": seed,
        "difficulty": difficulty,
        "truth_defects": truth_map,
        "predicted_defects": predicted_map,
        "confidence": confidence,
        "f1": f1,
        "concentration_mae": concentration_mae,
        "reward": model_reward,
        "baseline_reward": float(baseline["reward"]),
        "reward_delta_vs_baseline": reward_delta,
        "steps": int(env.state.step_count),
        "scan_cost": float(env.state.total_scan_cost),
        "done": bool(env.state.done),
        "tool_failure": bool(tool_failure),
        "repeated_tool_calls": int(repeated),
        "strict_parse_rate": float(strict_parse_count / max(1, len(raw_outputs))),
        "normalized_parse_rate": float(normalized_parse_count / max(1, len(raw_outputs))),
        "normalized_repair_rate": float(normalized_repair_count / max(1, len(raw_outputs))),
        "missing_species": missing_species,
        "extra_species": extra_species,
        "confidence_gap": confidence_gap,
        "action_pattern": classify_action_pattern(actions, env.state.done, tool_failure),
        "issue_bucket": classify_issue_bucket(
            tool_failure=tool_failure,
            done=bool(env.state.done),
            missing_species=missing_species,
            extra_species=extra_species,
            concentration_mae=concentration_mae,
            confidence_gap=confidence_gap,
        ),
        "actions": actions,
        "raw_outputs": raw_outputs,
        "normalized_outputs": normalized_outputs,
        "error": error,
    }
    return row


def summarize_mined_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts = Counter(row["issue_bucket"] for row in rows)
    action_pattern_counts = Counter(row["action_pattern"] for row in rows)
    worse_rows = [row for row in rows if row["reward_delta_vs_baseline"] < 0.0]
    worse_bucket_counts = Counter(row["issue_bucket"] for row in worse_rows)

    return {
        "episodes": len(rows),
        "mean_reward": _mean(row["reward"] for row in rows),
        "baseline_mean_reward": _mean(row["baseline_reward"] for row in rows),
        "mean_reward_delta_vs_baseline": _mean(
            row["reward_delta_vs_baseline"] for row in rows
        ),
        "mean_f1": _mean(row["f1"] for row in rows),
        "mean_concentration_mae": _mean(row["concentration_mae"] for row in rows),
        "worse_than_baseline_count": len(worse_rows),
        "equal_or_better_than_baseline_count": len(rows) - len(worse_rows),
        "issue_bucket_counts": dict(sorted(bucket_counts.items())),
        "issue_bucket_counts_when_worse_than_baseline": dict(
            sorted(worse_bucket_counts.items())
        ),
        "action_pattern_counts": dict(sorted(action_pattern_counts.items())),
        "top_negative_deltas": [
            {
                "seed": row["seed"],
                "reward_delta_vs_baseline": row["reward_delta_vs_baseline"],
                "issue_bucket": row["issue_bucket"],
                "action_pattern": row["action_pattern"],
            }
            for row in rows[:10]
        ],
    }


def build_markdown_report(report: dict[str, Any], top_k: int) -> str:
    summary = report["summary"]
    rows = report["episodes_by_regression"][:top_k]
    bucket_lines = "\n".join(
        f"- `{bucket}`: `{count}`"
        for bucket, count in summary["issue_bucket_counts_when_worse_than_baseline"].items()
    ) or "- none"

    top_rows = "\n".join(
        (
            f"| {row['seed']} | {row['reward_delta_vs_baseline']:.4f} | "
            f"{row['issue_bucket']} | {row['action_pattern']} | "
            f"{','.join(row['missing_species']) or '-'} | "
            f"{','.join(row['extra_species']) or '-'} | "
            f"{row['concentration_mae']:.4f} | {row['confidence_gap']:.4f} |"
        )
        for row in rows
    )
    if not top_rows:
        top_rows = "| - | 0.0000 | none | none | - | - | 0.0000 | 0.0000 |"

    return "\n".join(
        [
            "# AtomicVision Hard Error Mining",
            "",
            "## Run Setup",
            "",
            f"- Adapter: `{report['adapter']}`",
            f"- Base model: `{report['base_model']}`",
            f"- Difficulty: `{report['difficulty']}`",
            f"- Episodes: `{report['episodes']}`",
            f"- Seed start: `{report['seed_start']}`",
            f"- Mode: `{report['mode']}`",
            "",
            "## Summary",
            "",
            f"- mean reward: `{summary['mean_reward']:.4f}`",
            f"- baseline mean reward: `{summary['baseline_mean_reward']:.4f}`",
            f"- mean reward delta vs baseline: `{summary['mean_reward_delta_vs_baseline']:.4f}`",
            f"- worse than baseline: `{summary['worse_than_baseline_count']}` / `{summary['episodes']}`",
            "",
            "## Buckets Among Regressed Seeds",
            "",
            bucket_lines,
            "",
            "## Top Regressions",
            "",
            "| seed | delta vs baseline | issue bucket | action pattern | missing | extra | mae | confidence gap |",
            "| --- | ---: | --- | --- | --- | --- | ---: | ---: |",
            top_rows,
        ]
    )


def classify_issue_bucket(
    *,
    tool_failure: bool,
    done: bool,
    missing_species: list[str],
    extra_species: list[str],
    concentration_mae: float,
    confidence_gap: float,
) -> str:
    if tool_failure:
        return "tool_failure"
    if not done:
        return "no_terminal_submit"
    if missing_species and extra_species:
        return "mixed_species_error"
    if missing_species:
        return "missed_defects"
    if extra_species:
        return "false_positive_defects"
    if concentration_mae >= 0.03:
        return "concentration_error"
    if confidence_gap >= 0.25:
        return "confidence_miscalibration"
    return "clean_success"


def classify_action_pattern(
    actions: list[str],
    done: bool,
    tool_failure: bool,
) -> str:
    if tool_failure:
        return "tool_failure"
    if not done or "submit_defect_map" not in actions:
        return "no_submit"
    if actions == ["ask_prior", "submit_defect_map"]:
        return "prior_then_submit"
    if "compare_reference" in actions:
        return "reference_then_submit"
    if any(action in {"request_scan", "zoom_band"} for action in actions):
        return "scan_then_submit"
    return "other_submit_path"


def _prediction_map(final_submit: dict[str, Any] | None) -> dict[str, float]:
    if final_submit is None:
        return {}

    predictions: dict[str, float] = {}
    for species, concentration in zip(
        final_submit["predicted_defects"],
        final_submit["predicted_concentrations"],
        strict=True,
    ):
        predictions[species] = round(float(concentration), 6)
    return predictions


def _confidence_accuracy_gap(
    *,
    confidence: float,
    f1: float,
    concentration_mae: float,
) -> float:
    concentration_score = max(0.0, 1.0 - concentration_mae / 0.25)
    accuracy_proxy = 0.5 * f1 + 0.5 * concentration_score
    return round(abs(confidence - accuracy_proxy), 6)


def _mean(values: Any) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
