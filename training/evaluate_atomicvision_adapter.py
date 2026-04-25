"""Held-out adapter evaluation for AtomicVision with strict and normalized verifier modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomicvision_env.models import AtomicVisionAction  # noqa: E402
from atomicvision_env.server.environment import AtomicVisionEnvironment  # noqa: E402
from atomicvision.rewards import reward_component_dict, reward_source_totals  # noqa: E402
from training.train_grpo_atomicvision import (  # noqa: E402
    DEFAULT_PROMPT,
    TOOL_SYSTEM_PROMPT,
    _format_observation,
    canonicalize_tool_call_text,
    parse_strict_tool_call,
)
from training.seed_ranges import (  # noqa: E402
    HELDOUT_EVAL_BAND,
    HELDOUT_EVAL_SEED_START,
    seed_policy_dict,
)

EVAL_MODES = ("strict", "normalized")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an AtomicVision LoRA adapter.")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--difficulties", nargs="+", default=["medium", "hard"])
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=HELDOUT_EVAL_SEED_START)
    parser.add_argument("--max-tool-steps", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument(
        "--allow-non-heldout-seeds",
        action="store_true",
        help=(
            "Allow evaluation outside the official held-out seed band. "
            "Use only for debugging or historical comparisons."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=EVAL_MODES,
        default=list(EVAL_MODES),
        help="Verifier modes to run. Default runs both strict and normalized.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/evaluation/atomicvision_adapter_eval.json",
        help="Path for the JSON report.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = evaluate_adapter(
        adapter_dir=Path(args.adapter_dir),
        base_model=args.base_model,
        difficulties=tuple(args.difficulties),
        episodes=args.episodes,
        seed_start=args.seed_start,
        max_tool_steps=args.max_tool_steps,
        max_new_tokens=args.max_new_tokens,
        modes=tuple(args.modes),
        allow_non_heldout_seeds=args.allow_non_heldout_seeds,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(_table(report))
    print(f"\nWrote JSON report to {output_path}")


def evaluate_adapter(
    adapter_dir: Path,
    base_model: str,
    difficulties: tuple[str, ...],
    episodes: int,
    seed_start: int,
    max_tool_steps: int,
    max_new_tokens: int,
    modes: tuple[str, ...] = EVAL_MODES,
    allow_non_heldout_seeds: bool = False,
) -> dict[str, Any]:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    unknown_modes = [mode for mode in modes if mode not in EVAL_MODES]
    if unknown_modes:
        raise ValueError(f"Unknown modes: {', '.join(unknown_modes)}")
    _validate_heldout_seed_band(
        seed_start=seed_start,
        episodes=episodes,
        allow_non_heldout_seeds=allow_non_heldout_seeds,
    )

    torch, tokenizer, model = _load_model(adapter_dir=adapter_dir, base_model=base_model)

    report = {
        "base_model": base_model,
        "adapter": str(adapter_dir),
        "episodes_per_difficulty": episodes,
        "seed_start": seed_start,
        "seed_policy": seed_policy_dict(),
        "heldout_seed_enforced": not allow_non_heldout_seeds,
        "max_tool_steps": max_tool_steps,
        "max_new_tokens": max_new_tokens,
        "modes": list(modes),
        "results": {},
    }

    for difficulty in difficulties:
        seeds = list(range(seed_start, seed_start + episodes))
        baseline_rows = [prior_submit_baseline(seed, difficulty) for seed in seeds]
        difficulty_result: dict[str, Any] = {
            "baseline_prior_submit": summarize(baseline_rows),
        }
        for mode in modes:
            mode_rows = [
                run_model_episode(
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
            difficulty_result[f"{mode}_adapter"] = summarize(mode_rows)
            difficulty_result[f"{mode}_failures"] = [
                {
                    "seed": seed,
                    "actions": row["actions"],
                    "error": row["error"],
                    "raw_outputs": row["raw_outputs"],
                    "normalized_outputs": row["normalized_outputs"],
                }
                for seed, row in zip(seeds, mode_rows, strict=True)
                if row["tool_failure"] or not row["done"]
            ][:10]
        report["results"][difficulty] = difficulty_result
    return report


def _validate_heldout_seed_band(
    seed_start: int,
    episodes: int,
    allow_non_heldout_seeds: bool,
) -> None:
    if allow_non_heldout_seeds:
        return

    seed_stop = seed_start + episodes
    if seed_start < HELDOUT_EVAL_BAND.start or seed_stop > HELDOUT_EVAL_BAND.stop:
        raise ValueError(
            "Held-out eval seeds must stay inside the official eval band "
            f"{HELDOUT_EVAL_BAND.label}. "
            "Pass --allow-non-heldout-seeds only for debugging."
        )


def _load_model(adapter_dir: Path, base_model: str):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing evaluation dependencies. Install training/requirements-grpo.txt plus bitsandbytes."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.eval()
    return torch, tokenizer, model


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def user_message(observation_text: str) -> str:
    return f"{DEFAULT_PROMPT}\n\nObservation:\n{observation_text}"


def tool_response(observation_text: str) -> str:
    return f"<tool_response>\n{observation_text}\n</tool_response>"


def generate_tool_call(
    torch: Any,
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    prompt = render_prompt(tokenizer, messages)
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_tool_call(text: str, mode: str) -> tuple[dict[str, Any] | None, str, dict[str, float]]:
    strict_call = parse_strict_tool_call(text)
    normalized_text = canonicalize_tool_call_text(text)
    normalized_call = parse_strict_tool_call(normalized_text)

    if mode == "strict":
        active_call = strict_call
        active_text = text
    elif mode == "normalized":
        active_call = normalized_call
        active_text = normalized_text
    else:
        raise ValueError(f"Unknown mode: {mode}")

    metrics = {
        "strict_parse": 1.0 if strict_call is not None else 0.0,
        "normalized_parse": 1.0 if normalized_call is not None else 0.0,
        "normalized_repair": 1.0 if strict_call is None and normalized_call is not None else 0.0,
    }
    return active_call, active_text, metrics


def action_from_call(call: dict[str, Any]) -> AtomicVisionAction:
    name = call["name"]
    args = dict(call["arguments"])
    if name == "ask_prior":
        return AtomicVisionAction(action_type="ask_prior")
    if name == "compare_reference":
        return AtomicVisionAction(action_type="compare_reference")
    if name == "request_scan":
        return AtomicVisionAction(
            action_type="request_scan",
            scan_mode=args.get("scan_mode") or "standard_pdos",
            resolution=args.get("resolution") or "medium",
        )
    if name == "zoom_band":
        return AtomicVisionAction(
            action_type="zoom_band",
            scan_mode=args.get("scan_mode") or "high_res_pdos",
            resolution=args.get("resolution") or "high",
            freq_min=args.get("freq_min"),
            freq_max=args.get("freq_max"),
        )
    if name == "submit_defect_map":
        return AtomicVisionAction(
            action_type="submit_defect_map",
            predicted_defects=list(args.get("predicted_defects") or []),
            predicted_concentrations=[float(item) for item in (args.get("predicted_concentrations") or [])],
            confidence=float(args.get("confidence") if args.get("confidence") is not None else 0.0),
        )
    raise ValueError(f"Unsupported tool name: {name}")


def prior_submit_baseline(seed: int, difficulty: str) -> dict[str, Any]:
    env = AtomicVisionEnvironment(difficulty=difficulty)
    obs = env.reset(seed=seed)
    obs = env.step(AtomicVisionAction(action_type="ask_prior"))
    prior = obs.prior_prediction
    if prior is None:
        action = AtomicVisionAction(
            action_type="submit_defect_map",
            predicted_defects=[],
            predicted_concentrations=[],
            confidence=0.45,
        )
    else:
        action = AtomicVisionAction(
            action_type="submit_defect_map",
            predicted_defects=list(prior.predicted_defects),
            predicted_concentrations=list(prior.predicted_concentrations),
            confidence=float(prior.confidence),
        )
    obs = env.step(action)
    reward_breakdown = obs.reward_breakdown or {}
    return {
        "reward": float(obs.reward or 0.0),
        "f1": float(reward_breakdown.get("f1", 0.0)),
        "mae": float(reward_breakdown.get("concentration_mae", 1.0)),
        "steps": int(env.state.step_count),
        "scan_cost": float(env.state.total_scan_cost),
        "done": bool(env.state.done),
        "tool_failure": False,
        "repeated_tool_calls": 0,
        "strict_parse_rate": 1.0,
        "normalized_parse_rate": 1.0,
        "normalized_repair_rate": 0.0,
        "first_action_valid_rate": 1.0,
        "first_action_ask_prior_rate": 1.0,
        "submit_action_rate": 1.0,
        "identity_reward": float(reward_breakdown.get("identity_reward", 0.0)),
        "concentration_reward": float(reward_breakdown.get("concentration_reward", 0.0)),
        "confidence_reward": float(reward_breakdown.get("confidence_reward", 0.0)),
        "false_positive_penalty": float(reward_breakdown.get("false_positive_penalty", 0.0)),
        "missed_defect_penalty": float(reward_breakdown.get("missed_defect_penalty", 0.0)),
        "timeout_penalty": float(reward_breakdown.get("timeout_penalty", 0.0)),
        "outcome_reward_total": float(reward_source_totals(reward_breakdown)["outcome_reward_total"]),
        "penalty_total": float(reward_source_totals(reward_breakdown)["penalty_total"]),
        "actions": ["ask_prior", "submit_defect_map"],
        "raw_outputs": [],
        "normalized_outputs": [],
        "error": "",
    }


def run_model_episode(
    torch: Any,
    tokenizer: Any,
    model: Any,
    seed: int,
    difficulty: str,
    max_tool_steps: int,
    max_new_tokens: int,
    mode: str,
) -> dict[str, Any]:
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
    repeated = 0
    strict_parse_count = 0
    normalized_parse_count = 0
    normalized_repair_count = 0
    first_action_valid = 0.0
    first_action_is_prior = 0.0
    submit_action_seen = 0.0
    tool_failure = False
    error = ""
    final_obs = obs

    for step_index in range(max_tool_steps):
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
            if step_index == 0:
                first_action_valid = 0.0
            tool_failure = True
            error = f"no valid tool_call after {mode} verification | raw={raw_text[:700]}"
            break

        if step_index == 0:
            first_action_valid = 1.0
            first_action_is_prior = 1.0 if call["name"] == "ask_prior" else 0.0

        name = call["name"]
        if actions and actions[-1] == name and name != "submit_defect_map":
            repeated += 1
        actions.append(name)
        if name == "submit_defect_map":
            submit_action_seen = 1.0

        try:
            action = action_from_call(call)
            final_obs = env.step(action)
        except Exception as exc:
            tool_failure = True
            error = f"action failed: {exc} | call={call}"
            break

        messages.append({"role": "assistant", "content": normalized_text})
        messages.append({"role": "user", "content": tool_response(_format_observation(final_obs.model_dump()))})

        if env.state.done:
            break

    reward_breakdown = final_obs.reward_breakdown or {}
    component_values = reward_component_dict(reward_breakdown if env.state.done and not tool_failure else None)
    source_totals = reward_source_totals(reward_breakdown if env.state.done and not tool_failure else None)
    attempted_turns = max(1, len(raw_outputs))
    return {
        "reward": float(final_obs.reward or 0.0) if env.state.done and not tool_failure else -1.0,
        "f1": float(reward_breakdown.get("f1", 0.0)) if env.state.done and not tool_failure else 0.0,
        "mae": float(reward_breakdown.get("concentration_mae", 1.0)) if env.state.done and not tool_failure else 1.0,
        "steps": int(env.state.step_count),
        "scan_cost": float(env.state.total_scan_cost),
        "done": bool(env.state.done),
        "tool_failure": bool(tool_failure),
        "repeated_tool_calls": int(repeated),
        "strict_parse_rate": float(strict_parse_count / attempted_turns),
        "normalized_parse_rate": float(normalized_parse_count / attempted_turns),
        "normalized_repair_rate": float(normalized_repair_count / attempted_turns),
        "first_action_valid_rate": float(first_action_valid),
        "first_action_ask_prior_rate": float(first_action_is_prior),
        "submit_action_rate": float(submit_action_seen),
        "identity_reward": component_values["identity_reward"],
        "concentration_reward": component_values["concentration_reward"],
        "confidence_reward": component_values["confidence_reward"],
        "false_positive_penalty": component_values["false_positive_penalty"],
        "missed_defect_penalty": component_values["missed_defect_penalty"],
        "timeout_penalty": component_values["timeout_penalty"],
        "outcome_reward_total": source_totals["outcome_reward_total"],
        "penalty_total": source_totals["penalty_total"],
        "actions": actions,
        "raw_outputs": raw_outputs,
        "normalized_outputs": normalized_outputs,
        "error": error,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "mean_reward": mean(row["reward"] for row in rows),
        "mean_f1": mean(row["f1"] for row in rows),
        "mean_mae": mean(row["mae"] for row in rows),
        "mean_steps": mean(row["steps"] for row in rows),
        "mean_scan_cost": mean(row["scan_cost"] for row in rows),
        "done_rate": mean(1.0 if row["done"] else 0.0 for row in rows),
        "tool_failure_rate": mean(1.0 if row["tool_failure"] else 0.0 for row in rows),
        "mean_repeated_tool_calls": mean(row["repeated_tool_calls"] for row in rows),
        "strict_tool_call_pass_rate": mean(row["strict_parse_rate"] for row in rows),
        "normalized_tool_call_pass_rate": mean(row["normalized_parse_rate"] for row in rows),
        "normalized_tool_call_repair_rate": mean(row["normalized_repair_rate"] for row in rows),
        "first_action_valid_rate": mean(row["first_action_valid_rate"] for row in rows),
        "first_action_ask_prior_rate": mean(row["first_action_ask_prior_rate"] for row in rows),
        "submit_action_rate": mean(row["submit_action_rate"] for row in rows),
        "mean_identity_reward": mean(row["identity_reward"] for row in rows),
        "mean_concentration_reward": mean(row["concentration_reward"] for row in rows),
        "mean_confidence_reward": mean(row["confidence_reward"] for row in rows),
        "mean_false_positive_penalty": mean(row["false_positive_penalty"] for row in rows),
        "mean_missed_defect_penalty": mean(row["missed_defect_penalty"] for row in rows),
        "mean_timeout_penalty": mean(row["timeout_penalty"] for row in rows),
        "mean_outcome_reward_total": mean(row["outcome_reward_total"] for row in rows),
        "mean_penalty_total": mean(row["penalty_total"] for row in rows),
    }


def mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _table(report: dict[str, Any]) -> str:
    lines = [
        "| difficulty | policy | reward | f1 | mae | steps | cost | fail | done | repeat | strict | normalized | first_valid | first_prior | submit |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for difficulty, result in report["results"].items():
        row = result["baseline_prior_submit"]
        lines.append(_table_row(difficulty, "baseline", row))
        for mode in report["modes"]:
            row = result[f"{mode}_adapter"]
            lines.append(_table_row(difficulty, f"{mode}_adapter", row))
    return "\n".join(lines)


def _table_row(difficulty: str, label: str, row: dict[str, Any]) -> str:
    return (
        f"| {difficulty} | {label} | "
        f"{row['mean_reward']:.4f} | {row['mean_f1']:.4f} | {row['mean_mae']:.5f} | "
        f"{row['mean_steps']:.2f} | {row['mean_scan_cost']:.2f} | "
        f"{row['tool_failure_rate']:.3f} | {row['done_rate']:.3f} | "
        f"{row['mean_repeated_tool_calls']:.2f} | "
        f"{row['strict_tool_call_pass_rate']:.2f} | {row['normalized_tool_call_pass_rate']:.2f} | "
        f"{row['first_action_valid_rate']:.2f} | {row['first_action_ask_prior_rate']:.2f} | "
        f"{row['submit_action_rate']:.2f} |"
    )


if __name__ == "__main__":
    main()
