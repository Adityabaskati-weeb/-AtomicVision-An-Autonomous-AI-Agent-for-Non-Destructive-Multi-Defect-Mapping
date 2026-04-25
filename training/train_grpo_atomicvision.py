"""Phase 11 GRPO fine-tuning scaffold for AtomicVision.

This script is designed for Colab/Kaggle GPU runtimes. It follows TRL's
OpenEnv `environment_factory` pattern, exposing meaningful tools rather than a
generic `step(action)` method.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomicvision_env.client import AtomicVisionEnv
from atomicvision_env.models import AtomicVisionAction
from atomicvision.rewards import reward_component_dict, reward_source_totals, score_submission
from atomicvision_env.server.environment import AtomicVisionEnvironment
from training.seed_ranges import GRPO_TRAIN_SEED_START


DEFAULT_ENV_URL = "https://prodigyhuh-atomicvision-openenv.hf.space"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
POST_TERMINAL_TOOL_PENALTY = 2.0
VALID_TOOL_CALL_FORMAT_REWARD = 0.15
RECOVERABLE_TOOL_CALL_FORMAT_PENALTY = 0.10
INVALID_TOOL_CALL_FORMAT_PENALTY = 0.75
STRICT_FINAL_SUBMIT_FORMAT_REWARD = 0.05
RECOVERABLE_FINAL_SUBMIT_FORMAT_PENALTY = 0.0
EXACT_PRIOR_COPY_REWARD = 0.05
CONFIDENT_PRIOR_MIS_COPY_PENALTY = 0.25
CONFIDENT_PRIOR_COPY_THRESHOLD = 0.65
PRIOR_SUBMIT_THRESHOLD = 0.50
ScanMode = Literal["quick_pdos", "standard_pdos", "high_res_pdos", "raman_proxy"]
Resolution = Literal["low", "medium", "high"]
VALID_SCAN_MODES = ("quick_pdos", "standard_pdos", "high_res_pdos", "raman_proxy")
VALID_RESOLUTIONS = ("low", "medium", "high")
VALID_TOOL_NAMES = (
    "ask_prior",
    "compare_reference",
    "request_scan",
    "zoom_band",
    "submit_defect_map",
)
PROMPT_FOCI = (
    "all",
    "heldout",
    "borderline",
    "reference-improvement",
    "grpo-frontier",
)
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
TOOL_SYSTEM_PROMPT = (
    "You are using AtomicVision tools. Return exactly one XML-wrapped JSON tool call "
    "and nothing else. The only valid format is "
    '<tool_call>{"name":"ask_prior","arguments":{}}</tool_call> '
    "or "
    '<tool_call>{"name":"submit_defect_map","arguments":{"predicted_defects":["O"],'
    '"predicted_concentrations":[0.12],"confidence":0.73}}</tool_call>. '
    "Do not write placeholder text or ellipsis examples inside the tool-call tags. "
    "Do not write ask_prior or submit_defect_map outside the JSON object. "
    "Do not emit reasoning markup, assistant labels, or prose before or after the "
    "tool call. "
    "Use ask_prior first. After the prior appears, copy high-confidence priors into "
    "submit_defect_map. For borderline priors, one cheap valid scan or "
    "compare_reference call is allowed if it can improve the final map. "
    "Do not invent unsupported tool names, species, or concentration formats. "
    "Stop immediately after the closing </tool_call> tag."
)
DEFAULT_PROMPT = (
    "You are AtomicVision, an autonomous materials characterization agent. "
    "Your task is to infer hidden atomic defects from non-invasive spectral evidence. "
    "Maximize reward by submitting accurate defect identities and concentrations while "
    "avoiding unnecessary scan cost. "
    "Observations include compact spectral summaries: current peaks, defect-reference "
    "delta peaks when a reference is visible, and candidate signature bands/scores. "
    "Tool protocol: valid scan_mode values are exactly quick_pdos, standard_pdos, "
    "high_res_pdos, and raman_proxy. Valid resolution values are exactly low, medium, "
    "and high. The frequency axis is synthetic PDoS units from 0.0 to 20.0; valid "
    "zoom examples are 1.0-5.0, 5.0-10.0, and 10.0-18.0. Never use Raman/cm^-1 "
    "values such as 300, 1200, or 3000 for zoom_band. Strong default strategy: "
    "first call ask_prior. If the prior returns predicted_defects with confidence "
    "0.65 or higher, usually submit_defect_map using the prior predicted_defects, "
    "predicted_concentrations, and confidence. If confidence is 0.50-0.65, either "
    "submit the prior or request one cheap extra signal only if it can improve the "
    "final map. "
    "When you submit, output one XML-wrapped tool call and stop; do not add "
    "assistant labels, reasoning markup, or extra text. "
    "Extra evidence is expensive: ask_prior costs 1.5, compare_reference costs "
    "0.5, quick_pdos costs 1.0, standard_pdos costs 2.0, raman_proxy costs 2.5, "
    "and high_res_pdos or zoom_band costs 4.0. Request another scan only when the "
    "prior is missing, low-confidence, or you will change the final defect map. "
    "If requesting extra evidence, use at most one valid scan or one valid zoom "
    "band before submitting. submit_defect_map is terminal: after it returns, "
    "do not call any more tools. Extra tool calls after final submission are penalized."
)
TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "model": "Qwen/Qwen3-0.6B",
        "samples": 32,
        "max_steps": 5,
        "num_generations": 4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_completion_length": 1024,
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 50,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "run_name": "atomicvision-grpo-smoke",
    },
    "variance-probe": {
        "model": "Qwen/Qwen3-1.7B",
        "samples": 32,
        "seed_start": GRPO_TRAIN_SEED_START,
        "prompt_focus": "grpo-frontier",
        "max_seed_candidates": 1024,
        "max_steps": 3,
        "num_generations": 8,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_completion_length": 768,
        "temperature": 1.25,
        "top_p": 0.95,
        "top_k": 50,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "run_name": "atomicvision-grpo-variance-probe",
    },
    "cost-aware-variance-probe": {
        "model": "Qwen/Qwen3-1.7B",
        "samples": 32,
        "seed_start": GRPO_TRAIN_SEED_START,
        "prompt_focus": "grpo-frontier",
        "max_seed_candidates": 1024,
        "max_steps": 3,
        "num_generations": 8,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_completion_length": 768,
        "temperature": 1.25,
        "top_p": 0.95,
        "top_k": 50,
        "learning_rate": 1.0e-6,
        "scale_rewards": "batch",
        "loss_type": "dapo",
        "use_peft": True,
        "run_name": "atomicvision-cost-aware-grpo-variance-probe",
    },
    "cost-aware-grpo-20": {
        "model": "Qwen/Qwen3-1.7B",
        "samples": 64,
        "seed_start": GRPO_TRAIN_SEED_START,
        "prompt_focus": "grpo-frontier",
        "max_seed_candidates": 2048,
        "max_steps": 20,
        "num_generations": 8,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_completion_length": 768,
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 50,
        "learning_rate": 1.0e-6,
        "scale_rewards": "batch",
        "loss_type": "dapo",
        "use_peft": True,
        "run_name": "atomicvision-cost-aware-grpo-20step",
    },
    "cost-aware-grpo-100": {
        "model": "Qwen/Qwen3-1.7B",
        "samples": 256,
        "seed_start": GRPO_TRAIN_SEED_START,
        "prompt_focus": "grpo-frontier",
        "max_seed_candidates": 4096,
        "max_steps": 100,
        "num_generations": 8,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_completion_length": 768,
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 50,
        "learning_rate": 8.0e-7,
        "scale_rewards": "batch",
        "loss_type": "dapo",
        "use_peft": True,
        "run_name": "atomicvision-cost-aware-grpo-100step",
    },
    "colab-20": {
        "model": "Qwen/Qwen3-0.6B",
        "samples": 64,
        "max_steps": 20,
        "num_generations": 4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_completion_length": 1024,
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 50,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "run_name": "atomicvision-grpo-0p6b-20step",
    },
    "qwen-1p7b-50": {
        "model": "Qwen/Qwen3-1.7B",
        "samples": 128,
        "max_steps": 50,
        "num_generations": 8,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_completion_length": 768,
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 50,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "run_name": "atomicvision-grpo-1p7b-50step",
    },
    "hf-4b-100": {
        "model": "Qwen/Qwen3-4B",
        "samples": 256,
        "max_steps": 100,
        "num_generations": 8,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_completion_length": 768,
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 50,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "run_name": "atomicvision-grpo-4b-100step",
    },
}


class AtomicVisionToolEnv:
    """TRL environment factory wrapper for AtomicVision."""

    max_retries = 3
    retry_sleep_seconds = 1.5

    def __init__(self):
        self.client = None
        self.reward = 0.0
        self.done = False
        self.last_message = ""
        self.last_prior_prediction: dict[str, Any] | None = None
        self.last_submit_action: AtomicVisionAction | None = None
        self.last_reward_breakdown: dict[str, float] | None = None
        self._connected = False
        self.post_terminal_tool_calls = 0

    def reset(self, **kwargs) -> str:
        seed = kwargs.get("seed")
        difficulty = kwargs.get("difficulty", "medium")
        result = self._reset_with_retry(seed=seed, difficulty=difficulty)
        observation = result.observation
        self.reward = float(result.reward or 0.0)
        self.done = result.done
        self.last_message = observation.message
        self.last_prior_prediction = None
        self.last_submit_action = None
        self.last_reward_breakdown = (
            dict(observation.reward_breakdown) if observation.reward_breakdown is not None else None
        )
        self.post_terminal_tool_calls = 0
        return _format_observation(observation.model_dump())

    def request_scan(
        self,
        scan_mode: ScanMode = "standard_pdos",
        resolution: Resolution = "medium",
    ) -> str:
        """
        Request a non-invasive full-spectrum scan.

        Args:
            scan_mode: Must be exactly one of quick_pdos, standard_pdos, high_res_pdos, or raman_proxy.
                Costs are quick_pdos=1.0, standard_pdos=2.0, high_res_pdos=4.0, raman_proxy=2.5.
                Prefer not to call this after ask_prior unless it can improve the submitted map.
            resolution: Must be exactly one of low, medium, or high.

        Returns:
            A concise summary of the new scan, budget, and reward.
        """
        return self._step(
            AtomicVisionAction(
                action_type="request_scan",
                scan_mode=scan_mode,
                resolution=resolution,
            )
        )

    def zoom_band(
        self,
        freq_min: float,
        freq_max: float,
        resolution: Resolution = "high",
    ) -> str:
        """
        Request a higher-detail scan over one frequency band.

        Args:
            freq_min: Lower frequency bound in synthetic units. Must be within 0.0 to 20.0.
            freq_max: Upper frequency bound in synthetic units. Must be within 0.0 to 20.0 and greater than freq_min.
                A zoom costs 4.0, so use at most one and only when the prior is weak.
            resolution: Must be exactly one of low, medium, or high.

        Returns:
            A concise summary of the zoom scan, budget, and reward.
        """
        return self._step(
            AtomicVisionAction(
                action_type="zoom_band",
                freq_min=freq_min,
                freq_max=freq_max,
                resolution=resolution,
            )
        )

    def compare_reference(self) -> str:
        """
        Reveal the pristine reference spectrum for comparison.

        Returns:
            A concise summary of the reference comparison result.
        """
        return self._step(AtomicVisionAction(action_type="compare_reference"))

    def ask_prior(self) -> str:
        """
        Ask DefectNet-lite for a candidate defect map.

        Returns:
            Candidate defect species, concentrations, confidence, and current budget.
            If confidence is 0.50 or higher, usually submit this map immediately.
        """
        return self._step(AtomicVisionAction(action_type="ask_prior"))

    def submit_defect_map(
        self,
        predicted_defects: list[str],
        predicted_concentrations: list[float],
        confidence: float,
    ) -> str:
        """
        Submit the final defect map and end the episode.

        Args:
            predicted_defects: List of predicted defect species.
            predicted_concentrations: Concentrations matching predicted_defects.
            confidence: Confidence from 0.0 to 1.0.

        Returns:
            Final reward and reward breakdown. This is a terminal tool; after it
            returns, stop calling tools and respond with a short final sentence.
        """
        return self._step(
            AtomicVisionAction(
                action_type="submit_defect_map",
                predicted_defects=predicted_defects,
                predicted_concentrations=predicted_concentrations,
                confidence=confidence,
            )
        )

    def _close(self) -> None:
        if self._connected:
            try:
                self.client.close()
            except Exception:
                pass
            self._connected = False

    def _step(self, action: AtomicVisionAction) -> str:
        if self.client is None or not self._connected:
            raise ValueError("Environment is not connected. Call reset() before using tools.")
        if self.done:
            self.post_terminal_tool_calls += 1
            penalty = POST_TERMINAL_TOOL_PENALTY * self.post_terminal_tool_calls
            self.reward -= POST_TERMINAL_TOOL_PENALTY
            self.last_message = "Episode already ended after final submission."
            return (
                "episode_done=True\n"
                "message=Episode already ended after submit_defect_map. "
                "Stop calling tools and return a concise final answer.\n"
                f"post_terminal_tool_calls={self.post_terminal_tool_calls}\n"
                f"post_terminal_penalty=-{penalty:.3f}\n"
                f"reward={self.reward:.6f} done=True"
            )
        result = self._step_with_retry(action)
        observation = result.observation
        self.reward = float(result.reward or 0.0)
        self.done = result.done
        self.last_message = observation.message
        self.last_reward_breakdown = (
            dict(observation.reward_breakdown) if observation.reward_breakdown is not None else None
        )
        prior = observation.prior_prediction
        if prior is not None:
            self.last_prior_prediction = prior.model_dump()
        if action.action_type == "submit_defect_map":
            self.last_submit_action = action
        return _format_observation(observation.model_dump())

    def _new_client(self):
        return AtomicVisionEnv(base_url=_env_url()).sync()

    def _ensure_connected(self) -> None:
        if self.client is None:
            self.client = self._new_client()
        if not self._connected:
            self.client.connect()
            self._connected = True

    def _reset_connection(self) -> None:
        self._close()
        self.client = None

    def _reset_with_retry(self, seed, difficulty: str):
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Start each episode with a fresh WebSocket session. This avoids
                # stale sockets when TRL reuses environment objects across rollouts.
                self._reset_connection()
                self._ensure_connected()
                return self.client.reset(seed=seed, difficulty=difficulty)
            except Exception as exc:
                last_error = exc
                self._reset_connection()
                if not _is_retryable_connection_error(exc) or attempt == self.max_retries:
                    break
                time.sleep(self.retry_sleep_seconds * attempt)
        raise RuntimeError(
            "AtomicVision environment reset failed after reconnect attempts. "
            "Use lower rollout concurrency: --per-device-train-batch-size 2 "
            "--gradient-accumulation-steps 1 --num-generations 2."
        ) from last_error

    def _step_with_retry(self, action: AtomicVisionAction):
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.step(action)
            except Exception as exc:
                last_error = exc
                if not _is_retryable_connection_error(exc) or attempt == self.max_retries:
                    break
                time.sleep(self.retry_sleep_seconds * attempt)
        raise RuntimeError(
            "AtomicVision environment step failed after reconnect attempts. "
            "The current episode socket closed during tool execution."
        ) from last_error


def reward_func(environments, **kwargs) -> list[float]:
    """Return AtomicVision rewards plus small tool-format/copy shaping."""

    completion_texts = _extract_completion_texts(kwargs, expected_count=len(environments))
    env_rewards: list[float] = []
    format_rewards: list[float] = []
    copy_rewards: list[float] = []
    done_values: list[float] = []
    post_terminal_values: list[float] = []
    strict_parse_values: list[float] = []
    normalized_parse_values: list[float] = []
    normalized_repair_values: list[float] = []
    ask_prior_values: list[float] = []
    submit_values: list[float] = []
    identity_rewards: list[float] = []
    concentration_rewards: list[float] = []
    confidence_rewards: list[float] = []
    false_positive_penalties: list[float] = []
    missed_defect_penalties: list[float] = []
    scan_cost_penalties: list[float] = []
    timeout_penalties: list[float] = []
    outcome_reward_totals: list[float] = []
    penalty_totals: list[float] = []
    process_shaping_rewards: list[float] = []
    strict_submit_rewards: list[float] = []
    rewards: list[float] = []
    for index, env in enumerate(environments):
        completion_text = completion_texts[index] if index < len(completion_texts) else ""
        env_reward = float(env.reward)
        format_reward = _tool_call_format_reward(completion_text)
        copy_reward = _prior_copy_reward(env)
        strict_submit_reward = _strict_terminal_submit_reward(completion_text, env)
        component_values = reward_component_dict(getattr(env, "last_reward_breakdown", None))
        source_totals = reward_source_totals(getattr(env, "last_reward_breakdown", None))
        strict_call = parse_terminal_strict_tool_call(completion_text)
        repaired_call = repair_tool_call(completion_text)
        env_rewards.append(env_reward)
        format_rewards.append(format_reward)
        copy_rewards.append(copy_reward)
        strict_submit_rewards.append(strict_submit_reward)
        done_values.append(1.0 if getattr(env, "done", False) else 0.0)
        post_terminal_values.append(float(getattr(env, "post_terminal_tool_calls", 0)))
        strict_parse_values.append(1.0 if strict_call is not None else 0.0)
        normalized_parse_values.append(1.0 if repaired_call is not None else 0.0)
        normalized_repair_values.append(
            1.0 if repaired_call is not None and strict_call is None else 0.0
        )
        tool_name = repaired_call["name"] if repaired_call is not None else None
        ask_prior_values.append(1.0 if tool_name == "ask_prior" else 0.0)
        submit_values.append(1.0 if tool_name == "submit_defect_map" else 0.0)
        identity_rewards.append(component_values["identity_reward"])
        concentration_rewards.append(component_values["concentration_reward"])
        confidence_rewards.append(component_values["confidence_reward"])
        false_positive_penalties.append(component_values["false_positive_penalty"])
        missed_defect_penalties.append(component_values["missed_defect_penalty"])
        scan_cost_penalties.append(component_values["scan_cost_penalty"])
        timeout_penalties.append(component_values["timeout_penalty"])
        outcome_reward_totals.append(source_totals["outcome_reward_total"])
        penalty_totals.append(source_totals["penalty_total"])
        process_shaping_rewards.append(format_reward + copy_reward + strict_submit_reward)
        rewards.append(env_reward + format_reward + copy_reward + strict_submit_reward)
    _log_reward_metrics(
        kwargs,
        env_rewards=env_rewards,
        format_rewards=format_rewards,
        copy_rewards=copy_rewards,
        strict_submit_rewards=strict_submit_rewards,
        done_values=done_values,
        post_terminal_values=post_terminal_values,
        strict_parse_values=strict_parse_values,
        normalized_parse_values=normalized_parse_values,
        normalized_repair_values=normalized_repair_values,
        ask_prior_values=ask_prior_values,
        submit_values=submit_values,
        identity_rewards=identity_rewards,
        concentration_rewards=concentration_rewards,
        confidence_rewards=confidence_rewards,
        false_positive_penalties=false_positive_penalties,
        missed_defect_penalties=missed_defect_penalties,
        scan_cost_penalties=scan_cost_penalties,
        timeout_penalties=timeout_penalties,
        outcome_reward_totals=outcome_reward_totals,
        penalty_totals=penalty_totals,
        process_shaping_rewards=process_shaping_rewards,
        total_rewards=rewards,
    )
    return rewards


def build_prompt_rows(
    samples: int,
    difficulty: str = "medium",
    include_tool_system_prompt: bool = True,
    seed_start: int = GRPO_TRAIN_SEED_START,
    prompt_focus: str = "all",
    min_prior_confidence: float = 0.45,
    max_prior_confidence: float = 0.65,
    min_reference_improvement: float = 0.25,
    max_seed_candidates: int | None = None,
) -> dict[str, Any]:
    """Build plain Python prompt rows without optional training dependencies."""

    seeds = _select_prompt_seeds(
        samples=samples,
        difficulty=difficulty,
        seed_start=seed_start,
        prompt_focus=prompt_focus,
        min_prior_confidence=min_prior_confidence,
        max_prior_confidence=max_prior_confidence,
        min_reference_improvement=min_reference_improvement,
        max_seed_candidates=max_seed_candidates,
    )
    prompt = [{"role": "user", "content": DEFAULT_PROMPT}]
    if include_tool_system_prompt:
        prompt = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}, *prompt]
    return {
        "prompt": [prompt for _ in seeds],
        "seed": seeds,
        "difficulty": [difficulty] * len(seeds),
        "prompt_focus": [prompt_focus] * len(seeds),
    }


def build_dataset(
    samples: int,
    difficulty: str = "medium",
    include_tool_system_prompt: bool = True,
    seed_start: int = GRPO_TRAIN_SEED_START,
    prompt_focus: str = "all",
    min_prior_confidence: float = 0.45,
    max_prior_confidence: float = 0.65,
    min_reference_improvement: float = 0.25,
    max_seed_candidates: int | None = None,
):
    """Build a simple prompt dataset for AtomicVision episodes."""

    try:
        from datasets import Dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The `datasets` package is required for GRPO training. Install the "
            "training extras with `pip install -r training/requirements-grpo.txt`."
        ) from exc

    return Dataset.from_dict(
        build_prompt_rows(
            samples=samples,
            difficulty=difficulty,
            include_tool_system_prompt=include_tool_system_prompt,
            seed_start=seed_start,
            prompt_focus=prompt_focus,
            min_prior_confidence=min_prior_confidence,
            max_prior_confidence=max_prior_confidence,
            min_reference_improvement=min_reference_improvement,
            max_seed_candidates=max_seed_candidates,
        )
    )


def _select_prompt_seeds(
    *,
    samples: int,
    difficulty: str,
    seed_start: int,
    prompt_focus: str,
    min_prior_confidence: float,
    max_prior_confidence: float,
    min_reference_improvement: float,
    max_seed_candidates: int | None,
) -> list[int]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if prompt_focus not in PROMPT_FOCI:
        raise ValueError(f"Unknown prompt_focus: {prompt_focus}")
    if prompt_focus in {"all", "heldout"}:
        return list(range(seed_start, seed_start + samples))
    if min_prior_confidence > max_prior_confidence:
        raise ValueError("min_prior_confidence must be <= max_prior_confidence")
    if min_reference_improvement < 0.0:
        raise ValueError("min_reference_improvement must be non-negative")

    search_limit = max_seed_candidates or max(256, samples * 24)
    selected: list[int] = []
    for seed in range(seed_start, seed_start + search_limit):
        profile = _profile_seed_for_grpo(seed=seed, difficulty=difficulty)
        is_borderline = (
            min_prior_confidence
            <= profile["prior_confidence"]
            <= max_prior_confidence
        )
        improves_with_reference = (
            profile["reference_reward_improvement"] >= min_reference_improvement
        )
        if prompt_focus == "borderline":
            keep = is_borderline
        elif prompt_focus == "reference-improvement":
            keep = improves_with_reference
        else:
            keep = is_borderline or improves_with_reference
        if keep:
            selected.append(seed)
        if len(selected) >= samples:
            return selected

    raise ValueError(
        f"Only found {len(selected)} {prompt_focus} seeds for difficulty={difficulty} "
        f"after scanning {search_limit} candidates from seed_start={seed_start}. "
        "Increase --max-seed-candidates, lower --min-reference-improvement, "
        "or widen the prior-confidence range."
    )


def _profile_seed_for_grpo(seed: int, difficulty: str) -> dict[str, float]:
    env = AtomicVisionEnvironment(difficulty=difficulty)
    env.reset(seed=seed)
    prior_observation = env.step(AtomicVisionAction(action_type="ask_prior"))
    case = env._require_case()
    prior = prior_observation.prior_prediction
    if prior is None:
        prior_defects: list[str] = []
        prior_concentrations: list[float] = []
        prior_confidence = 0.0
    else:
        prior_defects = list(prior.predicted_defects)
        prior_concentrations = list(prior.predicted_concentrations)
        prior_confidence = float(prior.confidence)

    prior_score = score_submission(
        case,
        prior_defects,
        prior_concentrations,
        confidence=prior_confidence,
        scan_cost=1.5,
    )
    oracle_after_reference = score_submission(
        case,
        [defect.species for defect in case.defects],
        [defect.concentration for defect in case.defects],
        confidence=0.95,
        scan_cost=2.0,
    )
    return {
        "prior_confidence": prior_confidence,
        "prior_reward": float(prior_score.total_reward),
        "reference_oracle_reward": float(oracle_after_reference.total_reward),
        "reference_reward_improvement": (
            float(oracle_after_reference.total_reward)
            - float(prior_score.total_reward)
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the GRPO training CLI parser."""

    parser = argparse.ArgumentParser(description="Fine-tune an AtomicVision agent with TRL GRPO.")
    parser.add_argument("--preset", choices=sorted(TRAINING_PRESETS), default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-url", default=None)
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=GRPO_TRAIN_SEED_START)
    parser.add_argument(
        "--prompt-focus",
        choices=PROMPT_FOCI,
        default="all",
        help=(
            "Which seeds to feed into GRPO. Use grpo-frontier after SFT so "
            "training focuses on borderline/reference-improvement cases."
        ),
    )
    parser.add_argument("--min-prior-confidence", type=float, default=0.45)
    parser.add_argument("--max-prior-confidence", type=float, default=0.65)
    parser.add_argument("--min-reference-improvement", type=float, default=0.25)
    parser.add_argument("--max-seed-candidates", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/grpo-atomicvision")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument(
        "--scale-rewards",
        choices=["group", "batch", "none"],
        default="group",
        help="Reward scaling strategy passed to TRL GRPOConfig.",
    )
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--loss-type",
        choices=["grpo", "dapo", "dr_grpo", "bnpo"],
        default="dapo",
    )
    parser.add_argument("--report-to", default="trackio")
    parser.add_argument(
        "--trackio-project",
        default="atomicvision-grpo",
        help="Trackio project name when report_to includes trackio.",
    )
    parser.add_argument(
        "--trackio-space-id",
        default=None,
        help=(
            "Optional Hugging Face Space ID for persistent Trackio logging, "
            "for example prodigyhuh/atomicvision-trackio."
        ),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument(
        "--adapter-model-id",
        default=None,
        help=(
            "Optional PEFT adapter repo or local path to continue training from. "
            "When set, --model is treated as the base model and --use-peft is ignored."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument(
        "--no-tool-system-prompt",
        action="store_true",
        help="Disable the system tool-contract prompt. Only use for ablations.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    _apply_preset(args)
    if args.run_name is None:
        args.run_name = "atomicvision-grpo-smoke"

    if args.env_url:
        os.environ["ATOMICVISION_ENV_URL"] = args.env_url

    if args.dry_run:
        rows = build_prompt_rows(
            samples=2,
            difficulty=args.difficulty,
            include_tool_system_prompt=not args.no_tool_system_prompt,
            seed_start=args.seed_start,
            prompt_focus=args.prompt_focus,
            min_prior_confidence=args.min_prior_confidence,
            max_prior_confidence=args.max_prior_confidence,
            min_reference_improvement=args.min_reference_improvement,
            max_seed_candidates=args.max_seed_candidates,
        )
        env = AtomicVisionToolEnv()
        try:
            initial = env.reset(seed=0, difficulty=args.difficulty)
            prior = env.ask_prior()
        finally:
            env._close()
        print(rows["prompt"][0][0]["content"][:160])
        print(initial[:240])
        print(prior[:240])
        return

    from trl import GRPOConfig, GRPOTrainer

    model = args.model
    peft_config = None
    if args.adapter_model_id:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_model_id,
            is_trainable=True,
        )
    elif args.use_peft:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
        )

    dataset = build_dataset(
        samples=args.samples,
        difficulty=args.difficulty,
        include_tool_system_prompt=not args.no_tool_system_prompt,
        seed_start=args.seed_start,
        prompt_focus=args.prompt_focus,
        min_prior_confidence=args.min_prior_confidence,
        max_prior_confidence=args.max_prior_confidence,
        min_reference_improvement=args.min_reference_improvement,
        max_seed_candidates=args.max_seed_candidates,
    )
    config_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "log_completions": True,
        "learning_rate": args.learning_rate,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "scale_rewards": args.scale_rewards,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "report_to": args.report_to,
        "run_name": args.run_name,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    grpo_config_parameters = inspect.signature(GRPOConfig).parameters
    if "project" in grpo_config_parameters:
        config_kwargs["project"] = args.trackio_project
    if args.trackio_space_id and "trackio_space_id" in grpo_config_parameters:
        config_kwargs["trackio_space_id"] = args.trackio_space_id
    trainer = GRPOTrainer(
        model=model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=GRPOConfig(**config_kwargs),
        environment_factory=AtomicVisionToolEnv,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    metrics_summary = _build_training_metrics_summary(
        train_metrics=getattr(train_result, "metrics", None),
        log_history=getattr(getattr(trainer, "state", None), "log_history", None),
        run_name=args.run_name,
        difficulty=args.difficulty,
        prompt_focus=args.prompt_focus,
        seed_start=args.seed_start,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "grpo_train_metrics_summary.json"
    summary_path.write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("FINAL_TRAIN_METRICS_SUMMARY")
    print(json.dumps(metrics_summary, sort_keys=True))
    print(f"Metrics summary saved at: {summary_path}")
    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()


def _env_url() -> str:
    return os.environ.get("ATOMICVISION_ENV_URL", DEFAULT_ENV_URL)


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset is None:
        return
    for key, value in TRAINING_PRESETS[args.preset].items():
        if key == "run_name" and args.run_name is not None:
            continue
        setattr(args, key, value)


def _log_reward_metrics(
    kwargs: dict[str, Any],
    *,
    env_rewards: list[float],
    format_rewards: list[float],
    copy_rewards: list[float],
    strict_submit_rewards: list[float],
    done_values: list[float],
    post_terminal_values: list[float],
    strict_parse_values: list[float],
    normalized_parse_values: list[float],
    normalized_repair_values: list[float],
    ask_prior_values: list[float],
    submit_values: list[float],
    identity_rewards: list[float],
    concentration_rewards: list[float],
    confidence_rewards: list[float],
    false_positive_penalties: list[float],
    missed_defect_penalties: list[float],
    scan_cost_penalties: list[float],
    timeout_penalties: list[float],
    outcome_reward_totals: list[float],
    penalty_totals: list[float],
    process_shaping_rewards: list[float],
    total_rewards: list[float],
) -> None:
    log_metric = kwargs.get("log_metric")
    if not callable(log_metric):
        return
    log_metric("atomicvision/env_reward_mean", _safe_mean(env_rewards))
    log_metric("atomicvision/format_reward_mean", _safe_mean(format_rewards))
    log_metric("atomicvision/prior_copy_reward_mean", _safe_mean(copy_rewards))
    log_metric("atomicvision/strict_submit_reward_mean", _safe_mean(strict_submit_rewards))
    log_metric("atomicvision/done_rate", _safe_mean(done_values))
    log_metric("atomicvision/post_terminal_tool_calls_mean", _safe_mean(post_terminal_values))
    log_metric("atomicvision/strict_tool_call_pass_rate", _safe_mean(strict_parse_values))
    log_metric("atomicvision/normalized_tool_call_pass_rate", _safe_mean(normalized_parse_values))
    log_metric("atomicvision/normalized_tool_call_repair_rate", _safe_mean(normalized_repair_values))
    log_metric("atomicvision/ask_prior_tool_rate", _safe_mean(ask_prior_values))
    log_metric("atomicvision/submit_tool_rate", _safe_mean(submit_values))
    log_metric("atomicvision/identity_reward_mean", _safe_mean(identity_rewards))
    log_metric("atomicvision/concentration_reward_mean", _safe_mean(concentration_rewards))
    log_metric("atomicvision/confidence_reward_mean", _safe_mean(confidence_rewards))
    log_metric("atomicvision/outcome_reward_mean", _safe_mean(outcome_reward_totals))
    log_metric("atomicvision/false_positive_penalty_mean", _safe_mean(false_positive_penalties))
    log_metric("atomicvision/missed_defect_penalty_mean", _safe_mean(missed_defect_penalties))
    log_metric("atomicvision/scan_cost_penalty_mean", _safe_mean(scan_cost_penalties))
    log_metric("atomicvision/timeout_penalty_mean", _safe_mean(timeout_penalties))
    log_metric("atomicvision/penalty_total_mean", _safe_mean(penalty_totals))
    log_metric("atomicvision/process_shaping_reward_mean", _safe_mean(process_shaping_rewards))
    log_metric("atomicvision/total_reward_mean", _safe_mean(total_rewards))


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _build_training_metrics_summary(
    *,
    train_metrics: dict[str, Any] | None,
    log_history: list[dict[str, Any]] | None,
    run_name: str,
    difficulty: str,
    prompt_focus: str,
    seed_start: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "run_name": run_name,
        "difficulty": difficulty,
        "prompt_focus": prompt_focus,
        "seed_start": seed_start,
    }
    for source in (train_metrics or {}, *_iter_log_entries(log_history)):
        for key, value in source.items():
            if isinstance(value, bool):
                summary[key] = value
            elif isinstance(value, (int, float)):
                summary[key] = float(value)
            elif isinstance(value, str) and key.endswith(("_runtime", "_per_second")):
                summary[key] = value
    return summary


def _iter_log_entries(log_history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not log_history:
        return []
    return [entry for entry in log_history if isinstance(entry, dict)]


def _is_retryable_connection_error(exc: Exception) -> bool:
    name = type(exc).__name__
    message = str(exc)
    return (
        "ConnectionClosed" in name
        or "ConnectionError" in name
        or "Connection refused" in message
        or "closed" in message.lower()
        or "CAPACITY_REACHED" in message
    )


def _extract_completion_texts(kwargs: dict[str, Any], expected_count: int) -> list[str]:
    completions = kwargs.get("completions")
    if completions is None:
        return [""] * expected_count
    texts = [_completion_to_text(completion) for completion in completions]
    if len(texts) != expected_count:
        return [""] * expected_count
    return texts


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text") or "")
    if isinstance(completion, list):
        return "\n".join(_completion_to_text(item) for item in completion)
    return str(completion or "")


def render_tool_call_text(call: dict[str, Any]) -> str:
    """Serialize one canonical AtomicVision tool call."""

    payload = json.dumps(call, separators=(",", ":"), ensure_ascii=True)
    return f"<tool_call>{payload}</tool_call>"


def _strict_submit_template(prior: dict[str, Any]) -> str | None:
    defects = prior.get("predicted_defects") or []
    concentrations = prior.get("predicted_concentrations") or []
    confidence = float(prior.get("confidence") or 0.0)
    if isinstance(concentrations, dict):
        concentrations = [concentrations.get(defect, 0.0) for defect in defects]
    if not isinstance(defects, list) or not isinstance(concentrations, list):
        return None
    if not defects or len(defects) != len(concentrations):
        return None
    if confidence < CONFIDENT_PRIOR_COPY_THRESHOLD:
        return None
    call = {
        "name": "submit_defect_map",
        "arguments": {
            "predicted_defects": [str(item) for item in defects],
            "predicted_concentrations": _rounded_floats(
                [float(item) for item in concentrations]
            ),
            "confidence": round(confidence, 5),
        },
    }
    return render_tool_call_text(call)


def _parse_all_strict_tool_calls_with_spans(text: str) -> list[tuple[dict[str, Any], int, int]]:
    """Parse every strictly valid XML-wrapped JSON tool call in a transcript."""

    calls: list[tuple[dict[str, Any], int, int]] = []
    for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        try:
            call = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []
        if not _is_valid_tool_call(call):
            return []
        calls.append((call, match.start(), match.end()))
    return calls


def _parse_all_strict_tool_calls(text: str) -> list[dict[str, Any]]:
    return [call for call, _, _ in _parse_all_strict_tool_calls_with_spans(text)]


def parse_terminal_strict_tool_call(text: str) -> dict[str, Any] | None:
    """Return the last strict tool call only when it is the terminal tool mention."""

    calls = _parse_all_strict_tool_calls_with_spans(text)
    if not calls:
        return None
    _, last_tool_pos = _last_tool_name(text)
    if last_tool_pos < 0:
        return calls[-1][0]
    call, start, end = calls[-1]
    if start <= last_tool_pos < end:
        return call
    return None


def parse_strict_tool_call(text: str) -> dict[str, Any] | None:
    """Parse one exact XML-wrapped JSON tool call, returning None on any mismatch."""

    calls = _parse_all_strict_tool_calls(text)
    if len(calls) != 1:
        return None
    return calls[0]


def parse_last_strict_tool_call(text: str) -> dict[str, Any] | None:
    """Return the last strict tool call in a multi-turn transcript, if any."""

    calls = _parse_all_strict_tool_calls(text)
    if not calls:
        return None
    return calls[-1]


def repair_tool_call(text: str) -> dict[str, Any] | None:
    """Recover a valid tool call from near-miss generations when possible."""

    terminal_strict = parse_terminal_strict_tool_call(text)
    if terminal_strict is not None:
        return terminal_strict

    tool_name, tool_pos = _last_tool_name(text)
    if tool_name is None:
        return None

    if tool_name in {"ask_prior", "compare_reference"}:
        return {"name": tool_name, "arguments": {}}
    if tool_name == "request_scan":
        return _repair_request_scan_call(text, start_pos=tool_pos)
    if tool_name == "zoom_band":
        return _repair_zoom_band_call(text, start_pos=tool_pos)
    if tool_name == "submit_defect_map":
        return _repair_submit_defect_map_call(text, start_pos=tool_pos)
    return None


def canonicalize_tool_call_text(text: str) -> str:
    """Normalize a model completion into the strict AtomicVision tool-call envelope."""

    call = repair_tool_call(text)
    if call is None:
        return text
    return render_tool_call_text(call)


def _tool_call_format_reward(text: str) -> float:
    if not text:
        return 0.0
    if parse_terminal_strict_tool_call(text) is not None:
        return VALID_TOOL_CALL_FORMAT_REWARD
    if repair_tool_call(text) is not None:
        return -RECOVERABLE_TOOL_CALL_FORMAT_PENALTY
    return -INVALID_TOOL_CALL_FORMAT_PENALTY


def _strict_terminal_submit_reward(text: str, env: Any) -> float:
    if getattr(env, "last_submit_action", None) is None:
        return 0.0
    if getattr(env, "last_prior_prediction", None) is None:
        return 0.0
    repaired_call = repair_tool_call(text)
    if repaired_call is None or repaired_call.get("name") != "submit_defect_map":
        return 0.0
    strict_call = parse_terminal_strict_tool_call(text)
    if strict_call is not None and strict_call.get("name") == "submit_defect_map":
        return STRICT_FINAL_SUBMIT_FORMAT_REWARD
    if getattr(env, "done", False):
        return -RECOVERABLE_FINAL_SUBMIT_FORMAT_PENALTY
    return 0.0


def _is_valid_tool_call(call: Any) -> bool:
    if not isinstance(call, dict):
        return False
    name = call.get("name")
    arguments = call.get("arguments")
    return isinstance(name, str) and name in VALID_TOOL_NAMES and isinstance(arguments, dict)


def _last_tool_name(text: str) -> tuple[str | None, int]:
    last_match = None
    for tool_name in VALID_TOOL_NAMES:
        for match in re.finditer(rf"\b{re.escape(tool_name)}\b", text):
            if last_match is None or match.start() > last_match[1]:
                last_match = (tool_name, match.start())
    if last_match is None:
        return None, -1
    return last_match[0], last_match[1]


def _first_json_object(text: str, start_pos: int = 0) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index in range(max(0, start_pos), len(text)):
        if text[index] != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _last_json_object(text: str, start_pos: int = 0) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    last_obj: dict[str, Any] | None = None
    for index in range(max(0, start_pos), len(text)):
        if text[index] != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            last_obj = obj
    return last_obj


def _repair_request_scan_call(text: str, start_pos: int = 0) -> dict[str, Any]:
    args = _repair_arguments_object(text, start_pos=start_pos)
    return {
        "name": "request_scan",
        "arguments": {
            "scan_mode": args.get("scan_mode") or "standard_pdos",
            "resolution": args.get("resolution") or "medium",
        },
    }


def _repair_zoom_band_call(text: str, start_pos: int = 0) -> dict[str, Any] | None:
    args = _repair_arguments_object(text, start_pos=start_pos)
    freq_min = args.get("freq_min")
    freq_max = args.get("freq_max")
    if freq_min is None or freq_max is None:
        return None
    return {
        "name": "zoom_band",
        "arguments": {
            "scan_mode": args.get("scan_mode") or "high_res_pdos",
            "resolution": args.get("resolution") or "high",
            "freq_min": float(freq_min),
            "freq_max": float(freq_max),
        },
    }


def _repair_submit_defect_map_call(text: str, start_pos: int = 0) -> dict[str, Any] | None:
    args = _repair_arguments_object(text, start_pos=start_pos)
    if args:
        defects = args.get("predicted_defects") or []
        concentrations = args.get("predicted_concentrations") or []
        confidence = args.get("confidence")
        if isinstance(defects, list) and isinstance(concentrations, list):
            return {
                "name": "submit_defect_map",
                "arguments": {
                    "predicted_defects": [str(item) for item in defects],
                    "predicted_concentrations": [float(item) for item in concentrations],
                    "confidence": float(confidence if confidence is not None else 0.65),
                },
            }
    prior = _extract_prior_payload(text)
    if prior is None:
        return None
    defects = prior.get("predicted_defects") or []
    concentrations = prior.get("predicted_concentrations") or []
    if isinstance(concentrations, dict):
        concentrations = [concentrations.get(defect, 0.0) for defect in defects]
    if not isinstance(defects, list) or not isinstance(concentrations, list):
        return None
    return {
        "name": "submit_defect_map",
        "arguments": {
            "predicted_defects": [str(item) for item in defects],
            "predicted_concentrations": [float(item) for item in concentrations],
            "confidence": float(prior.get("confidence") or 0.65),
        },
    }


def _repair_arguments_object(text: str, start_pos: int = 0) -> dict[str, Any]:
    obj = _first_json_object(text, start_pos=start_pos)
    if obj is None:
        obj = _last_json_object(text, start_pos=start_pos)
    if obj is None and start_pos > 0:
        obj = _first_json_object(text)
    if obj is None:
        obj = _last_json_object(text)
    if obj is None:
        return {}
    if _is_valid_tool_call(obj):
        return dict(obj["arguments"])
    if isinstance(obj.get("arguments"), dict):
        return dict(obj["arguments"])
    if "defect_map" in obj and isinstance(obj["defect_map"], dict):
        defect_map = obj["defect_map"]
        defects = list(defect_map.keys())
        return {
            "predicted_defects": defects,
            "predicted_concentrations": [float(defect_map[defect]) for defect in defects],
            "confidence": obj.get("confidence", 0.65),
        }
    if isinstance(obj.get("defects"), list):
        defects = [str(item) for item in obj["defects"]]
        concentrations_obj = obj.get("concentrations")
        if isinstance(concentrations_obj, dict):
            concentrations = [float(concentrations_obj.get(defect, 0.0)) for defect in defects]
        elif isinstance(concentrations_obj, list):
            concentrations = [float(item) for item in concentrations_obj]
        else:
            concentrations = [0.0] * len(defects)
        return {
            "predicted_defects": defects,
            "predicted_concentrations": concentrations,
            "confidence": obj.get("confidence", 0.65),
        }
    if isinstance(obj.get("predicted_defects"), list) and (
        isinstance(obj.get("predicted_concentrations"), list)
        or isinstance(obj.get("predicted_concentrations"), dict)
    ):
        defects = [str(item) for item in obj["predicted_defects"]]
        concentrations_obj = obj["predicted_concentrations"]
        if isinstance(concentrations_obj, dict):
            concentrations = [float(concentrations_obj.get(defect, 0.0)) for defect in defects]
        else:
            concentrations = [float(item) for item in concentrations_obj]
        return {
            "predicted_defects": defects,
            "predicted_concentrations": concentrations,
            "confidence": obj.get("confidence", 0.65),
            "scan_mode": obj.get("scan_mode"),
            "resolution": obj.get("resolution"),
            "freq_min": obj.get("freq_min"),
            "freq_max": obj.get("freq_max"),
        }
    return dict(obj)


def _extract_prior_payload(text: str) -> dict[str, Any] | None:
    match = re.search(r"prior\s*=\s*(\{.*)", text, re.S)
    if match is None:
        return None
    return _first_json_object(match.group(1))


def _prior_copy_reward(env: AtomicVisionToolEnv) -> float:
    prior = env.last_prior_prediction
    submit = env.last_submit_action
    if prior is None or submit is None:
        return 0.0
    confidence = float(prior.get("confidence") or 0.0)
    is_exact = (
        list(prior.get("predicted_defects") or []) == list(submit.predicted_defects)
        and _rounded_floats(prior.get("predicted_concentrations") or [])
        == _rounded_floats(submit.predicted_concentrations)
        and round(confidence, 5) == round(float(submit.confidence or 0.0), 5)
    )
    if is_exact:
        return EXACT_PRIOR_COPY_REWARD
    if confidence >= CONFIDENT_PRIOR_COPY_THRESHOLD:
        return -CONFIDENT_PRIOR_MIS_COPY_PENALTY
    return 0.0


def _rounded_floats(values: list[float]) -> list[float]:
    return [round(float(value), 5) for value in values]


def _format_observation(observation: dict) -> str:
    prior = observation.get("prior_prediction") or {}
    reward_breakdown = observation.get("reward_breakdown") or {}
    axis = observation.get("frequency_axis") or []
    scan_history = observation.get("scan_history") or []
    scan_cost_so_far = _scan_cost_from_history(scan_history)
    recommended_next_action = _recommended_next_action(prior)
    strict_submit_template = _strict_submit_template(prior)
    spectral_summary = _spectral_summary(observation)
    if axis:
        frequency_range = f"{min(axis):.3f}-{max(axis):.3f}"
    else:
        frequency_range = "0.000-20.000"
    done = bool(observation.get("done"))
    terminal_instruction = (
        "\nterminal_instruction=stop_tool_calls_return_final_answer"
        if done
        else ""
    )
    has_prior = bool(prior.get("predicted_defects"))
    submit_shape_instruction = (
        "\nsubmit_field_order=predicted_defects,predicted_concentrations,confidence"
        "\nsubmit_output_rule=if_submitting_use_one_xml_tool_call_then_stop"
        if has_prior
        else ""
    )
    strict_submit_instruction = (
        f"\nstrict_submit_template={strict_submit_template}"
        "\nstrict_submit_rule=copy_template_only_when_prior_is_high_confidence"
        if strict_submit_template is not None
        else ""
    )
    return (
        f"message={observation.get('message')}\n"
        f"material={observation.get('material_id')} difficulty={observation.get('difficulty')}\n"
        f"budget_remaining={observation.get('budget_remaining')} "
        f"step={observation.get('step_count')}/{observation.get('max_steps')}\n"
        f"valid_scan_modes={VALID_SCAN_MODES}\n"
        f"valid_resolutions={VALID_RESOLUTIONS}\n"
        f"valid_frequency_range={frequency_range}\n"
        f"tool_costs=ask_prior:1.5, compare_reference:0.5, quick_pdos:1.0, "
        f"standard_pdos:2.0, raman_proxy:2.5, high_res_pdos_or_zoom:4.0\n"
        f"scan_cost_so_far={scan_cost_so_far:.3f}\n"
        f"cost_discipline=submit_high_confidence_prior; one_cheap_scan_only_when_borderline\n"
        "tool_output_rule=one_xml_tool_call_only_no_reasoning_markup_no_extra_text\n"
        f"recommended_first_action=ask_prior\n"
        f"recommended_next_action={recommended_next_action}\n"
        f"candidate_defects={observation.get('candidate_defects')}\n"
        f"spectral_summary={spectral_summary}\n"
        f"prior={prior}\n"
        f"reward={observation.get('reward')} done={observation.get('done')}\n"
        f"reward_breakdown={reward_breakdown}"
        f"{submit_shape_instruction}"
        f"{strict_submit_instruction}"
        f"{terminal_instruction}"
    )


def _recommended_next_action(prior: dict) -> str:
    defects = prior.get("predicted_defects") or []
    confidence = float(prior.get("confidence") or 0.0)
    if defects and confidence >= CONFIDENT_PRIOR_COPY_THRESHOLD:
        return "submit_defect_map_with_prior"
    if defects and confidence >= PRIOR_SUBMIT_THRESHOLD:
        return "copy_prior_or_one_cheap_scan_then_submit"
    if defects:
        return "optional_one_scan_then_submit"
    return "ask_prior"


def _scan_cost_from_history(scan_history: list) -> float:
    total = 0.0
    for record in scan_history:
        if isinstance(record, dict):
            total += float(record.get("cost") or 0.0)
        else:
            total += float(getattr(record, "cost", 0.0) or 0.0)
    return total


def _spectral_summary(observation: dict) -> str:
    axis = observation.get("frequency_axis") or []
    current = observation.get("current_spectrum") or []
    reference = observation.get("pristine_reference") or []
    candidates = observation.get("candidate_defects") or []
    if not axis or not current:
        return "unavailable"

    summary: dict[str, Any] = {
        "current_peak_freqs": _top_frequency_values(axis, current, top_k=4),
    }
    if candidates:
        summary["candidate_signature_bands"] = [
            _candidate_signature_bands(species) for species in candidates
        ]
    if reference and len(reference) == len(current) == len(axis):
        deltas = [
            round(float(current_value) - float(reference_value), 6)
            for current_value, reference_value in zip(current, reference, strict=True)
        ]
        summary["spectrum_delta_top_abs"] = _top_frequency_values(
            axis,
            deltas,
            top_k=6,
            absolute=True,
        )
        if candidates:
            summary["candidate_signature_scores"] = _candidate_signature_scores(
                axis,
                deltas,
                candidates,
            )
    return json.dumps(summary, separators=(",", ":"), ensure_ascii=True)


def _top_frequency_values(
    axis: list,
    values: list,
    top_k: int,
    absolute: bool = False,
) -> list[dict[str, float]]:
    ranked = sorted(
        zip(axis, values, strict=True),
        key=lambda pair: abs(float(pair[1])) if absolute else float(pair[1]),
        reverse=True,
    )[:top_k]
    return [
        {"freq": round(float(freq), 3), "value": round(float(value), 4)}
        for freq, value in ranked
    ]


def _candidate_signature_bands(species: str) -> dict[str, float | str]:
    signature = _species_signature(species)
    return {
        "species": species,
        "add": round(signature["center"], 3),
        "soften": round(signature["soften_center"], 3),
        "broad": round(signature["broad_center"], 3),
    }


def _candidate_signature_scores(
    axis: list,
    deltas: list[float],
    candidates: list[str],
) -> list[dict[str, float | str]]:
    scored = []
    for species in candidates:
        signature = _species_signature(species)
        add_delta = _nearest_spectral_value(axis, deltas, signature["center"])
        soften_delta = _nearest_spectral_value(axis, deltas, signature["soften_center"])
        broad_delta = _nearest_spectral_value(axis, deltas, signature["broad_center"])
        score = max(0.0, add_delta) + max(0.0, broad_delta) + max(0.0, -soften_delta)
        scored.append(
            {
                "species": species,
                "score": round(score, 4),
                "add_delta": round(add_delta, 4),
                "soften_delta": round(soften_delta, 4),
                "broad_delta": round(broad_delta, 4),
            }
        )
    return sorted(scored, key=lambda item: float(item["score"]), reverse=True)


def _nearest_spectral_value(axis: list, values: list[float], target: float) -> float:
    if not axis or not values:
        return 0.0
    nearest_index = min(
        range(len(axis)),
        key=lambda index: abs(float(axis[index]) - target),
    )
    return float(values[nearest_index])


def _species_signature(species: str) -> dict[str, float]:
    code = sum((index + 1) * ord(char) for index, char in enumerate(species))
    center = 1.2 + (code % 170) / 10.0
    soften_center = 1.0 + ((code * 7) % 180) / 10.0
    broad_center = 2.0 + ((code * 11) % 150) / 10.0
    return {
        "center": min(center, 19.2),
        "soften_center": min(soften_center, 19.0),
        "broad_center": min(broad_center, 18.5),
    }


if __name__ == "__main__":
    main()
