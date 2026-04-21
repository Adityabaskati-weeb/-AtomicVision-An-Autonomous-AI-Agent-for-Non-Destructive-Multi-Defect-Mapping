"""Phase 11 GRPO fine-tuning scaffold for AtomicVision.

This script is designed for Colab/Kaggle GPU runtimes. It follows TRL's
OpenEnv `environment_factory` pattern, exposing meaningful tools rather than a
generic `step(action)` method.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomicvision_env.client import AtomicVisionEnv
from atomicvision_env.models import AtomicVisionAction


DEFAULT_ENV_URL = "https://prodigyhuh-atomicvision-openenv.hf.space"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
POST_TERMINAL_TOOL_PENALTY = 2.0
ScanMode = Literal["quick_pdos", "standard_pdos", "high_res_pdos", "raman_proxy"]
Resolution = Literal["low", "medium", "high"]
VALID_SCAN_MODES = ("quick_pdos", "standard_pdos", "high_res_pdos", "raman_proxy")
VALID_RESOLUTIONS = ("low", "medium", "high")
DEFAULT_PROMPT = (
    "You are AtomicVision, an autonomous materials characterization agent. "
    "Your task is to infer hidden atomic defects from non-invasive spectral evidence. "
    "Maximize reward by submitting accurate defect identities and concentrations while "
    "avoiding unnecessary scan cost. "
    "Tool protocol: valid scan_mode values are exactly quick_pdos, standard_pdos, "
    "high_res_pdos, and raman_proxy. Valid resolution values are exactly low, medium, "
    "and high. The frequency axis is synthetic PDoS units from 0.0 to 20.0; valid "
    "zoom examples are 1.0-5.0, 5.0-10.0, and 10.0-18.0. Never use Raman/cm^-1 "
    "values such as 300, 1200, or 3000 for zoom_band. Strong default strategy: "
    "first call ask_prior. If the prior returns predicted_defects with confidence "
    "0.50 or higher, immediately submit_defect_map using the prior "
    "predicted_defects, predicted_concentrations, and confidence. "
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
        "num_generations": 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_completion_length": 1024,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "run_name": "atomicvision-grpo-smoke",
    },
    "colab-20": {
        "model": "Qwen/Qwen3-0.6B",
        "samples": 64,
        "max_steps": 20,
        "num_generations": 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_completion_length": 1024,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "run_name": "atomicvision-grpo-0p6b-20step",
    },
    "qwen-1p7b-50": {
        "model": "Qwen/Qwen3-1.7B",
        "samples": 128,
        "max_steps": 50,
        "num_generations": 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_completion_length": 768,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "run_name": "atomicvision-grpo-1p7b-50step",
    },
    "hf-4b-100": {
        "model": "Qwen/Qwen3-4B",
        "samples": 256,
        "max_steps": 100,
        "num_generations": 4,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_completion_length": 768,
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
    """Return final AtomicVision environment rewards for GRPO."""

    return [env.reward for env in environments]


def build_prompt_rows(samples: int, difficulty: str = "medium") -> dict[str, Any]:
    """Build plain Python prompt rows without optional training dependencies."""

    return {
        "prompt": [[{"role": "user", "content": DEFAULT_PROMPT}] for _ in range(samples)],
        "seed": list(range(samples)),
        "difficulty": [difficulty] * samples,
    }


def build_dataset(samples: int, difficulty: str = "medium"):
    """Build a simple prompt dataset for AtomicVision episodes."""

    try:
        from datasets import Dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The `datasets` package is required for GRPO training. Install the "
            "training extras with `pip install -r training/requirements-grpo.txt`."
        ) from exc

    return Dataset.from_dict(build_prompt_rows(samples=samples, difficulty=difficulty))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune an AtomicVision agent with TRL GRPO.")
    parser.add_argument("--preset", choices=sorted(TRAINING_PRESETS), default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-url", default=None)
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--output-dir", default="outputs/grpo-atomicvision")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--report-to", default="trackio")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _apply_preset(args)
    if args.run_name is None:
        args.run_name = "atomicvision-grpo-smoke"

    if args.env_url:
        os.environ["ATOMICVISION_ENV_URL"] = args.env_url

    if args.dry_run:
        rows = build_prompt_rows(samples=2, difficulty=args.difficulty)
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

    peft_config = None
    if args.use_peft:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    dataset = build_dataset(samples=args.samples, difficulty=args.difficulty)
    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=GRPOConfig(
            output_dir=args.output_dir,
            log_completions=True,
            learning_rate=args.learning_rate,
            max_completion_length=args.max_completion_length,
            num_generations=args.num_generations,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            report_to=args.report_to,
            run_name=args.run_name,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            chat_template_kwargs={"enable_thinking": False},
        ),
        environment_factory=AtomicVisionToolEnv,
        peft_config=peft_config,
    )
    trainer.train()
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


def _format_observation(observation: dict) -> str:
    prior = observation.get("prior_prediction") or {}
    reward_breakdown = observation.get("reward_breakdown") or {}
    axis = observation.get("frequency_axis") or []
    scan_history = observation.get("scan_history") or []
    scan_cost_so_far = _scan_cost_from_history(scan_history)
    recommended_next_action = _recommended_next_action(prior)
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
        f"cost_discipline=submit_prior_when_confident; extra_scans_lower_reward\n"
        f"recommended_first_action=ask_prior\n"
        f"recommended_next_action={recommended_next_action}\n"
        f"candidate_defects={observation.get('candidate_defects')}\n"
        f"prior={prior}\n"
        f"reward={observation.get('reward')} done={observation.get('done')}\n"
        f"reward_breakdown={reward_breakdown}"
        f"{terminal_instruction}"
    )


def _recommended_next_action(prior: dict) -> str:
    defects = prior.get("predicted_defects") or []
    confidence = float(prior.get("confidence") or 0.0)
    if defects and confidence >= 0.5:
        return "submit_defect_map_with_prior"
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


if __name__ == "__main__":
    main()
