from __future__ import annotations

import inspect
from argparse import Namespace
from typing import get_args, get_type_hints

import pytest

from training.train_grpo_atomicvision import (
    AtomicVisionToolEnv,
    CONFIDENT_PRIOR_COPY_THRESHOLD,
    CONFIDENT_PRIOR_MIS_COPY_PENALTY,
    EXACT_PRIOR_COPY_REWARD,
    TRAINING_PRESETS,
    VALID_TOOL_CALL_FORMAT_REWARD,
    TOOL_SYSTEM_PROMPT,
    _apply_preset,
    _env_url,
    _format_observation,
    _is_retryable_connection_error,
    _tool_call_format_reward,
    build_arg_parser,
    build_dataset,
    build_prompt_rows,
    reward_func,
)
from atomicvision_env.models import AtomicVisionAction


def test_prompt_rows_match_grpo_dataset_shape() -> None:
    rows = build_prompt_rows(samples=3, difficulty="hard")

    assert len(rows["prompt"]) == 3
    assert rows["prompt"][0][0]["role"] == "system"
    assert rows["prompt"][0][0]["content"] == TOOL_SYSTEM_PROMPT
    assert rows["prompt"][0][1]["role"] == "user"
    assert "AtomicVision" in rows["prompt"][0][1]["content"]
    assert "quick_pdos" in rows["prompt"][0][1]["content"]
    assert "0.0 to 20.0" in rows["prompt"][0][1]["content"]
    assert "ask_prior" in rows["prompt"][0][1]["content"]
    assert "confidence 0.65 or higher" in rows["prompt"][0][1]["content"]
    assert "0.50-0.65" in rows["prompt"][0][1]["content"]
    assert "standard_pdos costs 2.0" in rows["prompt"][0][1]["content"]
    assert "submit_defect_map is terminal" in rows["prompt"][0][1]["content"]
    assert rows["seed"] == [0, 1, 2]
    assert rows["difficulty"] == ["hard", "hard", "hard"]


def test_prompt_rows_allow_system_prompt_ablation() -> None:
    rows = build_prompt_rows(
        samples=1,
        difficulty="medium",
        include_tool_system_prompt=False,
    )

    assert len(rows["prompt"][0]) == 1
    assert rows["prompt"][0][0]["role"] == "user"


def test_build_dataset_has_clear_optional_dependency_error() -> None:
    try:
        import datasets  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(RuntimeError, match="training/requirements-grpo.txt"):
            build_dataset(samples=1)
    else:
        assert len(build_dataset(samples=1)) == 1


def test_reward_func_reads_environment_rewards() -> None:
    class DummyEnv:
        def __init__(self, reward: float):
            self.reward = reward
            self.last_prior_prediction = None
            self.last_submit_action = None

    assert reward_func([DummyEnv(1.25), DummyEnv(-0.5)]) == [1.25, -0.5]


def test_reward_func_adds_format_and_exact_copy_shaping() -> None:
    env = AtomicVisionToolEnv()
    env.reward = 2.0
    env.last_prior_prediction = {
        "predicted_defects": ["Zn", "P"],
        "predicted_concentrations": [0.19021, 0.04792],
        "confidence": 0.51,
    }
    env.last_submit_action = AtomicVisionAction(
        action_type="submit_defect_map",
        predicted_defects=["Zn", "P"],
        predicted_concentrations=[0.19021, 0.04792],
        confidence=0.51,
    )

    rewards = reward_func(
        [env],
        completions=[
            (
                '<tool_call>{"name":"submit_defect_map","arguments":'
                '{"predicted_defects":["Zn","P"],'
                '"predicted_concentrations":[0.19021,0.04792],'
                '"confidence":0.51}}</tool_call>'
            )
        ],
    )

    assert rewards == [2.0 + VALID_TOOL_CALL_FORMAT_REWARD + EXACT_PRIOR_COPY_REWARD]


def test_prior_copy_penalty_only_applies_to_high_confidence_priors() -> None:
    env = AtomicVisionToolEnv()
    env.reward = 2.0
    env.last_prior_prediction = {
        "predicted_defects": ["Zn", "P"],
        "predicted_concentrations": [0.19021, 0.04792],
        "confidence": CONFIDENT_PRIOR_COPY_THRESHOLD - 0.01,
    }
    env.last_submit_action = AtomicVisionAction(
        action_type="submit_defect_map",
        predicted_defects=["Zn"],
        predicted_concentrations=[0.19021],
        confidence=0.5,
    )

    assert reward_func([env]) == [2.0]

    env.last_prior_prediction["confidence"] = CONFIDENT_PRIOR_COPY_THRESHOLD

    assert reward_func([env]) == [2.0 - CONFIDENT_PRIOR_MIS_COPY_PENALTY]


def test_tool_call_format_reward_penalizes_missing_or_invalid_tool_call() -> None:
    assert _tool_call_format_reward("message=Initial scan") < 0.0
    assert _tool_call_format_reward("<tool_call>{bad json}</tool_call>") < 0.0


def test_training_presets_keep_grpo_generation_batch_valid() -> None:
    for preset in TRAINING_PRESETS.values():
        generation_batch_size = (
            preset["per_device_train_batch_size"]
            * preset["gradient_accumulation_steps"]
        )

        assert generation_batch_size % preset["num_generations"] == 0
        assert preset["temperature"] >= 1.2
        assert preset["top_p"] <= 1.0
        assert preset["use_peft"] is True


def test_apply_preset_overrides_training_args() -> None:
    args = Namespace(preset="qwen-1p7b-50", model="old", samples=1, run_name=None)

    _apply_preset(args)

    assert args.model == "Qwen/Qwen3-1.7B"
    assert args.samples == 128
    assert args.run_name == "atomicvision-grpo-1p7b-50step"


def test_apply_preset_preserves_explicit_run_name() -> None:
    args = Namespace(preset="qwen-1p7b-50", model="old", samples=1, run_name="custom")

    _apply_preset(args)

    assert args.model == "Qwen/Qwen3-1.7B"
    assert args.run_name == "custom"


def test_cli_accepts_adapter_continuation_args() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--model",
            "Qwen/Qwen3-1.7B",
            "--adapter-model-id",
            "prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora",
            "--max-steps",
            "10",
            "--temperature",
            "1.25",
            "--top-p",
            "0.9",
            "--top-k",
            "40",
            "--scale-rewards",
            "batch",
            "--beta",
            "0.001",
            "--loss-type",
            "dr_grpo",
        ]
    )

    assert args.model == "Qwen/Qwen3-1.7B"
    assert args.adapter_model_id == "prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora"
    assert args.max_steps == 10
    assert args.temperature == 1.25
    assert args.top_p == 0.9
    assert args.top_k == 40
    assert args.scale_rewards == "batch"
    assert args.beta == 0.001
    assert args.loss_type == "dr_grpo"
    assert args.no_tool_system_prompt is False


def test_tool_env_is_lazy_and_requires_reset_before_tools() -> None:
    env = AtomicVisionToolEnv()

    assert env.client is None
    with pytest.raises(ValueError, match="Call reset"):
        env.ask_prior()


def test_post_terminal_tool_calls_are_penalized_without_tool_failure() -> None:
    env = AtomicVisionToolEnv()
    env.client = object()
    env._connected = True
    env.done = True
    env.reward = 6.0

    response = env.ask_prior()

    assert "episode_done=True" in response
    assert "Stop calling tools" in response
    assert env.post_terminal_tool_calls == 1
    assert env.reward == 4.0


def test_retryable_connection_errors_are_detected() -> None:
    class ConnectionClosedOK(Exception):
        pass

    assert _is_retryable_connection_error(ConnectionClosedOK("closed normally"))
    assert _is_retryable_connection_error(ConnectionError("Connection refused"))
    assert _is_retryable_connection_error(RuntimeError("CAPACITY_REACHED"))
    assert not _is_retryable_connection_error(ValueError("bad defect map"))


def test_tool_env_exposes_only_model_facing_tools() -> None:
    exposed = {
        name
        for name, value in inspect.getmembers(AtomicVisionToolEnv, predicate=callable)
        if not name.startswith("_") and name != "reset"
    }

    assert exposed == {
        "ask_prior",
        "compare_reference",
        "request_scan",
        "submit_defect_map",
        "zoom_band",
    }


def test_tool_annotations_constrain_common_invalid_values() -> None:
    scan_hints = get_type_hints(AtomicVisionToolEnv.request_scan)
    zoom_hints = get_type_hints(AtomicVisionToolEnv.zoom_band)

    scan_mode_annotation = scan_hints["scan_mode"]
    resolution_annotation = scan_hints["resolution"]
    zoom_resolution_annotation = zoom_hints["resolution"]

    assert set(get_args(scan_mode_annotation)) == {
        "quick_pdos",
        "standard_pdos",
        "high_res_pdos",
        "raman_proxy",
    }
    assert set(get_args(resolution_annotation)) == {"low", "medium", "high"}
    assert set(get_args(zoom_resolution_annotation)) == {"low", "medium", "high"}


def test_env_url_can_be_overridden_with_environment_variable(monkeypatch) -> None:
    monkeypatch.setenv("ATOMICVISION_ENV_URL", "https://example.test")

    assert _env_url() == "https://example.test"


def test_observation_formatter_includes_training_signal() -> None:
    text = _format_observation(
        {
            "message": "scan complete",
            "material_id": "synthetic-medium-1",
            "difficulty": "medium",
            "budget_remaining": 6.5,
            "step_count": 2,
            "max_steps": 5,
            "candidate_defects": ["B", "N"],
            "prior_prediction": {"predicted_defects": ["B"], "confidence": 0.7},
            "reward": 3.0,
            "done": True,
            "reward_breakdown": {"f1": 0.8},
            "frequency_axis": [0.0, 5.0, 20.0],
            "scan_history": [
                {"action_type": "initial_scan", "cost": 0.0},
                {"action_type": "ask_prior", "cost": 1.5},
            ],
        }
    )

    assert "synthetic-medium-1" in text
    assert "budget_remaining=6.5" in text
    assert "valid_frequency_range=0.000-20.000" in text
    assert "scan_cost_so_far=1.500" in text
    assert "recommended_next_action=submit_defect_map_with_prior" in text
    assert "recommended_first_action=ask_prior" in text
    assert "reward=3.0 done=True" in text
    assert "terminal_instruction=stop_tool_calls_return_final_answer" in text
    assert "f1" in text


def test_observation_formatter_marks_borderline_prior_for_variance() -> None:
    text = _format_observation(
        {
            "message": "prior returned",
            "material_id": "synthetic-medium-1",
            "difficulty": "medium",
            "budget_remaining": 7.5,
            "step_count": 1,
            "max_steps": 5,
            "candidate_defects": ["B", "N"],
            "prior_prediction": {"predicted_defects": ["B"], "confidence": 0.55},
            "reward": -0.6,
            "done": False,
            "reward_breakdown": {"scan_cost_penalty": -0.6},
            "frequency_axis": [0.0, 20.0],
            "scan_history": [{"action_type": "ask_prior", "cost": 1.5}],
        }
    )

    assert "recommended_next_action=copy_prior_or_one_cheap_scan_then_submit" in text
    assert "one_cheap_scan_only_when_borderline" in text
