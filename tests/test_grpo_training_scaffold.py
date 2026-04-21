from __future__ import annotations

import inspect
from argparse import Namespace
from typing import get_args, get_type_hints

import pytest

from training.train_grpo_atomicvision import (
    AtomicVisionToolEnv,
    TRAINING_PRESETS,
    _apply_preset,
    _env_url,
    _format_observation,
    _is_retryable_connection_error,
    build_dataset,
    build_prompt_rows,
    reward_func,
)


def test_prompt_rows_match_grpo_dataset_shape() -> None:
    rows = build_prompt_rows(samples=3, difficulty="hard")

    assert len(rows["prompt"]) == 3
    assert rows["prompt"][0][0]["role"] == "user"
    assert "AtomicVision" in rows["prompt"][0][0]["content"]
    assert "quick_pdos" in rows["prompt"][0][0]["content"]
    assert "0.0 to 20.0" in rows["prompt"][0][0]["content"]
    assert "ask_prior" in rows["prompt"][0][0]["content"]
    assert "confidence 0.50 or higher" in rows["prompt"][0][0]["content"]
    assert "standard_pdos costs 2.0" in rows["prompt"][0][0]["content"]
    assert "submit_defect_map is terminal" in rows["prompt"][0][0]["content"]
    assert rows["seed"] == [0, 1, 2]
    assert rows["difficulty"] == ["hard", "hard", "hard"]


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

    assert reward_func([DummyEnv(1.25), DummyEnv(-0.5)]) == [1.25, -0.5]


def test_training_presets_keep_grpo_generation_batch_valid() -> None:
    for preset in TRAINING_PRESETS.values():
        generation_batch_size = (
            preset["per_device_train_batch_size"]
            * preset["gradient_accumulation_steps"]
        )

        assert generation_batch_size % preset["num_generations"] == 0
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
