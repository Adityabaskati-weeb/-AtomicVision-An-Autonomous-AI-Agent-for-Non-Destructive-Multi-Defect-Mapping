from __future__ import annotations

from pathlib import Path

import pytest

from atomicvision.models import TrainingConfig, train_defectnet_lite
from atomicvision_env.client import AtomicVisionEnv
from atomicvision_env.models import AtomicVisionAction
from atomicvision_env.server.app import app
from atomicvision_env.server.environment import AtomicVisionEnvironment


def _truth(env: AtomicVisionEnvironment):
    case = env._require_case()
    return [defect.species for defect in case.defects], [
        defect.concentration for defect in case.defects
    ]


def test_reset_returns_initial_observation() -> None:
    env = AtomicVisionEnvironment(difficulty="easy")

    observation = env.reset(seed=1)

    assert observation.done is False
    assert observation.step_count == 0
    assert observation.budget_remaining == 10.0
    assert len(observation.frequency_axis) == len(observation.current_spectrum)
    assert observation.scan_history[0].action_type == "initial_scan"


def test_request_scan_updates_budget_and_history() -> None:
    env = AtomicVisionEnvironment(difficulty="medium")
    env.reset(seed=2)

    observation = env.step(
        AtomicVisionAction(
            action_type="request_scan",
            scan_mode="standard_pdos",
            resolution="medium",
        )
    )

    assert observation.step_count == 1
    assert observation.budget_remaining == 7.0
    assert observation.last_reward == pytest.approx(-0.8)
    assert observation.scan_history[-1].action_type == "request_scan"


def test_compare_reference_reveals_pristine_spectrum() -> None:
    env = AtomicVisionEnvironment(difficulty="easy")
    env.reset(seed=3)

    observation = env.step(AtomicVisionAction(action_type="compare_reference"))

    assert observation.pristine_reference is not None
    assert len(observation.pristine_reference) == len(env._require_case().pristine_spectrum)


def test_ask_prior_returns_candidate_prediction() -> None:
    env = AtomicVisionEnvironment(difficulty="medium")
    env.reset(seed=4)

    observation = env.step(AtomicVisionAction(action_type="ask_prior"))

    assert observation.prior_prediction is not None
    assert observation.prior_prediction.source == "heuristic"
    assert len(observation.prior_prediction.predicted_defects) == len(
        observation.prior_prediction.predicted_concentrations
    )


def test_model_prior_mode_returns_model_source() -> None:
    env = AtomicVisionEnvironment(
        difficulty="easy",
        prior_mode="model",
        prior_threshold=0.0,
    )
    env.reset(seed=8)

    observation = env.step(AtomicVisionAction(action_type="ask_prior"))

    assert observation.prior_prediction is not None
    assert observation.prior_prediction.source == "model"
    assert observation.prior_prediction.predicted_defects
    assert len(observation.prior_prediction.predicted_defects) == len(
        observation.prior_prediction.predicted_concentrations
    )


def test_model_prior_uses_checkpoint_path() -> None:
    checkpoint = Path("outputs/test-artifacts/test_model_prior/defectnet_lite.pt")
    train_defectnet_lite(
        TrainingConfig(
            train_samples=4,
            val_samples=2,
            epochs=1,
            batch_size=2,
            difficulty="easy",
            seed=22,
        ),
        checkpoint_path=checkpoint,
    )
    env = AtomicVisionEnvironment(
        difficulty="easy",
        prior_mode="model",
        prior_checkpoint_path=str(checkpoint),
        prior_threshold=0.0,
    )
    env.reset(seed=9)

    observation = env.step(AtomicVisionAction(action_type="ask_prior"))

    assert observation.prior_prediction is not None
    assert observation.prior_prediction.source == "model"
    assert observation.prior_prediction.checkpoint_path == str(checkpoint)


def test_truth_submission_ends_episode_with_positive_reward() -> None:
    env = AtomicVisionEnvironment(difficulty="easy")
    env.reset(seed=5)
    defects, concentrations = _truth(env)

    observation = env.step(
        AtomicVisionAction(
            action_type="submit_defect_map",
            predicted_defects=defects,
            predicted_concentrations=concentrations,
            confidence=0.95,
        )
    )

    assert observation.done is True
    assert observation.reward is not None
    assert observation.reward > 0.0
    assert observation.reward_breakdown is not None
    assert observation.reward_breakdown["f1"] == 1.0


def test_invalid_zoom_band_returns_penalty_without_crashing() -> None:
    env = AtomicVisionEnvironment(difficulty="easy")
    env.reset(seed=6)

    observation = env.step(
        AtomicVisionAction(
            action_type="zoom_band",
            freq_min=8.0,
            freq_max=2.0,
        )
    )

    assert observation.done is False
    assert observation.last_reward == -1.0
    assert "Invalid action" in observation.message


def test_step_limit_triggers_timeout() -> None:
    env = AtomicVisionEnvironment(difficulty="easy")
    env.reset(seed=7)

    observation = None
    for _ in range(env.state.max_steps):
        observation = env.step(AtomicVisionAction(action_type="compare_reference"))

    assert observation is not None
    assert observation.done is True
    assert observation.reward_breakdown is not None
    assert observation.reward_breakdown["timeout_penalty"] == -2.0


def test_openenv_app_imports() -> None:
    assert app is not None


def test_atomicvision_client_payload_and_parse_hooks() -> None:
    client = AtomicVisionEnv(base_url="http://localhost:7860")
    env = AtomicVisionEnvironment(difficulty="easy")
    observation = env.reset(seed=1)

    payload = client._step_payload(AtomicVisionAction(action_type="ask_prior"))
    parsed = client._parse_result(
        {
            "observation": observation.model_dump(),
            "reward": observation.reward,
            "done": observation.done,
        }
    )

    assert payload["action_type"] == "ask_prior"
    assert parsed.observation.material_id == observation.material_id
    assert parsed.done is False
