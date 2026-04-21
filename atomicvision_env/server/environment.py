"""OpenEnv environment for AtomicVision."""

from __future__ import annotations

import random
from pathlib import Path

from openenv.core import Environment

from atomicvision.models import (
    DefectNetLite,
    load_defectnet_lite_checkpoint,
    predict_case,
    set_reproducible_seed,
)
from atomicvision.rewards import (
    RewardBreakdown,
    scan_cost_penalty_for,
    score_submission,
)
from atomicvision.synthetic import MaterialCase, ScanResult, generate_case, simulate_scan
from atomicvision_env.models import (
    AtomicVisionAction,
    AtomicVisionObservation,
    AtomicVisionState,
    PriorPrediction,
    ScanRecord,
)


class AtomicVisionEnvironment(
    Environment[AtomicVisionAction, AtomicVisionObservation, AtomicVisionState]
):
    """A simulated non-destructive materials characterization lab."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        difficulty: str = "medium",
        default_seed: int = 0,
        prior_mode: str = "heuristic",
        prior_checkpoint_path: str | None = None,
        prior_threshold: float = 0.5,
    ):
        super().__init__()
        self.default_difficulty = difficulty
        self.default_seed = default_seed
        self.prior_mode = prior_mode
        self.prior_checkpoint_path = prior_checkpoint_path
        self.prior_threshold = prior_threshold
        self._model_prior: DefectNetLite | None = None
        self._case: MaterialCase | None = None
        self._state = AtomicVisionState()
        self._current_axis: list[float] = []
        self._current_spectrum: list[float] = []
        self._scan_history: list[ScanRecord] = []
        self._prior_prediction: PriorPrediction | None = None
        self._reference_visible = False
        self._last_reward = 0.0
        self._reward_breakdown: dict[str, float] | None = None
        self._done = False
        self._message = "AtomicVision environment is ready."

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> AtomicVisionObservation:
        difficulty = kwargs.get("difficulty", self.default_difficulty)
        actual_seed = self.default_seed if seed is None else seed
        self._case = generate_case(seed=actual_seed, difficulty=difficulty)
        initial_scan = simulate_scan(
            self._case,
            scan_mode="quick_pdos",
            resolution="low",
            seed_offset=0,
        )
        self._current_axis = list(initial_scan.frequency_axis)
        self._current_spectrum = list(initial_scan.spectrum)
        self._scan_history = [
            ScanRecord(
                action_type="initial_scan",
                scan_mode=initial_scan.scan_mode,
                resolution=initial_scan.resolution,
                cost=0.0,
            )
        ]
        self._prior_prediction = None
        self._reference_visible = False
        self._last_reward = 0.0
        self._reward_breakdown = None
        self._done = False
        self._message = "Initial low-cost PDoS scan acquired."
        self._state = AtomicVisionState(
            episode_id=episode_id or self._case.material_id,
            step_count=0,
            seed=actual_seed,
            difficulty=difficulty,
            material_id=self._case.material_id,
            done=False,
            budget_remaining=self._case.budget,
            total_scan_cost=0.0,
            max_steps=self._case.max_steps,
            reward_history=[],
        )
        return self._observation()

    def step(
        self,
        action: AtomicVisionAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> AtomicVisionObservation:
        if self._case is None:
            return self.reset()
        if self._done:
            self._message = "Episode is already complete. Call reset() to start another sample."
            return self._observation()

        self._state.step_count += 1
        try:
            if action.action_type == "request_scan":
                self._handle_request_scan(action)
            elif action.action_type == "zoom_band":
                self._handle_zoom_band(action)
            elif action.action_type == "compare_reference":
                self._handle_compare_reference()
            elif action.action_type == "ask_prior":
                self._handle_ask_prior()
            elif action.action_type == "submit_defect_map":
                self._handle_submit(action)
        except ValueError as exc:
            self._last_reward = -1.0
            self._reward_breakdown = {"invalid_action_penalty": -1.0}
            self._message = f"Invalid action: {exc}"

        if not self._done and self._state.step_count >= self._case.max_steps:
            self._handle_timeout("Step limit reached before final submission.")
        if not self._done and self._state.budget_remaining <= 0:
            self._handle_timeout("Scan budget exhausted before final submission.")

        self._state.done = self._done
        self._state.reward_history.append(self._last_reward)
        return self._observation()

    @property
    def state(self) -> AtomicVisionState:
        return self._state

    def _handle_request_scan(self, action: AtomicVisionAction) -> None:
        scan = simulate_scan(
            self._require_case(),
            scan_mode=action.scan_mode or "standard_pdos",
            resolution=action.resolution or "medium",
            seed_offset=self._state.step_count,
        )
        self._apply_scan(action_type="request_scan", scan=scan)
        self._message = f"Acquired {scan.resolution} {scan.scan_mode} scan."

    def _handle_zoom_band(self, action: AtomicVisionAction) -> None:
        if action.freq_min is None or action.freq_max is None:
            raise ValueError("zoom_band requires freq_min and freq_max")
        scan = simulate_scan(
            self._require_case(),
            scan_mode=action.scan_mode or "high_res_pdos",
            resolution=action.resolution or "high",
            freq_min=action.freq_min,
            freq_max=action.freq_max,
            seed_offset=self._state.step_count,
        )
        self._apply_scan(action_type="zoom_band", scan=scan)
        self._message = f"Zoomed into {action.freq_min:.2f}-{action.freq_max:.2f} THz."

    def _handle_compare_reference(self) -> None:
        cost = 0.5
        self._charge_tool_cost(cost)
        self._reference_visible = True
        self._scan_history.append(ScanRecord(action_type="compare_reference", cost=cost))
        self._last_reward = scan_cost_penalty_for(cost)
        self._reward_breakdown = {"scan_cost_penalty": round(self._last_reward, 6)}
        self._message = "Pristine reference spectrum is now visible."

    def _handle_ask_prior(self) -> None:
        cost = 1.5
        self._charge_tool_cost(cost)
        self._prior_prediction = self._build_prior_prediction()
        self._scan_history.append(ScanRecord(action_type="ask_prior", cost=cost))
        self._last_reward = scan_cost_penalty_for(cost)
        self._reward_breakdown = {"scan_cost_penalty": round(self._last_reward, 6)}
        self._message = "DefectNet-lite prior returned a candidate defect map."

    def _handle_submit(self, action: AtomicVisionAction) -> None:
        confidence = 0.0 if action.confidence is None else action.confidence
        breakdown = score_submission(
            self._require_case(),
            action.predicted_defects,
            action.predicted_concentrations,
            confidence=confidence,
            scan_cost=self._state.total_scan_cost,
        )
        self._apply_final_reward(breakdown)
        self._done = True
        self._message = "Final defect map submitted and scored."

    def _handle_timeout(self, message: str) -> None:
        breakdown = score_submission(
            self._require_case(),
            [],
            [],
            confidence=0.0,
            scan_cost=self._state.total_scan_cost,
            timed_out=True,
        )
        self._apply_final_reward(breakdown)
        self._done = True
        self._message = message

    def _apply_scan(self, action_type: str, scan: ScanResult) -> None:
        self._charge_tool_cost(scan.cost)
        self._current_axis = list(scan.frequency_axis)
        self._current_spectrum = list(scan.spectrum)
        self._scan_history.append(
            ScanRecord(
                action_type=action_type,
                scan_mode=scan.scan_mode,
                resolution=scan.resolution,
                cost=scan.cost,
                freq_min=scan.freq_min,
                freq_max=scan.freq_max,
            )
        )
        self._last_reward = scan_cost_penalty_for(scan.cost)
        self._reward_breakdown = {"scan_cost_penalty": round(self._last_reward, 6)}

    def _charge_tool_cost(self, cost: float) -> None:
        self._state.total_scan_cost = round(self._state.total_scan_cost + cost, 6)
        self._state.budget_remaining = round(self._state.budget_remaining - cost, 6)

    def _build_prior_prediction(self) -> PriorPrediction:
        if self.prior_mode == "model":
            return self._build_model_prior_prediction()
        if self.prior_mode != "heuristic":
            raise ValueError(f"Unknown prior_mode: {self.prior_mode}")
        return self._build_heuristic_prior_prediction()

    def _build_heuristic_prior_prediction(self) -> PriorPrediction:
        case = self._require_case()
        rng = random.Random(case.seed + 50_000 + self._state.step_count)
        predicted_defects: list[str] = []
        predicted_concentrations: list[float] = []
        for defect in case.defects:
            if rng.random() <= 0.78:
                predicted_defects.append(defect.species)
                noisy = max(0.0, defect.concentration + rng.uniform(-0.015, 0.015))
                predicted_concentrations.append(round(noisy, 5))
        if rng.random() <= 0.35:
            false_candidates = [
                species
                for species in case.candidate_defects
                if species not in predicted_defects
            ]
            if false_candidates:
                predicted_defects.append(rng.choice(false_candidates))
                predicted_concentrations.append(round(rng.uniform(0.005, 0.04), 5))
        confidence = min(0.9, 0.35 + 0.08 * len(predicted_defects))
        return PriorPrediction(
            predicted_defects=predicted_defects,
            predicted_concentrations=predicted_concentrations,
            confidence=round(confidence, 3),
            source="heuristic",
        )

    def _build_model_prior_prediction(self) -> PriorPrediction:
        case = self._require_case()
        model = self._get_model_prior()
        prediction = predict_case(model, case, threshold=self.prior_threshold)
        return PriorPrediction(
            predicted_defects=prediction.predicted_defects,
            predicted_concentrations=prediction.predicted_concentrations,
            confidence=prediction.confidence,
            source="model",
            checkpoint_path=self.prior_checkpoint_path,
        )

    def _get_model_prior(self) -> DefectNetLite:
        if self._model_prior is not None:
            return self._model_prior

        if self.prior_checkpoint_path:
            checkpoint_path = Path(self.prior_checkpoint_path)
            if not checkpoint_path.exists():
                raise ValueError(f"Prior checkpoint not found: {checkpoint_path}")
            self._model_prior = load_defectnet_lite_checkpoint(checkpoint_path)
        else:
            set_reproducible_seed(self.default_seed)
            self._model_prior = DefectNetLite()
            self._model_prior.eval()
        return self._model_prior

    def _apply_final_reward(self, breakdown: RewardBreakdown) -> None:
        self._last_reward = breakdown.total_reward
        self._reward_breakdown = {
            "identity_reward": breakdown.identity_reward,
            "concentration_reward": breakdown.concentration_reward,
            "confidence_reward": breakdown.confidence_reward,
            "false_positive_penalty": breakdown.false_positive_penalty,
            "missed_defect_penalty": breakdown.missed_defect_penalty,
            "scan_cost_penalty": breakdown.scan_cost_penalty,
            "timeout_penalty": breakdown.timeout_penalty,
            "f1": breakdown.f1,
            "concentration_mae": breakdown.concentration_mae,
        }

    def _observation(self) -> AtomicVisionObservation:
        case = self._require_case()
        return AtomicVisionObservation(
            done=self._done,
            reward=self._last_reward,
            episode_id=self._state.episode_id or case.material_id,
            material_id=case.material_id,
            difficulty=case.difficulty,
            host_family=case.host_family,
            frequency_axis=self._current_axis,
            current_spectrum=self._current_spectrum,
            pristine_reference=list(case.pristine_spectrum) if self._reference_visible else None,
            scan_history=list(self._scan_history),
            candidate_defects=list(case.candidate_defects),
            prior_prediction=self._prior_prediction,
            budget_remaining=self._state.budget_remaining,
            step_count=self._state.step_count,
            max_steps=case.max_steps,
            last_reward=self._last_reward,
            reward_breakdown=self._reward_breakdown,
            message=self._message,
        )

    def _require_case(self) -> MaterialCase:
        if self._case is None:
            raise RuntimeError("environment has not been reset")
        return self._case
