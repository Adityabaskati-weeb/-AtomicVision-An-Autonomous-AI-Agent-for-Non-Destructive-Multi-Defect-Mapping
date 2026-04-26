"""Upload-driven heuristic analysis for the AtomicVision Space UI."""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, Field

from atomicvision.synthetic.generator import (
    HOST_FAMILIES,
    _frequency_axis,
    _gaussian,
    _host_spectrum,
    _normalize,
    _species_signature,
)
from atomicvision.synthetic.types import CANDIDATE_DEFECTS, DIFFICULTY_CONFIGS
from atomicvision_env.models import PriorPrediction


class UploadAnalysisRequest(BaseModel):
    """Client payload for upload-driven spectrum analysis."""

    difficulty: Literal["easy", "medium", "hard", "expert"] = "medium"
    filename: str | None = None
    spectrum: list[float] = Field(min_length=4)


class UploadAnalysisMetrics(BaseModel):
    """Metrics surfaced in the premium dashboard."""

    signal_fidelity: float
    certainty: float
    input_bins: int
    analysis_bins: int


class UploadAnalysisObservation(BaseModel):
    """Observation-like payload returned to the frontend."""

    episode_id: str
    material_id: str
    difficulty: str
    host_family: str
    frequency_axis: list[float]
    current_spectrum: list[float]
    pristine_reference: list[float]
    candidate_defects: list[str]
    prior_prediction: PriorPrediction
    budget_remaining: float
    step_count: int
    max_steps: int
    message: str


class UploadAnalysisResponse(BaseModel):
    """Full upload-driven analysis payload."""

    analysis_mode: Literal["upload_driven"] = "upload_driven"
    observation: UploadAnalysisObservation
    metrics: UploadAnalysisMetrics
    difference_spectrum: list[float]


def analyze_uploaded_spectrum(payload: UploadAnalysisRequest) -> UploadAnalysisResponse:
    """Infer a defect map from an uploaded spectrum using synthetic priors."""

    normalized_input = _normalize_input_series(payload.spectrum)
    input_bins = len(normalized_input)
    analysis_bins = max(64, min(128, input_bins if input_bins > 0 else 64))
    analysis_series = _resample_series(normalized_input, analysis_bins)
    axis = _frequency_axis(analysis_bins, 20.0)

    best_host, pristine_reference = _match_reference(axis, analysis_series)
    difference = [abs(current - reference) for current, reference in zip(analysis_series, pristine_reference, strict=True)]

    raw_scores = _score_defects(axis, analysis_series, pristine_reference)
    selected = _select_defect_candidates(raw_scores, payload.difficulty)
    prior = _build_prior_prediction(selected, raw_scores, pristine_reference, analysis_series, payload.difficulty)
    fidelity = _compute_signal_fidelity(analysis_series, pristine_reference)

    label = (payload.filename or "upload").rsplit(".", 1)[0]
    episode_id = f"upload-{payload.difficulty}-{_stable_hash(label)}"
    message = (
        f"Upload matched a {best_host.replace('_', ' ')} reference and surfaced "
        f"{len(prior.predicted_defects)} defect candidate{'s' if len(prior.predicted_defects) != 1 else ''}."
    )

    return UploadAnalysisResponse(
        observation=UploadAnalysisObservation(
            episode_id=episode_id,
            material_id=label,
            difficulty=payload.difficulty,
            host_family=best_host,
            frequency_axis=axis,
            current_spectrum=analysis_series,
            pristine_reference=pristine_reference,
            candidate_defects=list(CANDIDATE_DEFECTS),
            prior_prediction=prior,
            budget_remaining=DIFFICULTY_CONFIGS[payload.difficulty].budget,
            step_count=2,
            max_steps=DIFFICULTY_CONFIGS[payload.difficulty].max_steps,
            message=message,
        ),
        metrics=UploadAnalysisMetrics(
            signal_fidelity=fidelity,
            certainty=prior.confidence,
            input_bins=input_bins,
            analysis_bins=analysis_bins,
        ),
        difference_spectrum=[round(value, 6) for value in difference],
    )


def _normalize_input_series(values: list[float]) -> list[float]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return []
    minimum = min(numeric)
    if minimum < 0.0:
        numeric = [value - minimum for value in numeric]
    return _normalize(numeric)


def _resample_series(values: list[float], target_length: int) -> list[float]:
    if not values:
        return [0.0] * target_length
    if len(values) == target_length:
        return list(values)
    if len(values) == 1:
        return [values[0]] * target_length

    last_index = len(values) - 1
    resampled: list[float] = []
    for index in range(target_length):
        position = (index / max(target_length - 1, 1)) * last_index
        lower = int(position)
        upper = min(lower + 1, last_index)
        blend = position - lower
        resampled.append(values[lower] * (1.0 - blend) + values[upper] * blend)
    return resampled


def _match_reference(axis: list[float], analysis_series: list[float]) -> tuple[str, list[float]]:
    best_host = HOST_FAMILIES[0]
    best_reference = _mean_host_reference(axis, best_host)
    best_error = _mean_absolute_error(analysis_series, best_reference)

    for host_family in HOST_FAMILIES[1:]:
        candidate = _mean_host_reference(axis, host_family)
        candidate_error = _mean_absolute_error(analysis_series, candidate)
        if candidate_error < best_error:
            best_host = host_family
            best_reference = candidate
            best_error = candidate_error
    return best_host, best_reference


def _mean_host_reference(axis: list[float], host_family: str) -> list[float]:
    accumulator = [0.0] * len(axis)
    host_seed = sum(ord(char) for char in host_family)
    for seed_offset in range(4):
        rng = random.Random(17_000 + host_seed * 13 + seed_offset * 101)
        sample = _host_spectrum(axis, host_family, rng)
        for index, value in enumerate(sample):
            accumulator[index] += value
    averaged = [value / 4.0 for value in accumulator]
    return _normalize(averaged)


def _score_defects(
    axis: list[float],
    analysis_series: list[float],
    pristine_reference: list[float],
) -> dict[str, float]:
    positive_delta = [
        max(current - reference, 0.0)
        for current, reference in zip(analysis_series, pristine_reference, strict=True)
    ]
    negative_delta = [
        max(reference - current, 0.0)
        for current, reference in zip(analysis_series, pristine_reference, strict=True)
    ]
    magnitude_delta = [abs(current - reference) for current, reference in zip(analysis_series, pristine_reference, strict=True)]

    scores: dict[str, float] = {}
    for species in CANDIDATE_DEFECTS:
        signature = _species_signature(species)
        added_peak = _weighted_band_energy(axis, positive_delta, signature["center"], signature["width"])
        softened_band = _weighted_band_energy(
            axis,
            negative_delta,
            signature["soften_center"],
            signature["soften_width"],
        )
        broadening = _weighted_band_energy(
            axis,
            magnitude_delta,
            signature["broad_center"],
            signature["broad_width"],
        )
        score = 1.45 * added_peak + 0.95 * softened_band + 0.65 * broadening
        scores[species] = round(score, 6)
    return scores


def _weighted_band_energy(
    axis: list[float],
    values: list[float],
    center: float,
    width: float,
) -> float:
    weighted_total = 0.0
    weight_sum = 0.0
    for frequency, value in zip(axis, values, strict=True):
        weight = _gaussian(frequency, center, max(width, 0.18))
        weighted_total += value * weight
        weight_sum += weight
    if weight_sum <= 0.0:
        return 0.0
    return weighted_total / weight_sum


def _select_defect_candidates(
    scores: dict[str, float],
    difficulty: str,
) -> list[tuple[str, float]]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return []

    top_score = ranked[0][1]
    relative_gate = 0.56 if difficulty in {"easy", "medium"} else 0.45
    absolute_gate = 0.018 if difficulty in {"easy", "medium"} else 0.014
    max_candidates = 3 if difficulty in {"easy", "medium"} else 5

    selected = [
        (species, score)
        for species, score in ranked
        if score >= absolute_gate and score >= top_score * relative_gate
    ][:max_candidates]

    if not selected and top_score > absolute_gate * 0.6:
        selected = [ranked[0]]
    return selected


def _build_prior_prediction(
    selected: list[tuple[str, float]],
    raw_scores: dict[str, float],
    pristine_reference: list[float],
    analysis_series: list[float],
    difficulty: str,
) -> PriorPrediction:
    config = DIFFICULTY_CONFIGS[difficulty]
    top_score = selected[0][1] if selected else max(raw_scores.values(), default=0.0)
    second_score = selected[1][1] if len(selected) > 1 else 0.0
    mae = _mean_absolute_error(analysis_series, pristine_reference)

    predicted_defects = [species for species, _ in selected]
    predicted_concentrations = [
        _score_to_concentration(score, config.max_concentration, config.min_concentration)
        for _, score in selected
    ]

    host_fit = max(0.0, 1.0 - mae * 1.8)
    signal_strength = min(1.0, top_score / 0.08) if top_score else 0.0
    margin_strength = min(1.0, max(top_score - second_score, 0.0) / 0.05)
    confidence = _clamp(
        0.43 + 0.24 * host_fit + 0.21 * signal_strength + 0.10 * margin_strength,
        0.42,
        0.97,
    )

    if not predicted_defects:
        confidence = min(confidence, 0.58)

    return PriorPrediction(
        predicted_defects=predicted_defects,
        predicted_concentrations=predicted_concentrations,
        confidence=round(confidence, 5),
        source="upload_heuristic",
    )


def _score_to_concentration(
    score: float,
    max_concentration: float,
    min_concentration: float,
) -> float:
    floor = max(0.004, min_concentration * 0.8)
    strength = min(1.0, score / 0.08)
    value = floor + strength * (max_concentration * 0.55 - floor)
    return round(_clamp(value, floor, max_concentration), 5)


def _compute_signal_fidelity(current: list[float], reference: list[float]) -> float:
    normalized_error = _mean_absolute_error(current, reference)
    return round(_clamp(100.0 - normalized_error * 55.0, 82.0, 99.6), 1)


def _mean_absolute_error(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / len(left)


def _stable_hash(label: str) -> str:
    total = 0
    for index, char in enumerate(label.lower(), start=1):
        total += index * ord(char)
    return format(total % 0xFFFF, "04x")


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))
