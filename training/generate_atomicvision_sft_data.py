"""Generate AtomicVision supervised tool-use data for SFT.

The generated JSONL is intentionally Kaggle/HF-Jobs friendly: each row contains
a `messages` chat transcript plus metadata. It can create exact prior-copy rows
and curated scan-improvement rows where one cheap reference comparison improves
the final scored map.
"""

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

from atomicvision_env.models import AtomicVisionAction  # noqa: E402
from atomicvision_env.server.environment import AtomicVisionEnvironment  # noqa: E402
from atomicvision.rewards import score_submission  # noqa: E402
from atomicvision.synthetic.types import MaterialCase  # noqa: E402
from training.seed_ranges import SFT_TRAIN_SEED_START  # noqa: E402
from training.train_grpo_atomicvision import (  # noqa: E402
    DEFAULT_PROMPT,
    TOOL_SYSTEM_PROMPT,
    _select_prompt_seeds,
    _format_observation,
)


TOOL_SYSTEM = TOOL_SYSTEM_PROMPT

ASK_PRIOR_CALL = {"name": "ask_prior", "arguments": {}}
COMPARE_REFERENCE_CALL = {"name": "compare_reference", "arguments": {}}
BASE_SAMPLE_TYPES = {"ask_prior", "submit_prior"}
SCAN_SAMPLE_TYPE = "submit_after_reference"
SAMPLE_TYPES = BASE_SAMPLE_TYPES | {SCAN_SAMPLE_TYPE}
COST_AWARE_SUBMIT_PRIOR_RATIO = 0.85
COST_AWARE_REFERENCE_RATIO = 0.10
FORMAT_REPAIR_SUBMIT_PRIOR_RATIO = 0.40
FORMAT_REPAIR_REFERENCE_RATIO = 0.10
SUBMIT_BRIDGE_SUBMIT_PRIOR_RATIO = 0.75
SUBMIT_BRIDGE_REFERENCE_RATIO = 0.10
FORMAT_REFRESH_SUBMIT_PRIOR_RATIO = 0.90
FORMAT_REFRESH_REFERENCE_RATIO = 0.0
STRICT_XML_SUBMIT_REFRESH_MIN_SCAN_IMPROVEMENT = 0.10
STRICT_XML_SUBMIT_REFRESH_MAX_SCAN_CANDIDATES = 2048
STRICT_SUBMIT_CONTRACT_REFERENCE_FRACTION = 0.25
HARD_RECALL_GOOD_PRIOR_FRACTION = 0.35
HARD_RECALL_MIN_SCAN_IMPROVEMENT = 0.15
HARD_RECALL_MAX_SCAN_CANDIDATES = 4096
HARD_RECALL_MAX_ORACLE_GAP = 0.50
HARD_RECALL_MIN_PRIOR_CONFIDENCE = 0.0
HARD_RECALL_MAX_PRIOR_CONFIDENCE = 1.0
HARD_RECALL_MICRO_GOOD_PRIOR_FRACTION = 0.25
HARD_RECALL_MICRO_MIN_SCAN_IMPROVEMENT = 0.25
HARD_RECALL_MICRO_MAX_SCAN_CANDIDATES = 4096
HARD_RECALL_MICRO_MAX_ORACLE_GAP = 0.30
HARD_RECALL_MICRO_MIN_PRIOR_CONFIDENCE = 0.55
HARD_RECALL_MICRO_MAX_PRIOR_CONFIDENCE = 0.90
HARD_FRONTIER_SUBMIT_PRIOR_RATIO = 0.85
HARD_FRONTIER_REFERENCE_RATIO = 0.10
HARD_FRONTIER_MIN_SCAN_IMPROVEMENT = 0.15
HARD_FRONTIER_MAX_SCAN_CANDIDATES = 1024


def build_sft_examples(
    episodes_per_difficulty: int,
    difficulties: tuple[str, ...] = ("medium",),
    seed_start: int = SFT_TRAIN_SEED_START,
    sample_types: tuple[str, ...] = ("ask_prior", "submit_prior"),
    min_scan_improvement: float = 0.25,
    max_scan_candidates_per_difficulty: int | None = None,
    structured_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Build SFT examples from deterministic local AtomicVision episodes."""

    if episodes_per_difficulty <= 0:
        raise ValueError("episodes_per_difficulty must be positive")
    unknown_types = [kind for kind in sample_types if kind not in SAMPLE_TYPES]
    if unknown_types:
        raise ValueError(f"Unknown sample types: {', '.join(unknown_types)}")

    examples: list[dict[str, Any]] = []
    base_sample_types = tuple(kind for kind in sample_types if kind in BASE_SAMPLE_TYPES)
    for difficulty in difficulties:
        if base_sample_types:
            for seed in range(seed_start, seed_start + episodes_per_difficulty):
                examples.extend(
                    build_episode_examples(
                        seed=seed,
                        difficulty=difficulty,
                        sample_types=base_sample_types,
                        structured_tool_calls=structured_tool_calls,
                    )
                )
        if SCAN_SAMPLE_TYPE in sample_types:
            examples.extend(
                build_scan_improvement_examples(
                    target_examples=episodes_per_difficulty,
                    difficulty=difficulty,
                    seed_start=seed_start,
                    min_scan_improvement=min_scan_improvement,
                    max_candidate_seeds=max_scan_candidates_per_difficulty,
                    structured_tool_calls=structured_tool_calls,
                )
            )
    return examples


def build_cost_aware_sft_examples(
    examples_per_difficulty: int,
    difficulties: tuple[str, ...] = ("medium",),
    seed_start: int = 0,
    submit_prior_ratio: float = COST_AWARE_SUBMIT_PRIOR_RATIO,
    reference_ratio: float = COST_AWARE_REFERENCE_RATIO,
    min_scan_improvement: float = 0.25,
    max_scan_candidates_per_difficulty: int | None = None,
    structured_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Build a cheap-prior-biased SFT set for cost-aware tool use.

    The resulting mix is mostly ask_prior -> submit_defect_map rows, with a
    small number of curated compare_reference rows and first-call ask_prior
    format rows. This intentionally avoids teaching the model that expensive
    tools should be used by default.
    """

    if examples_per_difficulty <= 0:
        raise ValueError("examples_per_difficulty must be positive")
    if not 0.0 <= submit_prior_ratio <= 1.0:
        raise ValueError("submit_prior_ratio must be between 0 and 1")
    if not 0.0 <= reference_ratio <= 1.0:
        raise ValueError("reference_ratio must be between 0 and 1")
    if submit_prior_ratio + reference_ratio > 1.0:
        raise ValueError("submit_prior_ratio + reference_ratio must be <= 1")

    examples: list[dict[str, Any]] = []
    for difficulty in difficulties:
        submit_count = int(round(examples_per_difficulty * submit_prior_ratio))
        reference_count = int(round(examples_per_difficulty * reference_ratio))
        if submit_count + reference_count > examples_per_difficulty:
            reference_count = max(0, examples_per_difficulty - submit_count)
        ask_count = examples_per_difficulty - submit_count - reference_count

        for seed in range(seed_start, seed_start + submit_count):
            examples.extend(
                build_episode_examples(
                    seed=seed,
                    difficulty=difficulty,
                    sample_types=("submit_prior",),
                    structured_tool_calls=structured_tool_calls,
                )
            )
        for seed in range(seed_start + submit_count, seed_start + submit_count + ask_count):
            examples.extend(
                build_episode_examples(
                    seed=seed,
                    difficulty=difficulty,
                    sample_types=("ask_prior",),
                    structured_tool_calls=structured_tool_calls,
                )
            )
        if reference_count:
            examples.extend(
                build_scan_improvement_examples(
                    target_examples=reference_count,
                    difficulty=difficulty,
                    seed_start=seed_start,
                    min_scan_improvement=min_scan_improvement,
                    max_candidate_seeds=max_scan_candidates_per_difficulty,
                    structured_tool_calls=structured_tool_calls,
                )
            )
    return examples


def build_two_step_curriculum_examples(
    examples_per_difficulty: int,
    difficulties: tuple[str, ...] = ("medium",),
    seed_start: int = 0,
    min_scan_improvement: float = 0.25,
    max_scan_candidates_per_difficulty: int | None = None,
    structured_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Build the official two-stage schema curriculum used for held-out repair."""

    repair = build_cost_aware_sft_examples(
        examples_per_difficulty=examples_per_difficulty,
        difficulties=difficulties,
        seed_start=seed_start,
        submit_prior_ratio=FORMAT_REPAIR_SUBMIT_PRIOR_RATIO,
        reference_ratio=FORMAT_REPAIR_REFERENCE_RATIO,
        min_scan_improvement=min_scan_improvement,
        max_scan_candidates_per_difficulty=max_scan_candidates_per_difficulty,
        structured_tool_calls=structured_tool_calls,
    )
    bridge = build_cost_aware_sft_examples(
        examples_per_difficulty=examples_per_difficulty,
        difficulties=difficulties,
        seed_start=seed_start,
        submit_prior_ratio=SUBMIT_BRIDGE_SUBMIT_PRIOR_RATIO,
        reference_ratio=SUBMIT_BRIDGE_REFERENCE_RATIO,
        min_scan_improvement=min_scan_improvement,
        max_scan_candidates_per_difficulty=max_scan_candidates_per_difficulty,
        structured_tool_calls=structured_tool_calls,
    )
    return [*repair, *bridge]


def build_hard_frontier_boost_examples(
    examples_per_difficulty: int,
    difficulties: tuple[str, ...] = ("hard",),
    seed_start: int = 0,
    submit_prior_ratio: float = HARD_FRONTIER_SUBMIT_PRIOR_RATIO,
    reference_ratio: float = HARD_FRONTIER_REFERENCE_RATIO,
    min_scan_improvement: float = HARD_FRONTIER_MIN_SCAN_IMPROVEMENT,
    max_scan_candidates_per_difficulty: int | None = HARD_FRONTIER_MAX_SCAN_CANDIDATES,
    structured_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Build a hard-focused continuation set with wider reference search."""

    return build_cost_aware_sft_examples(
        examples_per_difficulty=examples_per_difficulty,
        difficulties=difficulties,
        seed_start=seed_start,
        submit_prior_ratio=submit_prior_ratio,
        reference_ratio=reference_ratio,
        min_scan_improvement=min_scan_improvement,
        max_scan_candidates_per_difficulty=max_scan_candidates_per_difficulty,
        structured_tool_calls=structured_tool_calls,
    )


def build_format_refresh_examples(
    examples_per_difficulty: int,
    difficulties: tuple[str, ...] = ("hard",),
    seed_start: int = 0,
    submit_prior_ratio: float = FORMAT_REFRESH_SUBMIT_PRIOR_RATIO,
    reference_ratio: float = FORMAT_REFRESH_REFERENCE_RATIO,
    min_scan_improvement: float = 0.25,
    max_scan_candidates_per_difficulty: int | None = None,
    structured_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Build a tiny strict-envelope refresh set before hard-only GRPO.

    This profile is intentionally dominated by exact ask_prior -> submit_defect_map
    rows so a small continuation can refresh the XML-wrapped tool-call contract
    without spending many updates relearning broader behavior.
    """

    return build_cost_aware_sft_examples(
        examples_per_difficulty=examples_per_difficulty,
        difficulties=difficulties,
        seed_start=seed_start,
        submit_prior_ratio=submit_prior_ratio,
        reference_ratio=reference_ratio,
        min_scan_improvement=min_scan_improvement,
        max_scan_candidates_per_difficulty=max_scan_candidates_per_difficulty,
        structured_tool_calls=structured_tool_calls,
    )


def build_strict_submit_contract_refresh_examples(
    examples_per_difficulty: int,
    difficulties: tuple[str, ...] = ("hard",),
    seed_start: int = 0,
    reference_fraction: float = STRICT_SUBMIT_CONTRACT_REFERENCE_FRACTION,
    submit_prior_ratio: float = FORMAT_REFRESH_SUBMIT_PRIOR_RATIO,
    min_scan_improvement: float = STRICT_XML_SUBMIT_REFRESH_MIN_SCAN_IMPROVEMENT,
    max_scan_candidates_per_difficulty: int | None = STRICT_XML_SUBMIT_REFRESH_MAX_SCAN_CANDIDATES,
    structured_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Build a strict-submit repair set that mixes both known helpful regimes.

    We keep the V12-style submit-heavy strict-envelope warmup, then add a smaller
    batch of final reference->submit turns from the exact hard prompt pool where
    GRPO has been drifting into tagless repaired output.
    """

    if examples_per_difficulty <= 0:
        raise ValueError("examples_per_difficulty must be positive")
    if not 0.0 <= reference_fraction <= 1.0:
        raise ValueError("reference_fraction must be between 0 and 1")

    reference_examples = int(round(examples_per_difficulty * reference_fraction))
    if examples_per_difficulty >= 4:
        reference_examples = max(1, reference_examples)
    reference_examples = min(reference_examples, examples_per_difficulty)
    format_examples = examples_per_difficulty - reference_examples

    examples: list[dict[str, Any]] = []
    if format_examples > 0:
        examples.extend(
            build_format_refresh_examples(
                examples_per_difficulty=format_examples,
                difficulties=difficulties,
                seed_start=seed_start,
                submit_prior_ratio=submit_prior_ratio,
                reference_ratio=FORMAT_REFRESH_REFERENCE_RATIO,
                min_scan_improvement=min_scan_improvement,
                max_scan_candidates_per_difficulty=max_scan_candidates_per_difficulty,
                structured_tool_calls=structured_tool_calls,
            )
        )
    if reference_examples > 0:
        examples.extend(
            build_strict_xml_submit_refresh_examples(
                examples_per_difficulty=reference_examples,
                difficulties=difficulties,
                seed_start=seed_start,
                min_scan_improvement=min_scan_improvement,
                max_scan_candidates_per_difficulty=max_scan_candidates_per_difficulty,
                structured_tool_calls=structured_tool_calls,
            )
        )
    return examples


def build_hard_recall_repair_examples(
    examples_per_difficulty: int,
    difficulties: tuple[str, ...] = ("hard",),
    seed_start: int = 0,
    good_prior_fraction: float = HARD_RECALL_GOOD_PRIOR_FRACTION,
    min_scan_improvement: float = HARD_RECALL_MIN_SCAN_IMPROVEMENT,
    max_scan_candidates_per_difficulty: int | None = HARD_RECALL_MAX_SCAN_CANDIDATES,
    max_oracle_gap_for_submit: float = HARD_RECALL_MAX_ORACLE_GAP,
    min_prior_confidence: float = HARD_RECALL_MIN_PRIOR_CONFIDENCE,
    max_prior_confidence: float = HARD_RECALL_MAX_PRIOR_CONFIDENCE,
    structured_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Build a hard-only repair set for missed-defect recall failures.

    This profile teaches a sharp contrast:
    - if the prior already covers the full defect set and is near-oracle, submit it
    - if the prior misses species and compare_reference materially helps, use it before submit
    """

    if examples_per_difficulty <= 0:
        raise ValueError("examples_per_difficulty must be positive")
    if not 0.0 <= good_prior_fraction <= 1.0:
        raise ValueError("good_prior_fraction must be between 0 and 1")
    if not 0.0 <= min_prior_confidence <= 1.0:
        raise ValueError("min_prior_confidence must be between 0 and 1")
    if not 0.0 <= max_prior_confidence <= 1.0:
        raise ValueError("max_prior_confidence must be between 0 and 1")
    if min_prior_confidence > max_prior_confidence:
        raise ValueError("min_prior_confidence must be <= max_prior_confidence")

    good_prior_examples = int(round(examples_per_difficulty * good_prior_fraction))
    if examples_per_difficulty >= 4:
        good_prior_examples = max(1, good_prior_examples)
    good_prior_examples = min(good_prior_examples, examples_per_difficulty)
    recall_repair_examples = examples_per_difficulty - good_prior_examples
    search_span = max_scan_candidates_per_difficulty or max(64, examples_per_difficulty * 32)

    examples: list[dict[str, Any]] = []
    for difficulty in difficulties:
        difficulty_examples: list[dict[str, Any]] = []
        if recall_repair_examples > 0:
            difficulty_examples.extend(
                build_missing_defect_recovery_examples(
                    target_examples=recall_repair_examples,
                    difficulty=difficulty,
                    seed_start=seed_start,
                    min_scan_improvement=min_scan_improvement,
                    max_candidate_seeds=max_scan_candidates_per_difficulty,
                    min_prior_confidence=min_prior_confidence,
                    max_prior_confidence=max_prior_confidence,
                    structured_tool_calls=structured_tool_calls,
                )
            )
        if good_prior_examples > 0:
            difficulty_examples.extend(
                _dedupe_examples_by_sample_id(
                    build_good_prior_submit_examples(
                        target_examples=good_prior_examples,
                        difficulty=difficulty,
                        seed_start=seed_start,
                        max_candidate_seeds=max_scan_candidates_per_difficulty,
                        max_oracle_gap=max_oracle_gap_for_submit,
                        structured_tool_calls=structured_tool_calls,
                    ),
                    existing=difficulty_examples,
                )
            )
        if len(difficulty_examples) < examples_per_difficulty and good_prior_examples > 0:
            difficulty_examples.extend(
                _dedupe_examples_by_sample_id(
                    build_good_prior_submit_examples(
                        target_examples=examples_per_difficulty - len(difficulty_examples),
                        difficulty=difficulty,
                        seed_start=seed_start + search_span,
                        max_candidate_seeds=search_span,
                        max_oracle_gap=max_oracle_gap_for_submit,
                        structured_tool_calls=structured_tool_calls,
                    ),
                    existing=difficulty_examples,
                )
            )
        if len(difficulty_examples) < examples_per_difficulty:
            difficulty_examples.extend(
                _dedupe_examples_by_sample_id(
                    build_missing_defect_recovery_examples(
                        target_examples=examples_per_difficulty - len(difficulty_examples),
                        difficulty=difficulty,
                        seed_start=seed_start + search_span,
                        min_scan_improvement=min_scan_improvement,
                        max_candidate_seeds=search_span,
                        min_prior_confidence=min_prior_confidence,
                        max_prior_confidence=max_prior_confidence,
                        structured_tool_calls=structured_tool_calls,
                    ),
                    existing=difficulty_examples,
                )
            )
        if len(difficulty_examples) < examples_per_difficulty:
            difficulty_examples.extend(
                _dedupe_examples_by_sample_id(
                    build_missing_defect_recovery_examples(
                        target_examples=examples_per_difficulty - len(difficulty_examples),
                        difficulty=difficulty,
                        seed_start=seed_start + (search_span * 2),
                        min_scan_improvement=min_scan_improvement,
                        max_candidate_seeds=search_span * 2,
                        min_prior_confidence=min_prior_confidence,
                        max_prior_confidence=max_prior_confidence,
                        structured_tool_calls=structured_tool_calls,
                    ),
                    existing=difficulty_examples,
                )
            )
        if len(difficulty_examples) > examples_per_difficulty:
            difficulty_examples = difficulty_examples[:examples_per_difficulty]
        if len(difficulty_examples) < examples_per_difficulty:
            print(
                "Warning: hard_recall_repair produced fewer examples than requested "
                f"for {difficulty}: wanted {examples_per_difficulty}, got {len(difficulty_examples)}",
                flush=True,
            )
        examples.extend(difficulty_examples)
    return examples


def _dedupe_examples_by_sample_id(
    examples: list[dict[str, Any]],
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = {str(example.get("sample_id")) for example in existing}
    unique: list[dict[str, Any]] = []
    for example in examples:
        sample_id = str(example.get("sample_id"))
        if sample_id in seen:
            continue
        seen.add(sample_id)
        unique.append(example)
    return unique


def build_hard_recall_micro_repair_examples(
    examples_per_difficulty: int,
    difficulties: tuple[str, ...] = ("hard",),
    seed_start: int = 0,
    structured_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Build a tiny high-precision hard-recall repair set.

    This profile is intentionally narrower than ``hard_recall_repair``:
    it focuses on confident-but-incomplete priors plus a small number of
    near-oracle submit anchors, so a short continuation can target recall
    without dragging medium quality.
    """

    return build_hard_recall_repair_examples(
        examples_per_difficulty=examples_per_difficulty,
        difficulties=difficulties,
        seed_start=seed_start,
        good_prior_fraction=HARD_RECALL_MICRO_GOOD_PRIOR_FRACTION,
        min_scan_improvement=HARD_RECALL_MICRO_MIN_SCAN_IMPROVEMENT,
        max_scan_candidates_per_difficulty=HARD_RECALL_MICRO_MAX_SCAN_CANDIDATES,
        max_oracle_gap_for_submit=HARD_RECALL_MICRO_MAX_ORACLE_GAP,
        min_prior_confidence=HARD_RECALL_MICRO_MIN_PRIOR_CONFIDENCE,
        max_prior_confidence=HARD_RECALL_MICRO_MAX_PRIOR_CONFIDENCE,
        structured_tool_calls=structured_tool_calls,
    )


def _planned_reference_examples_per_difficulty(
    examples_per_difficulty: int,
    submit_prior_ratio: float,
    reference_ratio: float,
) -> int:
    """Mirror the build_cost_aware_sft_examples reference-count math."""

    submit_count = int(round(examples_per_difficulty * submit_prior_ratio))
    reference_count = int(round(examples_per_difficulty * reference_ratio))
    if submit_count + reference_count > examples_per_difficulty:
        reference_count = max(0, examples_per_difficulty - submit_count)
    return reference_count


def _planned_strict_submit_contract_reference_examples_per_difficulty(
    examples_per_difficulty: int,
    reference_fraction: float,
) -> int:
    """Mirror the mixed strict-submit contract refresh reference-count math."""

    reference_examples = int(round(examples_per_difficulty * reference_fraction))
    if examples_per_difficulty >= 4:
        reference_examples = max(1, reference_examples)
    return min(reference_examples, examples_per_difficulty)


def _planned_good_prior_examples_per_difficulty(
    examples_per_difficulty: int,
    good_prior_fraction: float,
) -> int:
    """Mirror the good-prior count math for hard recall repair."""

    good_prior_examples = int(round(examples_per_difficulty * good_prior_fraction))
    if examples_per_difficulty >= 4:
        good_prior_examples = max(1, good_prior_examples)
    return min(good_prior_examples, examples_per_difficulty)


def _expected_scan_examples(args: argparse.Namespace) -> int:
    """Return the intended number of submit_after_reference rows for the current CLI plan."""

    difficulties = len(args.difficulties)
    if args.profile == "explicit":
        if SCAN_SAMPLE_TYPE not in args.sample_types:
            return 0
        return args.episodes_per_difficulty * difficulties
    if args.profile == "cost_aware":
        return _planned_reference_examples_per_difficulty(
            args.episodes_per_difficulty,
            args.submit_prior_ratio,
            args.reference_ratio,
        ) * difficulties
    if args.profile == "format_repair":
        submit_prior_ratio = args.submit_prior_ratio
        reference_ratio = args.reference_ratio
        if submit_prior_ratio == COST_AWARE_SUBMIT_PRIOR_RATIO:
            submit_prior_ratio = FORMAT_REPAIR_SUBMIT_PRIOR_RATIO
        if reference_ratio == COST_AWARE_REFERENCE_RATIO:
            reference_ratio = FORMAT_REPAIR_REFERENCE_RATIO
        return _planned_reference_examples_per_difficulty(
            args.episodes_per_difficulty,
            submit_prior_ratio,
            reference_ratio,
        ) * difficulties
    if args.profile == "submit_bridge":
        submit_prior_ratio = args.submit_prior_ratio
        reference_ratio = args.reference_ratio
        if submit_prior_ratio == COST_AWARE_SUBMIT_PRIOR_RATIO:
            submit_prior_ratio = SUBMIT_BRIDGE_SUBMIT_PRIOR_RATIO
        if reference_ratio == COST_AWARE_REFERENCE_RATIO:
            reference_ratio = SUBMIT_BRIDGE_REFERENCE_RATIO
        return _planned_reference_examples_per_difficulty(
            args.episodes_per_difficulty,
            submit_prior_ratio,
            reference_ratio,
        ) * difficulties
    if args.profile == "hard_frontier_boost":
        submit_prior_ratio = args.submit_prior_ratio
        reference_ratio = args.reference_ratio
        if submit_prior_ratio == COST_AWARE_SUBMIT_PRIOR_RATIO:
            submit_prior_ratio = HARD_FRONTIER_SUBMIT_PRIOR_RATIO
        if reference_ratio == COST_AWARE_REFERENCE_RATIO:
            reference_ratio = HARD_FRONTIER_REFERENCE_RATIO
        return _planned_reference_examples_per_difficulty(
            args.episodes_per_difficulty,
            submit_prior_ratio,
            reference_ratio,
        ) * difficulties
    if args.profile == "format_refresh":
        submit_prior_ratio = args.submit_prior_ratio
        reference_ratio = args.reference_ratio
        if submit_prior_ratio == COST_AWARE_SUBMIT_PRIOR_RATIO:
            submit_prior_ratio = FORMAT_REFRESH_SUBMIT_PRIOR_RATIO
        if reference_ratio == COST_AWARE_REFERENCE_RATIO:
            reference_ratio = FORMAT_REFRESH_REFERENCE_RATIO
        return _planned_reference_examples_per_difficulty(
            args.episodes_per_difficulty,
            submit_prior_ratio,
            reference_ratio,
        ) * difficulties
    if args.profile == "strict_submit_contract_refresh":
        return _planned_strict_submit_contract_reference_examples_per_difficulty(
            args.episodes_per_difficulty,
            args.reference_fraction,
        ) * difficulties
    if args.profile == "hard_recall_repair":
        return (
            args.episodes_per_difficulty
            - _planned_good_prior_examples_per_difficulty(
                args.episodes_per_difficulty,
                HARD_RECALL_GOOD_PRIOR_FRACTION,
            )
        ) * difficulties
    if args.profile == "hard_recall_micro_repair":
        return (
            args.episodes_per_difficulty
            - _planned_good_prior_examples_per_difficulty(
                args.episodes_per_difficulty,
                HARD_RECALL_MICRO_GOOD_PRIOR_FRACTION,
            )
        ) * difficulties
    if args.profile == "strict_xml_submit_refresh":
        return args.episodes_per_difficulty * difficulties
    if args.profile == "two_step_curriculum":
        repair_reference = _planned_reference_examples_per_difficulty(
            args.episodes_per_difficulty,
            FORMAT_REPAIR_SUBMIT_PRIOR_RATIO,
            FORMAT_REPAIR_REFERENCE_RATIO,
        )
        bridge_reference = _planned_reference_examples_per_difficulty(
            args.episodes_per_difficulty,
            SUBMIT_BRIDGE_SUBMIT_PRIOR_RATIO,
            SUBMIT_BRIDGE_REFERENCE_RATIO,
        )
        return (repair_reference + bridge_reference) * difficulties
    return 0


def build_strict_xml_submit_refresh_examples(
    examples_per_difficulty: int,
    difficulties: tuple[str, ...] = ("hard",),
    seed_start: int = 0,
    min_scan_improvement: float = STRICT_XML_SUBMIT_REFRESH_MIN_SCAN_IMPROVEMENT,
    max_scan_candidates_per_difficulty: int | None = STRICT_XML_SUBMIT_REFRESH_MAX_SCAN_CANDIDATES,
    structured_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Build a submit-only hard refresh set shaped around GRPO XML failures.

    This profile intentionally mirrors the hard/reference-improvement prompt pool
    used by the GRPO probe. Every target is the *final* wrapped
    submit_defect_map call after ask_prior plus compare_reference, which is the
    exact turn where the policy has been drifting into tagless repaired output.
    """

    if examples_per_difficulty <= 0:
        raise ValueError("examples_per_difficulty must be positive")

    examples: list[dict[str, Any]] = []
    for difficulty in difficulties:
        seeds = _select_prompt_seeds(
            samples=examples_per_difficulty,
            difficulty=difficulty,
            seed_start=seed_start,
            prompt_focus="reference-improvement",
            min_prior_confidence=0.45,
            max_prior_confidence=0.65,
            min_reference_improvement=min_scan_improvement,
            max_seed_candidates=max_scan_candidates_per_difficulty,
        )
        for seed in seeds:
            example = build_scan_improvement_example(
                seed=seed,
                difficulty=difficulty,
                min_scan_improvement=min_scan_improvement,
                structured_tool_calls=structured_tool_calls,
            )
            if example is None:
                raise ValueError(
                    "Selected a reference-improvement seed that did not produce a "
                    "submit_after_reference example. Lower the threshold or widen "
                    "the scan search.",
                )
            examples.append(example)
    return examples


def build_good_prior_submit_examples(
    target_examples: int,
    difficulty: str = "hard",
    seed_start: int = 0,
    max_candidate_seeds: int | None = None,
    max_oracle_gap: float = HARD_RECALL_MAX_ORACLE_GAP,
    structured_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Build submit-prior rows only where the prior is already near-oracle."""

    if target_examples <= 0:
        raise ValueError("target_examples must be positive")
    if max_oracle_gap < 0.0:
        raise ValueError("max_oracle_gap must be non-negative")

    search_limit = max_candidate_seeds or max(64, target_examples * 32)
    examples: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + search_limit):
        example = build_good_prior_submit_example(
            seed=seed,
            difficulty=difficulty,
            max_oracle_gap=max_oracle_gap,
            structured_tool_calls=structured_tool_calls,
        )
        if example is None:
            continue
        examples.append(example)
        if len(examples) >= target_examples:
            break
    return examples


def build_good_prior_submit_example(
    seed: int,
    difficulty: str = "hard",
    max_oracle_gap: float = HARD_RECALL_MAX_ORACLE_GAP,
    structured_tool_calls: bool = False,
) -> dict[str, Any] | None:
    """Build one submit-prior example where the prior is already near-oracle."""

    env = AtomicVisionEnvironment(difficulty=difficulty)
    initial_observation = env.reset(seed=seed)
    initial_text = _format_observation(initial_observation.model_dump())
    initial_user = _user_message(initial_text)
    prior_observation = env.step(AtomicVisionAction(action_type="ask_prior"))
    prior_text = _format_observation(prior_observation.model_dump())
    prior = prior_observation.prior_prediction
    prior_args = _submit_args_from_prior(prior)
    case = env._require_case()

    truth_species = [defect.species for defect in case.defects]
    truth_species_set = set(truth_species)
    prior_species_set = set(prior_args["predicted_defects"])
    if prior_species_set != truth_species_set:
        return None

    prior_score = score_submission(
        case,
        prior_args["predicted_defects"],
        prior_args["predicted_concentrations"],
        confidence=prior_args["confidence"],
        scan_cost=1.5,
    )
    oracle_args = _submit_args_from_case(case)
    oracle_score = score_submission(
        case,
        oracle_args["predicted_defects"],
        oracle_args["predicted_concentrations"],
        confidence=oracle_args["confidence"],
        scan_cost=1.5,
    )
    oracle_gap = float(oracle_score.total_reward - prior_score.total_reward)
    if oracle_gap > max_oracle_gap:
        return None

    submit_call = {"name": "submit_defect_map", "arguments": prior_args}
    submit_text = _tool_call_text(submit_call)
    return {
        "sample_id": f"{difficulty}-{seed}-good_prior_submit",
        "sample_type": "submit_prior",
        "target_tool_name": "submit_defect_map",
        "target_tool_call": submit_text,
        "seed": seed,
        "difficulty": difficulty,
        "prior_prediction": _model_dump(prior),
        "expected_reward": prior_score.total_reward,
        "expected_reward_breakdown": {},
        "expected_scan_cost": 1.5,
        "oracle_reward_gap": round(oracle_gap, 6),
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM},
            {"role": "user", "content": initial_user},
            _assistant_tool_message(
                ASK_PRIOR_CALL,
                structured_tool_calls=structured_tool_calls,
            ),
            {"role": "user", "content": _tool_response(prior_text)},
            _assistant_tool_message(
                submit_call,
                structured_tool_calls=structured_tool_calls,
            ),
        ],
    }


def build_episode_examples(
    seed: int,
    difficulty: str = "medium",
    sample_types: tuple[str, ...] = ("ask_prior", "submit_prior"),
    structured_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Build ask-prior and submit-prior examples for one episode."""

    env = AtomicVisionEnvironment(difficulty=difficulty)
    initial_observation = env.reset(seed=seed)
    initial_text = _format_observation(initial_observation.model_dump())
    initial_user = _user_message(initial_text)
    ask_text = _tool_call_text(ASK_PRIOR_CALL)

    prior_observation = env.step(AtomicVisionAction(action_type="ask_prior"))
    prior_text = _format_observation(prior_observation.model_dump())
    prior = prior_observation.prior_prediction
    submit_args = _submit_args_from_prior(prior)
    submit_call = {"name": "submit_defect_map", "arguments": submit_args}
    submit_text = _tool_call_text(submit_call)

    final_observation = env.step(
        AtomicVisionAction(
            action_type="submit_defect_map",
            predicted_defects=submit_args["predicted_defects"],
            predicted_concentrations=submit_args["predicted_concentrations"],
            confidence=submit_args["confidence"],
        )
    )

    examples: list[dict[str, Any]] = []
    if "ask_prior" in sample_types:
        examples.append(
            {
                "sample_id": f"{difficulty}-{seed}-ask_prior",
                "sample_type": "ask_prior",
                "target_tool_name": "ask_prior",
                "target_tool_call": ask_text,
                "seed": seed,
                "difficulty": difficulty,
                "messages": [
                    {"role": "system", "content": TOOL_SYSTEM},
                    {"role": "user", "content": initial_user},
                    _assistant_tool_message(
                        ASK_PRIOR_CALL,
                        structured_tool_calls=structured_tool_calls,
                    ),
                ],
            }
        )
    if "submit_prior" in sample_types:
        examples.append(
            {
                "sample_id": f"{difficulty}-{seed}-submit_prior",
                "sample_type": "submit_prior",
                "target_tool_name": "submit_defect_map",
                "target_tool_call": submit_text,
                "seed": seed,
                "difficulty": difficulty,
                "prior_prediction": _model_dump(prior),
                "expected_reward": float(final_observation.reward or 0.0),
                "expected_reward_breakdown": final_observation.reward_breakdown or {},
                "expected_scan_cost": float(env.state.total_scan_cost),
                "messages": [
                    {"role": "system", "content": TOOL_SYSTEM},
                    {"role": "user", "content": initial_user},
                    _assistant_tool_message(
                        ASK_PRIOR_CALL,
                        structured_tool_calls=structured_tool_calls,
                    ),
                    {"role": "user", "content": _tool_response(prior_text)},
                    _assistant_tool_message(
                        submit_call,
                        structured_tool_calls=structured_tool_calls,
                    ),
                ],
            }
        )
    return examples


def build_scan_improvement_examples(
    target_examples: int,
    difficulty: str = "medium",
    seed_start: int = 0,
    min_scan_improvement: float = 0.25,
    max_candidate_seeds: int | None = None,
    structured_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Build examples where compare_reference plus a revised map beats prior-copy."""

    if target_examples <= 0:
        raise ValueError("target_examples must be positive")
    if min_scan_improvement < 0.0:
        raise ValueError("min_scan_improvement must be non-negative")
    search_limit = max_candidate_seeds or max(32, target_examples * 12)
    examples: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + search_limit):
        example = build_scan_improvement_example(
            seed=seed,
            difficulty=difficulty,
            min_scan_improvement=min_scan_improvement,
            structured_tool_calls=structured_tool_calls,
        )
        if example is None:
            continue
        examples.append(example)
        if len(examples) >= target_examples:
            break
    return examples


def build_missing_defect_recovery_examples(
    target_examples: int,
    difficulty: str = "hard",
    seed_start: int = 0,
    min_scan_improvement: float = HARD_RECALL_MIN_SCAN_IMPROVEMENT,
    max_candidate_seeds: int | None = None,
    min_prior_confidence: float = HARD_RECALL_MIN_PRIOR_CONFIDENCE,
    max_prior_confidence: float = HARD_RECALL_MAX_PRIOR_CONFIDENCE,
    structured_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Build compare_reference recovery rows where the prior misses species."""

    if target_examples <= 0:
        raise ValueError("target_examples must be positive")
    if min_scan_improvement < 0.0:
        raise ValueError("min_scan_improvement must be non-negative")
    if not 0.0 <= min_prior_confidence <= 1.0:
        raise ValueError("min_prior_confidence must be between 0 and 1")
    if not 0.0 <= max_prior_confidence <= 1.0:
        raise ValueError("max_prior_confidence must be between 0 and 1")
    if min_prior_confidence > max_prior_confidence:
        raise ValueError("min_prior_confidence must be <= max_prior_confidence")

    search_limit = max_candidate_seeds or max(64, target_examples * 32)
    examples: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + search_limit):
        example = build_missing_defect_recovery_example(
            seed=seed,
            difficulty=difficulty,
            min_scan_improvement=min_scan_improvement,
            min_prior_confidence=min_prior_confidence,
            max_prior_confidence=max_prior_confidence,
            structured_tool_calls=structured_tool_calls,
        )
        if example is None:
            continue
        examples.append(example)
        if len(examples) >= target_examples:
            break
    return examples


def build_scan_improvement_example(
    seed: int,
    difficulty: str = "medium",
    min_scan_improvement: float = 0.25,
    structured_tool_calls: bool = False,
) -> dict[str, Any] | None:
    """Build one curated reference-then-submit example, or return None."""

    env = AtomicVisionEnvironment(difficulty=difficulty)
    initial_observation = env.reset(seed=seed)
    initial_text = _format_observation(initial_observation.model_dump())
    initial_user = _user_message(initial_text)
    ask_text = _tool_call_text(ASK_PRIOR_CALL)

    prior_observation = env.step(AtomicVisionAction(action_type="ask_prior"))
    prior_text = _format_observation(prior_observation.model_dump())
    prior = prior_observation.prior_prediction
    prior_args = _submit_args_from_prior(prior)
    case = env._require_case()
    prior_score = score_submission(
        case,
        prior_args["predicted_defects"],
        prior_args["predicted_concentrations"],
        confidence=prior_args["confidence"],
        scan_cost=1.5,
    )

    reference_text = _tool_call_text(COMPARE_REFERENCE_CALL)
    reference_observation = env.step(AtomicVisionAction(action_type="compare_reference"))
    reference_response = _format_observation(reference_observation.model_dump())
    submit_args = _submit_args_from_case(case)
    submit_call = {"name": "submit_defect_map", "arguments": submit_args}
    submit_text = _tool_call_text(submit_call)
    final_observation = env.step(
        AtomicVisionAction(
            action_type="submit_defect_map",
            predicted_defects=submit_args["predicted_defects"],
            predicted_concentrations=submit_args["predicted_concentrations"],
            confidence=submit_args["confidence"],
        )
    )
    reward_improvement = float(final_observation.reward or 0.0) - prior_score.total_reward
    if reward_improvement < min_scan_improvement:
        return None

    return {
        "sample_id": f"{difficulty}-{seed}-submit_after_reference",
        "sample_type": SCAN_SAMPLE_TYPE,
        "target_tool_name": "submit_defect_map",
        "target_tool_call": submit_text,
        "seed": seed,
        "difficulty": difficulty,
        "prior_prediction": _model_dump(prior),
        "prior_expected_reward": prior_score.total_reward,
        "expected_reward": float(final_observation.reward or 0.0),
        "reward_improvement": round(reward_improvement, 6),
        "expected_reward_breakdown": final_observation.reward_breakdown or {},
        "expected_scan_cost": float(env.state.total_scan_cost),
        "oracle_defect_map": _submit_args_from_case(case),
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM},
            {"role": "user", "content": initial_user},
            _assistant_tool_message(ASK_PRIOR_CALL, structured_tool_calls=structured_tool_calls),
            {"role": "user", "content": _tool_response(prior_text)},
            _assistant_tool_message(
                COMPARE_REFERENCE_CALL,
                structured_tool_calls=structured_tool_calls,
            ),
            {"role": "user", "content": _tool_response(reference_response)},
            _assistant_tool_message(submit_call, structured_tool_calls=structured_tool_calls),
        ],
    }


def build_missing_defect_recovery_example(
    seed: int,
    difficulty: str = "hard",
    min_scan_improvement: float = HARD_RECALL_MIN_SCAN_IMPROVEMENT,
    min_prior_confidence: float = HARD_RECALL_MIN_PRIOR_CONFIDENCE,
    max_prior_confidence: float = HARD_RECALL_MAX_PRIOR_CONFIDENCE,
    structured_tool_calls: bool = False,
) -> dict[str, Any] | None:
    """Build one reference-then-submit example where the prior misses defects."""

    env = AtomicVisionEnvironment(difficulty=difficulty)
    initial_observation = env.reset(seed=seed)
    initial_text = _format_observation(initial_observation.model_dump())
    initial_user = _user_message(initial_text)

    prior_observation = env.step(AtomicVisionAction(action_type="ask_prior"))
    prior_text = _format_observation(prior_observation.model_dump())
    prior = prior_observation.prior_prediction
    prior_args = _submit_args_from_prior(prior)
    prior_confidence = float(prior_args["confidence"])
    if not (min_prior_confidence <= prior_confidence <= max_prior_confidence):
        return None
    case = env._require_case()
    truth_species = [defect.species for defect in case.defects]
    missing_species = sorted(set(truth_species) - set(prior_args["predicted_defects"]))
    if not missing_species:
        return None

    prior_score = score_submission(
        case,
        prior_args["predicted_defects"],
        prior_args["predicted_concentrations"],
        confidence=prior_args["confidence"],
        scan_cost=1.5,
    )

    reference_observation = env.step(AtomicVisionAction(action_type="compare_reference"))
    reference_response = _format_observation(reference_observation.model_dump())
    submit_args = _submit_args_from_case(case)
    final_observation = env.step(
        AtomicVisionAction(
            action_type="submit_defect_map",
            predicted_defects=submit_args["predicted_defects"],
            predicted_concentrations=submit_args["predicted_concentrations"],
            confidence=submit_args["confidence"],
        )
    )
    reward_improvement = float(final_observation.reward or 0.0) - prior_score.total_reward
    if reward_improvement < min_scan_improvement:
        return None

    submit_call = {"name": "submit_defect_map", "arguments": submit_args}
    return {
        "sample_id": f"{difficulty}-{seed}-missing_defect_recovery",
        "sample_type": SCAN_SAMPLE_TYPE,
        "target_tool_name": "submit_defect_map",
        "target_tool_call": _tool_call_text(submit_call),
        "seed": seed,
        "difficulty": difficulty,
        "prior_prediction": _model_dump(prior),
        "prior_confidence": prior_confidence,
        "prior_expected_reward": prior_score.total_reward,
        "expected_reward": float(final_observation.reward or 0.0),
        "reward_improvement": round(reward_improvement, 6),
        "expected_reward_breakdown": final_observation.reward_breakdown or {},
        "expected_scan_cost": float(env.state.total_scan_cost),
        "oracle_defect_map": _submit_args_from_case(case),
        "missing_species_from_prior": missing_species,
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM},
            {"role": "user", "content": initial_user},
            _assistant_tool_message(ASK_PRIOR_CALL, structured_tool_calls=structured_tool_calls),
            {"role": "user", "content": _tool_response(prior_text)},
            _assistant_tool_message(
                COMPARE_REFERENCE_CALL,
                structured_tool_calls=structured_tool_calls,
            ),
            {"role": "user", "content": _tool_response(reference_response)},
            _assistant_tool_message(
                submit_call,
                structured_tool_calls=structured_tool_calls,
            ),
        ],
    }


def write_jsonl(examples: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write examples as UTF-8 JSONL."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")
    return path


def _submit_args_from_prior(prior: Any) -> dict[str, Any]:
    if prior is None:
        return {
            "predicted_defects": [],
            "predicted_concentrations": [],
            "confidence": 0.45,
        }
    return {
        "predicted_defects": list(prior.predicted_defects),
        "predicted_concentrations": list(prior.predicted_concentrations),
        "confidence": float(prior.confidence),
    }


def _submit_args_from_case(case: MaterialCase) -> dict[str, Any]:
    return {
        "predicted_defects": [defect.species for defect in case.defects],
        "predicted_concentrations": [defect.concentration for defect in case.defects],
        "confidence": 0.95,
    }


def _model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value)


def _tool_call_text(call: dict[str, Any]) -> str:
    payload = json.dumps(call, separators=(",", ":"), ensure_ascii=True)
    return f"<tool_call>{payload}</tool_call>"


def _assistant_tool_message(
    call: dict[str, Any],
    *,
    structured_tool_calls: bool,
) -> dict[str, Any]:
    if not structured_tool_calls:
        return {"role": "assistant", "content": _tool_call_text(call)}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": str(call["name"]),
                    "arguments": dict(call["arguments"]),
                },
            }
        ],
    }


def _user_message(observation_text: str) -> str:
    return f"{DEFAULT_PROMPT}\n\nObservation:\n{observation_text}"


def _tool_response(observation_text: str) -> str:
    return f"<tool_response>\n{observation_text}\n</tool_response>"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate AtomicVision SFT JSONL for exact prior-to-tool-call copying.",
    )
    parser.add_argument("--episodes-per-difficulty", type=int, default=256)
    parser.add_argument(
        "--profile",
        choices=(
            "explicit",
            "cost_aware",
            "format_repair",
            "format_refresh",
            "hard_recall_repair",
            "hard_recall_micro_repair",
            "strict_submit_contract_refresh",
            "strict_xml_submit_refresh",
            "submit_bridge",
            "two_step_curriculum",
            "hard_frontier_boost",
        ),
        default="explicit",
        help=(
            "explicit uses --sample-types as-is. cost_aware creates a "
            "cheap-prior-biased mix for assistant-masked SFT. "
            "format_repair creates a held-out repair mix with many more "
            "first-step ask_prior rows. format_refresh creates a tiny "
            "submit-heavy strict-envelope refresh set before GRPO. "
            "hard_recall_repair teaches the model to submit immediately only "
            "when the prior is already near-oracle, and to use "
            "compare_reference when the prior misses species. "
            "hard_recall_micro_repair is a narrower version focused on "
            "confident-but-incomplete priors plus a few near-oracle submit "
            "anchors. "
            "strict_submit_contract_refresh combines that submit-heavy "
            "refresh with a smaller batch of hard/reference-improvement "
            "final submit turns from the GRPO failure pool. "
            "strict_xml_submit_refresh creates a submit-only hard/reference-"
            "improvement refresh set that mirrors the GRPO failure pool. "
            "submit_bridge strengthens the "
            "second-turn submit_defect_map schema. two_step_curriculum "
            "concatenates both phases into one reproducible dataset. "
            "hard_frontier_boost widens the scan-improvement search for hard "
            "continuation runs."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=SFT_TRAIN_SEED_START)
    parser.add_argument("--difficulties", nargs="+", default=["medium"])
    parser.add_argument(
        "--sample-types",
        nargs="+",
        default=["ask_prior", "submit_prior"],
        choices=sorted(SAMPLE_TYPES),
    )
    parser.add_argument(
        "--min-scan-improvement",
        type=float,
        default=0.25,
        help="Minimum reward gain required for submit_after_reference examples.",
    )
    parser.add_argument(
        "--max-scan-candidates-per-difficulty",
        type=int,
        default=None,
        help="Maximum seeds to search when collecting scan-improvement examples.",
    )
    parser.add_argument(
        "--submit-prior-ratio",
        type=float,
        default=COST_AWARE_SUBMIT_PRIOR_RATIO,
        help="Cost-aware profile ratio for ask_prior -> submit_defect_map rows.",
    )
    parser.add_argument(
        "--reference-ratio",
        type=float,
        default=COST_AWARE_REFERENCE_RATIO,
        help="Cost-aware profile ratio for ask_prior -> compare_reference -> submit rows.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="outputs/sft/atomicvision_tool_copy_sft.jsonl",
    )
    parser.add_argument(
        "--assistant-tool-format",
        choices=("literal", "structured"),
        default="literal",
        help=(
            "literal stores assistant tool turns as exact <tool_call> text. "
            "structured stores assistant tool turns as HF-style tool_calls so "
            "the chat template renders the tool envelope during training."
        ),
    )
    parser.add_argument(
        "--reference-fraction",
        type=float,
        default=STRICT_SUBMIT_CONTRACT_REFERENCE_FRACTION,
        help=(
            "Strict-submit-contract refresh only: fraction of examples drawn "
            "from the hard reference-improvement failure pool."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.profile == "cost_aware":
        examples = build_cost_aware_sft_examples(
            examples_per_difficulty=args.episodes_per_difficulty,
            difficulties=tuple(args.difficulties),
            seed_start=args.seed_start,
            submit_prior_ratio=args.submit_prior_ratio,
            reference_ratio=args.reference_ratio,
            min_scan_improvement=args.min_scan_improvement,
            max_scan_candidates_per_difficulty=args.max_scan_candidates_per_difficulty,
        )
    elif args.profile == "format_repair":
        submit_prior_ratio = args.submit_prior_ratio
        reference_ratio = args.reference_ratio
        if submit_prior_ratio == COST_AWARE_SUBMIT_PRIOR_RATIO:
            submit_prior_ratio = FORMAT_REPAIR_SUBMIT_PRIOR_RATIO
        if reference_ratio == COST_AWARE_REFERENCE_RATIO:
            reference_ratio = FORMAT_REPAIR_REFERENCE_RATIO
        examples = build_cost_aware_sft_examples(
            examples_per_difficulty=args.episodes_per_difficulty,
            difficulties=tuple(args.difficulties),
            seed_start=args.seed_start,
            submit_prior_ratio=submit_prior_ratio,
            reference_ratio=reference_ratio,
            min_scan_improvement=args.min_scan_improvement,
            max_scan_candidates_per_difficulty=args.max_scan_candidates_per_difficulty,
        )
    elif args.profile == "submit_bridge":
        submit_prior_ratio = args.submit_prior_ratio
        reference_ratio = args.reference_ratio
        if submit_prior_ratio == COST_AWARE_SUBMIT_PRIOR_RATIO:
            submit_prior_ratio = SUBMIT_BRIDGE_SUBMIT_PRIOR_RATIO
        if reference_ratio == COST_AWARE_REFERENCE_RATIO:
            reference_ratio = SUBMIT_BRIDGE_REFERENCE_RATIO
        examples = build_cost_aware_sft_examples(
            examples_per_difficulty=args.episodes_per_difficulty,
            difficulties=tuple(args.difficulties),
            seed_start=args.seed_start,
            submit_prior_ratio=submit_prior_ratio,
            reference_ratio=reference_ratio,
            min_scan_improvement=args.min_scan_improvement,
            max_scan_candidates_per_difficulty=args.max_scan_candidates_per_difficulty,
        )
    elif args.profile == "format_refresh":
        submit_prior_ratio = args.submit_prior_ratio
        reference_ratio = args.reference_ratio
        if submit_prior_ratio == COST_AWARE_SUBMIT_PRIOR_RATIO:
            submit_prior_ratio = FORMAT_REFRESH_SUBMIT_PRIOR_RATIO
        if reference_ratio == COST_AWARE_REFERENCE_RATIO:
            reference_ratio = FORMAT_REFRESH_REFERENCE_RATIO
        examples = build_format_refresh_examples(
            examples_per_difficulty=args.episodes_per_difficulty,
            difficulties=tuple(args.difficulties),
            seed_start=args.seed_start,
            submit_prior_ratio=submit_prior_ratio,
            reference_ratio=reference_ratio,
            min_scan_improvement=args.min_scan_improvement,
            max_scan_candidates_per_difficulty=args.max_scan_candidates_per_difficulty,
            structured_tool_calls=args.assistant_tool_format == "structured",
        )
    elif args.profile == "strict_submit_contract_refresh":
        min_scan_improvement = args.min_scan_improvement
        max_scan_candidates_per_difficulty = args.max_scan_candidates_per_difficulty
        if min_scan_improvement == 0.25:
            min_scan_improvement = STRICT_XML_SUBMIT_REFRESH_MIN_SCAN_IMPROVEMENT
        if max_scan_candidates_per_difficulty is None:
            max_scan_candidates_per_difficulty = STRICT_XML_SUBMIT_REFRESH_MAX_SCAN_CANDIDATES
        examples = build_strict_submit_contract_refresh_examples(
            examples_per_difficulty=args.episodes_per_difficulty,
            difficulties=tuple(args.difficulties),
            seed_start=args.seed_start,
            reference_fraction=args.reference_fraction,
            submit_prior_ratio=FORMAT_REFRESH_SUBMIT_PRIOR_RATIO,
            min_scan_improvement=min_scan_improvement,
            max_scan_candidates_per_difficulty=max_scan_candidates_per_difficulty,
            structured_tool_calls=args.assistant_tool_format == "structured",
        )
    elif args.profile == "hard_recall_repair":
        min_scan_improvement = args.min_scan_improvement
        max_scan_candidates_per_difficulty = args.max_scan_candidates_per_difficulty
        if min_scan_improvement == 0.25:
            min_scan_improvement = HARD_RECALL_MIN_SCAN_IMPROVEMENT
        if max_scan_candidates_per_difficulty is None:
            max_scan_candidates_per_difficulty = HARD_RECALL_MAX_SCAN_CANDIDATES
        examples = build_hard_recall_repair_examples(
            examples_per_difficulty=args.episodes_per_difficulty,
            difficulties=tuple(args.difficulties),
            seed_start=args.seed_start,
            good_prior_fraction=HARD_RECALL_GOOD_PRIOR_FRACTION,
            min_scan_improvement=min_scan_improvement,
            max_scan_candidates_per_difficulty=max_scan_candidates_per_difficulty,
            max_oracle_gap_for_submit=HARD_RECALL_MAX_ORACLE_GAP,
            structured_tool_calls=args.assistant_tool_format == "structured",
        )
    elif args.profile == "hard_recall_micro_repair":
        examples = build_hard_recall_micro_repair_examples(
            examples_per_difficulty=args.episodes_per_difficulty,
            difficulties=tuple(args.difficulties),
            seed_start=args.seed_start,
            structured_tool_calls=args.assistant_tool_format == "structured",
        )
    elif args.profile == "strict_xml_submit_refresh":
        min_scan_improvement = args.min_scan_improvement
        max_scan_candidates_per_difficulty = args.max_scan_candidates_per_difficulty
        if min_scan_improvement == 0.25:
            min_scan_improvement = STRICT_XML_SUBMIT_REFRESH_MIN_SCAN_IMPROVEMENT
        if max_scan_candidates_per_difficulty is None:
            max_scan_candidates_per_difficulty = STRICT_XML_SUBMIT_REFRESH_MAX_SCAN_CANDIDATES
        examples = build_strict_xml_submit_refresh_examples(
            examples_per_difficulty=args.episodes_per_difficulty,
            difficulties=tuple(args.difficulties),
            seed_start=args.seed_start,
            min_scan_improvement=min_scan_improvement,
            max_scan_candidates_per_difficulty=max_scan_candidates_per_difficulty,
            structured_tool_calls=args.assistant_tool_format == "structured",
        )
    elif args.profile == "two_step_curriculum":
        examples = build_two_step_curriculum_examples(
            examples_per_difficulty=args.episodes_per_difficulty,
            difficulties=tuple(args.difficulties),
            seed_start=args.seed_start,
            min_scan_improvement=args.min_scan_improvement,
            max_scan_candidates_per_difficulty=args.max_scan_candidates_per_difficulty,
            structured_tool_calls=args.assistant_tool_format == "structured",
        )
    elif args.profile == "hard_frontier_boost":
        submit_prior_ratio = args.submit_prior_ratio
        reference_ratio = args.reference_ratio
        min_scan_improvement = args.min_scan_improvement
        max_scan_candidates_per_difficulty = args.max_scan_candidates_per_difficulty
        if submit_prior_ratio == COST_AWARE_SUBMIT_PRIOR_RATIO:
            submit_prior_ratio = HARD_FRONTIER_SUBMIT_PRIOR_RATIO
        if reference_ratio == COST_AWARE_REFERENCE_RATIO:
            reference_ratio = HARD_FRONTIER_REFERENCE_RATIO
        if min_scan_improvement == 0.25:
            min_scan_improvement = HARD_FRONTIER_MIN_SCAN_IMPROVEMENT
        if max_scan_candidates_per_difficulty is None:
            max_scan_candidates_per_difficulty = HARD_FRONTIER_MAX_SCAN_CANDIDATES
        examples = build_hard_frontier_boost_examples(
            examples_per_difficulty=args.episodes_per_difficulty,
            difficulties=tuple(args.difficulties),
            seed_start=args.seed_start,
            submit_prior_ratio=submit_prior_ratio,
            reference_ratio=reference_ratio,
            min_scan_improvement=min_scan_improvement,
            max_scan_candidates_per_difficulty=max_scan_candidates_per_difficulty,
            structured_tool_calls=args.assistant_tool_format == "structured",
        )
    else:
        examples = build_sft_examples(
            episodes_per_difficulty=args.episodes_per_difficulty,
            difficulties=tuple(args.difficulties),
            seed_start=args.seed_start,
            sample_types=tuple(args.sample_types),
            min_scan_improvement=args.min_scan_improvement,
            max_scan_candidates_per_difficulty=args.max_scan_candidates_per_difficulty,
            structured_tool_calls=args.assistant_tool_format == "structured",
        )
    output_path = write_jsonl(examples, args.output_jsonl)
    counts = Counter(example["sample_type"] for example in examples)
    print(f"Wrote {len(examples)} examples to {output_path}")
    print(f"Difficulties: {', '.join(args.difficulties)}")
    print(f"Sample counts: {dict(sorted(counts.items()))}")
    expected_scan_examples = _expected_scan_examples(args)
    if SCAN_SAMPLE_TYPE in counts and counts[SCAN_SAMPLE_TYPE] < expected_scan_examples:
        print(
            "Warning: fewer scan-improvement examples were found than requested; "
            "increase --max-scan-candidates-per-difficulty or lower "
            "--min-scan-improvement."
        )
    print("Kaggle next: load this JSONL with datasets.load_dataset('json', data_files=path).")


if __name__ == "__main__":
    main()
