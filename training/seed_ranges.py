"""Shared seed-range policy for AtomicVision training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


SFT_TRAIN_SEED_START = 1000
SFT_TRAIN_SEED_STOP = 4000
GRPO_TRAIN_SEED_START = 4000
GRPO_TRAIN_SEED_STOP = 8000
HELDOUT_EVAL_SEED_START = 10000
HELDOUT_EVAL_SEED_STOP = 11000


@dataclass(frozen=True)
class SeedBand:
    start: int
    stop: int

    @property
    def label(self) -> str:
        return f"{self.start}-{self.stop - 1}"


SFT_TRAIN_BAND = SeedBand(SFT_TRAIN_SEED_START, SFT_TRAIN_SEED_STOP)
GRPO_TRAIN_BAND = SeedBand(GRPO_TRAIN_SEED_START, GRPO_TRAIN_SEED_STOP)
HELDOUT_EVAL_BAND = SeedBand(HELDOUT_EVAL_SEED_START, HELDOUT_EVAL_SEED_STOP)


def seed_policy_dict() -> dict[str, dict[str, int]]:
    return {
        "sft_train": {"start": SFT_TRAIN_BAND.start, "stop": SFT_TRAIN_BAND.stop},
        "grpo_train": {"start": GRPO_TRAIN_BAND.start, "stop": GRPO_TRAIN_BAND.stop},
        "heldout_eval": {"start": HELDOUT_EVAL_BAND.start, "stop": HELDOUT_EVAL_BAND.stop},
    }


def assert_non_overlapping_seed_policy() -> None:
    ordered = [SFT_TRAIN_BAND, GRPO_TRAIN_BAND, HELDOUT_EVAL_BAND]
    for current, nxt in zip(ordered[:-1], ordered[1:], strict=True):
        if current.stop > nxt.start:
            raise ValueError(f"Seed bands overlap: {current.label} and {nxt.label}")


assert_non_overlapping_seed_policy()
