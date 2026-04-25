from __future__ import annotations

from pathlib import Path

from training.publish_targeted_sft_winner import build_model_card, should_publish_file


def test_should_publish_file_filters_training_state_artifacts() -> None:
    scratch = Path("outputs/test_publish_targeted_sft_winner")
    scratch.mkdir(parents=True, exist_ok=True)
    keep = scratch / "adapter_model.safetensors"
    keep.write_text("ok", encoding="utf-8")
    drop = scratch / "optimizer.pt"
    drop.write_text("nope", encoding="utf-8")

    assert should_publish_file(keep)
    assert not should_publish_file(drop)


def test_build_model_card_mentions_parent_and_eval_table() -> None:
    content = build_model_card(
        model="Qwen/Qwen3-1.7B",
        init_adapter_dir="prodigyhuh/atomicvision-medium-fidelity-boost-lora",
        target_repo="prodigyhuh/atomicvision-hard-recall-micro-boost-lora",
        promotion_candidate="checkpoint-1",
        source_job_id="69ed269fd70108f37acdef6d",
        source_commit="3838f9048bce4c6bc81e57f5c0dab00980c7fa08",
        medium={
            "reward": 4.5065,
            "f1": 0.7891,
            "mae": 0.02,
            "strict": 1.0,
            "normalized": 1.0,
            "done": 1.0,
            "submit": 1.0,
        },
        hard={
            "reward": 4.7148,
            "f1": 0.8207,
            "mae": 0.03,
            "strict": 1.0,
            "normalized": 1.0,
            "done": 1.0,
            "submit": 1.0,
        },
    )

    assert "checkpoint-1" in content
    assert "69ed269fd70108f37acdef6d" in content
    assert "4.7148" in content
    assert "4.5065" in content
