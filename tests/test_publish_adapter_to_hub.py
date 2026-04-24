from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from training.publish_adapter_to_hub import (
    REQUIRED_FILES,
    build_model_card,
    build_publish_manifest,
    validate_adapter_dir,
)


TEST_ROOT = Path("tests-publish-tmp")


def _fresh_dir(name: str) -> Path:
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{name}-", dir=TEST_ROOT))


def test_validate_adapter_dir_accepts_minimal_peft_layout() -> None:
    adapter_dir = _fresh_dir("valid-adapter")
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_text("stub", encoding="utf-8")

    summary = validate_adapter_dir(adapter_dir)

    assert summary["adapter_dir"] == str(adapter_dir)
    assert summary["required_files"] == list(REQUIRED_FILES)
    assert summary["extra_files"] == []


def test_validate_adapter_dir_rejects_missing_required_files() -> None:
    adapter_dir = _fresh_dir("missing-adapter")
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        validate_adapter_dir(adapter_dir)


def test_model_card_mentions_repo_and_base_model() -> None:
    card = build_model_card(
        repo_id="prodigyhuh/atomicvision-test-adapter",
        base_model="Qwen/Qwen3-1.7B",
        adapter_dir=Path("atomicvision-test-adapter"),
    )

    assert "prodigyhuh/atomicvision-test-adapter" in card
    assert "Qwen/Qwen3-1.7B" in card
    assert "PEFT adapter artifact for AtomicVision." in card


def test_publish_manifest_is_machine_readable() -> None:
    manifest = build_publish_manifest(
        repo_id="prodigyhuh/atomicvision-test-adapter",
        base_model="Qwen/Qwen3-1.7B",
        adapter_dir=Path("atomicvision-test-adapter"),
    )

    payload = json.loads(json.dumps(manifest))
    assert payload["repo_id"] == "prodigyhuh/atomicvision-test-adapter"
    assert payload["base_model"] == "Qwen/Qwen3-1.7B"
