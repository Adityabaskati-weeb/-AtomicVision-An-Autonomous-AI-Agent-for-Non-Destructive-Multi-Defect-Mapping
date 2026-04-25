from __future__ import annotations

from pathlib import Path


def test_readme_links_validator_artifacts() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv" in readme
    assert "notebooks/AtomicVision_Judge_Repro_Colab.ipynb" in readme
    assert "docs/judge-writeup.md" in readme
    assert "docs/training-loss-curve.png" in readme
    assert "docs/training-reward-curve.png" in readme
    assert "training/train_sft_atomicvision_safe.py" in readme
    assert "docs/hf-jobs-training-playbook.md" in readme
    assert "docs/experiment-lineage.md" in readme


def test_validator_files_exist() -> None:
    assert Path("openenv.yaml").exists()
    assert Path("docs/judge-writeup.md").exists()
    assert Path("docs/training-loss-curve.png").exists()
    assert Path("docs/training-reward-curve.png").exists()
    assert Path("docs/hf-jobs-training-playbook.md").exists()
    assert Path("docs/experiment-lineage.md").exists()
