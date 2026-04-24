from __future__ import annotations

import json
from pathlib import Path


def test_judge_colab_notebook_exists_and_targets_repro_flow() -> None:
    notebook_path = Path("notebooks/AtomicVision_Judge_Repro_Colab.ipynb")

    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    source_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
    )

    assert payload["nbformat"] == 4
    assert "AtomicVision Judge Repro Colab" in source_text
    assert 'BRANCH = "main"' in source_text
    assert "/blob/main/notebooks/AtomicVision_Judge_Repro_Colab.ipynb" in source_text
    assert "two_step_curriculum" in source_text
    assert "train_sft_atomicvision_safe.py" in source_text
    assert "publish_adapter_to_hub.py" in source_text
    assert "evaluate_atomicvision_adapter.py" in source_text
    assert "strict" in source_text
    assert "normalized" in source_text
    assert "prodigyhuh/atomicvision-format-submit-merged-lora" in source_text
