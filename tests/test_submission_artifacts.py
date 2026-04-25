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
    assert "docs/hard-only-grpo-reference-probe-results.md" in readme
    assert "docs/hard-only-grpo-reference-probe-metrics.json" in readme
    assert "uv.lock" in readme
    assert "pip install git+https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv" in readme
    assert "uv sync --frozen" in readme
    assert "uv run server" in readme
    assert "openenv push --repo-id prodigyhuh/atomicvision-openenv" in readme
    assert "docker build -t atomicvision-openenv:latest ." in readme
    assert "docker run -d -p 7860:7860 --name atomicvision-openenv atomicvision-openenv:latest" in readme
    assert "from atomicvision_env import AtomicVisionAction, AtomicVisionEnv" in readme
    assert "/health" in readme
    assert "/docs" in readme
    assert "/ws" in readme
    assert "/reset" in readme
    assert "/step" in readme
    assert "/state" in readme


def test_validator_files_exist() -> None:
    assert Path("openenv.yaml").exists()
    assert Path("pyproject.toml").exists()
    assert Path("uv.lock").exists()
    assert Path("docs/judge-writeup.md").exists()
    assert Path("docs/training-loss-curve.png").exists()
    assert Path("docs/training-reward-curve.png").exists()
    assert Path("docs/hf-jobs-training-playbook.md").exists()
    assert Path("docs/experiment-lineage.md").exists()
    assert Path("docs/hard-only-grpo-reference-probe-results.md").exists()
    assert Path("docs/hard-only-grpo-reference-probe-metrics.json").exists()
    assert Path("docs/replay-mix-sft-continuation-results.md").exists()
    assert Path("docs/replay-mix-sft-continuation-metrics.json").exists()
    assert Path("docs/hard-error-mining-results.md").exists()
    assert Path("docs/hard-error-mining-metrics.json").exists()
    assert Path("docs/hard-recall-micro-repair-results.md").exists()
    assert Path("docs/hard-recall-micro-repair-metrics.json").exists()


def test_pyproject_exposes_installable_space_package() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "atomicvision-openenv"' in pyproject
    assert 'readme = "README.md"' in pyproject
    assert 'server = "atomicvision_env.server.run:main"' in pyproject
    assert 'atomicvision-server = "atomicvision_env.server.run:main"' in pyproject


def test_dockerfile_runs_packaged_server_entrypoint() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "RUN pip install --no-cache-dir ." in dockerfile
    assert 'CMD ["server"]' in dockerfile


def test_uv_lock_exists_for_reproducible_local_sync() -> None:
    lockfile = Path("uv.lock").read_text(encoding="utf-8")

    assert 'version = 1' in lockfile
    assert 'requires-python = ">=3.11"' in lockfile
