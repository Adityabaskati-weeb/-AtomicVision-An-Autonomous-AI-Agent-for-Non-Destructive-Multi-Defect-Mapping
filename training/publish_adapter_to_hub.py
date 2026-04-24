"""Publish a PEFT adapter directory to the Hugging Face Hub.

This script exists because Kaggle and Colab working directories are ephemeral.
The safest post-train workflow is:

1. validate the adapter directory,
2. create or reuse a model repo on the Hub,
3. upload the adapter folder,
4. optionally upload a zip archive alongside it for easy recovery.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish an AtomicVision adapter to Hugging Face Hub.")
    parser.add_argument("--adapter-dir", required=True, help="Local adapter directory to upload.")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face model repo, e.g. prodigyhuh/atomicvision-format-submit-merged-lora")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B", help="Base model name for the generated model card.")
    parser.add_argument(
        "--commit-message",
        default="Upload AtomicVision PEFT adapter",
        help="Commit message for Hub upload.",
    )
    parser.add_argument(
        "--include-zip",
        action="store_true",
        help="Also upload a zip archive of the adapter directory for easy Kaggle recovery.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token override. By default the script uses HF_TOKEN or local auth.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the upload plan without performing network calls.",
    )
    return parser


def validate_adapter_dir(adapter_dir: Path) -> dict[str, Any]:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if not adapter_dir.is_dir():
        raise ValueError(f"Adapter path is not a directory: {adapter_dir}")

    missing = [name for name in REQUIRED_FILES if not (adapter_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Adapter directory is missing required files:\n" + "\n".join(missing)
        )

    extra_files = sorted(
        path.name
        for path in adapter_dir.iterdir()
        if path.is_file() and path.name not in REQUIRED_FILES and path.name != "README.md"
    )
    return {
        "adapter_dir": str(adapter_dir),
        "required_files": list(REQUIRED_FILES),
        "extra_files": extra_files,
    }


def build_model_card(repo_id: str, base_model: str, adapter_dir: Path) -> str:
    return "\n".join(
        [
            "---",
            "library_name: peft",
            "tags:",
            "- atomicvision",
            "- peft",
            "- lora",
            "- openenv",
            f"base_model: {base_model}",
            "---",
            "",
            f"# {repo_id}",
            "",
            "PEFT adapter artifact for AtomicVision.",
            "",
            "This repository is intended as a durable storage target for adapters",
            "trained in transient Kaggle/Colab runtimes.",
            "",
            "## Files",
            "",
            "- `adapter_config.json`",
            "- `adapter_model.safetensors`",
            "",
            f"Local source directory at publish time: `{adapter_dir.name}`",
            "",
        ]
    ) + "\n"


def build_publish_manifest(repo_id: str, base_model: str, adapter_dir: Path) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "base_model": base_model,
        "adapter_dir_name": adapter_dir.name,
        "required_files": list(REQUIRED_FILES),
    }


def publish_adapter(
    *,
    adapter_dir: Path,
    repo_id: str,
    base_model: str,
    commit_message: str,
    include_zip: bool,
    private: bool,
    token: str | None,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing huggingface_hub. Install it with `pip install huggingface_hub`."
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="atomicvision-publish-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        staging_dir = tmp_dir / adapter_dir.name
        shutil.copytree(adapter_dir, staging_dir)
        readme_path = staging_dir / "README.md"
        if not readme_path.exists():
            readme_path.write_text(
                build_model_card(repo_id=repo_id, base_model=base_model, adapter_dir=adapter_dir),
                encoding="utf-8",
            )
        manifest_path = staging_dir / "adapter_publish_manifest.json"
        manifest_path.write_text(
            json.dumps(
                build_publish_manifest(repo_id=repo_id, base_model=base_model, adapter_dir=adapter_dir),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(staging_dir),
            repo_type="model",
            commit_message=commit_message,
        )

        if include_zip:
            archive_base = tmp_dir / adapter_dir.name
            archive_path = Path(
                shutil.make_archive(
                    str(archive_base),
                    "zip",
                    root_dir=adapter_dir.parent,
                    base_dir=adapter_dir.name,
                )
            )
            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=str(archive_path),
                path_in_repo=archive_path.name,
                commit_message=f"{commit_message} (zip archive)",
            )


def main() -> None:
    args = build_arg_parser().parse_args()
    adapter_dir = Path(args.adapter_dir).resolve()
    validation = validate_adapter_dir(adapter_dir)
    plan = {
        "repo_id": args.repo_id,
        "base_model": args.base_model,
        "adapter_dir": str(adapter_dir),
        "include_zip": bool(args.include_zip),
        "private": bool(args.private),
        "validation": validation,
    }

    if args.dry_run:
        print("DRY RUN")
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    publish_adapter(
        adapter_dir=adapter_dir,
        repo_id=args.repo_id,
        base_model=args.base_model,
        commit_message=args.commit_message,
        include_zip=args.include_zip,
        private=args.private,
        token=args.token,
    )
    print("PUBLISH COMPLETE")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
