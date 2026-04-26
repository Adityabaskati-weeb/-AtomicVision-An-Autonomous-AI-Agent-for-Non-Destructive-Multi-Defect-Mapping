"""FastAPI app entrypoint for the AtomicVision OpenEnv server."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openenv.core import create_app

from atomicvision_env.models import AtomicVisionAction, AtomicVisionObservation
from atomicvision_env.server.environment import AtomicVisionEnvironment
from atomicvision_env.server.frontend import render_home_html
from atomicvision_env.server.upload_analysis import (
    UploadAnalysisRequest,
    UploadAnalysisResponse,
    analyze_uploaded_spectrum,
)


app = create_app(
    lambda: AtomicVisionEnvironment(),
    AtomicVisionAction,
    AtomicVisionObservation,
    env_name="atomicvision_env",
    max_concurrent_envs=32,
)


def _resolve_static_dir() -> Path:
    """Find bundled UI assets in either the package or the source tree."""

    package_static = Path(__file__).resolve().parent / "static"
    if package_static.exists():
        return package_static

    repo_static = Path.cwd() / "atomicvision_env" / "server" / "static"
    return repo_static


app.mount("/static", StaticFiles(directory=_resolve_static_dir()), name="static")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Browser workbench for the Hugging Face Space App tab."""

    return render_home_html()


@app.post("/analyze_upload", response_model=UploadAnalysisResponse)
def analyze_upload(payload: UploadAnalysisRequest) -> UploadAnalysisResponse:
    """Run upload-driven spectral analysis for the premium Space UI."""

    try:
        return analyze_uploaded_spectrum(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
