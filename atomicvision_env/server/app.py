"""FastAPI app entrypoint for the AtomicVision OpenEnv server."""

from __future__ import annotations

from fastapi.responses import HTMLResponse
from openenv.core import create_app

from atomicvision_env.models import AtomicVisionAction, AtomicVisionObservation
from atomicvision_env.server.environment import AtomicVisionEnvironment


app = create_app(
    lambda: AtomicVisionEnvironment(),
    AtomicVisionAction,
    AtomicVisionObservation,
    env_name="atomicvision_env",
    max_concurrent_envs=32,
)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Browser landing page for the Hugging Face Space App tab."""

    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AtomicVision OpenEnv</title>
    <style>
      :root {
        color-scheme: dark;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #080c12;
        color: #f4f7fb;
      }
      body {
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        background:
          radial-gradient(circle at 20% 20%, rgba(43, 196, 161, 0.18), transparent 32rem),
          linear-gradient(135deg, #080c12 0%, #111923 55%, #0a1017 100%);
      }
      main {
        width: min(880px, calc(100vw - 40px));
      }
      h1 {
        margin: 0 0 12px;
        font-size: clamp(2.2rem, 5vw, 4.5rem);
        line-height: 1;
        letter-spacing: 0;
      }
      p {
        margin: 0;
        color: #b9c5d6;
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 760px;
      }
      nav {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 28px;
      }
      a {
        border: 1px solid rgba(244, 247, 251, 0.22);
        color: #f4f7fb;
        padding: 10px 14px;
        border-radius: 8px;
        text-decoration: none;
        background: rgba(255, 255, 255, 0.06);
      }
      a:hover {
        border-color: rgba(43, 196, 161, 0.75);
      }
      .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 24px;
        color: #77e0c2;
        font-weight: 700;
      }
      .dot {
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: #39d59f;
      }
    </style>
  </head>
  <body>
    <main>
      <div class="status"><span class="dot"></span>OpenEnv server running</div>
      <h1>AtomicVision</h1>
      <p>
        An OpenEnv materials characterization lab for non-destructive atomic defect mapping.
        Agents inspect synthetic vibrational spectra, request lab tools, and submit defect maps
        while balancing accuracy against scan cost.
      </p>
      <nav aria-label="Space links">
        <a href="/health">Health</a>
        <a href="/docs">API Docs</a>
        <a href="https://github.com/Adityabaskati-weeb/-AtomicVision-An-Autonomous-AI-Agent-for-Non-Destructive-Multi-Defect-Mapping">GitHub</a>
      </nav>
    </main>
  </body>
</html>"""
