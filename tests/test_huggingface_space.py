from __future__ import annotations

from fastapi.testclient import TestClient

from atomicvision_env.server.app import app


def test_space_root_route_has_browser_landing_page() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AtomicVision" in response.text
    assert "OpenEnv server running" in response.text


def test_openenv_health_route_for_space() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"healthy", "ok"}
