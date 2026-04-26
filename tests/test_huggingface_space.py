from __future__ import annotations

from fastapi.testclient import TestClient

from atomicvision_env.server.app import app


def test_space_root_route_has_browser_landing_page() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AtomicVision | AI Defect Mapping Platform" in response.text
    assert "ATOMICVISION" in response.text
    assert "Autonomous AI for Non-Destructive Multi-Defect Mapping" in response.text
    assert "/static/space-ui.css" in response.text
    assert "/static/space-ui.js" in response.text
    assert "Static architecture view" in response.text
    assert "hero-frame" in response.text
    assert "/reset" in response.text
    assert "/step" in response.text
    assert "/ws" in response.text
    assert "/analyze_upload" in response.text


def test_openenv_health_route_for_space() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"healthy", "ok"}


def test_space_static_assets_are_served() -> None:
    client = TestClient(app)

    css_response = client.get("/static/space-ui.css")
    js_response = client.get("/static/space-ui.js")

    assert css_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]
    assert "Neue Machina" in css_response.text

    assert js_response.status_code == 200
    assert "javascript" in js_response.headers["content-type"]
    assert "setupHeroVideoScrub" in js_response.text
    assert "openDemoSocket" in js_response.text


def test_space_live_demo_websocket_reaches_prior_prediction() -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "reset", "data": {"difficulty": "medium"}})
        reset_payload = websocket.receive_json()
        assert reset_payload["type"] == "observation"
        assert reset_payload["data"]["observation"]["step_count"] == 0

        websocket.send_json({"type": "step", "data": {"action_type": "ask_prior"}})
        prior_payload = websocket.receive_json()
        assert prior_payload["type"] == "observation"
        prior = prior_payload["data"]["observation"]["prior_prediction"]
        assert prior is not None
        assert prior["predicted_defects"]


def test_space_upload_analysis_route_returns_prediction_payload() -> None:
    client = TestClient(app)

    spectrum = [0.0] * 64
    spectrum[12] = 1.0
    spectrum[20] = 0.72
    spectrum[28] = 0.44

    response = client.post(
        "/analyze_upload",
        json={
            "difficulty": "hard",
            "filename": "sample-a.csv",
            "spectrum": spectrum,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis_mode"] == "upload_driven"
    assert payload["observation"]["host_family"]
    assert payload["observation"]["prior_prediction"]["source"] == "upload_heuristic"
    assert payload["metrics"]["input_bins"] == 64
    assert len(payload["difference_spectrum"]) == payload["metrics"]["analysis_bins"]


def test_space_upload_analysis_changes_with_distinct_uploaded_spectra() -> None:
    client = TestClient(app)

    early_peak = [0.0] * 64
    early_peak[10] = 1.0
    early_peak[14] = 0.68

    late_peak = [0.0] * 64
    late_peak[45] = 1.0
    late_peak[51] = 0.74

    early_response = client.post(
        "/analyze_upload",
        json={
            "difficulty": "hard",
            "filename": "early.json",
            "spectrum": early_peak,
        },
    )
    late_response = client.post(
        "/analyze_upload",
        json={
            "difficulty": "hard",
            "filename": "late.json",
            "spectrum": late_peak,
        },
    )

    assert early_response.status_code == 200
    assert late_response.status_code == 200

    early_payload = early_response.json()
    late_payload = late_response.json()
    early_prior = early_payload["observation"]["prior_prediction"]
    late_prior = late_payload["observation"]["prior_prediction"]

    assert (
        early_prior["predicted_defects"] != late_prior["predicted_defects"]
        or early_prior["confidence"] != late_prior["confidence"]
        or early_payload["observation"]["host_family"] != late_payload["observation"]["host_family"]
    )
