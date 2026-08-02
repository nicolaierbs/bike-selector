"""Webhook contract tests.

Environment is set before importing the app module because the route paths are
derived from settings at import time.
"""

from __future__ import annotations

import os

os.environ.update(
    {
        "STRAVA_CLIENT_ID": "1",
        "STRAVA_CLIENT_SECRET": "secret",
        "STRAVA_REFRESH_TOKEN": "refresh",
        "WEBHOOK_VERIFY_TOKEN": "test-verify-token",
        "WEBHOOK_PATH_SECRET": "s3cr3t",
        "PUBLIC_BASE_URL": "",
        "KEEPALIVE_MINUTES": "0",
        "STATE_DIR": "/tmp/bike-selector-tests",
    }
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from bike_selector import app as app_module  # noqa: E402
from bike_selector.model import BikeClassifier  # noqa: E402

from .conftest import FakeActivity, FakeGateway  # noqa: E402

WEBHOOK = "/webhook/s3cr3t"


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    service = app_module.service
    service.settings.state_dir = tmp_path
    monkeypatch.setattr(service, "gateway", FakeGateway())
    monkeypatch.setattr(service, "classifier", BikeClassifier(service.settings))
    app_module.RECENT.clear()
    with TestClient(app_module.app) as test_client:
        yield test_client


# ------------------------------------------------------- subscription handshake


def test_validation_echoes_the_challenge(client):
    response = client.get(
        WEBHOOK,
        params={
            "hub.mode": "subscribe",
            "hub.challenge": "abc123",
            "hub.verify_token": "test-verify-token",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"hub.challenge": "abc123"}


def test_validation_rejects_a_wrong_verify_token(client):
    response = client.get(
        WEBHOOK,
        params={
            "hub.mode": "subscribe",
            "hub.challenge": "abc123",
            "hub.verify_token": "wrong",
        },
    )
    assert response.status_code == 403


def test_secret_path_hides_the_default_route(client):
    assert client.get("/webhook", params={"hub.mode": "subscribe"}).status_code == 404


# ---------------------------------------------------------------- event intake


def test_create_event_is_acked_and_processed(client):
    app_module.service.gateway.by_id[900] = FakeActivity(id=900, gear_id=None)

    response = client.post(
        WEBHOOK,
        json={
            "aspect_type": "create",
            "object_type": "activity",
            "object_id": 900,
            "owner_id": 42,
            "subscription_id": 1,
            "event_time": 1770000000,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}
    # TestClient runs BackgroundTasks before returning, so the write already happened.
    assert app_module.service.gateway.updates[0]["activity_id"] == 900
    assert app_module.RECENT[0]["status"] == "updated"


def test_malformed_body_still_returns_200(client):
    """A non-200 makes Strava retry, so never fail on junk input."""
    response = client.post(WEBHOOK, content=b"not json", headers={"Content-Type": "text/plain"})
    assert response.status_code == 200


def test_worker_errors_do_not_surface_as_a_failed_ack(client, monkeypatch):
    def boom(event):
        raise RuntimeError("Strava is down")

    monkeypatch.setattr(app_module.service, "handle_event", boom)
    response = client.post(
        WEBHOOK, json={"aspect_type": "create", "object_type": "activity", "object_id": 1}
    )
    assert response.status_code == 200
    assert app_module.RECENT[0]["status"] == "error"


def test_update_events_are_ignored(client):
    client.post(
        WEBHOOK,
        json={"aspect_type": "update", "object_type": "activity", "object_id": 900},
    )
    assert app_module.service.gateway.updates == []


# -------------------------------------------------------------- info endpoints


def test_healthz(client):
    payload = client.get("/healthz").json()
    assert payload["ok"] is True


def test_model_endpoint_before_and_after_training(client):
    app_module.service.ensure_model()
    payload = client.get("/model").json()
    assert payload["trained"] is True
    assert payload["n_classes"] == 3
    assert payload["cv_accuracy"] > payload["baseline_accuracy"]


def test_manual_predict_endpoint_is_a_dry_run_by_default(client):
    app_module.service.gateway.by_id[901] = FakeActivity(id=901, gear_id=None)
    payload = client.post("/activities/901/predict").json()
    assert payload["status"] == "dry-run"
    assert app_module.service.gateway.updates == []
