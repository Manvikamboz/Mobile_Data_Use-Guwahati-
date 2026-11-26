from fastapi.testclient import TestClient
import pytest
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test GET /health returns 200 and {'status': 'ok'} (Handbook Section 9)."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_predict_valid_payload():
    """Test POST /predict with a valid request returns 200 and a numeric prediction."""
    payload = {
        "age": 28.0,
        "screen_hours": 4.5,
        "apps_installed": 32,
        "plan_gb_limit": 2.5,
        "video_minutes": 150.0,
        "wifi_hours": 3.0,
        "phone_tier": "mid",
        "circle": "Urban",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))


def test_predict_client_rule_1_video_minutes_above_550_rejected():
    """
    Client Rule 1 (PROBLEM.md Section 5):
    'video_minutes above 550 is impossible for us. Reject it.'
    Must return 422 Unprocessable Entity.
    """
    payload = {
        "age": 30.0,
        "screen_hours": 5.0,
        "apps_installed": 25,
        "plan_gb_limit": 3.0,
        "video_minutes": 551.0,  # Exceeds 550
        "wifi_hours": 2.0,
        "phone_tier": "flagship",
        "circle": "Urban",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_client_rule_2_apps_installed_below_5_rejected():
    """
    Client Rule 2 (PROBLEM.md Section 5):
    'apps_installed below 5 is always a typing mistake. Reject it.'
    Must return 422 Unprocessable Entity.
    """
    payload = {
        "age": 30.0,
        "screen_hours": 5.0,
        "apps_installed": 4,  # Below 5
        "plan_gb_limit": 3.0,
        "video_minutes": 200.0,
        "wifi_hours": 2.0,
        "phone_tier": "basic",
        "circle": "Rural",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_boundary_values_accepted():
    """
    Test exact boundary conditions:
    video_minutes == 550 (allowed)
    apps_installed == 5 (allowed)
    """
    payload = {
        "age": 22.0,
        "screen_hours": 3.5,
        "apps_installed": 5,  # Minimum allowable
        "plan_gb_limit": 1.5,
        "video_minutes": 550.0,  # Maximum allowable
        "wifi_hours": 1.0,
        "phone_tier": "basic",
        "circle": "Semi-urban",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data


@pytest.mark.parametrize("missing_field", [
    "age",
    "screen_hours",
    "apps_installed",
    "plan_gb_limit",
    "video_minutes",
    "wifi_hours",
    "phone_tier",
    "circle",
])
def test_predict_missing_required_fields(missing_field):
    """Test that omitting any required field triggers 422 validation error."""
    payload = {
        "age": 25.0,
        "screen_hours": 4.0,
        "apps_installed": 20,
        "plan_gb_limit": 2.0,
        "video_minutes": 100.0,
        "wifi_hours": 2.0,
        "phone_tier": "mid",
        "circle": "Urban",
    }
    del payload[missing_field]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
