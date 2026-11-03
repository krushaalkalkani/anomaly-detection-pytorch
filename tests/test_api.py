"""API tests for /health and /predict using FastAPI TestClient."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results" / "autoencoder.pt"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Boot the app against the trained checkpoint under results/."""
    if not CHECKPOINT.exists():
        pytest.skip(f"Missing checkpoint: {CHECKPOINT}. Train with: python -m src.autoencoder")

    monkeypatch.setenv("MODEL_PATH", str(CHECKPOINT))
    # Import after env is set so startup resolves the right path.
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_ok(client: TestClient) -> None:
    """GET /health returns status ok when the model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_score_fields(client: TestClient) -> None:
    """POST /predict returns score, threshold, and is_anomaly for one block."""
    payload = {"events": ["E5", "E22", "E5", "E11", "E9", "E26"]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"score", "threshold", "is_anomaly"}
    assert isinstance(body["score"], float)
    assert isinstance(body["threshold"], float)
    assert isinstance(body["is_anomaly"], bool)
    assert body["score"] >= 0.0
