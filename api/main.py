"""FastAPI app: score one HDFS block with the trained autoencoder."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.autoencoder import EventCountAutoencoder, reconstruction_errors, transform
from src.features import events_to_count_vector

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "results" / "autoencoder.pt"

# Filled once at startup from results/autoencoder.pt.
_MODEL: EventCountAutoencoder | None = None
_MEAN: np.ndarray | None = None
_STD: np.ndarray | None = None
_THRESHOLD: float | None = None
_N_FEATURES: int | None = None


class PredictRequest(BaseModel):
    """One block represented as an ordered list of event template IDs."""

    events: list[str] = Field(..., description="Event IDs for one block, e.g. E5, E22")


class PredictResponse(BaseModel):
    """Reconstruction score vs the validation-tuned anomaly threshold."""

    score: float
    threshold: float
    is_anomaly: bool


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str


def checkpoint_path() -> Path:
    """Resolve checkpoint path from MODEL_PATH or the default results file."""
    return Path(os.environ.get("MODEL_PATH", str(DEFAULT_CHECKPOINT)))


def load_artifacts(path: Path | None = None) -> dict[str, Any]:
    """Load autoencoder weights, scaler stats, and threshold from disk."""
    ckpt_path = path or checkpoint_path()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_features = int(payload["n_features"])
    model = EventCountAutoencoder(n_features=n_features)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return {
        "model": model,
        "mean": np.asarray(payload["mean"], dtype=np.float32),
        "std": np.asarray(payload["std"], dtype=np.float32),
        "threshold": float(payload["threshold"]),
        "n_features": n_features,
    }


def _set_globals(artifacts: dict[str, Any]) -> None:
    """Store loaded artifacts in module-level state for request handlers."""
    global _MODEL, _MEAN, _STD, _THRESHOLD, _N_FEATURES
    _MODEL = artifacts["model"]
    _MEAN = artifacts["mean"]
    _STD = artifacts["std"]
    _THRESHOLD = artifacts["threshold"]
    _N_FEATURES = artifacts["n_features"]


def score_events(events: list[str]) -> PredictResponse:
    """Build count features, scale, reconstruct, and compare to threshold."""
    if _MODEL is None or _MEAN is None or _STD is None or _THRESHOLD is None or _N_FEATURES is None:
        raise RuntimeError("Model is not loaded")

    # Same event-count row layout as the training matrix (E1..En).
    counts = events_to_count_vector(events, n_features=_N_FEATURES)
    scaled = transform(counts.reshape(1, -1), _MEAN, _STD)
    score = float(reconstruction_errors(_MODEL, scaled)[0])
    return PredictResponse(
        score=score,
        threshold=_THRESHOLD,
        is_anomaly=score >= _THRESHOLD,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the checkpoint and scaler stats once when the process starts."""
    _set_globals(load_artifacts())
    yield


app = FastAPI(title="HDFS Log Anomaly Detection", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return ok when the process is up and the model is in memory."""
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    """Score one block's event sequence for anomalies."""
    try:
        return score_events(body.events)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
