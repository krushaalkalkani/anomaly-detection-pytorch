"""PyTorch autoencoder for HDFS_v1 event-count anomaly detection."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baseline import ensure_splits
from src.evaluate import evaluate_and_save

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SEED = 42
BATCH_SIZE = 256
EPOCHS = 20
HIDDEN = 64
LATENT = 16


def set_seeds(seed: int = SEED) -> None:
    """Fix Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_preprocessor(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit log1p + z-score stats on the training set only."""
    # Compress heavy-tailed counts before scaling.
    logged = np.log1p(x_train.astype(np.float64))
    mean = logged.mean(axis=0)
    std = logged.std(axis=0)
    # Avoid divide-by-zero on constant event columns.
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply the train-fitted log1p standardization to any split."""
    logged = np.log1p(x.astype(np.float64))
    return ((logged - mean) / std).astype(np.float32)


class EventCountAutoencoder(nn.Module):
    """Bottleneck autoencoder: input -> 64 -> 16 -> 64 -> input."""

    def __init__(self, n_features: int) -> None:
        """Build encoder/decoder Linear stacks with ReLU on hidden layers."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, LATENT),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the standardized event-count vector."""
        return self.decoder(self.encoder(x))


def make_loader(x: np.ndarray, batch_size: int = BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    """Wrap a NumPy matrix as a shuffled or ordered DataLoader."""
    tensor = torch.from_numpy(x)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_autoencoder(
    model: EventCountAutoencoder,
    x_normal: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> EventCountAutoencoder:
    """Train with Adam + MSE on normal training blocks only."""
    device = torch.device("cpu")
    model = model.to(device)
    # Unsupervised AE: learn to reconstruct normal behavior.
    loader = make_loader(x_normal, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        # Track convergence: average MSE over mini-batches this epoch.
        avg_loss = total_loss / max(n_batches, 1)
        print(f"epoch {epoch:02d}/{epochs}  loss={avg_loss:.6f}")
    return model


@torch.no_grad()
def reconstruction_errors(model: EventCountAutoencoder, x: np.ndarray) -> np.ndarray:
    """Mean squared reconstruction error per block (row)."""
    model.eval()
    tensor = torch.from_numpy(x)
    recon = model(tensor)
    # Per-sample MSE over event features.
    errors = torch.mean((recon - tensor) ** 2, dim=1)
    return errors.cpu().numpy()


def select_threshold(y_true: np.ndarray, errors: np.ndarray) -> float:
    """Pick the reconstruction-error cutoff that maximizes validation F1."""
    # Candidate cutoffs from unique sorted errors (skip the minimum).
    candidates = np.unique(errors)
    if len(candidates) < 2:
        return float(candidates[0]) if len(candidates) else 0.0

    best_f1 = -1.0
    best_threshold = float(candidates[0])
    # Sweep thresholds; score anomaly when error >= threshold.
    for threshold in candidates:
        y_pred = (errors >= threshold).astype(np.int8)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def predict_anomalies(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Label a block anomalous (1) when its error meets the threshold."""
    return (errors >= threshold).astype(np.int8)


def save_checkpoint(
    model: EventCountAutoencoder,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    """Persist weights plus preprocessing stats and decision threshold."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "n_features": mean.shape[0],
            "mean": mean,
            "std": std,
            "threshold": threshold,
        },
        path,
    )


def main() -> None:
    """Train AE on normal train blocks, tune threshold on val, evaluate on test."""
    set_seeds(SEED)

    # Reuse the same 60/20/20 splits as the IsolationForest baseline.
    splits = ensure_splits()
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]

    # Scale with train statistics only to avoid leakage into val/test.
    mean, std = fit_preprocessor(x_train)
    x_train_s = transform(x_train, mean, std)
    x_val_s = transform(x_val, mean, std)
    x_test_s = transform(x_test, mean, std)

    # One-class training: normal blocks only (label 0).
    x_normal = x_train_s[y_train == 0]

    model = EventCountAutoencoder(n_features=x_train_s.shape[1])
    model = train_autoencoder(model, x_normal)

    # Threshold chosen where validation F1 peaks.
    val_errors = reconstruction_errors(model, x_val_s)
    threshold = select_threshold(y_val, val_errors)
    print(f"threshold (max val F1): {threshold:.6f}")

    # Final report on held-out test blocks.
    test_errors = reconstruction_errors(model, x_test_s)
    y_pred = predict_anomalies(test_errors, threshold)
    evaluate_and_save(y_test, y_pred, out_path=RESULTS_DIR / "autoencoder.json")

    save_checkpoint(model, mean, std, threshold, RESULTS_DIR / "autoencoder.pt")
    print(f"saved: {RESULTS_DIR / 'autoencoder.pt'}")


if __name__ == "__main__":
    main()
