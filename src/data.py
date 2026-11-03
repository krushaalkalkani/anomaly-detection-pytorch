"""Load HDFS_v1, build an event-count matrix, and save stratified splits."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
BLOCK_ID_RE = re.compile(r"blk_-?\d+")
SEED = 42
SPLIT_NAMES = ("train", "val", "test")


def find_label_csv() -> Path:
    """Return the path to anomaly_label.csv under preprocessed or data/."""
    for path in (PREPROCESSED_DIR / "anomaly_label.csv", DATA_DIR / "anomaly_label.csv"):
        if path.exists():
            return path
    raise FileNotFoundError("anomaly_label.csv not found in data/preprocessed or data/")


def load_labels(label_path: Path | None = None) -> pd.Series:
    """Load block labels; Anomaly -> 1, Normal -> 0."""
    path = label_path or find_label_csv()
    labels = pd.read_csv(path)
    mapping = {"Anomaly": 1, "Normal": 0}
    if not set(labels["Label"].unique()).issubset(mapping):
        raise ValueError(f"Unexpected labels in {path}: {labels['Label'].unique()}")
    return labels.set_index("BlockId")["Label"].map(mapping).astype(np.int8)


def _event_columns(frame: pd.DataFrame) -> list[str]:
    """Return E1, E2, ... column names present in a frame."""
    return [c for c in frame.columns if re.fullmatch(r"E\d+", str(c))]


def load_from_occurrence_matrix(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load a precomputed BlockId x event-type count matrix."""
    frame = pd.read_csv(path)
    event_cols = _event_columns(frame)
    counts = frame.set_index("BlockId")[event_cols].astype(np.float32)
    return counts, event_cols


def _parse_feature_list(features: str) -> list[str]:
    """Parse a Features cell like '[E5,E22,E5]' into event ids."""
    text = str(features).strip().strip('"')
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def build_from_traces(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Build an event-count matrix from per-block event traces."""
    traces = pd.read_csv(path, usecols=["BlockId", "Features"])
    counters: dict[str, Counter] = {}
    event_ids: set[str] = set()
    for block_id, features in zip(traces["BlockId"], traces["Features"]):
        events = _parse_feature_list(features)
        counts = Counter(events)
        counters[block_id] = counts
        event_ids.update(counts)
    event_cols = sorted(event_ids, key=lambda e: int(e[1:]) if e[1:].isdigit() else e)
    rows = [[counters[bid].get(col, 0) for col in event_cols] for bid in counters]
    frame = pd.DataFrame(rows, index=list(counters.keys()), columns=event_cols, dtype=np.float32)
    frame.index.name = "BlockId"
    return frame, event_cols


def load_templates(path: Path) -> list[tuple[str, re.Pattern[str]]]:
    """Load event templates and compile them as regex patterns."""
    templates = pd.read_csv(path)
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for _, row in templates.iterrows():
        parts = [re.escape(part) for part in str(row["EventTemplate"]).split("[*]")]
        pattern = re.compile(".*?".join(parts))
        compiled.append((str(row["EventId"]), pattern))
    return compiled


def match_template(line: str, templates: list[tuple[str, re.Pattern[str]]]) -> str | None:
    """Map a raw log line to an event template id, if any matches."""
    for event_id, pattern in templates:
        if pattern.search(line):
            return event_id
    return None


def build_from_raw_log(
    log_path: Path,
    templates_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Parse HDFS.log into a BlockId x event-type count matrix."""
    templates = load_templates(templates_path)
    counters: dict[str, Counter] = {}
    event_ids: set[str] = set()
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            block_ids = BLOCK_ID_RE.findall(line)
            if not block_ids:
                continue
            event_id = match_template(line, templates)
            if event_id is None:
                continue
            event_ids.add(event_id)
            for block_id in block_ids:
                counters.setdefault(block_id, Counter())[event_id] += 1
    event_cols = sorted(event_ids, key=lambda e: int(e[1:]) if e[1:].isdigit() else e)
    rows = [[counters[bid].get(col, 0) for col in event_cols] for bid in counters]
    frame = pd.DataFrame(rows, index=list(counters.keys()), columns=event_cols, dtype=np.float32)
    frame.index.name = "BlockId"
    return frame, event_cols


def load_event_count_matrix() -> tuple[pd.DataFrame, list[str]]:
    """Load or build the event-count matrix from preprocessed traces or raw logs."""
    occurrence = PREPROCESSED_DIR / "Event_occurrence_matrix.csv"
    traces = PREPROCESSED_DIR / "Event_traces.csv"
    if occurrence.exists() or traces.exists():
        if occurrence.exists():
            return load_from_occurrence_matrix(occurrence)
        return build_from_traces(traces)

    log_path = DATA_DIR / "HDFS.log"
    templates_path = PREPROCESSED_DIR / "HDFS.log_templates.csv"
    if not templates_path.exists():
        templates_path = DATA_DIR / "HDFS.log_templates.csv"
    if not log_path.exists():
        raise FileNotFoundError(
            "No preprocessed traces and data/HDFS.log is missing. "
            "Place HDFS_v1 files under data/preprocessed or data/HDFS.log."
        )
    if not templates_path.exists():
        raise FileNotFoundError("HDFS.log_templates.csv required to parse raw logs.")
    return build_from_raw_log(log_path, templates_path)


def join_labels(counts: pd.DataFrame, labels: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Align count rows with labels; drop blocks missing either side."""
    aligned = counts.join(labels.rename("y"), how="inner")
    if aligned.empty:
        raise ValueError("No overlapping BlockIds between counts and labels.")
    y = aligned["y"].to_numpy(dtype=np.int8)
    x = aligned.drop(columns=["y"]).to_numpy(dtype=np.float32)
    return x, y, aligned.index


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split into train/val/test with a 60/20/20 stratified ratio."""
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }


def save_splits(splits: dict[str, tuple[np.ndarray, np.ndarray]], out_dir: Path = DATA_DIR) -> None:
    """Save each split as a compressed .npz with arrays X and y."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (x, y) in splits.items():
        path = out_dir / f"{name}.npz"
        np.savez_compressed(path, X=x, y=y)


def load_splits(data_dir: Path = DATA_DIR) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load train/val/test splits previously saved under data/."""
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in SPLIT_NAMES:
        path = data_dir / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}. Run src.data first.")
        with np.load(path) as payload:
            splits[name] = (payload["X"], payload["y"])
    return splits


def prepare_and_save(seed: int = SEED) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build the matrix, join labels, split 60/20/20, and write .npz files."""
    counts, _ = load_event_count_matrix()
    labels = load_labels()
    x, y, _ = join_labels(counts, labels)
    splits = stratified_split(x, y, seed=seed)
    save_splits(splits)
    return splits


def main() -> None:
    """CLI entry: prepare HDFS_v1 splits and print shapes."""
    splits = prepare_and_save()
    for name, (x, y) in splits.items():
        n_anom = int(y.sum())
        print(f"{name}: X={x.shape}, anomalies={n_anom}/{len(y)}")


if __name__ == "__main__":
    main()
