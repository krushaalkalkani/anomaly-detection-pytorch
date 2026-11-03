"""Unit tests for the event-count feature builder."""

import numpy as np

from src.features import events_to_count_vector


def test_events_to_count_vector_counts_and_order() -> None:
    """E1 maps to column 0; duplicates increment the same bin."""
    vector = events_to_count_vector(["E1", "E5", "E1", "E29"], n_features=29)
    assert vector.shape == (29,)
    assert vector.dtype == np.float32
    assert vector[0] == 2.0
    assert vector[4] == 1.0
    assert vector[28] == 1.0
    assert vector.sum() == 4.0


def test_events_to_count_vector_ignores_invalid_ids() -> None:
    """Unknown or out-of-range event IDs do not change the vector."""
    vector = events_to_count_vector(["E2", "bad", "E99", "E0"], n_features=29)
    expected = np.zeros(29, dtype=np.float32)
    expected[1] = 1.0
    np.testing.assert_array_equal(vector, expected)


def test_events_to_count_vector_empty() -> None:
    """An empty event list yields an all-zero count row."""
    vector = events_to_count_vector([], n_features=29)
    assert vector.shape == (29,)
    assert float(vector.sum()) == 0.0
