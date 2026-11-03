"""Build event-count feature vectors from raw event ID lists."""

from __future__ import annotations

import re

import numpy as np

EVENT_ID_RE = re.compile(r"^E(\d+)$", re.IGNORECASE)


def events_to_count_vector(
    events: list[str],
    n_features: int = 29,
) -> np.ndarray:
    """Turn one block's event IDs into a length-n_features count row (E1..En).

    Matches the BlockId x event-type matrix used in ``src.data``: column ``i``
    is the count of event ``E{i+1}``. Unknown or out-of-range IDs are ignored.
    """
    counts = np.zeros(n_features, dtype=np.float32)
    for event_id in events:
        match = EVENT_ID_RE.fullmatch(str(event_id).strip())
        if match is None:
            continue
        index = int(match.group(1)) - 1
        if 0 <= index < n_features:
            counts[index] += 1.0
    return counts
