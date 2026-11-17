"""Match an aligned fractional-coordinate site under the dataset's 5-decimal convention."""
import numpy as np


def contains_coordinate(ideal, defect_coords):
    """Return whether a single defect site matches all three components of ideal."""
    target = np.around(np.asarray(ideal, dtype=float).reshape(3), decimals=5)
    candidates = np.around(np.asarray(defect_coords, dtype=float).reshape(-1, 3), decimals=5)
    return bool(np.any(np.all(candidates == target, axis=1)))
