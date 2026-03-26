"""
State vector construction and normalization utilities.
"""
import numpy as np
from volt_sim.config import TOTAL_STATE_SIZE


def validate_state(state: np.ndarray) -> bool:
    """Check that the state vector has the expected shape and no NaN."""
    if state.shape != (TOTAL_STATE_SIZE,):
        return False
    if np.any(np.isnan(state)):
        return False
    return True


def normalize_state(state: np.ndarray, running_mean: np.ndarray,
                    running_var: np.ndarray, clip: float = 10.0) -> np.ndarray:
    """Apply running normalization to state vector."""
    std = np.sqrt(running_var + 1e-8)
    normalized = (state - running_mean) / std
    return np.clip(normalized, -clip, clip)


class RunningStats:
    """Welford's online algorithm for running mean/variance."""

    def __init__(self, size: int):
        self.n = 0
        self.mean = np.zeros(size, dtype=np.float32)
        self.var = np.ones(size, dtype=np.float32)
        self._m2 = np.zeros(size, dtype=np.float32)

    def update(self, x: np.ndarray):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2
        if self.n > 1:
            self.var = self._m2 / (self.n - 1)

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        return normalize_state(x, self.mean, self.var, clip)
