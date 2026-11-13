"""Sediment continuity constraint expressed as a differentiable residual."""

import numpy as np


def continuity_residual(
    transport_rate: np.ndarray,
    supply_rate: np.ndarray,
    storage_change: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Return the residual of d(storage)/dt = supply - transport."""
    if not (
        transport_rate.shape == supply_rate.shape == storage_change.shape
    ):
        raise ValueError("all flux arrays must share the same shape")
    return (storage_change / dt) - (supply_rate - transport_rate)


def physics_penalty(residual: np.ndarray, weight: float = 1.0) -> float:
    return weight * float(np.mean(residual**2))
