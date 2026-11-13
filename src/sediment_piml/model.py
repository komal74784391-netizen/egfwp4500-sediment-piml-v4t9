"""Hybrid transport network used for reach-scale bedload prediction."""

from dataclasses import dataclass

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


@dataclass
class TransportConfig:
    hidden_dim: int = 64
    depth: int = 4
    dropout: float = 0.1


class TransportNetwork:
    """A small MLP surrogate for dimensionless bedload transport rate."""

    def __init__(self, config: TransportConfig, seed: int = 0):
        self.config = config
        rng = np.random.default_rng(seed)
        self.layers = []
        width = 3  # shields stress, relative roughness, slope
        for _ in range(config.depth):
            w = rng.normal(scale=1.0 / np.sqrt(width), size=(width, config.hidden_dim))
            b = np.zeros(config.hidden_dim)
            self.layers.append((w, b))
            width = config.hidden_dim
        self.output = rng.normal(size=(config.hidden_dim, 1))

    def forward(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        for w, b in self.layers:
            x = gelu(x @ w + b)
        return x @ self.output

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.maximum(self.forward(features), 0.0).ravel()
