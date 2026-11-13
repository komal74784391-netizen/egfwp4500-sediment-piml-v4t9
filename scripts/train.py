"""Train the physics-informed transport surrogate."""

import argparse
from pathlib import Path

import numpy as np

from sediment_piml.model import TransportConfig, TransportNetwork
from sediment_piml.continuity import continuity_residual, physics_penalty


def load_dataset(path: Path):
    data = np.load(path)
    return data["features"], data["targets"], data["supply"], data["storage"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    features, targets, supply, storage = load_dataset(args.dataset)
    model = TransportNetwork(TransportConfig())
    for epoch in range(args.epochs):
        pred = model.predict(features)
        residual = continuity_residual(pred, supply, storage, dt=1.0)
        loss = float(np.mean((pred - targets) ** 2)) + physics_penalty(residual)
        print(f"epoch {epoch:03d} loss {loss:.6f}")


if __name__ == "__main__":
    main()
