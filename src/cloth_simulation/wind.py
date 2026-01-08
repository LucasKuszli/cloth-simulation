# src/project/cloth_simulation/wind.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindModel:
    """Simple laminar wind model for the cloth."""

    Cn: float = 0.02
    Ct: float = 0.05

    Ax: float = 3.0
    fx: float = 0.2

    Ay: float = 1.0
    fy: float = 0.5

    def wind_velocity(self, positions: np.ndarray, t: float) -> np.ndarray:
        """Compute spatially-uniform wind velocity at each vertex."""
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                "wind_velocity expects positions with shape (N, 3), "
                f"got {positions.shape}"
            )

        vx = self.Ax + self.Ax * np.sin(self.fx * t / (2.0 * np.pi))
        vy = self.Ay * np.sin(self.fy * t / (2.0 * np.pi))

        v_wind = np.zeros_like(positions)
        v_wind[:, 0] = vx
        v_wind[:, 1] = vy

        return v_wind
