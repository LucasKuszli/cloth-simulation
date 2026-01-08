# src/project/cloth_simulation/integrators.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from system import ClothSystem
from wind import WindModel


@dataclass
class EulerIntegrator:
    """Explicit (semi-implicit) Euler integrator for the cloth system."""

    def step(
        self,
        system: ClothSystem,
        dt: float,
        t: float,
        wind_model: WindModel,
    ) -> float:
        enforce_fixed = getattr(system, "enforce_fixed", None)
        if callable(enforce_fixed):
            enforce_fixed()

        a = system.compute_acceleration(t, wind_model)

        system.v += dt * a
        system.x += dt * system.v

        if callable(enforce_fixed):
            enforce_fixed()

        return t + dt


@dataclass
class RK4Integrator:
    """Classical 4th-order Runge-Kutta (RK4) integrator for the cloth system."""

    def step(
        self,
        system: ClothSystem,
        dt: float,
        t: float,
        wind_model: WindModel,
    ) -> float:
        x0 = system.x.copy()
        v0 = system.v.copy()

        enforce_fixed = getattr(system, "enforce_fixed", None)

        def compute_accel(x: np.ndarray, v: np.ndarray, time_value: float) -> np.ndarray:
            system.x = x.copy()
            system.v = v.copy()

            if callable(enforce_fixed):
                enforce_fixed()

            return system.compute_acceleration(time_value, wind_model)

        a1 = compute_accel(x0, v0, t)
        k1_v = a1
        k1_x = v0

        x2 = x0 + 0.5 * dt * k1_x
        v2 = v0 + 0.5 * dt * k1_v
        a2 = compute_accel(x2, v2, t + 0.5 * dt)
        k2_v = a2
        k2_x = v2

        x3 = x0 + 0.5 * dt * k2_x
        v3 = v0 + 0.5 * dt * k2_v
        a3 = compute_accel(x3, v3, t + 0.5 * dt)
        k3_v = a3
        k3_x = v3

        x4 = x0 + dt * k3_x
        v4 = v0 + dt * k3_v
        a4 = compute_accel(x4, v4, t + dt)
        k4_v = a4
        k4_x = v4

        system.x = x0 + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        system.v = v0 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        if callable(enforce_fixed):
            enforce_fixed()

        return t + dt
