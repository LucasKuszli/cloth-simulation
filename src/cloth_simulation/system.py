# cs/src/project/cloth_simulation/system.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from wind import WindModel


@dataclass
class SpringGroup:
    i: np.ndarray
    j: np.ndarray
    L0: np.ndarray
    k: float
    c: float


@dataclass
class ClothParams:
    k_struct: float
    k_shear: float
    k_bend: float
    c_struct: float
    c_shear: float
    c_bend: float
    mass: float
    gravity: float


class ClothSystem:
    """Holds cloth state and computes accelerations."""

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        fixed_mask: np.ndarray,
        springs: dict[str, SpringGroup],
        params: ClothParams,
        x_fixed: np.ndarray | None = None,
    ) -> None:
        assert positions.shape == velocities.shape
        assert positions.shape[1] == 3

        self.x = positions.astype(float)
        self.v = velocities.astype(float)

        self.N = self.x.shape[0]
        self.fixed = fixed_mask.astype(bool)
        self.springs = springs
        self.params = params

        self.x_fixed = self.x.copy() if x_fixed is None else x_fixed.copy()

        self.m = np.full(self.N, params.mass, dtype=float)

        self.normals = np.zeros_like(self.x)
        self.forces = np.zeros_like(self.x)

    def update_normals(self, triangles: np.ndarray) -> None:
        """Compute per-vertex normals via area-weighted face accumulation."""
        self.normals.fill(0.0)

        v0 = self.x[triangles[:, 0]]
        v1 = self.x[triangles[:, 1]]
        v2 = self.x[triangles[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)

        for idx in range(triangles.shape[0]):
            tri = triangles[idx]
            self.normals[tri] += face_normals[idx]

        lengths = np.linalg.norm(self.normals, axis=1)
        eps = 1e-8
        mask = lengths > eps
        self.normals[mask] /= lengths[mask, None]

    def _accumulate_spring_forces(self, group: SpringGroup) -> None:
        i = group.i
        j = group.j
        rest_lengths = group.L0
        stiffness = group.k
        damping = group.c

        dx = self.x[i] - self.x[j]
        lengths = np.linalg.norm(dx, axis=1)

        eps = 1e-8
        inv_lengths = 1.0 / np.maximum(lengths, eps)
        direction = dx * inv_lengths[:, None]

        stretch = lengths - rest_lengths
        spring_force_mag = -stiffness * stretch
        spring_force = spring_force_mag[:, None] * direction

        dv = self.v[i] - self.v[j]
        v_rel = np.sum(dv * direction, axis=1)
        damp_force_mag = -damping * v_rel
        damp_force = damp_force_mag[:, None] * direction

        force_ij = spring_force + damp_force

        np.add.at(self.forces, i, force_ij)
        np.add.at(self.forces, j, -force_ij)

    def _accumulate_internal_forces(self) -> None:
        for group in self.springs.values():
            self._accumulate_spring_forces(group)

    def _accumulate_gravity(self) -> None:
        g_vec = np.array([0.0, 0.0, -self.params.gravity])
        self.forces += self.m[:, None] * g_vec[None, :]

    def _accumulate_wind(self, wind_model: WindModel, t: float) -> None:
        x = self.x
        v = self.v
        normals = self.normals

        v_wind = wind_model.wind_velocity(x, t)
        v_rel = v_wind - v

        v_rel_dot_n = np.sum(v_rel * normals, axis=1)
        v_rel_n = v_rel_dot_n[:, None] * normals

        v_rel_t = v_rel - v_rel_n

        v_n_mag = np.linalg.norm(v_rel_n, axis=1)
        v_t_mag = np.linalg.norm(v_rel_t, axis=1)

        f_n = wind_model.Cn * v_n_mag[:, None] * v_rel_n
        f_t = wind_model.Ct * v_t_mag[:, None] * v_rel_t

        self.forces += f_n + f_t

    def compute_acceleration(
        self,
        t: float,
        wind_model: WindModel | None = None,
    ) -> np.ndarray:
        """Compute acceleration at the current state."""
        self.forces.fill(0.0)

        self._accumulate_internal_forces()
        self._accumulate_gravity()

        if wind_model is not None:
            self._accumulate_wind(wind_model, t)

        a = self.forces / self.m[:, None]
        a[self.fixed] = 0.0
        return a

    def enforce_fixed(self) -> None:
        self.v[self.fixed] = 0.0
        self.x[self.fixed] = self.x_fixed[self.fixed]
