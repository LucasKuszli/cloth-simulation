# cs/src/project/cloth_simulation/utils.py

from __future__ import annotations

from typing import Any

import numpy as np
from system import ClothParams, ClothSystem, SpringGroup


def compute_stretch_stats(system: ClothSystem) -> dict[str, float]:
    """Compute stretch statistics for each spring group."""
    x = system.x
    metrics: dict[str, float] = {}

    for name, group in system.springs.items():
        dx = x[group.i] - x[group.j]
        lengths = np.linalg.norm(dx, axis=1)
        ratios = lengths / np.maximum(group.L0, 1e-9)

        metrics[f"{name}_stretch_max"] = float(np.max(ratios))
        metrics[f"{name}_stretch_mean"] = float(np.mean(ratios))

    return metrics


def build_flag_grid(
    nx: int,
    ny: int,
    length_x: float,
    length_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build a rectangular grid for a flag in the x-z plane (y = 0)."""
    dx = length_x / (nx - 1)
    dz = length_y / (ny - 1)

    jj, ii = np.indices((ny, nx))

    x0 = np.zeros((nx * ny, 3), dtype=float)
    x0[:, 0] = (ii * dx).ravel()
    x0[:, 1] = 0.0
    x0[:, 2] = (jj * dz).ravel()

    v0 = np.zeros_like(x0)

    fixed = np.zeros(nx * ny, dtype=bool)

    def flat_index(i: int, j: int) -> int:
        return j * nx + i

    fixed[flat_index(0, 0)] = True
    fixed[flat_index(0, ny - 1)] = True

    return x0, v0, fixed, dx, dz


def build_flag_springs(
    nx: int,
    ny: int,
    params: ClothParams,
    dx: float,
    dy: float,
) -> dict[str, SpringGroup]:
    """Create structural, shear, and bending springs for a rectangular grid."""
    jj, ii = np.indices((ny, nx))
    index = (jj * nx + ii).astype(int)

    i_right = index[:, :-1].ravel()
    j_right = index[:, 1:].ravel()
    l0_right = np.full(i_right.shape[0], dx, dtype=float)

    i_up = index[:-1, :].ravel()
    j_up = index[1:, :].ravel()
    l0_up = np.full(i_up.shape[0], dy, dtype=float)

    struct_i = np.concatenate([i_right, i_up])
    struct_j = np.concatenate([j_right, j_up])
    struct_l0 = np.concatenate([l0_right, l0_up])

    struct_group = SpringGroup(
        i=struct_i,
        j=struct_j,
        L0=struct_l0,
        k=params.k_struct,
        c=params.c_struct,
    )

    diag = float(np.sqrt(dx**2 + dy**2))

    i_d1 = index[:-1, :-1].ravel()
    j_d1 = index[1:, 1:].ravel()

    i_d2 = index[1:, :-1].ravel()
    j_d2 = index[:-1, 1:].ravel()

    shear_i = np.concatenate([i_d1, i_d2])
    shear_j = np.concatenate([j_d1, j_d2])
    shear_l0 = np.full(shear_i.shape[0], diag, dtype=float)

    shear_group = SpringGroup(
        i=shear_i,
        j=shear_j,
        L0=shear_l0,
        k=params.k_shear,
        c=params.c_shear,
    )

    i_bx = index[:, :-2].ravel()
    j_bx = index[:, 2:].ravel()
    l0_bx = np.full(i_bx.shape[0], 2.0 * dx, dtype=float)

    i_by = index[:-2, :].ravel()
    j_by = index[2:, :].ravel()
    l0_by = np.full(i_by.shape[0], 2.0 * dy, dtype=float)

    bend_i = np.concatenate([i_bx, i_by])
    bend_j = np.concatenate([j_bx, j_by])
    bend_l0 = np.concatenate([l0_bx, l0_by])

    bend_group = SpringGroup(
        i=bend_i,
        j=bend_j,
        L0=bend_l0,
        k=params.k_bend,
        c=params.c_bend,
    )

    return {"struct": struct_group, "shear": shear_group, "bend": bend_group}


def build_flag_triangles(nx: int, ny: int) -> np.ndarray:
    """Generate triangle indices for the grid (each quad becomes two triangles)."""
    jj, ii = np.indices((ny, nx))
    index = (jj * nx + ii).astype(int)

    cells_j, cells_i = np.indices((ny - 1, nx - 1))

    p00 = index[cells_j, cells_i].ravel()
    p10 = index[cells_j, cells_i + 1].ravel()
    p01 = index[cells_j + 1, cells_i].ravel()
    p11 = index[cells_j + 1, cells_i + 1].ravel()

    tris1 = np.stack([p00, p10, p11], axis=1)
    tris2 = np.stack([p00, p11, p01], axis=1)

    return np.concatenate([tris1, tris2], axis=0).astype(int)


def make_wind_params(flag_speed: float = 5.0) -> dict[str, Any]:
    """Build a callable wind velocity model that blows along +x."""

    def v_wind(t: float) -> np.ndarray:
        base = flag_speed
        fluct = 0.5 * np.sin(2.0 * np.pi * 0.5 * t)
        v_x = base + fluct
        return np.array([v_x, 0.0, 0.0], dtype=float)

    return {"v_wind": v_wind}


def compute_energy(system: ClothSystem, triangles: np.ndarray) -> dict[str, float]:
    """Compute approximate kinetic and potential energy."""
    m = system.m
    v = system.v
    x = system.x
    params = system.params

    v2 = np.sum(v * v, axis=1)
    e_kin = 0.5 * np.sum(m * v2)

    z = x[:, 2]
    e_grav = np.sum(m * params.gravity * z)

    e_spring = 0.0
    for group in system.springs.values():
        dx = x[group.i] - x[group.j]
        lengths = np.linalg.norm(dx, axis=1)
        stretch = lengths - group.L0
        e_spring += 0.5 * group.k * np.sum(stretch * stretch)

    e_pot = e_spring + e_grav
    e_tot = e_kin + e_pot

    return {"kinetic": e_kin, "potential": e_pot, "total": e_tot}


def create_flag_geometry(
    nx: int,
    ny: int,
    length_x: float,
    length_y: float,
    params: ClothParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, SpringGroup], np.ndarray]:
    """Build geometry, spring groups, and triangle indices for the cloth system."""
    x0, v0, fixed_mask, dx, dy = build_flag_grid(
        nx=nx,
        ny=ny,
        length_x=length_x,
        length_y=length_y,
    )

    springs = build_flag_springs(
        nx=nx,
        ny=ny,
        params=params,
        dx=dx,
        dy=dy,
    )

    triangles = build_flag_triangles(nx, ny)

    return x0, v0, fixed_mask, springs, triangles
