# src/project/cloth_simulation/main.py

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from integrators import EulerIntegrator, RK4Integrator
from recorder import SimulationRecorder
from system import ClothParams, ClothSystem
from utils import compute_energy, create_flag_geometry
from wind import WindModel


def run_flag_simulation(
    nx: int,
    ny: int,
    length_x: float,
    length_y: float,
    total_time: float,
    dt: float,
    integrator_name: str,
    run_name: str,
    stiffness_scale: float = 1.0,
) -> None:
    """
    Run a cloth simulation and record results via SimulationRecorder.

    The integrator is expected to advance time via:
        new_t = integrator.step(system, dt, t, wind_model)
    """
    print(f"\nRunning simulation: {run_name} ({integrator_name}, dt={dt})")

    params = ClothParams(
        k_struct=1000.0 * stiffness_scale,
        k_shear=200.0 * stiffness_scale,
        k_bend=200.0 * stiffness_scale,
        c_struct=0.5,
        c_shear=0.5,
        c_bend=0.5,
        mass=0.01,
        gravity=9.81,
    )

    x0, v0, fixed_mask, springs, triangles = create_flag_geometry(
        nx=nx,
        ny=ny,
        length_x=length_x,
        length_y=length_y,
        params=params,
    )
    n_particles = nx * ny

    system = ClothSystem(
        positions=x0,
        velocities=v0,
        fixed_mask=fixed_mask,
        springs=springs,
        params=params,
    )
    system.update_normals(triangles)

    wind_model = WindModel()

    name = integrator_name.lower()
    if name == "euler":
        integrator = EulerIntegrator()
    elif name == "rk4":
        integrator = RK4Integrator()
    else:
        raise ValueError(f"Unknown integrator '{integrator_name}'")

    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{run_name}.npz"

    recorder = SimulationRecorder(out_path, n_particles)
    recorder.initialize_flutter(nx, ny, fixed_mask)
    recorder.triangles = np.asarray(triangles, dtype=np.int32)

    t = 0.0
    t_end = total_time
    start_wall = time.time()

    while t < t_end:
        system.update_normals(triangles)
        t = integrator.step(system, dt, t, wind_model)

        positions = system.x.copy()
        velocities = system.v.copy()
        energy = compute_energy(system, triangles)

        recorder.record_state(
            t,
            positions,
            velocities,
            energy["kinetic"],
            energy["potential"],
        )
        recorder.record_stretch(system)
        recorder.record_flutter(system)

        if recorder.check_divergence(system):
            print(f"Diverged: {recorder.divergence_reason}")
            break

    runtime_sec = time.time() - start_wall

    recorder.save(
        total_time=total_time,
        runtime_sec=runtime_sec,
        integrator_name=integrator_name,
        nx=nx,
        ny=ny,
        dt=dt,
        stiffness_scale=stiffness_scale,
    )

    print(f"Finished: {run_name} (runtime={runtime_sec:.2f}s)")


def main() -> None:
    import argparse

    from project.cloth_simulation.presets import SCENARIO_PRESETS

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True)
    args = parser.parse_args()

    if args.scenario not in SCENARIO_PRESETS:
        raise KeyError(f"Unknown scenario {args.scenario}")

    run_flag_simulation(**SCENARIO_PRESETS[args.scenario])


if __name__ == "__main__":
    main()
