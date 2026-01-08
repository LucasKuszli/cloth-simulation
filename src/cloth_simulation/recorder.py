# src/project/cloth_simulation/recorder.py

from __future__ import annotations

from pathlib import Path

import numpy as np
from utils import compute_stretch_stats


class SimulationRecorder:
    """Record cloth simulation state and derived metrics."""

    def __init__(self, out_path: Path, n_particles: int) -> None:
        self.out_path = Path(out_path)
        self.n_particles = n_particles

        self.times: list[float] = []
        self.positions: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []

        self.kinetic: list[float] = []
        self.potential: list[float] = []
        self.total: list[float] = []

        self.n_steps_done = 0

        self.stretch_stats: dict[str, list[float]] = {}

        self.free_node_id: int | None = None
        self.flutter_reference: np.ndarray | None = None
        self.flutter_series: list[float] = []

        self.diverged = False
        self.divergence_reason = ""

        self.triangles: np.ndarray | None = None

    def initialize_flutter(self, nx: int, ny: int, fixed_mask: np.ndarray) -> None:
        """Select a free vertex for flutter measurement."""
        candidate = (ny - 1) * nx + (nx - 1)

        if not fixed_mask[candidate]:
            self.free_node_id = candidate
            return

        for i, pinned in enumerate(fixed_mask):
            if not pinned:
                self.free_node_id = i
                return

    def record_state(
        self,
        t: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        kinetic: float,
        potential: float,
    ) -> None:
        self.times.append(t)
        self.positions.append(positions.copy())
        self.velocities.append(velocities.copy())

        self.kinetic.append(kinetic)
        self.potential.append(potential)
        self.total.append(kinetic + potential)

        self.n_steps_done += 1

    def record_stretch(self, system) -> None:
        metrics = compute_stretch_stats(system)

        for key, value in metrics.items():
            self.stretch_stats.setdefault(key, []).append(value)

    def record_flutter(self, system) -> None:
        if self.free_node_id is None:
            return

        pos = system.x[self.free_node_id]

        if self.flutter_reference is None:
            self.flutter_reference = pos.copy()

        amplitude = np.linalg.norm(pos - self.flutter_reference)
        self.flutter_series.append(float(amplitude))

    def check_divergence(self, system, threshold: float = 10.0) -> bool:
        if np.isnan(system.x).any() or np.isinf(system.x).any():
            self.diverged = True
            self.divergence_reason = "nan_or_inf"
            return True

        key = "struct_stretch_max"
        values = self.stretch_stats.get(key)
        if values and values[-1] > threshold:
            self.diverged = True
            self.divergence_reason = "excessive_stretch"
            return True

        return False

    def finalise_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positions = np.stack(self.positions, axis=0)
        velocities = np.stack(self.velocities, axis=0)
        times = np.asarray(self.times)
        kinetic = np.asarray(self.kinetic)
        potential = np.asarray(self.potential)
        total = np.asarray(self.total)

        return positions, velocities, times, kinetic, potential, total

    def save(
        self,
        total_time: float,
        runtime_sec: float,
        integrator_name: str,
        nx: int,
        ny: int,
        dt: float,
        stiffness_scale: float,
    ) -> None:
        positions, velocities, times, kinetic, potential, total = self.finalise_arrays()

        flutter_array = np.asarray(self.flutter_series)
        if flutter_array.size:
            flutter_rms = float(np.sqrt(np.mean(flutter_array**2)))
            flutter_mean = float(np.mean(flutter_array))
            flutter_max = float(np.max(flutter_array))
        else:
            flutter_rms = 0.0
            flutter_mean = 0.0
            flutter_max = 0.0

        steps_per_second = self.n_steps_done / runtime_sec
        sim_speed = total_time / runtime_sec

        np.savez_compressed(
            self.out_path,
            triangles=self.triangles,
            positions=positions,
            velocities=velocities,
            times=times,
            kinetic=kinetic,
            potential=potential,
            total=total,
            **{key: np.asarray(values) for key, values in self.stretch_stats.items()},
            flutter_series=flutter_array,
            flutter_rms=flutter_rms,
            flutter_mean=flutter_mean,
            flutter_max=flutter_max,
            n_steps_done=self.n_steps_done,
            runtime_sec=runtime_sec,
            steps_per_second=steps_per_second,
            sim_speed=sim_speed,
            nx=nx,
            ny=ny,
            dt=dt,
            total_time=total_time,
            stiffness_scale=stiffness_scale,
            integrator_name=integrator_name,
            diverged=self.diverged,
            divergence_reason=self.divergence_reason,
        )

        print(f"[Recorder] Saved: {self.out_path}")
