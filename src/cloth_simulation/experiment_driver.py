# cs/src/project/experiment_driver.py

from __future__ import annotations

import itertools
import multiprocessing as mp

from main import run_flag_simulation

INTEGRATORS = ["euler", "rk4"]

TIMESTEPS = [0.0001, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05]

STIFFNESS = [0.5, 1.0, 2.0]

NX, NY = 20, 15

LENGTH_X = 1
LENGTH_Y = 0.75

TOTAL_TIME = 10.0


def make_run_name(integrator: str, dt: float, stiffness_scale: float) -> str:
    dt_str = f"{dt:.6f}".rstrip("0").rstrip(".")
    k_str = f"{stiffness_scale:.2f}".rstrip("0").rstrip(".")
    return f"flag_{integrator}_dt{dt_str}_k{k_str}"


def run_single_task(task: dict) -> None:
    run_flag_simulation(**task)


def build_experiments() -> list[dict]:
    tasks: list[dict] = []

    for integrator, dt, stiffness_scale in itertools.product(
        INTEGRATORS, TIMESTEPS, STIFFNESS
    ):
        run_name = make_run_name(integrator, dt, stiffness_scale)
        tasks.append(
            {
                "nx": NX,
                "ny": NY,
                "length_x": LENGTH_X,
                "length_y": LENGTH_Y,
                "total_time": TOTAL_TIME,
                "dt": dt,
                "integrator_name": integrator,
                "run_name": run_name,
                "stiffness_scale": stiffness_scale,
            }
        )

    return tasks


def main(parallel: bool = True, num_workers: int = 4) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    tasks = build_experiments()

    print(f"\nLaunching experiment sweep: {len(tasks)} runs")
    print(f"Integrators: {INTEGRATORS}")
    print(f"Timesteps:   {TIMESTEPS}")
    print(f"Stiffness:   {STIFFNESS}")
    print(f"Parallel:    {parallel}")

    if not args.debug:
        workers = num_workers or mp.cpu_count()
        print(f"Using {workers} workers")

        with mp.Pool(workers) as pool:
            pool.map(run_single_task, tasks)
    else:
        for task in tasks:
            run_single_task(task)

    print("\nExperiment sweep complete!")


if __name__ == "__main__":
    main()
