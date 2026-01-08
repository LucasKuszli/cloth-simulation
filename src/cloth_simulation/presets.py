from __future__ import annotations

from typing import Any

SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "flag_euler_baseline": {
        "nx": 20,
        "ny": 15,
        "length_x": 1.0,
        "length_y": 0.75,
        "total_time": 5.0,
        "dt": 0.0002,
        "integrator_name": "euler",
        "run_name": "flag_euler_baseline",
    },
    "flag_euler_fine_dt": {
        "nx": 20,
        "ny": 15,
        "length_x": 1.0,
        "length_y": 0.75,
        "total_time": 5.0,
        "dt": 0.0001,
        "integrator_name": "euler",
        "run_name": "flag_euler_fine_dt",
    },
    "flag_rk4_baseline": {
        "nx": 20,
        "ny": 15,
        "length_x": 1.0,
        "length_y": 0.75,
        "total_time": 5.0,
        "dt": 0.004,
        "integrator_name": "rk4",
        "run_name": "flag_rk4_baseline",
    },
    "flag_rk4_high_res": {
        "nx": 40,
        "ny": 30,
        "length_x": 1.0,
        "length_y": 0.75,
        "total_time": 5.0,
        "dt": 0.003,
        "integrator_name": "rk4",
        "run_name": "flag_rk4_high_res",
    },
}


def resolve_scenario(
    name: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a config dict for a scenario name, with optional overrides."""
    try:
        cfg = SCENARIO_PRESETS[name].copy()
    except KeyError as exc:
        raise KeyError(f"Unknown scenario: {name}") from exc

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value

    return cfg
