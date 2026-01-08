from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

FIG_WIDTH = 5.0
FIG_HEIGHT = 3.0

plt.rcParams.update(
    {
        "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
        "savefig.dpi": 300,
        "figure.dpi": 300,
    }
)

INTEGRATOR_COLORS = {
    "euler": "tab:blue",
    "rk4": "tab:orange",
}

RATIO_COLORS = {0.5: "tab:blue", 1.0: "tab:orange", 2.0: "tab:green"}
STIFF_MARKERS = {0.5: "o", 1.0: "s", 2.0: "^"}


@dataclass
class RunMeta:
    path: Path
    run_name: str
    integrator: str
    dt: float
    stiffness_scale: float
    diverged: bool
    divergence_reason: str
    runtime_sec: float
    total_time: float
    energy_drift_rel: float
    max_stretch: float


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_npz_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.npz"))


def _safe_bool(data: np.lib.npyio.NpzFile, key: str, default: bool = False) -> bool:
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    return bool(arr.reshape(()))


def _safe_float(
    data: np.lib.npyio.NpzFile, key: str, default: float = np.nan
) -> float:
    if key not in data.files:
        return float(default)
    arr = np.asarray(data[key])
    try:
        return float(arr.reshape(()))
    except Exception:
        return float(default)


def _safe_str(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    elem = arr.reshape(()).item()
    if isinstance(elem, bytes):
        return elem.decode("utf-8")
    return str(elem)


def _compute_energy_drift_rel_from_timeseries(
    times: np.ndarray, total_energy: np.ndarray
) -> float:
    """Relative drift = (E_end - E_start) / max(|E_start|, eps)."""
    _ = times
    try:
        if total_energy.size < 2:
            return float("nan")
        if not np.isfinite(total_energy).all():
            return float("nan")
        e0 = float(total_energy[0])
        e1 = float(total_energy[-1])
        denom = max(abs(e0), 1e-12)
        return float((e1 - e0) / denom)
    except Exception:
        return float("nan")


def load_run_meta(path: Path) -> RunMeta | None:
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        print(f"[metrics] Skipping {path.name}: failed to load ({exc})")
        return None

    integrator = _safe_str(data, "integrator_name", default="").lower()
    if integrator not in {"euler", "rk4"}:
        print(f"[metrics] Skipping {path.name}: unknown integrator '{integrator}'")
        return None

    dt = _safe_float(data, "dt")
    stiffness_scale = _safe_float(data, "stiffness_scale", default=np.nan)
    diverged = _safe_bool(data, "diverged", default=False)
    divergence_reason = _safe_str(data, "divergence_reason", default="")

    runtime_sec = _safe_float(data, "runtime_sec", default=np.nan)
    total_time = _safe_float(data, "total_time", default=np.nan)

    energy_drift_rel = _safe_float(data, "energy_drift_rel", default=np.nan)
    if np.isnan(energy_drift_rel) and "times" in data.files and "total" in data.files:
        try:
            t_vals = np.asarray(data["times"], dtype=float)
            e_vals = np.asarray(data["total"], dtype=float)
            energy_drift_rel = _compute_energy_drift_rel_from_timeseries(t_vals, e_vals)
        except Exception:
            pass

    max_stretch = np.nan
    for key in (
        "struct_stretch_max_global_max",
        "struct_stretch_global_max",
        "stretch_max_global_max",
        "stretch_global_max",
    ):
        if key in data.files:
            max_stretch = _safe_float(data, key, default=np.nan)
            break

    if np.isnan(max_stretch):
        for key in ("struct_stretch_max", "stretch_max", "struct_stretch_max_global"):
            if key in data.files:
                try:
                    arr = np.asarray(data[key], dtype=float).reshape(-1)
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        max_stretch = float(np.max(arr))
                        break
                except Exception:
                    pass

    return RunMeta(
        path=path,
        run_name=path.stem,
        integrator=integrator,
        dt=float(dt),
        stiffness_scale=float(stiffness_scale),
        diverged=bool(diverged),
        divergence_reason=divergence_reason,
        runtime_sec=float(runtime_sec),
        total_time=float(total_time),
        energy_drift_rel=float(energy_drift_rel),
        max_stretch=float(max_stretch),
    )


def load_all_runs(data_dir: Path) -> list[RunMeta]:
    runs: list[RunMeta] = []
    for path in find_npz_files(data_dir):
        meta = load_run_meta(path)
        if meta is not None:
            runs.append(meta)
    print(f"[metrics] Loaded {len(runs)} runs from {data_dir}")
    return runs


def pair_runs_by_k_dt(
    runs: list[RunMeta],
) -> list[tuple[float, float, RunMeta, RunMeta]]:
    index: dict[tuple[str, float, float], RunMeta] = {}
    for run in runs:
        if np.isnan(run.dt) or np.isnan(run.stiffness_scale):
            continue
        k_key = round(float(run.stiffness_scale), 6)
        dt_key = round(float(run.dt), 10)
        index[(run.integrator.lower(), k_key, dt_key)] = run

    pairs: list[tuple[float, float, RunMeta, RunMeta]] = []
    for (integrator, k_key, dt_key), euler_run in index.items():
        if integrator != "euler":
            continue
        rk4_run = index.get(("rk4", k_key, dt_key))
        if rk4_run is None:
            continue
        pairs.append((k_key, dt_key, euler_run, rk4_run))

    pairs.sort(key=lambda t: (t[0], t[1]))
    return pairs


def pick_baseline_run(
    runs: list[RunMeta], integrator: str, target_k: float = 1.0
) -> RunMeta | None:
    candidates = [
        run
        for run in runs
        if run.integrator == integrator
        and not run.diverged
        and np.isfinite(run.dt)
        and np.isfinite(run.stiffness_scale)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda run: (abs(run.stiffness_scale - target_k), run.dt))
    return candidates[0]


def plot_baseline_energy(runs: list[RunMeta], img_dir: Path) -> None:
    euler_run = pick_baseline_run(runs, "euler", target_k=1.0)
    rk4_run = pick_baseline_run(runs, "rk4", target_k=1.0)

    if euler_run is None or rk4_run is None:
        print("[metrics] Could not find non-diverged baseline runs for energy plot.")
        return

    plt.figure()

    euler_data = np.load(euler_run.path, allow_pickle=False)
    rk4_data = np.load(rk4_run.path, allow_pickle=False)

    if "times" in euler_data.files and "total" in euler_data.files:
        t_vals = np.asarray(euler_data["times"], dtype=float)
        e_vals = np.asarray(euler_data["total"], dtype=float)
        if np.isfinite(e_vals).all():
            plt.plot(
                t_vals,
                e_vals,
                color=INTEGRATOR_COLORS["euler"],
                linestyle="-",
                label=f"Euler (dt={euler_run.dt:g})",
            )

    if "times" in rk4_data.files and "total" in rk4_data.files:
        t_vals = np.asarray(rk4_data["times"], dtype=float)
        e_vals = np.asarray(rk4_data["total"], dtype=float)
        if np.isfinite(e_vals).all():
            plt.plot(
                t_vals,
                e_vals,
                color=INTEGRATOR_COLORS["rk4"],
                linestyle="--",
                label=f"RK4 (dt={rk4_run.dt:g})",
            )

    plt.xlabel("time [s]")
    plt.ylabel("total energy")
    plt.title("Total energy (baseline)")
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = img_dir / "baseline_energy.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[metrics] Saved {out_path}")


def _plot_ratio_only_success(
    pairs: list[tuple[float, float, RunMeta, RunMeta]],
    get_value,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure()

    by_k: dict[float, list[tuple[float, float]]] = {}
    for k_val, dt_val, euler_run, rk4_run in pairs:
        if euler_run.diverged or rk4_run.diverged:
            continue

        e_val = float(get_value(euler_run))
        r_val = float(get_value(rk4_run))
        if not np.isfinite(e_val) or not np.isfinite(r_val):
            continue
        if e_val == 0.0:
            continue

        ratio = r_val / e_val
        if not np.isfinite(ratio):
            continue

        by_k.setdefault(k_val, []).append((dt_val, ratio))

    if not by_k:
        print(f"[metrics] No successful pairs for '{title}'.")
        plt.close()
        return

    all_ratios: list[float] = []
    for k_val, samples in sorted(by_k.items()):
        samples.sort(key=lambda t: t[0])
        xs = [dt for dt, _ in samples]
        ys = [ratio for _, ratio in samples]
        all_ratios.extend(ys)

        color = RATIO_COLORS.get(k_val)
        marker = STIFF_MARKERS.get(k_val, "x")
        plt.plot(xs, ys, marker + "-", color=color, label=f"k={k_val:g}")

    plt.xscale("log")
    plt.xlabel("dt [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")

    if all_ratios:
        ymin = min(all_ratios)
        ymax = max(all_ratios)
        if ymin == ymax:
            ymin -= 0.1
            ymax += 0.1
        pad = 0.08 * (ymax - ymin)
        plt.ylim(ymin - pad, ymax + pad)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[metrics] Saved {out_path}")


def plot_relative_energy_drift(runs: list[RunMeta], img_dir: Path) -> None:
    pairs = pair_runs_by_k_dt(runs)
    if not pairs:
        print("[metrics] No (k, dt) pairs found for energy drift plot.")
        return

    out_path = img_dir / "rel_energy_drift.png"
    _plot_ratio_only_success(
        pairs=pairs,
        get_value=lambda run: abs(run.energy_drift_rel),
        ylabel="RK4/Euler",
        title="Energy drift ratio",
        out_path=out_path,
    )


def plot_relative_runtime(runs: list[RunMeta], img_dir: Path) -> None:
    pairs = pair_runs_by_k_dt(runs)
    if not pairs:
        print("[metrics] No (k, dt) pairs found for runtime plot.")
        return

    out_path = img_dir / "rel_runtime.png"
    _plot_ratio_only_success(
        pairs=pairs,
        get_value=lambda run: run.runtime_sec,
        ylabel="RK4/Euler",
        title="Runtime ratio",
        out_path=out_path,
    )


def plot_relative_max_stretch(runs: list[RunMeta], img_dir: Path) -> None:
    pairs = pair_runs_by_k_dt(runs)
    if not pairs:
        print("[metrics] No (k, dt) pairs found for max stretch plot.")
        return

    out_path = img_dir / "rel_max_stretch.png"
    _plot_ratio_only_success(
        pairs=pairs,
        get_value=lambda run: run.max_stretch,
        ylabel="RK4/Euler",
        title="Max stretch ratio",
        out_path=out_path,
    )


def plot_stability_map(runs: list[RunMeta], img_dir: Path) -> None:
    points: dict[str, list[tuple[float, float, bool]]] = {"euler": [], "rk4": []}
    for run in runs:
        if run.integrator not in points:
            continue
        if not np.isfinite(run.dt) or not np.isfinite(run.stiffness_scale):
            continue
        points[run.integrator].append((run.dt, run.stiffness_scale, run.diverged))

    if not points["euler"] and not points["rk4"]:
        print("[metrics] No runs for stability map.")
        return

    plt.figure(figsize=(FIG_WIDTH * 2.0, FIG_HEIGHT))

    for plot_idx, integrator in enumerate(["euler", "rk4"], start=1):
        ax = plt.subplot(1, 2, plot_idx)

        pts = points[integrator]
        if pts:
            dts = np.asarray([p[0] for p in pts], dtype=float)
            ks = np.asarray([p[1] for p in pts], dtype=float)
            diverged = np.asarray([p[2] for p in pts], dtype=bool)

            if np.any(~diverged):
                ax.scatter(
                    dts[~diverged],
                    ks[~diverged],
                    marker="o",
                    s=40,
                    color=INTEGRATOR_COLORS[integrator],
                    label="success",
                )
            if np.any(diverged):
                ax.scatter(
                    dts[diverged],
                    ks[diverged],
                    marker="x",
                    s=50,
                    color="black",
                    label="diverged",
                )

        ax.set_xscale("log")
        ax.set_xlabel("dt [s]")
        ax.set_ylabel("k scale" if plot_idx == 1 else "")
        ax.set_title(f"{integrator.upper()} stability")
        ax.grid(True)
        ax.legend(loc="best")

    plt.tight_layout()
    out_path = img_dir / "stability_map.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[metrics] Saved {out_path}")


def load_energy_timeseries(run: RunMeta) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (times, total_energy) for a run, or None if missing/invalid."""
    try:
        data = np.load(run.path, allow_pickle=False)
        if "times" not in data.files or "total" not in data.files:
            return None
        times = np.asarray(data["times"], dtype=float).reshape(-1)
        total_energy = np.asarray(data["total"], dtype=float).reshape(-1)
        if times.size < 2 or total_energy.size < 2:
            return None
        if not np.isfinite(total_energy).all():
            return None
        return times, total_energy
    except Exception:
        return None


def compute_deltaE(run: RunMeta) -> float | None:
    """Compute delta energy = E_end - E_start from saved total energy."""
    ts = load_energy_timeseries(run)
    if ts is None:
        return None
    _, total_energy = ts
    return float(total_energy[-1] - total_energy[0])


def plot_energy_change_error_heatmap_stacked(
    runs: list[RunMeta],
    img_dir: Path,
    k_values: tuple[float, ...] = (0.5, 1.0, 2.0),
    dt_values: tuple[float, ...] = (
        0.0001,
        0.0005,
        0.00075,
        0.001,
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.05,
    ),
    dt_ref: float = 1e-4,
) -> None:
    """
    Stacked (2x1) heatmaps with square cells:
      - top: Euler
      - bottom: RK4

    Error metric is relative deviation in net energy change:
      err = |dE(dt) - dE_ref| / max(|dE_ref|, eps)

    Diverged runs are shown with black X markers.
    """
    eps = 1e-12
    integrators = ["euler", "rk4"]

    k_vals = [float(k) for k in k_values]
    dt_vals = [float(dt) for dt in dt_values]
    nk, ndt = len(k_vals), len(dt_vals)

    k_index = {round(k, 6): i for i, k in enumerate(k_vals)}
    dt_index = {round(dt, 10): j for j, dt in enumerate(dt_vals)}
    dt_ref_key = round(float(dt_ref), 10)

    ref_de: dict[tuple[str, float], float] = {}
    for integrator in integrators:
        for k_val in k_vals:
            k_key = round(float(k_val), 6)
            found: float | None = None
            for run in runs:
                if run.integrator != integrator or run.diverged:
                    continue
                if not np.isfinite(run.dt) or not np.isfinite(run.stiffness_scale):
                    continue
                if round(float(run.stiffness_scale), 6) != k_key:
                    continue
                if round(float(run.dt), 10) != dt_ref_key:
                    continue
                de_val = compute_deltaE(run)
                if de_val is None or not np.isfinite(de_val):
                    continue
                found = float(de_val)
                break
            if found is not None:
                ref_de[(integrator, k_key)] = found

    if not ref_de:
        print("[metrics] No reference runs found at dt_ref for any (k, integrator).")
        return

    z_grid = {i: np.full((nk, ndt), np.nan, dtype=float) for i in integrators}
    diverged_grid = {i: np.zeros((nk, ndt), dtype=bool) for i in integrators}

    for run in runs:
        if run.integrator not in z_grid:
            continue
        if not np.isfinite(run.dt) or not np.isfinite(run.stiffness_scale):
            continue

        k_key = round(float(run.stiffness_scale), 6)
        dt_key = round(float(run.dt), 10)
        if k_key not in k_index or dt_key not in dt_index:
            continue

        i = k_index[k_key]
        j = dt_index[dt_key]

        if run.diverged:
            diverged_grid[run.integrator][i, j] = True
            continue

        baseline_key = (run.integrator, k_key)
        if baseline_key not in ref_de:
            continue

        de_val = compute_deltaE(run)
        if de_val is None or not np.isfinite(de_val):
            continue

        de_ref = ref_de[baseline_key]
        denom = max(abs(de_ref), eps)
        err = abs(float(de_val) - de_ref) / denom
        z_grid[run.integrator][i, j] = err

    all_err = np.concatenate([z_grid["euler"].ravel(), z_grid["rk4"].ravel()])
    all_err = all_err[np.isfinite(all_err)]
    all_err = all_err[all_err > 0.0]
    if all_err.size == 0:
        print("[metrics] No finite positive errors to plot.")
        return

    vmin = float(np.min(all_err))
    vmax = float(np.max(all_err))
    if vmin == vmax:
        vmin = max(vmin / 10.0, 1e-16)
        vmax = vmax * 10.0

    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(alpha=0.0)

    cell = 0.60
    panel_w = ndt * cell
    panel_h = nk * cell
    fig_w = panel_w + 2.2
    fig_h = panel_h * 2 + 2.0

    fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)

    last_im = None
    for ax, integrator in zip(axs, integrators):
        im = ax.imshow(
            z_grid[integrator],
            origin="lower",
            aspect="equal",
            norm=norm,
            cmap=cmap,
            interpolation="nearest",
        )
        last_im = im

        div = diverged_grid[integrator]
        if np.any(div):
            ys, xs = np.where(div)
            ax.scatter(xs, ys, marker="x", s=60, color="black", linewidths=1.5)

        ax.set_yticks(range(nk))
        ax.set_yticklabels([f"{k:g}" for k in k_vals])
        ax.set_ylabel("k scale")
        ax.set_title(f"{integrator.upper()}")

        ax.set_xticks(np.arange(-0.5, ndt, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nk, 1), minor=True)
        ax.grid(which="minor", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)

    axs[-1].set_xticks(range(ndt))
    axs[-1].set_xticklabels([f"{dt:.0e}" for dt in dt_vals], rotation=45, ha="right")
    axs[-1].set_xlabel("dt [s]")

    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.28, top=0.92, hspace=0.35)

    if last_im is not None:
        fig.colorbar(last_im, ax=axs, fraction=0.035, pad=0.02)

    out_path = img_dir / "energy_change_error_heatmap_stacked.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[metrics] Saved {out_path}")


def main() -> None:
    project_root = get_project_root()
    data_dir = project_root / "data"
    img_dir = project_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[metrics] Project root: {project_root}")
    print(f"[metrics] Data dir:    {data_dir}")
    print(f"[metrics] Image dir:   {img_dir}")

    runs = load_all_runs(data_dir)
    if not runs:
        return

    plot_baseline_energy(runs, img_dir)
    plot_relative_energy_drift(runs, img_dir)
    plot_relative_runtime(runs, img_dir)
    plot_relative_max_stretch(runs, img_dir)
    plot_stability_map(runs, img_dir)
    plot_energy_change_error_heatmap_stacked(runs, img_dir)


if __name__ == "__main__":
    main()
