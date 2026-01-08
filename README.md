# Cloth Simulation

A small cloth (flag) simulation project with a simple wind model, Euler and RK4 time
integrators, experiment sweeps, and optional Blender-based rendering.

## Repository layout

- `src/cloth_simulation/`
  - `main.py`  
    Runs a single simulation scenario and writes results to `.npz` in `data/`.
  - `system.py`  
    Cloth state, spring forces, gravity, wind forces, and acceleration evaluation.
  - `integrators.py`  
    `EulerIntegrator` and `RK4Integrator` with a common `step(...)` interface.
  - `utils.py`  
    Geometry generation (grid, springs, triangles) and metrics helpers.
  - `wind.py`  
    Spatially-uniform laminar wind model.
  - `recorder.py`  
    Records states and metrics during simulation and saves them to `.npz`.

- `src/cloth_simulation/experiment_driver.py`  
  Runs a parameter sweep over integrators, timesteps, and stiffness scales (optionally
  in parallel).

- `src/visualization/blender/`
  - `run.py`  
    Launches Blender in background mode to render a sequence and optionally build a GIF.
  - `blender_bridge.py`  
    Blender-side entrypoint that loads `.npz`, builds a mesh, and renders frames.

- `visualization/blender/`
  - `build_lattice.py`  
    Blender utilities for scene setup, cloth mesh creation, and camera/light placement.
  - `render.py`  
    Renders a sequence by updating mesh vertices per frame.

- `data/`  
  Output directory for simulation `.npz` files (created automatically).

- `img/`  
  Output directory for plots (created automatically by the metrics script, if used).

- `renders/`  
  Output directory for Blender renders (created automatically by the Blender runner).

## Requirements

- Python 3.10+ recommended
- Core Python packages:
  - `numpy`
- Plotting (optional):
  - `matplotlib`
- Blender rendering (optional):
  - Blender installed and available on your PATH, or provide `--blender-path`
  - `bpy` is provided by Blender when running scripts from Blender
- GIF creation (optional):
  - `imageio`

Install project dependencies:
```bash
uv sync
```