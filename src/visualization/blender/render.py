# visualization/blender/render.py

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np


def render_sequence(
    cloth_obj: bpy.types.Object,
    positions: np.ndarray,
    times: np.ndarray,
    out_dir: Path,
    run_name: str,
    fps: int = 24,
) -> None:
    """Render a cloth animation sequence and save frames to disk."""
    out_dir = Path(out_dir)
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    n_steps, n_verts, _ = positions.shape

    scene.render.engine = "CYCLES"
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 32
        scene.cycles.preview_samples = 32

    scene.render.resolution_x = 600
    scene.render.resolution_y = 400
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    t0 = float(times[0])
    t_end = float(times[-1])
    total_time = t_end - t0

    n_frames = max(1, int(np.floor(total_time * fps)))

    scene.frame_start = 1
    scene.frame_end = n_frames
    scene.render.fps = fps

    verts = cloth_obj.data.vertices

    print(f"[render] Rendering run '{run_name}' ({n_frames} frames) to {run_dir}")

    selected_indices = np.linspace(0, n_steps - 1, num=n_frames, dtype=int)

    for frame_idx, step_idx in enumerate(selected_indices):
        frame = frame_idx + 1
        scene.frame_set(frame)

        pos = positions[step_idx]

        for vert_idx in range(n_verts):
            verts[vert_idx].co = pos[vert_idx]
        cloth_obj.data.update()

        frame_path = run_dir / f"{run_name}_{frame:04d}.png"
        scene.render.filepath = str(frame_path)

        bpy.ops.render.render(write_still=True)
