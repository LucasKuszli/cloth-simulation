# visualization/blender/blender_bridge.py

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from build_lattice import (
    build_cloth_object,
    create_camera_and_light_for_object,
    create_floor_for_object,
    init_scene,
)
from render import render_sequence


def parse_args_from_blender() -> argparse.Namespace:
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(description="Blender bridge for cloth visualization.")
    parser.add_argument("--npz", required=True, help="Path to simulation .npz file.")
    parser.add_argument("--output-dir", required=True, help="Output directory for renders.")
    parser.add_argument("--run-name", required=True, help="Run name (for file naming).")
    parser.add_argument("--fps", type=int, default=24)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args_from_blender()

    npz_path = Path(args.npz)
    out_dir = Path(args.output_dir)

    print(f"[bridge] Loading simulation from: {npz_path}")
    data = np.load(npz_path)
    positions = data["positions"]
    triangles = data["triangles"]
    times = data["times"]

    print(f"[bridge] positions.shape = {positions.shape}")
    print(f"[bridge] triangles.shape = {triangles.shape}")
    print(f"[bridge] times.shape = {times.shape}")

    init_scene()

    cloth_obj = build_cloth_object(positions, triangles, name=f"Cloth_{args.run_name}")

    create_camera_and_light_for_object(cloth_obj)

    render_sequence(
        cloth_obj=cloth_obj,
        positions=positions,
        times=times,
        out_dir=out_dir,
        run_name=args.run_name,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
