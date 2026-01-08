# src/project/visualization/blender/run.py

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio.v2 as imageio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Blender cloth visualization.")
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name of the simulation run (e.g. 'flag_euler_baseline').",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
        help="Directory where simulation .npz files are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "renders",
        help="Directory where rendered frames or videos will be saved.",
    )
    parser.add_argument(
        "--blender-path",
        type=Path,
        default=Path("blender"),
        help="Path to the Blender executable.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for the animation (also used for GIF).",
    )
    return parser.parse_args()


def frames_to_gif(frames_dir: Path, run_name: str, fps: int) -> Path | None:
    """Convert rendered PNG frames into an animated GIF."""
    pattern = f"{run_name}_*.png"
    frame_files = sorted(frames_dir.glob(pattern))

    if not frame_files:
        print(f"[gif] No frames found in {frames_dir} matching {pattern}")
        return None

    print(f"[gif] Found {len(frame_files)} frames, building GIF...")

    images = [imageio.imread(str(frame)) for frame in frame_files]
    duration = 1.0 / float(fps)

    gif_path = frames_dir / f"{run_name}.gif"
    imageio.mimsave(gif_path, images, duration=duration, loop=0)

    print(f"[gif] Saved GIF to {gif_path}")
    return gif_path


def main() -> None:
    args = parse_args()

    npz_path = args.data_dir / f"{args.run_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No simulation file found at: {npz_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    bridge_script = Path(__file__).resolve().parent / "blender_bridge.py"

    cmd = [
        str(args.blender_path),
        "--background",
        "--python",
        str(bridge_script),
        "--",
        "--npz",
        str(npz_path),
        "--output-dir",
        str(args.output_dir),
        "--run-name",
        args.run_name,
        "--fps",
        str(args.fps),
    ]

    print("Launching Blender with command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    frames_dir = args.output_dir / args.run_name
    frames_to_gif(frames_dir, args.run_name, args.fps)


if __name__ == "__main__":
    main()
