"""Convert a retargeted G1 pickle motion into the CSV format used by csv_to_npz.py."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def _as_array(data: dict, key: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing required key: {key}")
    value = np.asarray(data[key], dtype=np.float64)
    if value.ndim != 1 + len(shape_tail) or value.shape[1:] != shape_tail:
        raise ValueError(f"{key} must have shape (frames, {', '.join(map(str, shape_tail))}); got {value.shape}")
    return value


def convert(input_file: Path, output_file: Path) -> None:
    with input_file.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected pickle to contain a dict, got {type(data).__name__}")

    root_pos = _as_array(data, "root_pos", (3,))
    # GMR already exports root_rot as xyzw (its export scripts convert MuJoCo's wxyz qpos
    # via [1, 2, 3, 0] before saving), which matches the IsaacLab 3.x / csv_to_npz convention.
    # Do NOT reorder again here — doing so scrambles the quaternion and tips the robot ~90°.
    root_rot = _as_array(data, "root_rot", (4,))
    dof_pos = _as_array(data, "dof_pos", (29,))

    frame_count = root_pos.shape[0]
    if root_rot.shape[0] != frame_count or dof_pos.shape[0] != frame_count:
        raise ValueError(
            "root_pos, root_rot, and dof_pos must have the same frame count; "
            f"got {root_pos.shape[0]}, {root_rot.shape[0]}, {dof_pos.shape[0]}"
        )

    motion = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_file, motion, delimiter=",", fmt="%.10f")

    fps = data.get("fps", "unknown")
    print(f"Wrote {frame_count} frames at {fps} fps to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True, help="Input pickle motion.")
    parser.add_argument("--output_file", type=Path, required=True, help="Output CSV path.")
    args = parser.parse_args()

    convert(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
