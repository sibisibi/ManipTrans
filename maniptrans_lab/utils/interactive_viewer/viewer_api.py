from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .viewer_common import DEFAULT_TEMPLATE_PATH, render_template

Vec3 = Tuple[float, float, float]
ColorRGB = Tuple[float, float, float]


def _split_pose7(poses) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(poses, dtype=float)
    quats = arr[:, 3:]
    return arr[:, :3], quats / np.linalg.norm(quats, axis=1, keepdims=True)


def _serialize(array, decimals: int = 6) -> list:
    return np.round(np.asarray(array, dtype=float), decimals=decimals).tolist()


def build_trajectory_payload(
    *,
    joint_names: list[str],
    extra_animated: dict | None = None,
    robot_joint_positions,
    dt: float,
    object_poses: dict[str, np.ndarray] | None = None,
    robot_base_poses=None,
    robot_name: str = "robot",
) -> dict:
    joint_positions = np.asarray(robot_joint_positions, dtype=float)
    num_frames = joint_positions.shape[0]

    object_trajectories = {}
    for name, poses in (object_poses or {}).items():
        positions, quats = _split_pose7(poses)
        object_trajectories[name] = {
            "positions": _serialize(positions),
            "quats": _serialize(quats),
        }

    animated = {}
    for name, spec in (extra_animated or {}).items():
        entry = {
            "joint_names": list(spec["joint_names"]),
            "positions": _serialize(spec["positions"]),
        }
        if spec.get("base_poses") is not None:
            positions, quats = _split_pose7(spec["base_poses"])
            entry["base_trajectory"] = {
                "positions": _serialize(positions),
                "quats": _serialize(quats),
            }
        animated[name] = entry

    trajectory = {
        "robot_name": robot_name,
        "joint_names": list(joint_names),
        "dt": float(dt),
        "timestamps": _serialize(np.arange(num_frames, dtype=float) * float(dt)),
        "positions": _serialize(joint_positions),
        "object_trajectories": object_trajectories,
        "extra_animated": animated,
    }

    if robot_base_poses is not None:
        positions, quats = _split_pose7(robot_base_poses)
        trajectory["base_trajectory"] = {
            "positions": _serialize(positions),
            "quats": _serialize(quats),
        }

    return trajectory


def create_html(
    *,
    joint_names: list[str],
    extra_animated: dict | None = None,
    robot_joint_positions,
    robots: list[dict],
    object_poses: dict[str, np.ndarray] | None = None,
    robot_base_poses=None,
    dt: float,
    robot_name: str = "robot",
    template_path: Path = DEFAULT_TEMPLATE_PATH,
) -> str:
    trajectory = build_trajectory_payload(
        joint_names=joint_names,
        extra_animated=extra_animated,
        robot_joint_positions=robot_joint_positions,
        dt=dt,
        object_poses=object_poses,
        robot_base_poses=robot_base_poses,
        robot_name=robot_name,
    )
    return render_template(template_path, {"robots": robots, "trajectory": trajectory})


def make_embedded_robot(
    *,
    name: str,
    urdf_text: str,
    position: Vec3 = (0.0, 0.0, 0.0),
    rpy: Vec3 = (0.0, 0.0, 0.0),
    animated: bool = False,
    color_override: ColorRGB | None = None,
) -> dict:
    robot = {
        "name": name,
        "urdf_text": urdf_text,
        "position": list(position),
        "rpy": list(rpy),
        "animated": animated,
    }
    if color_override is not None:
        robot["color_override"] = list(color_override)
    return robot
