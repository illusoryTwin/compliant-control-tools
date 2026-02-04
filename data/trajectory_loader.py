import numpy as np
import os
import pickle
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrajectoryData:
    """Container for trajectory data"""

    joint_positions: np.ndarray  # Shape: (n_frames, n_joints)
    joint_velocities: Optional[np.ndarray] = None  # Shape: (n_frames, n_joints)
    root_positions: Optional[np.ndarray] = None  # Shape: (n_frames, 3)
    root_quaternions: Optional[np.ndarray] = None  # Shape: (n_frames, 4) - (w, x, y, z)
    joint_order: Optional[list[str]] = None
    dt: Optional[float] = None

    @property
    def n_frames(self) -> int:
        return len(self.joint_positions)

    @property
    def n_joints(self) -> int:
        return self.joint_positions.shape[1] if self.joint_positions.ndim > 1 else 0


class TrajectoryLoader:
    def __init__(self, root_name: str = "root"):
        self.root_name = root_name

    def load(self, file_path: str) -> TrajectoryData:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        joint_positions = np.asarray(data["joint_positions"], dtype=np.float64)

        joint_velocities = None
        if "joint_velocities" in data:
            joint_velocities = np.asarray(data["joint_velocities"], dtype=np.float64)

        joint_order = None
        if "joint_order" in data:
            joint_order = list(data["joint_order"])

        dt = data.get("dt")

        # Extract root/base transform data
        root_positions = None
        root_quaternions = None
        if "transforms" in data and self.root_name in data["transforms"]:
            root_data = data["transforms"][self.root_name]
            if "position" in root_data:
                root_positions = np.asarray(root_data["position"], dtype=np.float64)
            if "quaternion" in root_data:
                # in (w, x, y, z) format
                root_quaternions = np.asarray(root_data["quaternion"], dtype=np.float64)

        return TrajectoryData(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            root_positions=root_positions,
            root_quaternions=root_quaternions,
            joint_order=joint_order,
            dt=dt,
        )

    @staticmethod
    def get_frame(traj: TrajectoryData, frame_idx: int) -> dict:
        """Get data for a specific frame."""
        frame_data = {
            "joint_positions": traj.joint_positions[frame_idx],
        }

        if traj.joint_velocities is not None:
            frame_data["joint_velocities"] = traj.joint_velocities[frame_idx]

        if traj.root_positions is not None:
            frame_data["root_position"] = traj.root_positions[frame_idx]

        if traj.root_quaternions is not None:
            frame_data["root_quaternion"] = traj.root_quaternions[frame_idx]

        return frame_data
