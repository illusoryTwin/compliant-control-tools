"""Launch Unitree G1 in MuJoCo viewer — kinematic arm waving.

Logs joint positions to data/stance.pkl on viewer close.

Usage:
    python experiments/g1/launch_scene.py
"""

import pickle
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
G1_SCENE_XML = str(
    PROJECT_ROOT.parent
    / "unitree_robotics"
    / "unitree_mujoco"
    / "unitree_robots"
    / "g1"
    / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(G1_SCENE_XML)
data = mujoco.MjData(model)

# Joint indices in qpos (offset 7 for free joint)
# Shoulders: pitch=15,22  roll=16,23  yaw=17,24
# Elbows: 18, 25
IDX = 7  # free joint offset
n_joints = model.nu  # 29 actuated joints

# Joint names (skip the free joint)
joint_names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0]

mujoco.mj_resetData(model, data)

t = 0.0
dt = 0.01  # display timestep

# Logging buffers
log_joint_pos = []
log_root_pos = []
log_root_quat = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # All joints zero (standing), then override arms
        data.qpos[IDX:] = 0.0

        # Left arm wave
        data.qpos[IDX + 15] = -0.3 + 0.5 * np.sin(2.0 * t)    # shoulder pitch (forward/back)
        data.qpos[IDX + 16] = 1.3                               # shoulder roll (+ve = outward for left)
        data.qpos[IDX + 18] = 0.5 + 0.3 * np.sin(2.0 * t)     # elbow bend

        # Right arm wave (opposite phase)
        data.qpos[IDX + 22] = -0.3 + 0.5 * np.sin(2.0 * t + np.pi)  # shoulder pitch
        data.qpos[IDX + 23] = -1.3                                    # shoulder roll (-ve = outward for right)
        data.qpos[IDX + 25] = 0.5 + 0.3 * np.sin(2.0 * t + np.pi)   # elbow bend

        mujoco.mj_forward(model, data)
        viewer.sync()

        # Log frame
        log_joint_pos.append(data.qpos[IDX : IDX + n_joints].copy())
        log_root_pos.append(data.qpos[0:3].copy())
        log_root_quat.append(data.qpos[3:7].copy())

        t += dt
        elapsed = time.time() - step_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

# Save trajectory to data/stance.pkl
out_path = PROJECT_ROOT / "data" / "wave_arms.pkl"
traj_data = {
    "joint_positions": np.array(log_joint_pos),
    "joint_order": joint_names,
    "dt": dt,
    "transforms": {
        "root": {
            "position": np.array(log_root_pos),
            "quaternion": np.array(log_root_quat),
        }
    },
}

with open(out_path, "wb") as f:
    pickle.dump(traj_data, f)

print(f"Saved {len(log_joint_pos)} frames to {out_path}")
