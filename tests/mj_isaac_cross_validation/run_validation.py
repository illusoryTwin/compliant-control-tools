from __future__ import annotations

import argparse

from omni.isaac.lab.app import AppLauncher

import sys
from pathlib import Path

# Add project root and script directory to Python path to enable imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--robot",
    type=str,
    default="booster_t1",
    help="Robot to test",
)
parser.add_argument(
    "--zero-joints",
    action="store_true",
    help="Initialize all joints to zero",
)
parser.add_argument(
    "--trajectory",
    type=str,
    default=str(PROJECT_ROOT / "data" / "idle.pkl"),
    help="Path to trajectory pickle file for replay",
)
parser.add_argument(
    "--decimation",
    type=int,
    default=5,
    help="MSD integration sub-steps per frame",
)
parser.add_argument(
    "--mode",
    type=str,
    default="validation",
    choices=["validation", "trajectory"],
    help="Mode: 'validation' for force tests, 'trajectory' for trajectory replay",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import wrappers and runner
from core.sim_wrappers import IsaacWrapper, MujocoWrapper
from validator import ValidationRunner
from config import ROBOT_VALIDATION_CONFIGS
from robot_configs import ROBOT_CONFIGS


SIM_DT = 0.005
TOTAL_TIME = 3.0


if __name__ == "__main__":
    robot_cfg = ROBOT_CONFIGS[args.robot]
    validation_cfg = ROBOT_VALIDATION_CONFIGS[args.robot]

    compliance_config = validation_cfg["msd"]

    # Get URDF path
    if hasattr(robot_cfg.spawn, "asset_path"):
        urdf_path = robot_cfg.spawn.asset_path
    elif hasattr(robot_cfg.spawn, "usd_path"):
        urdf_path = robot_cfg.spawn.usd_path
    else:
        raise AttributeError(
            f"Cannot find URDF path in robot_cfg.spawn. Available attributes: {dir(robot_cfg.spawn)}"
        )

    print(f"Loading URDF: {urdf_path}")

    # Initialize Isaac Lab wrapper
    isaac_sim = IsaacWrapper(
        robot_cfg,
        SIM_DT,
        compliance_config=compliance_config,
        trajectory_path=args.trajectory,
    )

    # Initialize Mujoco wrapper
    mj_sim = MujocoWrapper(
        urdf_path,
        SIM_DT,
        compliance_config=compliance_config,
        trajectory_path=args.trajectory,
    )

    config = {
        "monitored_bodies": validation_cfg["monitored_bodies"],
        "decimation": args.decimation,
    }

    # Create and run validation
    validator = ValidationRunner(isaac_sim, mj_sim, config)
    validator.run(sim_dt=SIM_DT, total_time=TOTAL_TIME, simulation_app=simulation_app)

    simulation_app.close()
