# Import local URDF preprocessor
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from utils.urdf import preprocess_urdf

import sys
from pathlib import Path

# Add project root and script directory to Python path to enable imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


# Local URDF paths
BOOSTER_T1_URDF = str(PROJECT_ROOT / "models" / "booster" / "T1_serial.urdf")
G1_URDF = str(PROJECT_ROOT / "models" / "g1" / "g1_29dof_with_hand_rev_1_0.urdf")

# Booster T1 configuration
BOOSTER_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=preprocess_urdf(BOOSTER_T1_URDF),
        fix_base=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*Head.*": 0.0,
            ".*Left_Shoulder_Pitch.*": 0.2,
            ".*Left_Shoulder_Roll.*": -1.35,
            ".*Left_Elbow_Pitch.*": 0.0,
            ".*Left_Elbow_Yaw.*": -0.5,
            ".*Right_Shoulder_Pitch.*": 0.2,
            ".*Right_Shoulder_Roll.*": 1.35,
            ".*Right_Elbow_Pitch.*": 0.0,
            ".*Right_Elbow_Yaw.*": 0.5,
            ".*Waist.*": 0.0,
            ".*Hip_Pitch.*": -0.2,
            ".*Hip_Roll.*": 0.0,
            ".*Hip_Yaw.*": 0.0,
            ".*Knee.*": 0.4,
            ".*Ankle_Pitch.*": -0.25,
            ".*Ankle_Roll.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                ".*Hip.*": 200.0,
                ".*Knee.*": 200.0,
                ".*Ankle.*": 50.0,
            },
            damping={
                ".*Hip.*": 5.0,
                ".*Knee.*": 5.0,
                ".*Ankle.*": 3,
            },
            effort_limit={
                ".*Hip_Pitch.*": 60.0,
                ".*Hip_Roll.*": 25.0,
                ".*Hip_Yaw.*": 30.0,
                ".*Knee.*": 60.0,
                ".*Ankle_Pitch.*": 24.0,
                ".*Ankle_Roll.*": 15.0,
            },
            friction=1.0,
        )
    },
)

# Unitree G1 configuration (with active arms)
UNITREE_G1_FULL_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=preprocess_urdf(G1_URDF),
        fix_base=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.4,
            ".*_knee_joint": 0.8,
            ".*_ankle_pitch_joint": -0.4,
            ".*_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                ".*_hip_yaw_.*": 200,
                ".*_hip_roll_.*": 200,
                ".*_hip_pitch_.*": 200,
                ".*_knee_.*": 300,
                ".*_ankle_roll_.*": 40,
                ".*_ankle_pitch_.*": 40,
                "waist_yaw_.*": 100,
                "waist_roll_.*": 100,
                "waist_pitch_.*": 100,
                ".*_shoulder_.*": 40,
                ".*_elbow_.*": 40,
                ".*_wrist_.*": 20,
                ".*_hand_.*": 5,
            },
            damping={
                ".*_hip_yaw_.*": 5,
                ".*_hip_roll_.*": 5,
                ".*_hip_pitch_.*": 5,
                ".*_knee_.*": 6,
                ".*_ankle_roll_.*": 1,
                ".*_ankle_pitch_.*": 1,
                "waist_yaw_.*": 2,
                "waist_roll_.*": 2,
                "waist_pitch_.*": 2,
                ".*_shoulder_.*": 5,
                ".*_elbow_.*": 5,
                ".*_wrist_.*": 2,
                ".*_hand_.*": 1,
            },
            friction=1.0,
        )
    },
)

# Robot configuration mapping
ROBOT_CONFIGS = {
    "booster_t1": BOOSTER_T1_CFG,
    "unitree_g1": UNITREE_G1_FULL_CFG,
}