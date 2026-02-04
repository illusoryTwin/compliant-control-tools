"""Per-robot MSD and validation configurations."""

FLOATING_BASE_MAP = {"x": 0, "y": 1, "z": 2, "mx": 3, "my": 4, "mz": 5}


BOOSTER_T1_CONFIG = {
    "monitored_bodies": ["left_hand_link", "right_hand_link"],
    "msd": {
        "stiffness_config": {
            # Floating base DOFs
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "mx": 0.0,
            "my": 0.0,
            "mz": 0.0,
            # Upper body joints
            "Waist": 2.0,
            "Left_Shoulder_Pitch": 0.8,
            "Right_Shoulder_Pitch": 0.8,
            "Left_Shoulder_Roll": 0.8,
            "Right_Shoulder_Roll": 0.8,
            "Left_Elbow_Pitch": 0.6,
            "Right_Elbow_Pitch": 0.6,
            "Left_Elbow_Yaw": 0.6,
            "Right_Elbow_Yaw": 0.6,
            "AAHead_yaw": 0.5,
            "Head_pitch": 0.5,
        },
        "floating_base_map": FLOATING_BASE_MAP,
        "base_inertia": 0.5,
        "base_stiffness": 40.0,
    },
}

UNITREE_G1_CONFIG = {
    "monitored_bodies": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
    "msd": {
        "stiffness_config": {
            # Floating base DOFs
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "mx": 0.0,
            "my": 0.0,
            "mz": 0.0,
            # Waist joints
            "waist_yaw_joint": 2.0,
            "waist_roll_joint": 1.5,
            "waist_pitch_joint": 1.5,
            # Arm joints
            "left_shoulder_pitch_joint": 0.8,
            "right_shoulder_pitch_joint": 0.8,
            "left_shoulder_roll_joint": 0.8,
            "right_shoulder_roll_joint": 0.8,
            "left_shoulder_yaw_joint": 0.8,
            "right_shoulder_yaw_joint": 0.8,
            "left_elbow_joint": 0.6,
            "right_elbow_joint": 0.6,
            "left_wrist_roll_joint": 0.5,
            "right_wrist_roll_joint": 0.5,
            "left_wrist_pitch_joint": 0.5,
            "right_wrist_pitch_joint": 0.5,
            "left_wrist_yaw_joint": 0.5,
            "right_wrist_yaw_joint": 0.5,
        },
        "floating_base_map": FLOATING_BASE_MAP,
        "base_inertia": 0.5,
        "base_stiffness": 10.0, # 40.0,
    },
}

ROBOT_VALIDATION_CONFIGS = {
    "booster_t1": BOOSTER_T1_CONFIG,
    "unitree_g1": UNITREE_G1_CONFIG,
}
