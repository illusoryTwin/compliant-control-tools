# Compliant Control Tools 

## Overview

This repository contains experiments and simulation tests for developing and validating implementation hypotheses.

Cross-simulator validation of Mass-Spring-Damper (MSD) dynamics for joint compliance modeling — comparing Isaac Lab and MuJoCo implementations.


## `Experiments`

For generating kinematic trajectories for Unitree G1 (waving hands):
```
python experiments/g1/launch_scene.py
```

This will:

1. Open a Mujoco viewer with the G1 robot waving arms
2. Record joint positions while running
3. When you close the viewer window, it saves to data/wave_arms.pkl


## `Tests` (Validation across Mujoco and Issac simulation engines) 
`mj_isaac_cross_validation`

```
python tests/mj_isaac_cross_validation/run_validation.py --robot unitree_g1 --mode trajectory --trajectory data/wave_arms.pkl
```

This will:
1. Load the Unitree G1 robot in both Isaac Sim and Mujoco (in parallel)
2. Replay the trajectory from data/*.pkl in both simulators simultaneously
4. Compare results between the two simulators at each timestep:
    - Joint positions
    - Body positions
    - Jacobians
    - External wrenches (forces/torques)
    - Computed torques
    - MSD (mass-spring-damper) compliance state