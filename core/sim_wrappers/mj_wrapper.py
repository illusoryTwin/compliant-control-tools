from __future__ import annotations

import numpy as np
import torch

import mujoco
import mujoco.viewer
from .base import SimulationWrapper
from core.compliance_model.mass_spring_damper_model import MassSpringDamperModel
from data.trajectory_loader import TrajectoryLoader


class MujocoWrapper(SimulationWrapper):
    def __init__(
        self,
        model_path,
        step_dt,
        compliance_config=None,
        # active_joints=None,
        trajectory_path=None,
    ):
        """
        Initialize Mujoco wrapper.

        Args:
            model_path: Path to Mujoco XML/URDF file
            step_dt: Simulation timestep
            compliance_config: MSD configuration dictionary (optional)
            active_joints: List of active joint names (optional)
            trajectory_path: Path to trajectory pickle file (optional)
        """
        super().__init__(step_dt)

        # Load Mujoco model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = step_dt
        self.data = mujoco.MjData(self.model)

        # Get joint position indices
        self.qpos_idx = {
            self.model.joint(j).name: self.model.jnt_qposadr[j]
            for j in range(self.model.njnt)
        }

        self.ROOT_NAME = "root"

        # Check if model has floating base
        self._has_floating_base = self._check_floating_base()

        # Store reference positions for debugging
        self.qpos_ref = None

        # Load trajectory
        self.trajectory_loader = TrajectoryLoader(root_name=self.ROOT_NAME)
        self.trajectory_data = None
        if trajectory_path:
            try:
                self.trajectory_data = self.trajectory_loader.load(trajectory_path)
                print(
                    f"Loaded trajectory: {self.trajectory_data.n_frames} frames, {self.trajectory_data.n_joints} joints"
                )
            except (FileNotFoundError, KeyError) as e:
                print(f"Warning: Could not load trajectory from {trajectory_path}: {e}")
                self.trajectory_data = None

        # Create DOF mask for active joints (currently unused, kept for future use)
        self.dof_mask = None

        # Setup MSD system if configured
        self.msd_system = None
        if compliance_config:
            self._setup_msd_system(compliance_config)

    def _setup_msd_system(self, config):
        """Initialize MSD system from configuration.

        Args:
            config: MSD configuration dictionary with keys:
                - stiffness_config: Dict mapping joint/DOF names to stiffness scales
                - floating_base_map: Dict mapping floating base DOF names to indices (optional)
                - base_inertia: Base inertia value (default: 0.5)
                - base_stiffness: Base stiffness value (default: 60.0)
        """
        stiffness_config = config.get("stiffness_config", {})
        floating_base_map_local = config.get("floating_base_map", {})
        base_inertia = config.get("base_inertia", 0.5)
        base_stiffness = config.get("base_stiffness", 60.0)

        # Build DOF index to scale mapping
        stiffness_scales = {}  # Maps DOF index -> stiffness scale

        # Process floating base DOFs
        for name, dof_idx in floating_base_map_local.items():
            if name in stiffness_config:
                scale = stiffness_config[name]
                if scale > 0:
                    stiffness_scales[dof_idx] = scale

        # Process joint DOFs
        joint_configs = [
            (name, scale)
            for name, scale in stiffness_config.items()
            if name not in floating_base_map_local and scale > 0
        ]

        if len(joint_configs) > 0:
            joint_names, joint_scales = zip(*joint_configs)

            # Get joint IDs
            joint_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in joint_names
            ]

            # Filter valid joints
            valid_data = [
                (jid, scale) for jid, scale in zip(joint_ids, joint_scales) if jid >= 0
            ]

            if len(valid_data) > 0:
                valid_joint_ids, valid_scales = zip(*valid_data)
                valid_joint_ids = np.array(valid_joint_ids, dtype=np.int32)

                # Get DOF addresses
                joint_dof_addrs = self.model.jnt_dofadr[valid_joint_ids]

                # Map DOF indices to scales
                for dof_addr, scale in zip(joint_dof_addrs, valid_scales):
                    stiffness_scales[dof_addr] = scale

        # Get active DOF indices
        active_dof_indices = np.array(sorted(stiffness_scales.keys()), dtype=np.int32)
        print(f"Number of active DOFs: {len(active_dof_indices)}")

        # Create MSD system
        self.msd_system = MassSpringDamperModel(
            n_dofs=self.model.nv,
            dt=self.dt,
            base_inertia=base_inertia,
            base_stiffness=base_stiffness,
            stiffness_scales=stiffness_scales,
        )

    def reset(self, root_height: float = 0.9):
        """Reset simulation and MSD state.

        Args:
            root_height: Height of robot root in world frame (to match Isaac's fixed base height)
        """
        mujoco.mj_resetData(self.model, self.data)

        # Set all joint positions to zero
        if self._has_floating_base:
            self.data.qpos[:3] = [
                0,
                0,
                root_height,
            ]  # Position - match Isaac root height
            self.data.qpos[3:7] = [1, 0, 0, 0]  # Quaternion
            self.data.qpos[7:] = 0  # All joint positions to zero
        else:
            # No floating base (fixed base), just set everything to zero
            self.data.qpos[:] = 0

        self.data.qvel[:] = 0

        if self.msd_system:
            self.msd_system.reset()

        mujoco.mj_forward(self.model, self.data)

        print("SHAPE", self.data.qpos.shape)
        print("MuJoCo initial joint pos:", self.data.qpos)
        print(f"  Total qpos size: {self.model.nq}, Total joints: {self.model.njnt}")

    @property
    def msd_state(self):
        """MSD state dictionary."""
        return (
            self.msd_system.get_state_dict()
            if self.msd_system
            else {"q_def": np.array([]), "qd_def": np.array([])}
        )

    def apply_external_forces(
        self, body_names: list[str], forces: np.ndarray, torques: np.ndarray
    ) -> None:
        """Apply external forces and torques to specified bodies."""
        for idx, body_name in enumerate(body_names):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_XBODY, body_name
            )
            # xfrc_applied: (nbody, 6) array of external forces in world frame
            # Format: [force_x, force_y, force_z, torque_x, torque_y, torque_z]
            self.data.xfrc_applied[body_id, :3] = forces[idx]
            self.data.xfrc_applied[body_id, 3:] = torques[idx]

    def get_jacobians(self, body_names):
        """
        Get Jacobians for specified bodies in world-aligned frame.

        Args:
            mj_model: MuJoCo model
            mj_data: MuJoCo data
            body_id: ID of the body

        Returns:
            np.ndarray: 6xnv Jacobian matrix [linear; angular]
        """
        jacobians = []

        for body_name in body_names:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_XBODY, body_name
            )
            if body_id < 0:
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
                )

            jac_lin = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))

            # Use xipos (body CoM) as reference point to match where xfrc_applied acts
            # mj_jacBody uses xpos (frame origin), which is WRONG for tau = J^T @ F
            com_pos = self.data.xipos[body_id]
            mujoco.mj_jac(self.model, self.data, jac_lin, jac_rot, com_pos, body_id)

            # Combine linear and angular parts
            J = np.vstack([jac_lin, jac_rot])  # [6, nv]

            # Apply DOF mask if provided (zero out inactive DOFs)
            if self.dof_mask is not None:
                J_masked = np.zeros_like(J)
                J_masked[:, self.dof_mask] = J[:, self.dof_mask]
                J = J_masked

            jacobians.append(J)

        return np.array(jacobians)

    def _get_body_id(self, body_name: str):
        model = self.model
        """Get body ID from name."""
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in Mujoco model")
        return body_id

    def get_wrench(self, body_names: list[str]):
        """
        Get forces and torques on the specified bodies
        """
        ext_wrenches = []

        for body_name in body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            ext_force_torque = self.data.xfrc_applied[body_id].copy()
            ext_wrenches.append(ext_force_torque)

        return ext_wrenches

    def calculate_external_torques(self, body_names):
        """
        Calculate joint torques from external forces on specified bodies.
        """
        total_torques = np.zeros(self.model.nv)

        for body_name in body_names:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_XBODY, body_name
            )
            if body_id < 0:
                continue
            # Get external force and torque on this body [force, torque]
            ext_force_torque = self.data.xfrc_applied[body_id].copy()

            jac_lin = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))

            # Use xipos (body CoM) as reference point to match where xfrc_applied acts
            com_pos = self.data.xipos[body_id]
            mujoco.mj_jac(self.model, self.data, jac_lin, jac_rot, com_pos, body_id)

            # Combine linear and angular parts
            J = np.vstack([jac_lin, jac_rot])  # [6, nv]

            # Apply DOF mask if provided (zero out inactive DOFs)
            if self.dof_mask is not None:
                J_masked = np.zeros_like(J)
                J_masked[:, self.dof_mask] = J[:, self.dof_mask]
                J = J_masked

            # Compute joint torques: tau = J^T * F
            torques = J.T @ ext_force_torque
            total_torques += torques

        return total_torques

    def update_state(self, frame_idx=None, monitored_bodies=None, decimation=1):
        """Update MuJoCo state: optionally set trajectory position, then apply MSD deformation.

        Args:
            frame_idx: Trajectory frame index (None to use current state)
            monitored_bodies: List of body names for force calculation
            decimation: MSD integration sub-steps (default: 1)
        """
        # # Start with current state
        # qpos_ref = self.data.qpos.copy()

        # Build reference qpos from data
        qpos_ref = np.zeros(self.model.nq)

        # Override with trajectory data if provided
        if frame_idx is not None and self.trajectory_data is not None:
            # Loop trajectory if frame_idx exceeds bounds
            frame_idx = frame_idx % self.trajectory_data.n_frames

            # Set root if available
            if self.trajectory_data.root_positions is not None:
                qpos_root_start = self.qpos_idx.get(self.ROOT_NAME, 0)
                if qpos_root_start >= 0 and qpos_root_start + 7 <= self.model.nq:
                    qpos_ref[
                        qpos_root_start : qpos_root_start + 3
                    ] = self.trajectory_data.root_positions[frame_idx]
                    qpos_ref[
                        qpos_root_start + 3 : qpos_root_start + 7
                    ] = self.trajectory_data.root_quaternions[frame_idx]

            # Set joints
            if self.trajectory_data.joint_order is not None:
                for joint_idx, joint_name in enumerate(
                    self.trajectory_data.joint_order
                ):
                    if joint_name in self.qpos_idx:
                        qpos_addr = self.qpos_idx[joint_name]
                        if 0 <= qpos_addr < self.model.nq:
                            qpos_ref[qpos_addr] = self.trajectory_data.joint_positions[
                                frame_idx, joint_idx
                            ]

        # Store reference positions for debugging
        self.qpos_ref = qpos_ref.copy()

        # Apply MSD deformation if configured
        if self.msd_system and monitored_bodies and decimation > 0:
            for _ in range(decimation):
                # Map MSD deformation to full vector
                q_def_full = np.zeros(self.model.nv)
                q_def_full[self.msd_system.active_idx] = self.msd_system.state["q_def"][0].cpu().numpy()

                # Integrate deformation
                qpos_current = qpos_ref.copy()
                mujoco.mj_integratePos(self.model, qpos_current, q_def_full, 1.0)

                # Set positions and step physics (consistent with Isaac's sim.step())
                self.data.qpos[:] = qpos_current
                self.data.qvel[:] = 0
                mujoco.mj_step(self.model, self.data)

                # Calculate external torques
                external_torques = self.calculate_external_torques(monitored_bodies)

                # Update MSD state (convert to 2D tensor [1, n_dofs] for batched API)
                external_torques_tensor = torch.from_numpy(external_torques).float().unsqueeze(0)
                self.msd_system.update_msd_state_discrete(external_torques_tensor)

            # After MSD loop, re-apply final position to correct drift from last mj_step()
            # This matches Isaac behavior where final position is set after the loop
            q_def_full = np.zeros(self.model.nv)
            q_def_full[self.msd_system.active_idx] = self.msd_system.state["q_def"][0].cpu().numpy()

            qpos_actual = qpos_ref.copy()
            mujoco.mj_integratePos(self.model, qpos_actual, q_def_full, 1.0)
            self.data.qpos[:] = qpos_actual
            self.data.qvel[:] = 0
        else:
            # No MSD - just apply reference
            self.data.qpos[:] = qpos_ref

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

    def _check_floating_base(self) -> bool:
        """Check if the model has a floating base (free joint)."""
        if self.model.njnt == 0:
            return False

        # Check if first joint is a free joint (type 0 = mjJNT_FREE)
        first_joint_type = self.model.jnt_type[0]
        if first_joint_type == 0:  # mjJNT_FREE
            return True

        return False

    @property
    def has_floating_base(self) -> bool:
        return self._has_floating_base

    @staticmethod
    def _create_dof_mask(mj_model, active_joint_names: list[str]) -> np.ndarray:
        """
        Create a DOF mask for active joints.
        Returns:
            np.ndarray: Boolean mask array of size (nv,) with True for active DOFs
        """
        dof_mask = np.zeros(mj_model.nv, dtype=bool)

        for joint_name in active_joint_names:
            joint_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id >= 0:
                dof_addr = mj_model.jnt_dofadr[joint_id]
                dof_mask[dof_addr] = True

        return dof_mask
