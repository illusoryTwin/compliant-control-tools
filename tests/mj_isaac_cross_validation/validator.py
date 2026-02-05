from __future__ import annotations

import numpy as np
import mujoco
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.sim_wrappers import IsaacWrapper, MujocoWrapper


class ValidationRunner:
    def __init__(self, isaac_sim, mj_sim, config: dict = None):
        """
        Class to handle parallel computations in isaac and mujoco (for results comparison)
        Args:
            isaac_sim: Isaac Lab simulation wrapper
            mj_sim: Mujoco simulation wrapper
            config: configuration dict with:
                - monitored_bodies: List of body names to monitor
                - force_fn: Callable(step, time) -> forces tensor (default: uses default_force_fn)
                - torque_fn: Callable(step, time) -> torques tensor (default: uses default_torque_fn)
                - use_trajectory: If True, use trajectory replay mode (default: False)
                - decimation: MSD integration sub-steps per frame for trajectory mode
        """
        self.isaac_sim = isaac_sim
        self.mj_sim = mj_sim

        # Extract configuration
        config = config or {}
        self.monitored_bodies = config.get("monitored_bodies", [])
        self.force_fn = config.get("force_fn", self.default_force_fn)
        self.torque_fn = config.get("torque_fn", self.default_torque_fn)
        self.use_trajectory = config.get("use_trajectory", False)
        self.decimation = config.get("decimation", 1)

        self.current_time = 0.0
        self.step_count = 0
        self.current_frame = 0

    def default_force_fn(self, step: int, time: float):
        """Some random const force"""
        num_bodies = max(len(self.monitored_bodies), 1)
        return torch.tensor([[0.0, 0.0, 50.0]] * num_bodies)

    def default_torque_fn(self, step: int, time: float):
        """Some random const torque"""
        num_bodies = max(len(self.monitored_bodies), 1)
        torques = torch.tensor([[0.0, 0.0, 0.0]] * num_bodies)
        return torques

    def reset(self):
        """Reset both simulators."""
        self.isaac_sim.reset()
        self.mj_sim.reset()
        self.current_time = 0.0
        self.step_count = 0

        # Compare joint positions at reset
        print("\n=== INITIAL JOINT POSITION COMPARISON (after reset) ===")
        isaac_joint_names = self.isaac_sim.articulation.joint_names
        isaac_joint_pos = self.isaac_sim.articulation.data.joint_pos[0].cpu().numpy()

        print(f"{'Joint Name':<35} {'Isaac':>12} {'MuJoCo':>12} {'Diff':>12}")
        print("-" * 75)
        for i, joint_name in enumerate(isaac_joint_names):
            mj_joint_id = mujoco.mj_name2id(self.mj_sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if mj_joint_id >= 0:
                mj_qpos_addr = self.mj_sim.model.jnt_qposadr[mj_joint_id]
                mj_pos = self.mj_sim.data.qpos[mj_qpos_addr]
                diff = isaac_joint_pos[i] - mj_pos
                if abs(isaac_joint_pos[i]) > 1e-6 or abs(mj_pos) > 1e-6 or abs(diff) > 1e-6:
                    print(f"{joint_name:<35} {isaac_joint_pos[i]:>12.6f} {mj_pos:>12.6f} {diff:>12.6f}")

        # Compare Jacobians at initial pose (before any forces or MSD)
        print("\n=== INITIAL JACOBIAN COMPARISON (after reset, before any forces) ===")
        self._compare_jacobians()

        # Run analytical verification to investigate the offset
        self._verify_jacobians_analytically()

    def _get_forces_torques(self):
        """Get forces and torques for current timestep."""
        forces = self.force_fn(self.step_count, self.current_time)
        torques = self.torque_fn(self.step_count, self.current_time)
        return forces, torques

    def _apply_external_forces(self, forces, torques, body_names):
        self.isaac_sim.apply_external_forces(body_names, forces, torques)

        forces_np = (
            forces[0].cpu().numpy() if forces.dim() == 3 else forces.cpu().numpy()
        )
        torques_np = (
            torques[0].cpu().numpy() if torques.dim() == 3 else torques.cpu().numpy()
        )

        self.mj_sim.apply_external_forces(body_names, forces_np, torques_np)

    def _compare_ext_wrenches(self):
        """Compare external forces/torques between simulators."""

        isaac_ext_forces_torques = self.isaac_sim.get_wrench(self.monitored_bodies)
        print("ISAAC ext forces & torques", isaac_ext_forces_torques)

        mj_ext_forces_torques = self.mj_sim.get_wrench(self.monitored_bodies)
        print("MUJOCO ext forces & torques", np.array(mj_ext_forces_torques))

    def _verify_jacobians_analytically(self):
        """Analytically verify Jacobian values and investigate the ~0.0657m offset.

        For a revolute joint at position p_joint with axis a, the Jacobian for point p_end is:
            J_lin = a × (p_end - p_joint)  (linear velocity from rotation)
            J_ang = a                       (angular velocity contribution)
        """
        import mujoco

        print("\n" + "=" * 120)
        print("ANALYTICAL JACOBIAN VERIFICATION - Investigating 0.0657m offset")
        print("=" * 120)

        # Check Isaac's body_com_w - this is what PhysX uses for Jacobians
        print("\n=== Isaac Lab body_com_pos_w vs body_pos_w comparison ===")
        for body_name in self.monitored_bodies:
            body_idx = self.isaac_sim.articulation.body_names.index(body_name)
            isaac_pos_w = self.isaac_sim.articulation.data.body_pos_w[0, body_idx].cpu().numpy()
            isaac_com_w = self.isaac_sim.articulation.data.body_com_pos_w[0, body_idx].cpu().numpy()
            com_offset = isaac_com_w - isaac_pos_w

            print(f"{body_name}:")
            print(f"  body_pos_w (frame origin): {isaac_pos_w}")
            print(f"  body_com_pos_w (link CoM): {isaac_com_w}")
            print(f"  CoM offset from frame:     {com_offset} (norm: {np.linalg.norm(com_offset):.6f})")

        # Also check parent links
        print("\n=== Parent link CoM comparison ===")
        parent_links = ["left_elbow_yaw_link", "right_elbow_yaw_link"]
        for body_name in parent_links:
            if body_name in self.isaac_sim.articulation.body_names:
                body_idx = self.isaac_sim.articulation.body_names.index(body_name)
                isaac_pos_w = self.isaac_sim.articulation.data.body_pos_w[0, body_idx].cpu().numpy()
                isaac_com_w = self.isaac_sim.articulation.data.body_com_pos_w[0, body_idx].cpu().numpy()
                com_offset = isaac_com_w - isaac_pos_w

                print(f"{body_name}:")
                print(f"  body_pos_w (frame origin): {isaac_pos_w}")
                print(f"  body_com_pos_w (link CoM): {isaac_com_w}")
                print(f"  CoM offset from frame:     {com_offset}")

        for body_name in self.monitored_bodies:
            print(f"\n{'='*60}")
            print(f"Body: {body_name}")
            print(f"{'='*60}")

            # Get end-effector positions
            body_idx = self.isaac_sim.articulation.body_names.index(body_name)
            isaac_frame_pos = self.isaac_sim.articulation.data.body_pos_w[0, body_idx].cpu().numpy()

            mj_body_id = mujoco.mj_name2id(self.mj_sim.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            mj_xpos = self.mj_sim.data.xpos[mj_body_id].copy()
            mj_xipos = self.mj_sim.data.xipos[mj_body_id].copy()

            print(f"\nEnd-effector positions:")
            print(f"  Isaac body_pos_w:  {isaac_frame_pos}")
            print(f"  MuJoCo xpos:       {mj_xpos}")
            print(f"  MuJoCo xipos:      {mj_xipos}")

            # Check key joints
            test_joints = ["left_shoulder_roll_joint", "left_elbow_pitch_joint"] if "left" in body_name else \
                         ["right_shoulder_roll_joint", "right_elbow_pitch_joint"]

            print(f"\n--- Joint-by-joint analysis ---")

            isaac_jacobians = self.isaac_sim.get_jacobians(self.monitored_bodies)
            mj_jacobians = self.mj_sim.get_jacobians(self.monitored_bodies)

            for joint_name in test_joints:
                mj_joint_id = mujoco.mj_name2id(self.mj_sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if mj_joint_id < 0:
                    continue

                print(f"\n  Joint: {joint_name}")

                # Joint properties from MuJoCo
                joint_axis = self.mj_sim.data.xaxis[mj_joint_id].copy()
                joint_anchor = self.mj_sim.data.xanchor[mj_joint_id].copy()
                mj_dof = self.mj_sim.model.jnt_dofadr[mj_joint_id]

                print(f"    MuJoCo joint anchor: {joint_anchor}")
                print(f"    MuJoCo joint axis:   {joint_axis}")

                # Get Isaac's Jacobian
                body_idx_jac = self.monitored_bodies.index(body_name)
                if joint_name in self.isaac_sim.articulation.joint_names:
                    joint_idx = self.isaac_sim.articulation.joint_names.index(joint_name)
                    isaac_j_lin = isaac_jacobians[body_idx_jac, :3, joint_idx].cpu().numpy()
                    mj_j_lin = mj_jacobians[body_idx_jac, :3, mj_dof]

                    print(f"    Isaac J_lin:  {isaac_j_lin}")
                    print(f"    MuJoCo J_lin: {mj_j_lin}")
                    print(f"    Difference:   {isaac_j_lin - mj_j_lin}")

                    # Reverse-engineer the point Isaac uses
                    # J_lin = axis × (p_end - p_joint)
                    # For Y-axis rotation: J_lin_x = -r_z, J_lin_z = r_x
                    # So if we know J_lin and axis, we can compute r

                    # Compute analytical J_lin using MuJoCo's end-effector position
                    r_mj = mj_xipos - joint_anchor
                    j_lin_analytical_mj = np.cross(joint_axis, r_mj)

                    print(f"    Analytical (MuJoCo xipos): {j_lin_analytical_mj}")
                    print(f"    Match with MuJoCo? {np.allclose(j_lin_analytical_mj, mj_j_lin, atol=0.001)}")

                    # What end-effector position would Isaac need to produce its Jacobian?
                    # This tells us what reference point Isaac is using
                    if abs(joint_axis[1]) > 0.9:  # Y-axis joint (like elbow_pitch)
                        # For Y-axis: J_lin = [r_z, 0, -r_x] approximately
                        # Isaac J_lin_x = r_z => r_z = Isaac J_lin_x
                        # Isaac J_lin_z = -r_x => r_x = -Isaac J_lin_z
                        isaac_implied_r = np.array([-isaac_j_lin[2], 0, isaac_j_lin[0]])
                        isaac_implied_end = joint_anchor + isaac_implied_r
                        print(f"    Isaac implied end-effector pos: {isaac_implied_end}")
                        print(f"    Diff from MuJoCo xipos: {isaac_implied_end - mj_xipos}")

    def _compare_jacobians(self):
        """Compare Jacobians between simulators for active joints only."""
        import mujoco

        # First, compare body positions to verify robots are in same configuration
        print("\n=== BODY POSITION COMPARISON ===")
        for body_name in self.monitored_bodies:
            # Isaac body position
            body_idx = self.isaac_sim.articulation.body_names.index(body_name)
            isaac_pos = self.isaac_sim.articulation.data.body_pos_w[0, body_idx].cpu().numpy()

            # MuJoCo positions - show both xpos (frame origin) and xipos (CoM)
            mj_body_id = mujoco.mj_name2id(self.mj_sim.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            mj_xpos = self.mj_sim.data.xpos[mj_body_id].copy()
            mj_xipos = self.mj_sim.data.xipos[mj_body_id].copy()
            com_offset = mj_xipos - mj_xpos

            print(f"{body_name}:")
            print(f"  Isaac pos (frame):   {isaac_pos}")
            print(f"  MuJoCo xpos (frame): {mj_xpos}")
            print(f"  MuJoCo xipos (CoM):  {mj_xipos}  <-- Jacobian & force reference point")
            print(f"  CoM offset:          {com_offset} (norm: {np.linalg.norm(com_offset):.6f})")

        mj_jacobians = self.mj_sim.get_jacobians(self.monitored_bodies)
        isaac_jacobians = self.isaac_sim.get_jacobians(self.monitored_bodies)

        isaac_joint_names = self.isaac_sim.articulation.joint_names

        print("\nJacobian comparison (active joints only):")
        print(f"Bodies: {self.monitored_bodies}")
        print(
            f"Isaac Jacobian shape: {isaac_jacobians.shape}, MuJoCo Jacobian shape: {mj_jacobians.shape}"
        )

        for body_idx, body_name in enumerate(self.monitored_bodies):
            print(f"\n--- Body: {body_name} ---")
            print(
                f"{'Joint Name':<35} {'Row':<5} {'Isaac':>12} {'MuJoCo':>12} {'Diff':>12}"
            )
            print("-" * 80)

            for i, joint_name in enumerate(isaac_joint_names):
                # Check if joint is active in Isaac
                if (
                    self.isaac_sim.joint_mask is not None
                    and not self.isaac_sim.joint_mask[i]
                ):
                    continue  # Skip inactive joints

                # Isaac: joint index directly (fixed base)
                isaac_col = i

                # MuJoCo: get DOF address from joint name
                mj_joint_id = mujoco.mj_name2id(
                    self.mj_sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                if mj_joint_id < 0:
                    continue
                mj_col = self.mj_sim.model.jnt_dofadr[mj_joint_id]

                # Print all 6 rows (3 linear + 3 angular)
                row_names = ["lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z"]
                for row in range(6):
                    isaac_val = isaac_jacobians[body_idx, row, isaac_col].item()
                    mj_val = mj_jacobians[body_idx, row, mj_col]
                    diff = isaac_val - mj_val

                    # Only print if at least one value is non-zero
                    if abs(isaac_val) > 1e-6 or abs(mj_val) > 1e-6:
                        print(
                            f"{joint_name:<35} {row_names[row]:<5} {isaac_val:>12.6f} {mj_val:>12.6f} {diff:>12.6f}"
                        )

    def _compare_torques(self):
        """Compare computed torques between simulators for active joints only."""
        import mujoco

        # First, show Isaac world vs body frame comparison
        print("\n" + "=" * 80)
        print("ISAAC: World Frame vs Body Frame Torque Comparison")
        print("=" * 80)
        (
            isaac_torques_w,
            isaac_torques_b,
        ) = self.isaac_sim.calculate_external_torques_compare(self.monitored_bodies)

        # Get MuJoCo torques (world frame)
        mujoco_torques = self.mj_sim.calculate_external_torques(self.monitored_bodies)

        # Now compare Isaac (world frame) vs MuJoCo
        print("\n" + "=" * 80)
        print("ISAAC vs MUJOCO Torques Comparison (World Frame)")
        print("=" * 80)
        print(
            f"{'Joint Name':<35} {'Isaac(W)':>12} {'Isaac(B)':>12} {'MuJoCo':>12} {'Diff(W)':>12}"
        )
        print("-" * 90)

        # Get active joints from Isaac joint_mask
        isaac_joint_names = self.isaac_sim.articulation.joint_names

        # Determine offset based on fix_base setting
        base_offset = 0 if self.isaac_sim.fix_base else 6

        for i, joint_name in enumerate(isaac_joint_names):
            # Check if joint is active in Isaac
            isaac_dof_idx = base_offset + i
            if (
                self.isaac_sim.joint_mask is not None
                and not self.isaac_sim.joint_mask[i]
            ):
                continue  # Skip inactive joints

            isaac_val_w = isaac_torques_w[0, isaac_dof_idx].item()
            isaac_val_b = isaac_torques_b[0, isaac_dof_idx].item()

            # MuJoCo: get DOF address from joint name
            mj_joint_id = mujoco.mj_name2id(
                self.mj_sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if mj_joint_id >= 0:
                mj_dof_addr = self.mj_sim.model.jnt_dofadr[mj_joint_id]
                mj_val = mujoco_torques[mj_dof_addr]
            else:
                mj_val = float("nan")

            diff = isaac_val_w - mj_val

            print(
                f"{joint_name:<35} {isaac_val_w:>12.6f} {isaac_val_b:>12.6f} {mj_val:>12.6f} {diff:>12.6f}"
            )

    def _print_msd_state(self):
        """Print MSD state comparison for active joints: reference, deformation, and final trajectories."""
        import mujoco

        if not (self.isaac_sim.msd_system or self.mj_sim.msd_system):
            return

        print(f"\n[Step {self.step_count}] Trajectory comparison (active joints only):")
        print(
            f"{'Joint Name':<28} {'Isaac ref':>10} {'MJ ref':>10} {'Isaac def':>11} {'MJ def':>11} {'Isaac final':>12} {'MJ final':>12}"
        )
        print("-" * 108)

        isaac_joint_names = self.isaac_sim.articulation.joint_names

        # Get active DOF indices for both simulators
        isaac_active_idx = (
            self.isaac_sim.msd_system.active_idx if self.isaac_sim.msd_system else []
        )
        mj_active_idx = (
            self.mj_sim.msd_system.active_idx if self.mj_sim.msd_system else []
        )

        for i, joint_name in enumerate(isaac_joint_names):
            # Check if joint is active in Isaac (mask index is now just joint index for fixed base)
            if (
                self.isaac_sim.joint_mask is not None
                and not self.isaac_sim.joint_mask[i]
            ):
                continue  # Skip inactive joints

            # Isaac: get reference position
            isaac_ref = 0.0
            if self.isaac_sim.joint_pos_ref is not None:
                isaac_ref = self.isaac_sim.joint_pos_ref[0, i].item()

            # Isaac: get deformation (MSD uses joint index without offset)
            isaac_q_def = 0.0
            if self.isaac_sim.msd_system and i in isaac_active_idx:
                active_pos = np.where(isaac_active_idx == i)[0]
                if len(active_pos) > 0:
                    isaac_q_def = self.isaac_sim.msd_system.state["q_def"][
                        0, active_pos[0]
                    ].cpu().item()

            # MuJoCo: get reference position and deformation
            mj_ref = 0.0
            mj_q_def = 0.0
            mj_joint_id = mujoco.mj_name2id(
                self.mj_sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if mj_joint_id >= 0:
                # Get MuJoCo reference position
                qpos_addr = self.mj_sim.model.jnt_qposadr[mj_joint_id]
                if self.mj_sim.qpos_ref is not None and qpos_addr < len(
                    self.mj_sim.qpos_ref
                ):
                    mj_ref = self.mj_sim.qpos_ref[qpos_addr]

                # Get MuJoCo deformation
                if self.mj_sim.msd_system:
                    mj_dof_addr = self.mj_sim.model.jnt_dofadr[mj_joint_id]
                    if mj_dof_addr in mj_active_idx:
                        active_pos = np.where(mj_active_idx == mj_dof_addr)[0]
                        if len(active_pos) > 0:
                            mj_q_def = self.mj_sim.msd_system.state["q_def"][
                                0, active_pos[0]
                            ].cpu().item()

            # Calculate final positions (ref + deformation)
            isaac_final = isaac_ref + isaac_q_def
            mj_final = mj_ref + mj_q_def

            print(
                f"{joint_name:<28} {isaac_ref:>10.6f} {mj_ref:>10.6f} {isaac_q_def:>11.7f} {mj_q_def:>11.7f} {isaac_final:>12.6f} {mj_final:>12.6f}"
            )

    def run(self, sim_dt: float, total_time: float, simulation_app=None):
        """Run validation loop."""
        self.reset()
        frame_idx = 0

        def should_continue():
            if simulation_app is not None:
                return self.current_time < total_time and simulation_app.is_running()
            return self.current_time < total_time

        while should_continue():
            forces, torques = self._get_forces_torques()

            self._apply_external_forces(forces, torques, self.monitored_bodies)

            # Compare wrenches BEFORE update_state() to avoid body orientation changes
            # Print debug info for first 5 steps and every 50 steps after
            should_print_debug = self.step_count < 5 or self.step_count % 50 == 0
            if should_print_debug:
                self._compare_ext_wrenches()
                self._compare_jacobians()
                self._compare_torques()

            # Update both simulators with MSD
            self.isaac_sim.update_state(
                frame_idx=frame_idx if self.isaac_sim.trajectory_data else None,
                monitored_bodies=self.monitored_bodies,
                decimation=self.decimation,
            )

            self.mj_sim.update_state(
                frame_idx=frame_idx if self.mj_sim.trajectory_data else None,
                monitored_bodies=self.monitored_bodies,
                decimation=self.decimation,
            )

            if should_print_debug:
                self._print_msd_state()

            # Update state
            self.current_time += sim_dt
            self.step_count += 1
            frame_idx += 1
