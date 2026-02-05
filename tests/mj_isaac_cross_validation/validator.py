from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import mujoco
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.sim_wrappers import IsaacWrapper, MujocoWrapper

# =======================
# Comparator tools 
# =======================

@dataclass
class JointPositionComparison:
    joint_name: str
    isaac_pos: float
    mujoco_pos: float

    @property
    def diff(self) -> float:
        return self.isaac_pos - self.mujoco_pos


@dataclass
class BodyPositionComparison:
    body_name: str
    isaac_pos: np.ndarray
    mujoco_xpos: np.ndarray
    mujoco_xipos: np.ndarray

    @property
    def com_offset(self) -> np.ndarray:
        return self.mujoco_xipos - self.mujoco_xpos


@dataclass
class JacobianEntry:
    joint_name: str
    row_name: str
    isaac_val: float
    mujoco_val: float

    @property
    def diff(self) -> float:
        return self.isaac_val - self.mujoco_val


@dataclass
class JacobianComparison:
    body_name: str
    entries: list[JacobianEntry] = field(default_factory=list)


@dataclass
class TorqueEntry:
    joint_name: str
    isaac_world: float
    isaac_body: float
    mujoco_val: float

    @property
    def diff(self) -> float:
        return self.isaac_world - self.mujoco_val


@dataclass
class MSDStateEntry:
    joint_name: str
    isaac_ref: float
    mujoco_ref: float
    isaac_def: float
    mujoco_def: float

    @property
    def isaac_final(self) -> float:
        return self.isaac_ref + self.isaac_def

    @property
    def mujoco_final(self) -> float:
        return self.mujoco_ref + self.mujoco_def


@dataclass
class WrenchComparison:
    isaac_wrench: np.ndarray
    mujoco_wrench: np.ndarray


# Printer class to compare results
class ComparisonPrinter:
    """Handles all formatting and printing of comparison results."""

    @staticmethod
    def print_joint_positions(comparisons: list[JointPositionComparison], title: str = "JOINT POSITION COMPARISON"):
        print(f"\n=== {title} ===")
        print(f"{'Joint Name':<35} {'Isaac':>12} {'MuJoCo':>12} {'Diff':>12}")
        print("-" * 75)
        for c in comparisons:
            if abs(c.isaac_pos) > 1e-6 or abs(c.mujoco_pos) > 1e-6 or abs(c.diff) > 1e-6:
                print(f"{c.joint_name:<35} {c.isaac_pos:>12.6f} {c.mujoco_pos:>12.6f} {c.diff:>12.6f}")

    @staticmethod
    def print_body_positions(comparisons: list[BodyPositionComparison]):
        print("\n=== BODY POSITION COMPARISON ===")
        for c in comparisons:
            print(f"{c.body_name}:")
            print(f"  Isaac pos (frame):   {c.isaac_pos}")
            print(f"  MuJoCo xpos (frame): {c.mujoco_xpos}")
            print(f"  MuJoCo xipos (CoM):  {c.mujoco_xipos}  <-- Jacobian & force reference point")
            print(f"  CoM offset:          {c.com_offset} (norm: {np.linalg.norm(c.com_offset):.6f})")

    @staticmethod
    def print_jacobians(
        comparisons: list[JacobianComparison],
        isaac_shape: tuple,
        mujoco_shape: tuple,
        body_names: list[str],
    ):
        print("\nJacobian comparison (active joints only):")
        print(f"Bodies: {body_names}")
        print(f"Isaac Jacobian shape: {isaac_shape}, MuJoCo Jacobian shape: {mujoco_shape}")

        for jac_comp in comparisons:
            print(f"\n--- Body: {jac_comp.body_name} ---")
            print(f"{'Joint Name':<35} {'Row':<5} {'Isaac':>12} {'MuJoCo':>12} {'Diff':>12}")
            print("-" * 80)
            for e in jac_comp.entries:
                print(f"{e.joint_name:<35} {e.row_name:<5} {e.isaac_val:>12.6f} {e.mujoco_val:>12.6f} {e.diff:>12.6f}")

    @staticmethod
    def print_torques(entries: list[TorqueEntry]):
        print("\n" + "=" * 80)
        print("ISAAC: World Frame vs Body Frame Torque Comparison")
        print("=" * 80)
        print("\n" + "=" * 80)
        print("ISAAC vs MUJOCO Torques Comparison (World Frame)")
        print("=" * 80)
        print(f"{'Joint Name':<35} {'Isaac(W)':>12} {'Isaac(B)':>12} {'MuJoCo':>12} {'Diff(W)':>12}")
        print("-" * 90)
        for e in entries:
            print(f"{e.joint_name:<35} {e.isaac_world:>12.6f} {e.isaac_body:>12.6f} {e.mujoco_val:>12.6f} {e.diff:>12.6f}")

    @staticmethod
    def print_wrenches(comparison: WrenchComparison):
        print("ISAAC ext forces & torques", comparison.isaac_wrench)
        print("MUJOCO ext forces & torques", comparison.mujoco_wrench)

    @staticmethod
    def print_msd_state(entries: list[MSDStateEntry], step_count: int):
        if not entries:
            return
        print(f"\n[Step {step_count}] Trajectory comparison (active joints only):")
        print(
            f"{'Joint Name':<28} {'Isaac ref':>10} {'MJ ref':>10} {'Isaac def':>11} {'MJ def':>11} {'Isaac final':>12} {'MJ final':>12}"
        )
        print("-" * 108)
        for e in entries:
            print(
                f"{e.joint_name:<28} {e.isaac_ref:>10.6f} {e.mujoco_ref:>10.6f} "
                f"{e.isaac_def:>11.7f} {e.mujoco_def:>11.7f} {e.isaac_final:>12.6f} {e.mujoco_final:>12.6f}"
            )



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
        self.printer = ComparisonPrinter()

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
        joint_comparisons = self._collect_joint_positions()
        self.printer.print_joint_positions(joint_comparisons, "INITIAL JOINT POSITION COMPARISON (after reset)")

        # Compare Jacobians at initial pose (before any forces or compliance deformations)
        print("\n=== INITIAL JACOBIAN COMPARISON ===")
        self._compare_jacobians()

 
    def _get_forces_torques(self):
        forces = self.force_fn(self.step_count, self.current_time)
        torques = self.torque_fn(self.step_count, self.current_time)
        return forces, torques

    def _apply_external_forces(self, forces, torques, body_names):
        self.isaac_sim.apply_external_forces(body_names, forces, torques)

        forces_np = (forces[0].cpu().numpy() if forces.dim() == 3 else forces.cpu().numpy())
        torques_np = (torques[0].cpu().numpy() if torques.dim() == 3 else torques.cpu().numpy())
        self.mj_sim.apply_external_forces(body_names, forces_np, torques_np)


    def _collect_joint_positions(self) -> list[JointPositionComparison]:
        """Collect joint position comparisons between simulators."""
        comparisons = []
        isaac_joint_names = self.isaac_sim.articulation.joint_names
        isaac_joint_pos = self.isaac_sim.articulation.data.joint_pos[0].cpu().numpy()

        for i, joint_name in enumerate(isaac_joint_names):
            mj_joint_id = mujoco.mj_name2id(self.mj_sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if mj_joint_id >= 0:
                mj_qpos_addr = self.mj_sim.model.jnt_qposadr[mj_joint_id]
                mj_pos = self.mj_sim.data.qpos[mj_qpos_addr]
                comparisons.append(JointPositionComparison(
                    joint_name=joint_name,
                    isaac_pos=float(isaac_joint_pos[i]),
                    mujoco_pos=float(mj_pos),
                ))
        return comparisons

    def _collect_body_positions(self) -> list[BodyPositionComparison]:
        """Collect body position comparisons between simulators."""
        comparisons = []
        for body_name in self.monitored_bodies:
            body_idx = self.isaac_sim.articulation.body_names.index(body_name)
            isaac_pos = self.isaac_sim.articulation.data.body_pos_w[0, body_idx].cpu().numpy()

            mj_body_id = mujoco.mj_name2id(self.mj_sim.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            mj_xpos = self.mj_sim.data.xpos[mj_body_id].copy()
            mj_xipos = self.mj_sim.data.xipos[mj_body_id].copy()

            comparisons.append(BodyPositionComparison(
                body_name=body_name,
                isaac_pos=isaac_pos,
                mujoco_xpos=mj_xpos,
                mujoco_xipos=mj_xipos,
            ))
        return comparisons

    def _collect_jacobians(self) -> tuple[list[JacobianComparison], tuple, tuple]:
        """Collect jacobian comparisons between simulators."""
        mj_jacobians = self.mj_sim.get_jacobians(self.monitored_bodies)
        isaac_jacobians = self.isaac_sim.get_jacobians(self.monitored_bodies)
        isaac_joint_names = self.isaac_sim.articulation.joint_names

        row_names = ["lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z"]
        comparisons = []

        for body_idx, body_name in enumerate(self.monitored_bodies):
            body_comparison = JacobianComparison(body_name=body_name)

            for i, joint_name in enumerate(isaac_joint_names):
                if self.isaac_sim.joint_mask is not None and not self.isaac_sim.joint_mask[i]:
                    continue

                isaac_col = i
                mj_joint_id = mujoco.mj_name2id(self.mj_sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if mj_joint_id < 0:
                    continue
                mj_col = self.mj_sim.model.jnt_dofadr[mj_joint_id]

                for row in range(6):
                    isaac_val = isaac_jacobians[body_idx, row, isaac_col].item()
                    mj_val = mj_jacobians[body_idx, row, mj_col]

                    if abs(isaac_val) > 1e-6 or abs(mj_val) > 1e-6:
                        body_comparison.entries.append(JacobianEntry(
                            joint_name=joint_name,
                            row_name=row_names[row],
                            isaac_val=isaac_val,
                            mujoco_val=mj_val,
                        ))

            comparisons.append(body_comparison)

        return comparisons, isaac_jacobians.shape, mj_jacobians.shape

    def _collect_wrenches(self) -> WrenchComparison:
        isaac_wrench = self.isaac_sim.get_wrench(self.monitored_bodies)
        mj_wrench = self.mj_sim.get_wrench(self.monitored_bodies)
        return WrenchComparison(
            isaac_wrench=isaac_wrench,
            mujoco_wrench=np.array(mj_wrench),
        )

    def _collect_torques(self) -> list[TorqueEntry]:
        isaac_torques_w, isaac_torques_b = self.isaac_sim.calculate_external_torques_compare(self.monitored_bodies)
        mujoco_torques = self.mj_sim.calculate_external_torques(self.monitored_bodies)

        isaac_joint_names = self.isaac_sim.articulation.joint_names
        base_offset = 0 if self.isaac_sim.fix_base else 6

        entries = []
        for i, joint_name in enumerate(isaac_joint_names):
            isaac_dof_idx = base_offset + i
            if self.isaac_sim.joint_mask is not None and not self.isaac_sim.joint_mask[i]:
                continue

            isaac_val_w = isaac_torques_w[0, isaac_dof_idx].item()
            isaac_val_b = isaac_torques_b[0, isaac_dof_idx].item()

            mj_joint_id = mujoco.mj_name2id(self.mj_sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if mj_joint_id >= 0:
                mj_dof_addr = self.mj_sim.model.jnt_dofadr[mj_joint_id]
                mj_val = mujoco_torques[mj_dof_addr]
            else:
                mj_val = float("nan")

            entries.append(TorqueEntry(
                joint_name=joint_name,
                isaac_world=isaac_val_w,
                isaac_body=isaac_val_b,
                mujoco_val=mj_val,
            ))

        return entries

    def _collect_msd_state(self) -> list[MSDStateEntry]:
        if not (self.isaac_sim.msd_system or self.mj_sim.msd_system):
            return []

        isaac_joint_names = self.isaac_sim.articulation.joint_names
        isaac_active_idx = self.isaac_sim.msd_system.active_idx if self.isaac_sim.msd_system else []
        mj_active_idx = self.mj_sim.msd_system.active_idx if self.mj_sim.msd_system else []

        entries = []
        for i, joint_name in enumerate(isaac_joint_names):
            if self.isaac_sim.joint_mask is not None and not self.isaac_sim.joint_mask[i]:
                continue

            # Isaac reference position
            isaac_ref = 0.0
            if self.isaac_sim.joint_pos_ref is not None:
                isaac_ref = self.isaac_sim.joint_pos_ref[0, i].item()

            # Isaac deformation
            isaac_q_def = 0.0
            if self.isaac_sim.msd_system and i in isaac_active_idx:
                active_pos = np.where(isaac_active_idx == i)[0]
                if len(active_pos) > 0:
                    isaac_q_def = self.isaac_sim.msd_system.state["q_def"][0, active_pos[0]].cpu().item()

            # MuJoCo reference and deformation
            mj_ref = 0.0
            mj_q_def = 0.0
            mj_joint_id = mujoco.mj_name2id(self.mj_sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if mj_joint_id >= 0:
                qpos_addr = self.mj_sim.model.jnt_qposadr[mj_joint_id]
                if self.mj_sim.qpos_ref is not None and qpos_addr < len(self.mj_sim.qpos_ref):
                    mj_ref = self.mj_sim.qpos_ref[qpos_addr]

                if self.mj_sim.msd_system:
                    mj_dof_addr = self.mj_sim.model.jnt_dofadr[mj_joint_id]
                    if mj_dof_addr in mj_active_idx:
                        active_pos = np.where(mj_active_idx == mj_dof_addr)[0]
                        if len(active_pos) > 0:
                            mj_q_def = self.mj_sim.msd_system.state["q_def"][0, active_pos[0]].cpu().item()

            entries.append(MSDStateEntry(
                joint_name=joint_name,
                isaac_ref=isaac_ref,
                mujoco_ref=mj_ref,
                isaac_def=isaac_q_def,
                mujoco_def=mj_q_def,
            ))

        return entries

    # Compare results between Mujoco and Isaac
    def _compare_ext_wrenches(self):
        comparison = self._collect_wrenches()
        self.printer.print_wrenches(comparison)

    def _compare_jacobians(self):
        # body_comparisons = self._collect_body_positions()
        # self.printer.print_body_positions(body_comparisons)

        jacobian_comparisons, isaac_shape, mj_shape = self._collect_jacobians()
        self.printer.print_jacobians(jacobian_comparisons, isaac_shape, mj_shape, self.monitored_bodies)

    def _compare_torques(self):
        torque_entries = self._collect_torques()
        self.printer.print_torques(torque_entries)

    def _print_msd_state(self):
        msd_entries = self._collect_msd_state()
        self.printer.print_msd_state(msd_entries, self.step_count)

    def run(self, sim_dt: float, total_time: float, simulation_app=None):
        """Run validation loop."""
        self.reset()
        frame_idx = 0

        def should_continue():
            if simulation_app is not None:
                return self.current_time < total_time and simulation_app.is_running()
            return self.current_time < total_time

        while should_continue():
            forces, torques = self._get_forces_torques()  # initial check

            self._apply_external_forces(forces, torques, self.monitored_bodies)

            # Compare wrenches BEFORE update_state() to avoid body orientation changes
            should_print_debug = self.step_count < 5 or self.step_count % 50 == 0
            if should_print_debug:
                self._compare_ext_wrenches()
                self._compare_jacobians()
                self._compare_torques()

            # Update both simulators
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
