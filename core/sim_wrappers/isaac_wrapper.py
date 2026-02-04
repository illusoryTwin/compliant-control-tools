from __future__ import annotations

import torch

from .base import SimulationWrapper
from core.compliance_model.mass_spring_damper_model import MassSpringDamperModel
from data.trajectory_loader import TrajectoryLoader
from utils.dynamics import (                                                                                                          
    apply_external_forces,                                                                                                            
    calculate_external_torques,                                                                                                       
    calculate_external_torques_b,                                                                                                     
    calculate_external_torques_compare,                                                                                               
    create_joint_mask,                                                                                                                
    get_jacobians,                                                                                                                    
    get_jacobians_b,                                                                                                                  
    get_wrench,                                                                                                                       
    get_wrench_b,                                                                                                                     
) 

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext


class IsaacWrapper(SimulationWrapper):
    def __init__(
        self,                                                                                                                         
        robot_cfg,                                                                                                                    
        dt: float,                                                                                                                    
        compliance_config: dict | None = None,                                                                                               
        active_joints: list[str] | None = None,                                                                                       
        trajectory_path: str | None = None,     
    ):
        """
        Initialize Isaac Lab wrapper.

        Args:
            robot_cfg: Robot configuration
            dt: Simulation timestep
            compliance_config: MSD configuration dictionary (optional)
            active_joints: List of active joint names (optional)
            trajectory_path: Path to trajectory pickle file (optional)
        """
        super().__init__(dt)

        self.robot_cfg = robot_cfg.copy()
        self._setup_simulation()
        self._setup_articulation()


        # Store fix_base setting for mask creation
        self.fix_base = self.robot_cfg.spawn.fix_base

        # Create joint mask for active joints
        self.joint_mask = None                                                                                                        
        if active_joints:                                                                                                             
            active_indices = [                                                                                                        
                self.articulation.joint_names.index(name)                                                                             
                for name in active_joints                                                                                             
                if name in self.articulation.joint_names                                                                              
            ]                                                                                                                         
            self.joint_mask = create_joint_mask(                                                                                      
                num_joints=self.articulation.num_joints,                                                                              
                active_joint_indices=active_indices,                                                                                  
                fix_base=self.fix_base,                                                                                               
            ) 

        # self.joint_mask = None
        # if active_joints:
        #     self.joint_mask = self._create_joint_mask(
        #         self.articulation, active_joints, fix_base=self.fix_base
        #     )

        # Setup MSD system if configured
        self.msd_system = None
        if compliance_config:
            self._setup_msd_system(compliance_config)


        # Store reference positions for debugging
        self.joint_pos_ref = None

        self._setup_trajectory(trajectory_path)


    def _setup_simulation(self) -> None:
        """Setup Isaac simulation context, ground plane, and lights."""
        sim_cfg = sim_utils.SimulationCfg(device="cpu", dt=self.dt)
        self.sim = SimulationContext(sim_cfg)

        # Ground-plane
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg, translation=(0, 0, -1.0))

        # Lights
        cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        cfg.func("/World/Light", cfg)
        
        # Camera
        self.sim.set_camera_view(
            [1, 0.0, self.robot_cfg.init_state.pos[2]], self.robot_cfg.init_state.pos
        )

    def _setup_articulation(self) -> None:
        """Setup robot articulation."""
        # Fix base
        self.robot_cfg.spawn.fix_base = True
        self.robot_cfg.prim_path = "/Robot"
        self.articulation = Articulation(cfg=self.robot_cfg)
        
        self.sim.reset()

        root_state = self.articulation.data.default_root_state.clone()
        self.articulation.write_root_state_to_sim(root_state)
        self.articulation.write_joint_state_to_sim(
            self.articulation.data.default_joint_pos,
            self.articulation.data.default_joint_vel,
        )
        self.articulation.reset()

    def _setup_trajectory(self, trajectory_path: str | None) -> None:
        self.ROOT_NAME = "root"

        # Load trajectory if path provided
        self.trajectory_loader = TrajectoryLoader(root_name=self.ROOT_NAME)
        self.trajectory_data = None
        if trajectory_path:
            try:
                self.trajectory_data = self.trajectory_loader.load(trajectory_path)
                print(
                    f"Isaac loaded trajectory: {self.trajectory_data.n_frames} frames, {self.trajectory_data.n_joints} joints"
                )
            except (FileNotFoundError, KeyError) as e:
                print(f"Warning: Could not load trajectory from {trajectory_path}: {e}")
                self.trajectory_data = None


    def _setup_msd_system(self, config: dict) -> None:
        """Initialize MSD system from configuration.

        Args:
            config: MSD configuration dictionary with keys:
                - stiffness_config: Dict mapping joint names to stiffness scales
                - base_inertia: Base inertia value (default: 0.5)
                - base_stiffness: Base stiffness value (default: 60.0)
        """
        stiffness_config = config.get("stiffness_config", {})
        base_inertia = config.get("base_inertia", 0.5)
        base_stiffness = config.get("base_stiffness", 60.0)

        n_dofs = self.articulation.num_joints

        # Build DOF index to scale mapping
        stiffness_scales = {}
        for joint_name, scale in stiffness_config.items():
            if scale > 0 and joint_name in self.articulation.joint_names:
                joint_idx = self.articulation.joint_names.index(joint_name)
                stiffness_scales[joint_idx] = scale

        self.msd_system = MassSpringDamperModel(
            n_dofs=n_dofs,
            dt=self.dt,
            base_inertia=base_inertia,
            base_stiffness=base_stiffness,
            stiffness_scales=stiffness_scales,
        )

    def reset(self):
        """Reset simulation and MSD state."""
        joint_pos = torch.zeros_like(self.articulation.data.default_joint_pos)
        joint_vel = torch.zeros_like(self.articulation.data.default_joint_vel)

        root_state = self.articulation.data.default_root_state.clone()
        self.articulation.write_root_state_to_sim(root_state)

        self.articulation.write_joint_state_to_sim(joint_pos, joint_vel)

        self.sim.step()
        self.articulation.update(self.dt)

        if self.msd_system:
            self.msd_system.reset()

        print("Isaac initial joint pos:", self.articulation.data.joint_pos)


    # -------------------------------------------------------------------------                                                      
    # Dynamics wrappers                                                                     
    # -------------------------------------------------------------------------  

    def apply_external_forces(                                                                                                                
        self,                                                                                                                        
        body_names: list[str],                                                                                                       
        forces: torch.Tensor,                                                                                                        
        torques: torch.Tensor,                                                                                                       
    ) -> None:                                                                                                                       
        """Apply external forces and torques to specified bodies (world frame)."""                                                   
        apply_external_forces(self.articulation, body_names, forces, torques)  

    def get_jacobians(self, body_names: list[str]) -> torch.Tensor:                                                            
        """Get Jacobians for specified bodies in WORLD frame."""                                                                     
        return get_jacobians(self.articulation, body_names, self.joint_mask)
       
    def get_jacobians_b(self, body_names: list[str]) -> torch.Tensor:                                                             
        """Get Jacobians for specified bodies in BODY frame."""                                                                      
        return get_jacobians_b(self.articulation, body_names, self.joint_mask)    
    
    def get_wrench(self, body_names: list[str]) -> torch.Tensor:                                                               
        """Get external wrench in WORLD frame."""                                                                                    
        return get_wrench(self.articulation, body_names)  
    
    def get_wrench_b(self, body_names: list[str]) -> torch.Tensor:                                                                
        """Get external wrench in BODY frame."""                                                                                     
        return get_wrench_b(self.articulation, body_names) 
    
    def calculate_external_torques(                                                                                                             
        self,                                                                                                                        
        body_names: list[str],                                                                                                       
        verbose: bool = False,                                                                                                       
    ) -> torch.Tensor:                                                                                                               
        """Calculate joint torques from external forces (world frame)."""                                                            
        return calculate_external_torques(                                                                                           
            self.articulation, body_names, self.joint_mask, verbose                                                                  
        ) 
    
    def calculate_external_torques_b(                                                                                                        
        self,                                                                                                                        
        body_names: list[str],                                                                                                       
        verbose: bool = False,                                                                                                       
    ) -> torch.Tensor:                                                                                                               
        """Calculate joint torques from external forces (body frame)."""                                                             
        return calculate_external_torques_b(                                                                                         
            self.articulation, body_names, self.joint_mask, verbose                                                                  
        )
    
    def calculate_external_torques_compare(                                                                                                     
        self,                                                                                                                        
        body_names: list[str],                                                                                                       
    ) -> tuple[torch.Tensor, torch.Tensor]:                                                                                          
        """Calculate and compare torques in world and body frames."""                                                                
        return calculate_external_torques_compare(                                                                                   
            self.articulation, body_names, self.joint_mask                                                                           
        )   


    # -------------------------------------------------------------------------                                                      
    # State update                                                                                                                   
    # -------------------------------------------------------------------------   

    def update_state(
            self, 
            frame_idx: int | None = None, 
            monitored_bodies: list[str] | None = None, 
            decimation: int = 1
        ):
        """Update Isaac state: set trajectory position, then apply MSD deformation.

        Args:
            frame_idx: Trajectory frame index (None to use current state)
            monitored_bodies: List of body names for force calculation
            decimation: MSD integration sub-steps (default: 1)
        """
        # Start with current state
        joint_pos_ref = self.articulation.data.joint_pos.clone()
        joint_vel_ref = self.articulation.data.joint_vel.clone()

        # Override with trajectory data if provided
        if frame_idx is not None and self.trajectory_data is not None:
            # Clamp frame_idx to valid range (loop or clamp)
            frame_idx = frame_idx % self.trajectory_data.n_frames

            # Set joints from trajectory
            if self.trajectory_data.joint_order is not None:
                for joint_idx, joint_name in enumerate(
                    self.trajectory_data.joint_order
                ):
                    if joint_name in self.articulation.joint_names:
                        isaac_joint_idx = self.articulation.joint_names.index(joint_name)
                        joint_pos_ref[0, isaac_joint_idx] = self.trajectory_data.joint_positions[frame_idx, joint_idx]

        # Store reference positions for debugging
        self.joint_pos_ref = joint_pos_ref.clone()

        # Apply MSD deformation if configured
        if self.msd_system and monitored_bodies and decimation > 0:
            for _ in range(decimation):
                # Map MSD deformation to full vector
                q_def_full = torch.zeros(
                    self.articulation.num_joints,
                    device=joint_pos_ref.device,
                    dtype=joint_pos_ref.dtype,
                )
                q_def_full[self.msd_system.active_idx] = self.msd_system.state["q_def"][0].to(
                    device=joint_pos_ref.device, dtype=joint_pos_ref.dtype
                )

                # Apply deformation to reference position
                joint_pos_current = joint_pos_ref.clone()
                joint_pos_current[0, :] += q_def_full

                # Write positions and step physics to update Jacobians
                # Note: Unlike MuJoCo's mj_forward(), Isaac requires sim.step() to update physics state
                self.articulation.write_joint_state_to_sim(
                    joint_pos_current, torch.zeros_like(joint_vel_ref)
                )
                self.sim.step()
                self.articulation.update(self.dt)

                # Calculate external torques from forces on monitored bodies
                external_torques = calculate_external_torques(self.articulation, monitored_bodies, self.joint_mask)

                # Update MSD state (pass full batched tensor)
                self.msd_system.update_msd_state_discrete(external_torques)

            # After MSD loop, re-apply final position to correct drift from last sim.step()
            # This matches MuJoCo behavior where final position is set after the loop
            q_def_full = torch.zeros(
                self.articulation.num_joints,
                device=joint_pos_ref.device,
                dtype=joint_pos_ref.dtype,
            )
            q_def_full[self.msd_system.active_idx] = self.msd_system.state["q_def"][0].to(
                device=joint_pos_ref.device, dtype=joint_pos_ref.dtype
            )
            joint_pos_final = joint_pos_ref.clone()
            joint_pos_final[0, :] += q_def_full
            self.articulation.write_joint_state_to_sim(joint_pos_final, joint_vel_ref)
            self.articulation.update(self.dt)
        else:
            # No MSD - just apply reference and step once
            self.articulation.write_joint_state_to_sim(joint_pos_ref, joint_vel_ref)
            self.sim.step()
            self.articulation.update(self.dt)
