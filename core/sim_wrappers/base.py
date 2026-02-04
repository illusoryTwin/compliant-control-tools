import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional


class SimulationWrapper(ABC):
    """Abstract base class for simulator wrappers."""

    def __init__(self, dt: float):
        self.dt = dt
        self.trajectory_loader: Optional[Any] = None
        self.trajectory_data: Optional[Any] = None
        self.dof_mask: Optional[np.ndarray] = None  # active dofs
        self.msd_system: Optional[Any] = None  # mass-spring-damper model

    @abstractmethod
    def _setup_msd_system(self, config: dict) -> None:
        """Initialize MSD system from configuration.

        Args:
            config: Configuration dictionary containing MSD parameters
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset simulation state."""
        pass

    @abstractmethod
    def get_jacobians(self, body_name: str) -> np.ndarray:
        """Get geometric Jacobian for monitored bodies in their (body) frames."""
        pass

    @abstractmethod
    def get_wrench(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get external wrench (force and torque) applied to monitored bodies."""
        pass

    @abstractmethod
    def apply_external_forces(
        self, body_names: list[str], forces: np.ndarray, torques: np.ndarray
    ) -> None:
        """Apply external forces and torques to specified bodies."""
        pass

    @abstractmethod
    def calculate_external_torques(self, body_names: list[str]) -> np.ndarray:
        """Calculate joint torques emerging from applying external forces on bodies."""
        pass

    @abstractmethod
    def update_state(
        self,
        frame_idx: Optional[int] = None,
        monitored_bodies: Optional[list[str]] = None,
        decimation: int = 1,
    ) -> None:
        pass

    def get_msd_state(self) -> Optional[dict[str, np.ndarray]]:
        if self.msd_system is None:
            return None
        return {
            "q_def": self.msd_system.state["q_def"].copy(),
            "qd_def": self.msd_system.state["qd_def"].copy(),
        }
