import abc

import numpy as np
import pybullet as p

from neuro_robotics.utils.common import constants


class GoalEntity(abc.ABC):
    """force subclass to implement abstract method"""

    def __init__(self, client, model):
        self.sim_client = client
        self.model = model

    @abc.abstractmethod
    def _load_model(self):
        pass

    def _get_base_position(self) -> np.ndarray:
        """Get the position of the body.
        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.sim_client.getBasePositionAndOrientation(self.model)[0]
        return np.array(position)

    def _set_base_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """Set the position of the body.
        Args:
            position (np.ndarray): The position, as (x, y, z).
            orientation (np.ndarray): The target orientation as quaternion (x, y, z, w).
        """
        if len(orientation) == 3:
            orientation = self.physics_client.getQuaternionFromEuler(orientation)
        self.sim_client.resetBasePositionAndOrientation(
            bodyUniqueId=self.model, posObj=position, ornObj=orientation
        )
