from typing import Dict

import numpy as np
import pybullet as p

from .realm import Goal
from .realm import Plane
from .realm import Robot
from .realm import Table
from neuro_robotics.utils.common import constants
from neuro_robotics.utils.common import methods


class PandaEnv:

    distance_threshold = 0.05

    def __init__(self, client: int) -> None:
        self.panda_client = client

    def _set_camera(self, default=True):
        if default:
            default_position = (
                constants.FrankaEmikaPanda.DEFAULT_CAMERA_POSITION.value,
            )
            default_distance = (
                constants.FrankaEmikaPanda.DEFAULT_CAMERA_DISTANCE.value,
            )
            default_yaw = (constants.FrankaEmikaPanda.DEFAULT_CAMERA_YAW.value,)
            default_pitch = (constants.FrankaEmikaPanda.DEFAULT_CAMERA_PITCH.value,)

            p.resetDebugVisualizerCamera(
                cameraDistance=default_distance[0],
                cameraYaw=default_yaw[0],
                cameraPitch=default_pitch[0],
                cameraTargetPosition=default_position[0],
            )
        else:
            raise NotImplementedError("Method is not yet implemented")

    def set_env(self) -> None:
        self.robot = Robot(self.panda_client)
        self.goal = Goal(self.panda_client)
        self.plane = Plane(self.panda_client)
        self.table = Table(self.panda_client)
        self._set_camera()

    def reset_env(self):
        self.robot.reset_model()
        self.goal.reset_model(sample=False)

    def generate_observation_matrix(self) -> Dict[str, np.ndarray]:
        robot_observation: np.ndarray = self.robot.get_observation()
        achieved_goal: np.ndarray = self.goal.get_observation()
        desired_goal = self.goal.get_desired_goal()
        observation = np.concatenate(
            [robot_observation, achieved_goal], dtype=np.float32
        )

        observation_matrix = {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }
        return observation_matrix

    def calculate_reward(self, achieved_goal, desired_goal, info):
        computed_distance = methods.distance(achieved_goal, desired_goal)
        return -np.array(computed_distance > self.distance_threshold, dtype=np.float32)

    def is_success(self, achieved_goal, desired_goal):
        computed_distance = methods.distance(achieved_goal, desired_goal)
        return np.array(computed_distance < self.distance_threshold, dtype=np.float32)

    def recalculate_done(self, current_step, info):
        success = info["is_success"]
        if success:
            return True
        else:
            return False
