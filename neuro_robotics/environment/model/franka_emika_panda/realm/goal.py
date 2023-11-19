import numpy as np
import pybullet as p

from neuro_robotics.environment.abstract import GoalEntity
from neuro_robotics.environment.model.configuration import InjectEnvMetadata
from neuro_robotics.utils.common import constants


class GoalConfiguration(
    InjectEnvMetadata,
    metadata=constants.InjectMetadataDescription.PANDA_GOAL_METADATA.value,
):
    """InjectEnvMetadata will implant the dataclass configuration based on metadata key-word attr
    the metadata is assigned to self._metadata"""

    pass


class Goal(GoalEntity):

    x_range_min = 0.4
    x_range_max = 0.8
    y_range_min = -0.1
    y_range_max = 0.1
    z_range_min = 0.0
    z_range_max = 0.3
    object_size = 0.04  # TODO set proper object size

    def __init__(self, client):
        self._implant_metadata()
        self.goal_client = client
        self.model = self._load_model(self.goal_client)

        self.goal_range_low = np.array(
            [self.x_range_min, self.y_range_min, self.z_range_min], dtype=np.float32
        )

        self.goal_range_high = np.array(
            [self.x_range_max, self.y_range_max, self.z_range_max], dtype=np.float32
        )

        self.target_range_low = np.array(
            [self.x_range_min, self.y_range_min, self.z_range_min], dtype=np.float32
        )

        self.target_range_high = np.array(
            [self.x_range_max, self.y_range_max, self.z_range_min], dtype=np.float32
        )

        super().__init__(self.goal_client, self.model)

    def _implant_metadata(self):
        goal_metadata = GoalConfiguration()._metadata
        self.init_position = goal_metadata.init_position
        self.description_file = goal_metadata.description_file_location

    def _load_model(self, client):
        model = client.loadURDF(
            fileName=self.description_file, basePosition=self.init_position
        )
        return model

    def _update_desired_goal(self, position):
        self.goal_position = position

    def reset_model(self, sample=True):
        if not sample:
            self._set_base_pose(
                position=self.init_position, orientation=np.array([0, 0, 1, 1])
            )
        else:
            sampled_position = self.sample_target()
            self._set_base_pose(
                position=sampled_position, orientation=np.array([0, 0, 1, 1])
            )

        self.goal_position = self.sample_goal()
        self.target_position = self._get_base_position()

    def get_observation(self):
        observation = self._get_base_position()
        return observation.astype(np.float32)

    def get_desired_goal(self):
        return self.goal_position

    def sample_goal(self):
        desired_goal = np.array([0.0, 0.0, self.object_size / 2], dtype=np.float32)
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            """let the bodies hit the floor"""
            noise[2] = 0.0
        desired_goal += noise
        self._update_desired_goal(desired_goal)
        return desired_goal

    def sample_target(self):
        target_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.target_range_low, self.target_range_high)
        target_position += noise
        return target_position

    # TODO: remove that shit
    def set_random_seed(self, rnd_seed):
        self.np_random = rnd_seed
