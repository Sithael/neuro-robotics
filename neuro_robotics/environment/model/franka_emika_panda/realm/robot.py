import numpy as np
import pybullet as p

from neuro_robotics.environment.abstract import RobotEntity
from neuro_robotics.environment.model.configuration import InjectEnvMetadata
from neuro_robotics.utils.common import constants


class RobotConfiguration(
    InjectEnvMetadata,
    metadata=constants.InjectMetadataDescription.PANDA_ROBOT_METADATA.value,
):
    """InjectEnvMetadata will implant the dataclass configuration based on metadata key-word attr
    the metadata is assigned to self._metadata"""

    pass


class Robot(RobotEntity):
    def __init__(self, client):
        self._implant_metadata()
        self.robot_client = client
        self.model = self._load_model(self.robot_client)
        super().__init__(self.robot_client, self.model)

    def _implant_metadata(self):
        robot_metadata = RobotConfiguration()._metadata
        self.init_position = robot_metadata.init_position
        self.description_file = robot_metadata.description_file_location
        self.control_joints_id = robot_metadata.control_joints_id
        self.control_joints_neutral_position = (
            robot_metadata.control_joints_neutral_position
        )
        self.effector_joint_id = robot_metadata.effector_joint_id
        self.effector_link_id = robot_metadata.effector_link_id
        self.effector_displacement_limit = robot_metadata.effector_displacement_limit
        self.effector_lateral_friction = robot_metadata.effector_lateral_friction
        self.effector_spinning_friction = robot_metadata.effector_spinning_friction
        self.inverse_kinematics_displacement_limit = (
            robot_metadata.inverse_kinematics_displacement_limit
        )

    def _load_model(self, client):
        model = client.loadURDF(
            fileName=self.description_file,
            basePosition=self.init_position,
            useFixedBase=True,
        )
        return model

    def reset_model(self):
        self._set_joint_angles(
            joints=self.control_joints_id, angles=self.control_joints_neutral_position
        )
        self._set_ee_friction(
            self.effector_joint_id,
            self.effector_lateral_friction,
            self.effector_spinning_friction,
        )

    def act(self, action):
        fingers_ctr = (
            np.array(action[-1], dtype=np.float32) * self.effector_displacement_limit
        )
        fingers_width = self._get_fingers_width(self.effector_joint_id)
        target_fingers_width = fingers_width + fingers_ctr

        joint_control_position = self._effector_displacement_to_target_arm_angles(
            action, self.inverse_kinematics_displacement_limit, self.effector_link_id
        )

        robot_control_position = np.concatenate(
            (
                joint_control_position,
                [target_fingers_width / 2.0, target_fingers_width / 2.0],
            ),
            dtype=np.float32,
        )

        robot_control_parts = np.array(self.control_joints_id)
        self.robot_client.setJointMotorControlArray(
            self.model,
            jointIndices=robot_control_parts,
            controlMode=p.POSITION_CONTROL,
            targetPositions=robot_control_position,
        )

    def get_observation(self):
        ee_position = self._get_ee_position(self.effector_link_id)
        ee_velocity = self._get_ee_velocity(self.effector_link_id)
        fingers_width = self._get_fingers_width(self.effector_joint_id)
        observation = np.concatenate(
            (ee_position, ee_velocity, [fingers_width]), dtype=np.float32
        )
        return observation
