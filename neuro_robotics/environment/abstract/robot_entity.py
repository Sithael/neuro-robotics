import abc

import numpy as np


class RobotEntity(abc.ABC):
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

    def _get_link_position(self, link: int) -> np.ndarray:
        """Get the position of the link of the body.
        Args:
            link (int): Link index in the body.
        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.sim_client.getLinkState(self.model, link)[0]
        return np.array(position)

    def _get_link_velocity(self, link: int) -> np.ndarray:
        """Get the velocity of the link of the body.
        Args:
            link (int): Link index in the body.
        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.sim_client.getLinkState(
            self.model, link, computeLinkVelocity=True
        )[6]
        return np.array(velocity, dtype=np.float32)

    def _set_joint_angle(self, joint: int, angle: float) -> None:
        """Set the angle of the joint of the body.
        Args:
            joint (int): Joint index in the body.
            angle (float): Target angle.
        """
        self.sim_client.resetJointState(
            bodyUniqueId=self.model, jointIndex=joint, targetValue=angle
        )

    def _get_joint_angle(self, joint: int) -> float:
        """Get the angle of the joint of the body.
        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body
        Returns:
            float: The angle.
        """
        return self.sim_client.getJointState(self.model, joint)[0]

    def _set_joint_angles(self, joints: np.ndarray, angles: np.ndarray) -> None:
        """Set the angles of the joints of the body.
        Args:
            joints (np.ndarray): List of joint indices, as a list of ints.
            angles (np.ndarray): List of target angles, as a list of floats.
        """
        for joint, angle in zip(joints, angles):
            self._set_joint_angle(joint=joint, angle=angle)

    def _get_ee_position(self, ee_link_id: int) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self._get_link_position(ee_link_id)

    def _get_ee_velocity(self, ee_link_id) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self._get_link_velocity(ee_link_id)

    def _inverse_kinematics(
        self, link: int, position: np.ndarray, orientation: np.ndarray
    ) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint state.
        Args:
            link (int): Link index in the body.
            position (np.ndarray): Desired position of the end-effector, as (x, y, z).
            orientation (np.ndarray): Desired orientation of the end-effector as quaternion (x, y, z, w).
        Returns:
            np.ndarray: The new joint state.
        """
        joint_state = self.sim_client.calculateInverseKinematics(
            bodyIndex=self.model,
            endEffectorLinkIndex=link,
            targetPosition=position,
            targetOrientation=orientation,
        )
        return np.array(joint_state, dtype=np.float32)

    def _effector_displacement_to_target_arm_angles(
        self, ee_displacement: np.ndarray, ee_limit: float, ee_link_id: int
    ) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.
        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).
            ee_limit (float): Inverse kinematics limit.
            ee_link_id (int): Index of End-effector link.
        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        # limit maximum change in position
        ee_displacement = ee_displacement[:3] * ee_limit
        # get the current position and the target position
        ee_position = self._get_ee_position(ee_link_id)
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self._inverse_kinematics(
            link=ee_link_id,
            position=target_ee_position,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def _get_fingers_width(self, fingers_id: list) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.
        Args:
            fingers_id (list): Indexes of End-effector fingers.
        Returns:
            np.ndarray: Target finger width.
        """
        finger1 = self._get_joint_angle(fingers_id[0])
        finger2 = self._get_joint_angle(fingers_id[1])
        return np.array(finger1 + finger2, dtype=np.float32)

    def _set_lateral_friction(self, link: int, lateral_friction: float) -> None:
        """Set the lateral friction of a link.
        Args:
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        self.sim_client.changeDynamics(
            bodyUniqueId=self.model,
            linkIndex=link,
            lateralFriction=lateral_friction,
        )

    def _set_spinning_friction(self, link: int, spinning_friction: float) -> None:
        """Set the spinning friction of a link.
        Args:
            link (int): Link index in the body.
            spinning_friction (float): Spinning friction.
        """
        self.sim_client.changeDynamics(
            bodyUniqueId=self.model,
            linkIndex=link,
            spinningFriction=spinning_friction,
        )

    def _set_ee_friction(self, fingers_indices, lateral_friction, spinning_friction):
        """Set lateral and spinning friction for end effector"""
        self._set_lateral_friction(
            fingers_indices[0], lateral_friction=lateral_friction
        )
        self._set_lateral_friction(
            fingers_indices[1], lateral_friction=lateral_friction
        )
        self._set_spinning_friction(
            fingers_indices[0], spinning_friction=spinning_friction
        )
        self._set_spinning_friction(
            fingers_indices[1], spinning_friction=spinning_friction
        )
