from attrs import define


@define
class RobotDepiction:

    description_file_location: str

    init_position: list

    control_joints_id: list
    control_joints_neutral_position: list

    effector_joint_id: list
    effector_link_id: int
    effector_displacement_limit: float
    effector_lateral_friction: float
    effector_spinning_friction: float

    inverse_kinematics_displacement_limit: float
