from neuro_robotics.environment.model.configuration.franka_emika_panda.dataclass import (
    RobotDepiction,
)
from neuro_robotics.utils.common import constants
from neuro_robotics.utils.common import methods
from neuro_robotics.utils.common.constants import FrankaEmikaPanda as panda


class RobotDataclassEngram:
    def __init__(self):
        """load and resolve franka emika panda configuration yml file"""
        self.robot_configuration = methods.load_yaml(panda.ROBOT_YAML_CONFIG_PATH.value)

        self._init_position_attr = self.robot_configuration[
            panda.CFG_KEY.value.INIT_POSITION_ATTR.value
        ]
        self._control_joints_attr = self.robot_configuration[
            panda.CFG_KEY.value.CONTROL_JOINTS_ATTR.value
        ]
        self._effector_attr = self.robot_configuration[
            panda.CFG_KEY.value.EFFECTOR_ATTR.value
        ]
        self._inverse_kinematics_attr = self.robot_configuration[
            panda.CFG_KEY.value.INVERSE_KINEMATICS_ATTR.value
        ]

        self.dataclass_entity = self._initialize_dataclass_fields()

    def _initialize_dataclass_fields(self):
        robot_dataclass = RobotDepiction(
            description_file_location=self.description_file_location,
            init_position=self.init_position,
            control_joints_id=self.control_joints_id,
            control_joints_neutral_position=self.control_joints_neutral_position,
            effector_joint_id=self.effector_joint_id,
            effector_link_id=self.effector_link_id,
            effector_displacement_limit=self.effector_displacement_limit,
            effector_lateral_friction=self.effector_lateral_friction,
            effector_spinning_friction=self.effector_spinning_friction,
            inverse_kinematics_displacement_limit=self.inverse_kinematics_displacement_limit,
        )
        return robot_dataclass

    @property
    def description_file_location(self):
        location_str = (
            str(constants.CORE_DIR)
            + "/"
            + self.robot_configuration[panda.CFG_KEY.value.DESC_FILE_LOC_ATTR.value]
        )
        return location_str

    @property
    def init_position(self):
        x_pos = self._init_position_attr[panda.CFG_KEY.value._X_ATTR.value]
        y_pos = self._init_position_attr[panda.CFG_KEY.value._Y_ATTR.value]
        z_pos = self._init_position_attr[panda.CFG_KEY.value._Z_ATTR.value]
        return [x_pos, y_pos, z_pos]

    @property
    def control_joints_id(self):
        return self._control_joints_attr[panda.CFG_KEY.value._ID.value]

    @property
    def control_joints_neutral_position(self):
        return self._control_joints_attr[panda.CFG_KEY.value._NEUTRAL_POSITION.value]

    @property
    def effector_joint_id(self):
        return self._effector_attr[panda.CFG_KEY.value._JOINT_ID.value]

    @property
    def effector_link_id(self):
        return self._effector_attr[panda.CFG_KEY.value._LINK_ID.value]

    @property
    def effector_displacement_limit(self):
        return self._effector_attr[panda.CFG_KEY.value._SOFT_DISPLACEMENT_LIMIT.value]

    @property
    def effector_lateral_friction(self):
        return self._effector_attr[panda.CFG_KEY.value._LATERAL_FRICTION.value]

    @property
    def effector_spinning_friction(self):
        return self._effector_attr[panda.CFG_KEY.value._SPINNING_FRICTION.value]

    @property
    def inverse_kinematics_displacement_limit(self):
        return self._inverse_kinematics_attr[
            panda.CFG_KEY.value._DISPLACEMENT_LIMIT.value
        ]
