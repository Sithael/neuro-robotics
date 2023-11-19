from .franka_emika_panda import GoalDataclassEngram
from .franka_emika_panda import PlaneDataclassEngram
from .franka_emika_panda import RobotDataclassEngram
from .franka_emika_panda import TableDataclassEngram
from .franka_emika_panda import TrayDataclassEngram
from neuro_robotics.utils.common import constants
from neuro_robotics.utils.common.constants import InjectMetadataDescription


class InjectMetadataMeta(type):
    """Inject dataclass configuration attribute to child class"""

    @classmethod
    def __prepare__(cls, class_name, bases, metadata):
        return super().__prepare__(class_name, bases, metadata)

    def __new__(cls, class_name, bases, attrdict, metadata):
        if metadata is None:
            agent_dataclass = None
            attrdict[InjectMetadataDescription.METADATA_ATTR_KEY.value] = None

        if metadata == constants.InjectMetadataDescription.PANDA_ROBOT_METADATA.value:
            panda_robot_dataclass = RobotDataclassEngram().dataclass_entity
            attrdict[
                InjectMetadataDescription.METADATA_ATTR_KEY.value
            ] = panda_robot_dataclass
        elif metadata == constants.InjectMetadataDescription.PANDA_GOAL_METADATA.value:
            panda_goal_dataclass = GoalDataclassEngram().dataclass_entity
            attrdict[
                InjectMetadataDescription.METADATA_ATTR_KEY.value
            ] = panda_goal_dataclass
        elif metadata == constants.InjectMetadataDescription.PANDA_PLANE_METADATA.value:
            panda_plane_dataclass = PlaneDataclassEngram().dataclass_entity
            attrdict[
                InjectMetadataDescription.METADATA_ATTR_KEY.value
            ] = panda_plane_dataclass
        elif metadata == constants.InjectMetadataDescription.PANDA_TRAY_METADATA.value:
            panda_tray_dataclass = TrayDataclassEngram().dataclass_entity
            attrdict[
                InjectMetadataDescription.METADATA_ATTR_KEY.value
            ] = panda_tray_dataclass
        elif metadata == constants.InjectMetadataDescription.PANDA_TABLE_METADATA.value:
            panda_table_dataclass = TableDataclassEngram().dataclass_entity
            attrdict[
                InjectMetadataDescription.METADATA_ATTR_KEY.value
            ] = panda_table_dataclass
        return super().__new__(cls, class_name, bases, attrdict)


class InjectEnvMetadata(metaclass=InjectMetadataMeta, metadata=None):
    pass
