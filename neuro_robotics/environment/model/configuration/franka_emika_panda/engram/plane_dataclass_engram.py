from pathlib import Path

import yaml

from neuro_robotics.environment.model.configuration.franka_emika_panda.dataclass import (
    PlaneDepiction,
)
from neuro_robotics.utils.common import constants
from neuro_robotics.utils.common import methods
from neuro_robotics.utils.common.constants import FrankaEmikaPanda as panda


class PlaneDataclassEngram:
    def __init__(self):
        """load and resolve franka emika panda configuration yml file"""
        self.plane_configuration = methods.load_yaml(panda.PLANE_YAML_CONFIG_PATH.value)

        self._init_position_attr = self.plane_configuration[
            panda.CFG_KEY.value.INIT_POSITION_ATTR.value
        ]
        self.dataclass_entity = self._initialize_dataclass_fields()

    def _initialize_dataclass_fields(self):
        plane_dataclass = PlaneDepiction(
            description_file_location=self.description_file_location,
            init_position=self.init_position,
        )
        return plane_dataclass

    @property
    def description_file_location(self):
        location_str = (
            str(constants.CORE_DIR)
            + "/"
            + self.plane_configuration[panda.CFG_KEY.value.DESC_FILE_LOC_ATTR.value]
        )
        return location_str

    @property
    def init_position(self):
        x_pos = self._init_position_attr[panda.CFG_KEY.value._X_ATTR.value]
        y_pos = self._init_position_attr[panda.CFG_KEY.value._Y_ATTR.value]
        z_pos = self._init_position_attr[panda.CFG_KEY.value._Z_ATTR.value]
        return [x_pos, y_pos, z_pos]
