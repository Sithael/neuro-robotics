import pybullet as p

from neuro_robotics.environment.abstract import EnvEntity
from neuro_robotics.environment.model.configuration import InjectEnvMetadata
from neuro_robotics.utils.common import constants


class TrayConfiguration(
    InjectEnvMetadata,
    metadata=constants.InjectMetadataDescription.PANDA_TRAY_METADATA.value,
):
    """InjectEnvMetadata will implant the dataclass configuration based on metadata key-word attr
    the metadata is assigned to self._metadata"""

    pass


class Tray(EnvEntity):
    def __init__(self, client):
        self._implant_metadata()
        self.tray_client = client
        self.model = self._load_model(self.tray_client)

    def _implant_metadata(self):
        tray_metadata = TrayConfiguration()._metadata
        self.init_position = tray_metadata.init_position
        self.description_file = tray_metadata.description_file_location

    def _load_model(self, client):
        model = client.loadURDF(
            fileName=self.description_file, basePosition=self.init_position
        )
        return model
