import abc

import numpy as np
import pybullet as p

from neuro_robotics.utils.common import constants


class EnvEntity(abc.ABC):
    """force subclass to implement abstract method"""

    def __init__(self, client, model):
        self.sim_client = client
        self.model = model

    @abc.abstractmethod
    def _load_model(self):
        pass
