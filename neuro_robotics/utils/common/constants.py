import os
from enum import Enum
from pathlib import Path


RLROOT = os.getenv("RLROOT", ".")
CORE_DIR = Path(f"{RLROOT}/neuro_robotics")
DATA_DIR = Path(f"{RLROOT}/neuro_robotics/data")
ALGORITHM_DIR = Path(f"{RLROOT}/neuro_robotics/algorithm")
SETTINGS_DIR = Path(f"{RLROOT}/neuro_robotics/settings")
ENVIRONMENT_DIR = CORE_DIR / "environment"
DATA_SAVE_DIRECTORY_PATH = DATA_DIR / "inference"
PRETRAINED_MODEL_PATH = DATA_DIR / "pretrained" / "best_model"


class InjectMetadataDescription(Enum):
    METADATA_ATTR_KEY = "_metadata"

    SAC_AGENT_METADATA = "soft_actor_critic_agent"
    SAC_NETWORK_METADATA = "soft_actor_critic_network"

    DDPG_AGENT_METADATA = "ddpg_agent"
    DDPG_NETWORK_METADATA = "ddpg_network"

    PANDA_ROBOT_METADATA = "panda_robot"
    PANDA_GOAL_METADATA = "panda_goal"
    PANDA_PLANE_METADATA = "panda_plane"
    PANDA_TRAY_METADATA = "panda_tray"
    PANDA_TABLE_METADATA = "panda_table"


class _FrankaEmikaPandaConfigurationKey(Enum):
    DESC_FILE_LOC_ATTR = "description_file_location"

    INIT_POSITION_ATTR = "init_position"
    _X_ATTR = "x"
    _Z_ATTR = "z"
    _Y_ATTR = "y"

    CONTROL_JOINTS_ATTR = "control_joints"
    _ID = "id"
    _NEUTRAL_POSITION = "neutral_position"

    EFFECTOR_ATTR = "effector"
    _JOINT_ID = "joint_id"
    _LINK_ID = "link_id"
    _SOFT_DISPLACEMENT_LIMIT = "soft_displacement_limit"
    _LATERAL_FRICTION = "lateral_friction"
    _SPINNING_FRICTION = "spinning_friction"

    INVERSE_KINEMATICS_ATTR = "inverse_kinematics"
    _DISPLACEMENT_LIMIT = "displacement_limit"


class FrankaEmikaPanda(Enum):
    CFG_KEY = _FrankaEmikaPandaConfigurationKey

    METADATA_DIR = (
        ENVIRONMENT_DIR / "model" / "configuration" / "franka_emika_panda" / "metadata"
    )
    ROBOT_YAML_CONFIG_PATH = METADATA_DIR / "robot.yml"
    GOAL_YAML_CONFIG_PATH = METADATA_DIR / "goal.yml"
    PLANE_YAML_CONFIG_PATH = METADATA_DIR / "plane.yml"
    TRAY_YAML_CONFIG_PATH = METADATA_DIR / "traybox.yml"
    TABLE_YAML_CONFIG_PATH = METADATA_DIR / "table.yml"

    DEFAULT_CAMERA_POSITION = (0.5, 0.4, -0.8)
    DEFAULT_CAMERA_PITCH = -50
    DEFAULT_CAMERA_YAW = 0
    DEFAULT_CAMERA_DISTANCE = 2
