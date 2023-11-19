from gym.envs.registration import register
from utils.common import methods


date_of_instantiation = methods.get_current_timestamp()

register(
    id="NeuroRobotics-v1",
    entry_point="neuro_robotics.environment:NeuroRoboticsEnv",
    max_episode_steps=50,
)
