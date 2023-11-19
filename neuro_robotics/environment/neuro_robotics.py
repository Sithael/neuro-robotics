import time
from contextlib import contextmanager
from typing import Iterator

import gym
import numpy as np
import pybullet as p
from pybullet_utils import bullet_client as bc

from .model import PandaEnv


class NeuroRoboticsEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):
        self.connection_type = p.DIRECT
        self.physics_client = bc.BulletClient(connection_mode=self.connection_type)

        self._initialize_simulation()
        self.realm = PandaEnv(self.physics_client)
        self.realm.set_env()

        # TODO: Remove that shit
        self.seed()
        self.realm.goal.set_random_seed(self.np_random)

        obs = self.reset()
        action_shape = (4,)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=action_shape)
        observation_dict = self._create_observation_dict(obs)
        self.observation_space = gym.spaces.Dict(observation_dict)

    @property
    def dt(self):
        """Timestep."""
        return self.timestep * self.n_substeps

    def _create_observation_dict(self, obs):
        obs_shape = obs["observation"].shape
        desired_goal_shape = obs["desired_goal"].shape
        achieved_goal_shape = obs["achieved_goal"].shape
        observation_dict = dict(
            observation=gym.spaces.Box(-10.0, 10.0, shape=obs_shape),
            desired_goal=gym.spaces.Box(-10.0, 10.0, shape=desired_goal_shape),
            achieved_goal=gym.spaces.Box(-10.0, 10.0, shape=achieved_goal_shape),
        )
        return observation_dict

    def _initialize_simulation(self):
        self.physics_client.resetSimulation()

        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

        self.steps = 0
        self.n_substeps = 20
        self.timestep = 1.0 / 500
        self.physics_client.setTimeStep(self.timestep)
        self.physics_client.setGravity(0, 0, -9.81)

    def _discard_target_imposition(self, observation):
        desired_goal = observation["desired_goal"]
        achieved_goal = observation["achieved_goal"]
        is_success = self.realm.is_success(achieved_goal, desired_goal)
        imposing_targets: bool = np.allclose(
            achieved_goal, desired_goal, rtol=1e-1, atol=5e-2
        )
        initial_step: bool = True if not self.steps else False
        return initial_step and (imposing_targets or is_success)

    def step(self, action):
        self.steps += 1
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.realm.robot.act(action)

        for _ in range(self.n_substeps):
            self.physics_client.stepSimulation()

        observation = self.realm.generate_observation_matrix()
        achieved_goal = observation["achieved_goal"]
        desired_goal = observation["desired_goal"]

        info = {"is_success": self.realm.is_success(achieved_goal, desired_goal)}
        reward = self.realm.calculate_reward(achieved_goal, desired_goal, info)
        done = self.realm.recalculate_done(self.steps, info)
        return observation, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.steps = 0
        try:
            with self.no_rendering():
                self.realm.reset_env()
        except Exception:
            raise SystemError("Could not initialize simulator environment")
        observation = self.realm.generate_observation_matrix()
        while self._discard_target_imposition(observation):
            desired_goal: np.ndarray = self.realm.goal.sample_goal()
            observation["desired_goal"] = desired_goal
        return observation

    def close(self):
        self.physics_client.disconnect()

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        """Disable rendering within this context."""
        self.physics_client.configureDebugVisualizer(
            self.physics_client.COV_ENABLE_RENDERING, 0
        )
        yield
        self.physics_client.configureDebugVisualizer(
            self.physics_client.COV_ENABLE_RENDERING, 1
        )

    @property
    def compute_reward(self):
        return self.realm.calculate_reward

    # TODO: reimplement
    def render(self, mode="human"):
        self.physics_client.configureDebugVisualizer(
            self.physics_client.COV_ENABLE_SINGLE_STEP_RENDERING
        )
        time.sleep(self.dt)  # wait to seems like real speed

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.5, 0.4, -0.8],
            distance=3,
            yaw=0,
            pitch=-50,
            roll=1,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(960) / 720, nearVal=0.1, farVal=100.0
        )
        (_, _, px, _, _) = p.getCameraImage(
            width=960,
            height=720,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def __repr__(self):
        return "NeuroRobotics environment has been instantiated"
