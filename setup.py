from setuptools import find_packages
from setuptools import setup

setup(
    name="neuro_robotics",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "gym",
        "pybullet",
        "numpy",
        "matplotlib",
        "stable-baselines3",
        "tensorboard",
        "wandb",
        "black",
        "attrs",
        "click",
    ],
    author="Jakub Chojnacki",
)
