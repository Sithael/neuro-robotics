import json
from datetime import datetime
from functools import wraps
from os.path import expandvars
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import yaml


def get_current_timestamp(use_hour=True):
    if use_hour:
        return datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        return datetime.now().strftime("%Y%m%d")


def load_yaml(yaml_path):
    def process_dict(dict_to_process):
        for key, item in dict_to_process.items():
            if isinstance(item, dict):
                dict_to_process[key] = process_dict(item)
            elif isinstance(item, str):
                dict_to_process[key] = expandvars(item)
            elif isinstance(item, list):
                dict_to_process[key] = process_list(item)
        return dict_to_process

    def process_list(list_to_process):
        new_list = []
        for item in list_to_process:
            if isinstance(item, dict):
                new_list.append(process_dict(item))
            elif isinstance(item, str):
                new_list.append(expandvars(item))
            elif isinstance(item, list):
                new_list.append(process_list(item))
            else:
                new_list.append(item)
        return new_list

    with open(yaml_path) as yaml_file:
        yaml_content = yaml.safe_load(yaml_file)

    return process_dict(yaml_content)


def load_json(path):
    with open(path) as f:
        content = json.load(f)
    return content


def save_yaml(content, path):
    with open(path, "w") as file:
        yaml.dump(content, file, sort_keys=False)


def create_json_file(output_filename, out_content):
    """create output json files"""
    with open(output_filename, "w") as out_file:
        json.dump(out_content, out_file, indent=1)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = datetime.now()
        result = f(*args, **kw)
        te = datetime.now()
        print(f"func:{f.__qualname__} took: {te - ts}.")
        return result

    return wrap


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 scores")
    plt.savefig(figure_file)


def distance(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """Compute the distance between two array. This function is vectorized.
    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
    Returns:
        Union[float, np.ndarray]: The distance between the arrays.
    """
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)


def angle_distance(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """Compute the geodesic distance between two array of angles. This function is vectorized.
    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
    Returns:
        Union[float, np.ndarray]: The geodesic distance between the angles.
    """
    assert a.shape == b.shape
    dist = 1 - np.inner(a, b) ** 2
    return dist
