import json
import h5py
import omnigibson as og
import torch as th
th.set_printoptions(precision=3, sci_mode=False)
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from omnigibson.macros import create_module_macros
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
import omnigibson.lazy as lazy
from omnigibson import object_states
from omnigibson.objects.dataset_object import DatasetObject

import robomimic.utils.file_utils as FileUtils
from omnigibson.macros import gm

import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.env_interfaces.base import make_interface
from omnigibson.envs import DataPlaybackWrapper
import random

import argparse

gm.DATASET_PATH = "/cvgl2/u/chengshu/OmniGibson/omnigibson/data/og_dataset"
gm.ENABLE_TRANSITION_RULES = False
gm.RENDER_VIEWER_CAMERA = False

seed = 0
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)

def main(args):
    config_hdf5_path = args.config_hdf5_path
    data_hdf5_path = args.data_hdf5_path

    f_src = h5py.File(config_hdf5_path, "r")
    f_dst = h5py.File(data_hdf5_path, "a")
    f_dst["data"].attrs["config"] = f_src["data"].attrs["config"]
    f_dst["data"].attrs["scene_file"] = f_src["data"].attrs["scene_file"]
    n_episodes = 0
    while True:
        if f"demo_{n_episodes}" not in f_dst["data"]:
            break
        n_episodes += 1
    f_dst["data"].attrs["n_episodes"] = n_episodes
    f_src.close()
    f_dst.close()

    robot_obs_modalities = "seg_instance"
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": ["seg_instance"],
            "sensor_kwargs": {
                "image_height": 256,
                "image_width": 256,
            },
        },
    }

    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=data_hdf5_path,
        output_path=None,
        robot_obs_modalities=robot_obs_modalities,
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=None,
        n_render_iterations=1,
        only_successes=False,
        replay_state=True,
        append_to_input_path=True,
        load_room_instances=["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"],
    )
    env.playback_dataset_datagen()
    env.input_hdf5.close()
    og.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_hdf5_path",
        type=str,
        help="config hdf5 path",
        default=None,
    )
    parser.add_argument(
        "--data_hdf5_path",
        type=str,
        help="data hdf5 path",
        default=None,
    )
    args = parser.parse_args()
    main(args)
