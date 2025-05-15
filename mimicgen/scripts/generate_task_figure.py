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
from omnigibson.utils.python_utils import create_object_from_init_info, h5py_group_to_torch, assert_valid_key


import robomimic.utils.file_utils as FileUtils
from omnigibson.macros import gm

import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.env_interfaces.base import make_interface
from omnigibson.envs import DataPlaybackWrapper
import random

import argparse
from PIL import Image
import os
import shutil
import seaborn as sns


gm.DATASET_PATH = "/cvgl2/u/chengshu/OmniGibson/omnigibson/data/og_dataset"
gm.ENABLE_TRANSITION_RULES = False
gm.RENDER_VIEWER_CAMERA = True


def main():
    config_hdf5_path = "/cvgl2/u/chengshu/mimicgen/datasets/source_og/r1_pick_cup.hdf5"   
    # image_folder = "/scr/chengshu/Downloads/images"
    image_folder = "/vision/u/chengshu/figure_images"

    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=config_hdf5_path,
        output_path=None,
        robot_obs_modalities=(),
        robot_sensor_config=None,
        external_sensors_config=None,
        n_render_iterations=1,
        only_successes=False,
        replay_state=True,
        append_to_input_path=False,
    )

    og.sim.viewer_camera.image_width = 2560
    og.sim.viewer_camera.image_height = 1440
    og.sim.viewer_camera.add_modality("seg_semantic")
    og.sim.enable_viewer_camera_teleoperation()

    # Episode 14 is good
    episode_id = 14
    data_grp = env.input_hdf5["data"]
    assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
    traj_grp = data_grp[f"demo_{episode_id}"]

    # Grab episode data
    traj_grp = h5py_group_to_torch(traj_grp)
    state = traj_grp["state"]
    
    def save_image(image_file, robot_only=False):
        obs, obs_info = og.sim.viewer_camera.get_obs()
        rgb_numpy = obs["rgb"].cpu().numpy()
        if robot_only:
            seg_semantic_numpy = obs["seg_semantic"].cpu().numpy()
            agent_ids = [id for id, category in obs_info["seg_semantic"].items() if category == "agent"]
            assert len(agent_ids) == 1, "There should be only one agent in the scene"
            agent_id = agent_ids[0]
            rgb_numpy[seg_semantic_numpy != agent_id] = 0

        Image.fromarray(rgb_numpy).save(os.path.join(image_folder, image_file))

    def step_and_render():
        og.sim.step()
        for _ in range(10):
            og.sim.render()

    def save_timestep(timestep, image_file, robot_only=False):
        state_t = state[timestep]
        dicts, total_state_size = og.sim.deserialize(state_t)
        for obj_name in dicts[0]["object_registry"].keys():
            if env.scene.object_registry("name", obj_name).kinematic_only:
                del dicts[0]["object_registry"][obj_name]
        og.sim.load_state(dicts)
        step_and_render()
        save_image(image_file, robot_only=robot_only)

    # transformed eef
    viewer_camera_pos = [1.336, 0.872, 1.899]
    viewer_camera_orn = [0.250, 0.462, 0.748, 0.406]
    og.sim.viewer_camera.set_position_orientation(viewer_camera_pos, viewer_camera_orn)

    save_timestep(250, "pick_cup_1.png", robot_only=True)
    save_timestep(800, "pick_cup_2.png", robot_only=True)
    save_timestep(900, "pick_cup_3.png", robot_only=False)

    env.input_hdf5.close()
    og.shutdown()


if __name__ == "__main__":
    main()
