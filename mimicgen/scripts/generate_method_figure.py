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
    config_hdf5_path = "/cvgl2/u/chengshu/mimicgen/datasets/source_og/r1_tidy_table.hdf5"   
    # data_hdf5_path = "/mnt/chengshu/momagen/tidy_table_full/r1_tidy_table_worker_9/demo_src_r1_tidy_table_task_D1/demo.hdf5"
    data_hdf5_path = "/vision/u/chengshu/momagen/tidy_table_full/r1_tidy_table_worker_9/demo_src_r1_tidy_table_task_D1/demo.hdf5"
    # image_folder = "/mnt/chengshu/figure_images"
    image_folder = "/vision/u/chengshu/figure_images"

    # f_src = h5py.File(config_hdf5_path, "r")
    # f_dst = h5py.File(data_hdf5_path, "r")

    # f_dst["data"].attrs["config"] = f_src["data"].attrs["config"]
    # f_dst["data"].attrs["scene_file"] = f_src["data"].attrs["scene_file"]
    # n_episodes = 0
    # while True:
    #     if f"demo_{n_episodes}" not in f_dst["data"]:
    #         break
    #     n_episodes += 1
    # f_dst["data"].attrs["n_episodes"] = n_episodes
    # f_src.close()
    # f_dst.close()

    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=data_hdf5_path,
        output_path=None,
        robot_obs_modalities=(),
        robot_sensor_config=None,
        external_sensors_config=None,
        n_render_iterations=1,
        only_successes=False,
        replay_state=True,
        append_to_input_path=False,
        load_room_instances=["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"],
    )

    og.sim.viewer_camera.image_width = 1280
    og.sim.viewer_camera.image_height = 960
    og.sim.enable_viewer_camera_teleoperation()

    palette = sns.color_palette("deep")

    robot = env.robots[0]
    # Make sure robot is black
    for material in robot.materials:
        material.diffuse_color_constant = th.tensor([0.0, 0.0, 0.0])

    # Set all the room lights to be invisible to avoid occlusion
    for room_light in env.scene.object_registry("category", "room_light"):
        room_light.visible = False

    # Episode 0 is good
    episode_id = 0
    data_grp = env.input_hdf5["data"]
    assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
    traj_grp = data_grp[f"demo_{episode_id}"]

    # Grab episode data
    traj_grp = h5py_group_to_torch(traj_grp)
    state = traj_grp["states"]
    
    def save_timestep(timestep, image_file):
        state_t = state[timestep]
        og.sim.load_state(state_t, serialized=True)
        og.sim.step_physics()
        for _ in range(20): og.sim.render()
        Image.fromarray(og.sim.viewer_camera.get_obs()[0]["rgb"].cpu().numpy()).save(os.path.join(image_folder, image_file))

    # transformed eef
    viewer_camera_pos = [7.493, 0.250, 1.482]
    viewer_camera_orn = [0.390, 0.134, 0.308, 0.857]
    og.sim.viewer_camera.set_position_orientation(viewer_camera_pos, viewer_camera_orn)

    for link_name, link in env.robots[0].links.items():
        if link_name not in ["left_gripper_link1", "left_gripper_link2", "left_arm_link6"]:
            link.visible = False
    
    transformed_eef_timestep = [530, 550, 600]
    for timestep in transformed_eef_timestep:
        save_timestep(timestep, "transformed_eef_%05d.png" % timestep)

    for link_name, link in env.robots[0].links.items():
        if link_name not in ["left_gripper_link1", "left_gripper_link2", "left_arm_link6"]:
            link.visible = True

    # sample reachability base poses
    viewer_camera_pos = [7.870, -0.139, 2.384]
    viewer_camera_orn = [0.390, 0.134, 0.308, 0.857]
    og.sim.viewer_camera.set_position_orientation(viewer_camera_pos, viewer_camera_orn)

    robot.highlighted = True
    robot.set_highlight_properties(color=list(palette[0]), intensity=1000.0)
    save_timestep(550, "reachability_success.png")
    pos, orn = robot.get_position_orientation()
    yaw = T.quat2euler(orn)[2]
    
    robot.set_highlight_properties(color=list(palette[2]), intensity=1000.0)
    robot.set_position_orientation(pos + th.tensor([0.7, 0.7, 0.0]), T.euler2quat(th.tensor([0.0, 0.0, yaw - np.pi / 4])))
    og.sim.step()
    for _ in range(10): og.sim.render()
    Image.fromarray(og.sim.viewer_camera.get_obs()[0]["rgb"].cpu().numpy()).save(os.path.join(image_folder, "reachability_failure_1.png"))

    robot.set_highlight_properties(color=list(palette[3]), intensity=1000.0)
    robot.set_position_orientation(pos + th.tensor([-0.7, 0.7, 0.0]), T.euler2quat(th.tensor([0.0, 0.0, yaw])))
    og.sim.step()
    for _ in range(10): og.sim.render()
    Image.fromarray(og.sim.viewer_camera.get_obs()[0]["rgb"].cpu().numpy()).save(os.path.join(image_folder, "reachability_failure_2.png"))

    # sample visibility base pose
    robot.highlighted = True
    robot.set_highlight_properties(color=list(palette[0]), intensity=1000.0)
    save_timestep(550, "visibility_success.png")

    robot.set_highlight_properties(color=list(palette[2]), intensity=1000.0)
    robot.joints["torso_joint4"].set_pos(1.0)
    og.sim.step()
    for _ in range(10): og.sim.render()
    Image.fromarray(og.sim.viewer_camera.get_obs()[0]["rgb"].cpu().numpy()).save(os.path.join(image_folder, "visibility_failure_1.png"))

    robot.set_highlight_properties(color=list(palette[3]), intensity=1000.0)
    robot.joints["torso_joint4"].set_pos(0.0)
    robot.joints["base_footprint_rz_joint"].set_pos(-0.3)
    og.sim.step()
    for _ in range(10): og.sim.render()
    Image.fromarray(og.sim.viewer_camera.get_obs()[0]["rgb"].cpu().numpy()).save(os.path.join(image_folder, "visibility_failure_2.png"))
    robot.highlighted = False

    # base motion
    base_mp_timestep = [100, 150, 200]
    for timestep in base_mp_timestep:
        save_timestep(timestep, "base_mp_%05d.png" % timestep)

    # arm motion
    arm_mp_timestep = [250, 450, 650]
    for timestep in arm_mp_timestep:
        save_timestep(timestep, "arm_mp_%05d.png" % timestep)
    
    # retract motion
    retract_timestep = [650, 825, 1000]
    for timestep in retract_timestep:
        save_timestep(timestep, "retract_%05d.png" % timestep)

    env.input_hdf5.close()
    og.shutdown()


if __name__ == "__main__":
    main()
