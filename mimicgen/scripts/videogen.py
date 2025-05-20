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
gm.RENDER_VIEWER_CAMERA = True

seed = 0
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)

def update_env_post_creation_r1_clean_pan(env, baseline, dr="D0"):
    # breakpoint()
    # og.sim.viewer_camera.set_position_orientation(pos, orn)

    # Moving the scrub away from the faucet
    if baseline not in ["mimicgen", "skillgen"]: 
        scrub_brush_601 = env.scene.object_registry("name", "scrub_brush_601")
        scrub_brush_601.set_position_orientation(position=th.tensor([6.5, -1.856, 0.905]), orientation=th.tensor([0.796, -0.606, -0.001, -0.007]))
        for _ in range(5): og.sim.step()

    # Set the default orn of pan (around which D0 will sample)
    frying_pan_602 = env.scene.object_registry("name", "frying_pan_602")
    orientation = frying_pan_602.get_position_orientation()[1]
    rot_z = R.from_euler('z', -45, degrees=True)
    original_rot = R.from_quat(orientation)
    new_rot = rot_z * original_rot
    rotated_quat = new_rot.as_quat()
    frying_pan_602.set_position_orientation(orientation=rotated_quat)

    if dr == "D2":  
        distractor_objects = []
        obj = DatasetObject(
            name="instant_pot",
            category="instant_pot",
            model="wengzf",
            scale=th.tensor([0.5, 0.5, 0.5]),
        )
        distractor_objects.append(obj)

        obj = DatasetObject(
            name="can_of_oatmeal",
            category="can_of_oatmeal",
            model="qyukhm",
        )
        distractor_objects.append(obj)

        obj = DatasetObject(
            name="wine_bottle",
            category="wine_bottle",
            model="inkqch",
        )
        distractor_objects.append(obj)

        obj = DatasetObject(
            name="bowl_1",
            category="bowl",
            model="wtepsx",
        )
        distractor_objects.append(obj)

        obj = DatasetObject(
            name="bowl_2",
            category="bowl",
            model="tvtive",
        )
        distractor_objects.append(obj)

        state = og.sim.dump_state()
        og.sim.stop()

        # Load the objects into the scene
        og.sim.batch_add_objects(distractor_objects, [env.scene] * len(distractor_objects))
        og.sim.play()
        og.sim.load_state(state)
        
        # Set object pose to ensure no collision at spawn time
        x_pos = 5.0
        for distractor_object in distractor_objects:
            x_pos += 1.0
            distractor_object.set_position_orientation(position=th.tensor([x_pos,  0.0,  0.0]))
            # Open the laptop
            if distractor_object.name == "laptop":
                distractor_object.joints["j_screen"].set_pos(1.0, normalized=True)
        og.sim.step()

def main(args):
    config_hdf5_path = args.config_hdf5_path
    data_hdf5_path = args.data_hdf5_path
    video_folder_path = args.video_folder_path
    task = args.task
    dr = args.dr

    with h5py.File(config_hdf5_path, "r") as f:
        overwrite_config = f["data"].attrs["config"]
        if task != "clean_pan":
            overwrite_scene_file = f["data"].attrs["scene_file"]
        else:
            with open("/cvgl2/u/chengshu/OmniGibson/omnigibson/data/og_dataset/scenes/house_single_floor/json/house_single_floor_task_datagen_wash_dishes_0_0_template.json") as f:
                overwrite_scene_file = json.dumps(json.load(f))

    robot_obs_modalities = "seg_instance"
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": ["rgb"],
            "sensor_kwargs": {
                "image_width": 640,
                "image_height": 360,
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
        append_to_input_path=False,
        load_room_instances=["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"],
        overwrite_config=overwrite_config,
        overwrite_scene_file=overwrite_scene_file,
    )

    # Enable external sensors
    env.external_sensors["external_sensor2"].add_modality("rgb")
    env.external_sensors["external_sensor2"].image_width = 1920
    env.external_sensors["external_sensor2"].image_height = 1080

    # Place the viewer camera
    og.sim.enable_viewer_camera_teleoperation()

    # Hide objects
    for obj in env.scene.object_registry("category", "room_light"):
        obj.visible = False

    assert dr in ["D0", "D1", "D2"], f"dr {dr} not supported"

    if task == "clean_pan":
        update_env_post_creation_r1_clean_pan(env, "momagen", dr)
    else:
        assert False, f"task {task} not supported"

    env.playback_dataset_videogen(task=task, video_folder_path=video_folder_path)
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
    parser.add_argument(
        "--video_folder_path",
        type=str,
        help="video folder path",
        default=None,
    )
    parser.add_argument(
        "--task",
        type=str,
        help="task",
        default=None,
    )
    parser.add_argument(
        "--dr",
        type=str,
        help="domain randomization",
        default=None,
    )

    args = parser.parse_args()
    main(args)
