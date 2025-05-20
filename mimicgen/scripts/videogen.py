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
            distractor_object.set_position_orientation(position=th.tensor([x_pos, 0.0, 0.0]))
            # Open the laptop
            if distractor_object.name == "laptop":
                distractor_object.joints["j_screen"].set_pos(1.0, normalized=True)
        og.sim.step()

def update_env_post_creation_r1_pick_cup(env, baseline, dr="D0"):
    floor = env.scene.object_registry("name", "floors_ptwlei_0")

    temp_state = og.sim.dump_state(serialized=False)
    og.sim.stop()
    floor.scale = th.tensor([1.8, 1.0, 1.0])

    og.sim.play()
    og.sim.load_state(temp_state)
    og.sim.step()

    if dr == "D2":
        distractor_objects = []
        obj = DatasetObject(
            name="pot_plant",
            category="pot_plant",
            model="mqhlkf",
            # model="udqjui",
        )
        distractor_objects.append(obj)
        obj = DatasetObject(
            name="straight_chair_0",
            category="straight_chair",
            model="amgwaw",
            # For some reason, this pose does not work!
            position=th.tensor([5.0,  0.0028,  0.4485]),
            orientation=th.tensor([ 0.0016,  0.0020, -0.1448,  0.9895])
        )
        distractor_objects.append(obj)
        obj = DatasetObject(
            name="straight_chair_1",
            category="straight_chair",
            model="amgwaw",
        )
        distractor_objects.append(obj)
        obj = DatasetObject(
            name="gift_box",
            category="gift_box",
            model="mfalrc",
        )
        distractor_objects.append(obj)

        # Load the objects into the scene
        og.sim.batch_add_objects(distractor_objects, [env.scene] * len(distractor_objects))

        # Set object pose to ensure no collision at spawn time
        x_pos = 5.0
        for distractor_object in distractor_objects:
            x_pos += 1.0
            distractor_object.set_position_orientation(position=th.tensor([x_pos,  0.0,  0.0]))
        og.sim.step()


def update_env_post_creation_r1_tidy_table(env, baseline, dr="D0"):
    if dr == "D2":
        distractor_objects = []
        obj = DatasetObject(
            name="vacuum",
            category="vacuum",
            model="bdmsbr",
            scale=th.tensor([1.0, 1.0, 1.5]),
        )
        distractor_objects.append(obj)
        obj = DatasetObject(
            name="trash_can",
            category="trash_can",
            model="vasiit",
        )
        distractor_objects.append(obj)
        obj = DatasetObject(
            name="pot_plant",
            category="pot_plant",
            model="cqqyzp",
            scale=th.tensor([1.3, 1.3, 1.3]),
        )
        distractor_objects.append(obj)
        obj = DatasetObject(
            name="wine_bottle",
            category="wine_bottle",
            model="inkqch",
        )
        distractor_objects.append(obj)
        obj = DatasetObject(
            name="laptop",
            category="laptop",
            model="izydvb",
        )
        distractor_objects.append(obj)
        obj = DatasetObject(
            name="loudspeaker",
            category="loudspeaker",
            model="fsyioq",
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

def update_env_post_creation_r1_dishes_away(env, baseline, dr="D0"):
    shelf = env.scene.object_registry("name", "shelf_pfusrd_1")
    shelf.set_position_orientation(position=th.tensor([ 7.122, -2.029,  1.403]))
    for _ in range(5): og.sim.step()

    if dr == "D2":
        distractor_objects = []
        obj = DatasetObject(
            name="trash_can",
            category="trash_can",
            model="vasiit",
            scale=th.tensor([0.7, 0.7, 1.0]),
        )
        distractor_objects.append(obj)
        obj = DatasetObject(
            name="mop",
            category="mop",
            model="qclfvj",
        )
        distractor_objects.append(obj)
        obj = DatasetObject(
            name="can_of_oatmeal",
            category="can_of_oatmeal",
            model="qyukhm",
        )
        distractor_objects.append(obj)
        obj = DatasetObject(
            name="bowl_1",
            category="bowl",
            model="wtepsx",
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


def update_params_r1_pick_cup(kwargs, baseline):
    pass

def update_params_r1_tidy_table(kwargs, baseline):
    kwargs["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
    kwargs["scene"]["not_load_object_categories"] = ["taboret"]
    # NOTE: in mimicgen/skillgen we are reaplying the exact same base pose. So, we need the init robot pose to be the same as that in source demo
    if baseline not in ["mimicgen", "skillgen"]:
        original_quat = kwargs["robots"][0]["orientation"]
        rot_z_45 = R.from_euler('z', 45, degrees=True)
        original_rot = R.from_quat(original_quat)
        kwargs["robots"][0]["orientation"]
        new_rot = rot_z_45 * original_rot
        rotated_quat = new_rot.as_quat()
        kwargs["robots"][0]["orientation"] = rotated_quat

def update_params_r1_dishes_away(kwargs, baseline):
    kwargs["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
    # For the task of dishes away, we don't load the fridge
    kwargs["scene"]["not_load_object_categories"] = ["fridge"]
    if baseline not in ["mimicgen", "skillgen"]:
        kwargs["robots"][0]["position"] = [5.4, 1.7, kwargs["robots"][0]["position"][2]]
        kwargs["robots"][0]["orientation"] = R.from_euler('z', -2.3, degrees=False).as_quat().tolist()

def update_params_r1_clean_pan(kwargs, baseline):
    kwargs["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
    if baseline not in ["mimicgen", "skillgen"]:
        kwargs["robots"][0]["position"] = [5.4, 1.7, kwargs["robots"][0]["position"][2]]
        kwargs["robots"][0]["orientation"] = R.from_euler('z', -2.3, degrees=False).as_quat().tolist()

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

    kwargs = json.loads(overwrite_config)
    if task == "clean_pan":
        update_params_r1_clean_pan(kwargs, "momagen")
    elif task == "dishes_away":
        update_params_r1_dishes_away(kwargs, "momagen")
    elif task == "tidy_table":
        update_params_r1_tidy_table(kwargs, "momagen")
    elif task == "pick_cup":
        update_params_r1_pick_cup(kwargs, "momagen")
    else:
        assert False, f"task {task} not supported"
    overwrite_config = json.dumps(kwargs)

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
    if task != "pick_cup":
        for obj in env.scene.object_registry("category", "room_light"):
            obj.visible = False

    assert dr in ["D0", "D1", "D2"], f"dr {dr} not supported"

    if task == "clean_pan":
        update_env_post_creation_r1_clean_pan(env, "momagen", dr)
    elif task == "dishes_away":
        update_env_post_creation_r1_dishes_away(env, "momagen", dr)
    elif task == "tidy_table":
        update_env_post_creation_r1_tidy_table(env, "momagen", dr)
    elif task == "pick_cup":
        update_env_post_creation_r1_pick_cup(env, "momagen", dr)
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
