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


def add_instance_segmentation_image(action):
    print("in callback")
    pass

seed = 0
np.random.seed(seed)
th.manual_seed(seed)

# Load the scene from the hdf5 file
dataset_path = "/home/arpit/test_projects/OmniGibson/teleop_collected_data/r1_tidy_table.hdf5"
hdf5_path = "/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/r1_tidy_table_momagen/demo_src_r1_tidy_table_task_D0/tmp/date_05_04_2025_time_23_35_55.hdf5"
new_hdf5_path = "/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/r1_tidy_table_momagen/demo_src_r1_tidy_table_task_D0/tmp/date_05_04_2025_time_23_35_55_new.hdf5"
env_interface_type = "omnigibson_bimanual"
env_interface_name = "MG_R1TidyTable"
json_path = ""

FileUtils.preprocess_omnigibson_dataset(dataset_path)

# create environment that was to collect source demonstrations
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

print("==== Using environment with the following metadata ====")
print(env_meta)

gm.ENABLE_TRANSITION_RULES = False
# Option 1: Might not be able to use this as input_path should get scene_file info from dataset_path but episodes from hdf5_path
env = DataPlaybackWrapper.create_from_hdf5(
    input_path=dataset_path,
    output_path=None,
    robot_obs_modalities=(),
    robot_sensor_config=None,
    external_sensors_config=None,
    n_render_iterations=1,
    only_successes=False,
    replay_state=True,
    load_room_instances=["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"],
)

env_interface = make_interface(
    name=env_interface_name,
    interface_type=env_interface_type,
    # NOTE: env_interface takes underlying simulation environment, not robomimic wrapper
    env=env,
)

demos = MG_FileUtils.get_all_demos_from_dataset(
    dataset_path=hdf5_path,
    start=None,
)
breakpoint()

all_datagen_info = env.playback_dataset(record_data=False, callback=add_instance_segmentation_image)