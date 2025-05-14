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
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.objects.dataset_object import DatasetObject


seed = 0
np.random.seed(seed)
th.manual_seed(seed)

# Load the scene from the hdf5 file
f = h5py.File("/home/arpit/test_projects/OmniGibson/teleop_collected_data/r1_tidy_table.hdf5", "r")
config = f["data"].attrs["config"]
config = json.loads(config)

# Custom changes
config["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
config["robots"][0]["position"] = [0.0, 0.0, 0.0]
config["robots"][0]["orientation"] = [0.0, 0.0, 0.0, 1.0]

env = og.Environment(configs=config)

# Read the file
file_name = "/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/r1_tidy_table_momagen/demo_src_r1_tidy_table_task_D2/demo.hdf5"
f = h5py.File(file_name, "r")


# Hide objects
for obj in env.scene.object_registry("category", "ceiling_light"):
    for link in obj.links.values():
        for mesh in link.visual_meshes.values():
            mesh.purpose = "guide"

# Set camera
og.sim.viewer_camera.set_position_orientation(
    position=th.tensor([6.611, -0.367,  4.950]),
    orientation=th.tensor([0.002, -0.011,  1.000,  0.002])
)

# mark with pink color the or render the cup at those poses
num_demos = len(f["data"].keys())
marker_list = []
base_marker_list = []
for i in range(num_demos):
    marker = PrimitiveObject(
        relative_prim_path=f"/marker_{i}",
        primitive_type="Cube",
        name=f"marker_{i}",
        size=th.tensor([0.03, 0.03, 0.03]),
        visual_only=True,
        rgba=th.tensor([1, 0, 0, 1])
    )
    marker_list.append(marker)

    base_marker = PrimitiveObject(
        relative_prim_path=f"/base_marker_{i}",
        primitive_type="Cube",
        name=f"base_marker_{i}",
        size=th.tensor([0.03, 0.03, 0.03]),
        visual_only=True,
        rgba=th.tensor([0, 1, 0, 1])
    )
    base_marker_list.append(base_marker)


og.sim.batch_add_objects(marker_list, [env.scene] * len(marker_list))
og.sim.batch_add_objects(base_marker_list, [env.scene] * len(base_marker_list))

for i in range(num_demos):
    pos = f["data"][f"demo_{i}"]["datagen_info"]["object_poses"]["teacup_601"][0][:3,3]
    marker_list[i].set_position_orientation(position=pos)
    base_pos = f["data"][f"demo_{i}"]["datagen_info"]["base_pose"][-1][:3,3]
    base_marker_list[i].set_position_orientation(position=base_pos)



for _ in range(300): og.sim.step()
breakpoint()
