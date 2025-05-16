
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

from omnigibson.macros import gm
from PIL import Image
import os
import matplotlib.pyplot as plt

# image_folder = "/scr/chengshu/Downloads/images"
image_folder = "/vision/u/chengshu/figure_images
# data_folder = "/mnt/chengshu"
data_folder = "/vision/u/chengshu

gm.DATASET_PATH = "/cvgl2/u/chengshu/OmniGibson/omnigibson/data/og_dataset"

seed = 0
np.random.seed(seed)
th.manual_seed(seed)

# Load the scene from the hdf5 file
with h5py.File("/cvgl2/u/chengshu/mimicgen/datasets/source_og/r1_tidy_table.hdf5", "r") as f:
    config = f["data"].attrs["config"]

config = json.loads(config)

# Custom changes
config["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
config["robots"][0]["position"] = [0.0, 0.0, 0.0]
config["robots"][0]["orientation"] = [0.0, 0.0, 0.0, 1.0]

env = og.Environment(configs=config)
# -

# Set camera
rot = R.from_euler("xyz", [0, 0, 180], degrees=True).as_quat()
table_pos = env.task.object_scope["countertop.n.01_1"].get_position_orientation()[0]
og.sim.viewer_camera.set_position_orientation(
    position=th.tensor([table_pos[0] - 0.2, table_pos[1] - 0.2, 4]),
    orientation=th.tensor(rot)
)
og.sim.viewer_camera.image_height = 960
og.sim.viewer_camera.image_width = 1280
og.sim.stop()
# og.sim.viewer_camera.horizontal_aperture = 31.0

# +
# Hide objects
for obj in env.scene.object_registry("category", "room_light"):
    obj.visible = False
for obj in env.scene.object_registry("category", "ceilings"):
    obj.visible = False
for obj in env.scene.object_registry("category", "range_hood"):
    obj.visible = False
for obj in env.scene.object_registry("category", "downlight"):
    obj.visible = False
for obj in env.scene.object_registry("category", "taboret"):
    obj.visible = False

obj = env.scene.object_registry("name", "top_cabinet_lkxmne_0")
obj.visible = False
obj = env.scene.object_registry("name", "shelf_pfusrd_0")
obj.visible = False
obj = env.scene.object_registry("name", "teacup_601")
obj.visible = False
for _ in range(10): og.sim.step()
# -

og.sim.play()
for _ in range(10): og.sim.render()
viewer_camera_img = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3]
Image.fromarray(viewer_camera_img.cpu().numpy()).save(os.path.join(image_folder, f"diversity_background.png"))
og.sim.stop()

# +
# For multiple task-relevant objects

import seaborn as sns
palette = sns.color_palette("deep")
palette = [palette[0], palette[2]]
# color = th.cat((th.tensor(palette[0]), th.tensor([1.0])))

# paths = ["/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/r1_dishes_away_ablation_only_soft/demo_src_r1_dishes_away_task_D0/demo.hdf5",
#         "/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/r1_dishes_away_ablation_only_soft/demo_src_r1_dishes_away_task_D0/demo.hdf5"]
# objects = ["plate_601", "plate_602", "plate_603"]

paths = [
    f"{data_folder}/momagen/tidy_table_full_old/r1_tidy_table_worker_0/demo_src_r1_tidy_table_task_D0/demo.hdf5",
    f"{data_folder}/momagen/tidy_table_full/r1_tidy_table_worker_0/demo_src_r1_tidy_table_task_D1/demo.hdf5",
]
objects = ["teacup_601", "base", "left_eef"]

interval = 200
marker_counter = 0

for obj_j, obj in enumerate(objects):
    print(f"Plotting diversity for {obj}")
    marker_list_all = []
    for path_j, path in enumerate(paths):
        with h5py.File(path, "r") as f:
            num_demos = len(f["data"].keys())
            # If D0/D1 wise color
            color = th.cat((th.tensor(palette[path_j]), th.tensor([1.0])))
            marker_list = []
            pos_list = []
            for i in range(num_demos):
                print(f"Demo {i}")
                if i == 40:
                    break
                if obj == "base":
                    poses = f["data"][f"demo_{i}"]["datagen_info"]["base_pose"][::interval, :3, 3]
                elif obj == "left_eef":
                    poses = f["data"][f"demo_{i}"]["datagen_info"]["eef_pose"][::interval, :3, 3]
                elif obj == "right_eef":
                    poses = f["data"][f"demo_{i}"]["datagen_info"]["eef_pose"][::interval, 4:7, 3]
                else:
                    poses = f["data"][f"demo_{i}"]["datagen_info"]["object_poses"][obj][0:1, :3, 3]
                for pos in poses:
                    marker = PrimitiveObject(
                        relative_prim_path=f"/marker_{marker_counter}",
                        name=f"marker_{marker_counter}",
                        primitive_type="Cylinder",
                        # size=th.tensor([0.05, 0.05, 0.05]),
                        radius=0.05,
                        height=0.02,
                        visual_only=True,
                        rgba=color
                    )
                    marker_list.append(marker)
                    pos_list.append(pos)
                    print(marker_counter)
                    marker_counter += 1

            assert len(marker_list) == len(pos_list)
            print(f"Demo {i}: adding {len(marker_list)} markers")
            og.sim.batch_add_objects(marker_list, [env.scene] * len(marker_list))
            for marker, pos in zip(marker_list, pos_list):
                marker.set_position_orientation(position=pos)

            marker_list_all.extend(marker_list)


    og.sim.play()
    for _ in range(10): og.sim.step()
    print(f"Saving image for {obj}: {len(marker_list_all)} markers")
    viewer_camera_img = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3]
    Image.fromarray(viewer_camera_img.cpu().numpy()).save(os.path.join(image_folder, f"diversity_{obj}.png"))

    # Set marker invisible rather than removing
    for marker in marker_list_all:
        marker.visible = False

    # print(f"Removing markers for {obj}: {len(marker_list_all)} markers")
    # og.sim.batch_remove_objects(marker_list_all)
    # for _ in range(10): og.sim.step()

    og.sim.stop()
    for _ in range(10): og.sim.step()
