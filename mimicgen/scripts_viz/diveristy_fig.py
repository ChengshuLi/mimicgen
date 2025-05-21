# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# import h5py
# import numpy as np

# # Getting the data
# path = "/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/r1_dishes_away_ablation_only_soft/demo_src_r1_dishes_away_task_D0/demo.hdf5"
# f = h5py.File(path, "r")
# x, y = [], []
# for demo in f["data"].keys():
#     x.append(np.array(f[f"data/{demo}/datagen_info/object_poses/plate_601"])[0, 0, 3])
#     y.append(np.array(f[f"data/{demo}/datagen_info/object_poses/plate_601"])[0, 1, 3])
# x = np.array(x)
# y = np.array(y)

# +
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# fig_width = 1280 / 100  # = 12.8 inches
# fig_height = 720 / 100  # = 7.2 inches

# # # Example: replace with your actual data
# # x = np.random.normal(6.7, 0.8, 100)
# # y = np.random.normal(-0.5, 1.0, 100)

# # Create the plot
# plt.figure(figsize=(12.8, 7.2), dpi=100)
# sns.kdeplot(x=x, y=y, fill=True, cmap="Blues", bw_adjust=0.3, levels=50, thresh=0.05)
# # plt.scatter(x, y, c='orange', s=20)

# # Set plot limits and labels
# plt.xlim(6.7 - 3.5, 6.7 + 3.5)
# plt.ylim(-0.5 - 2.25, -0.5 + 2.25)
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.tight_layout()
# plt.show()

# +
# og.shutdown()

# +
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
image_folder = "/vision/u/chengshu/figure_images"
# data_folder_prefix = "/mnt"
data_folder_prefix = "/vision/u"

task = "pick_cup"

gm.DATASET_PATH = "/cvgl2/u/chengshu/OmniGibson/omnigibson/data/og_dataset"

seed = 0
np.random.seed(seed)
th.manual_seed(seed)

# Load the scene from the hdf5 file
with h5py.File(f"/cvgl2/u/chengshu/mimicgen/datasets/source_og/r1_{task}.hdf5", "r") as f:
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
if task != "pick_cup":
    table_pos = env.scene.object_registry("name", "bar_udatjt_0").get_position_orientation()[0]
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([table_pos[0] - 0.2, table_pos[1] - 0.2, 5.0]),
        orientation=th.tensor(rot)
    )
else:
    table_pos = env.scene.object_registry("name", "breakfast_table_6").get_position_orientation()[0]
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([table_pos[0] - 0.5, table_pos[1], 2.6]),
        orientation=th.tensor(rot)
    )
og.sim.viewer_camera.image_height = 960
og.sim.viewer_camera.image_width = 1280
og.sim.viewer_camera.horizontal_aperture = 31.0
og.sim.stop()

# +
# Hide objects
if task != "pick_cup":
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
else:
    obj = env.scene.object_registry("name", "floors_ptwlei_0")
    obj.scale = th.tensor([1.8, 1.0, 1.0])

robot = env.scene.robots[0]
robot.visible = False

if task == "pick_cup":
    obj = env.scene.object_registry("name", "coffee_cup_7")
    obj.visible = False
elif task == "tidy_table":
    obj = env.scene.object_registry("name", "teacup_601")
    obj.visible = False
elif task == "dishes_away":
    for obj in env.scene.object_registry("category", "plate"):
        obj.visible = False
elif task == "clean_pan":
    for obj in env.scene.object_registry("category", "frying_pan"):
        obj.visible = False
    for obj in env.scene.object_registry("category", "scrub_brush"):
        obj.visible = False

for _ in range(10): og.sim.step()

# +
# for _ in range(300): og.sim.render()
# -

og.sim.play()
for _ in range(10): og.sim.render()
viewer_camera_img = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3]
Image.fromarray(viewer_camera_img.cpu().numpy()).save(os.path.join(image_folder, f"diversity_{task}_background.png"))
og.sim.stop()

# +
# For multiple task-relevant objects

import seaborn as sns
palette = sns.color_palette("deep")
# blue, red, green
palette = [np.array(palette[0]), np.array(palette[3]), np.array(palette[2])]
# color = th.cat((th.tensor(palette[0]), th.tensor([1.0])))

# paths = ["/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/r1_dishes_away_ablation_only_soft/demo_src_r1_dishes_away_task_D0/demo.hdf5",
#         "/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/r1_dishes_away_ablation_only_soft/demo_src_r1_dishes_away_task_D0/demo.hdf5"]
# objects = ["plate_601", "plate_602", "plate_603"]

if task == "pick_cup":
    paths = [
        [f"{data_folder_prefix}/chengshu/momagen/{task}_full/r1_{task}_worker_{i}/demo_src_r1_{task}_task_D0/demo.hdf5" for i in range(2)],
        [f"{data_folder_prefix}/chengshu/momagen/{task}_full/r1_{task}_worker_{i}/demo_src_r1_{task}_task_D1/demo.hdf5" for i in range(1)],
        [f"{data_folder_prefix}/mengdixu/skillgen/{task}_full/r1_{task}_worker_{i}/demo_src_r1_{task}_skillgen_task_D0/demo.hdf5" for i in range(1)],
    ]
elif task == "tidy_table":
    paths = [
        [f"{data_folder_prefix}/chengshu/momagen/{task}_full/r1_{task}_worker_{i}/demo_src_r1_{task}_task_D0/demo.hdf5" for i in range(2)],
        [f"{data_folder_prefix}/chengshu/momagen/{task}_full/r1_{task}_worker_{i}/demo_src_r1_{task}_task_D1/demo.hdf5" for i in range(1)],
        [f"{data_folder_prefix}/mengdixu/skillgen/{task}_full/r1_{task}_worker_{i}/demo_src_r1_{task}_skillgen_task_D0/demo.hdf5" for i in range(1)],
    ]
elif task == "clean_pan":
    paths = [
        [f"{data_folder_prefix}/chengshu/momagen/{task}_full/r1_{task}_worker_{i}/demo_src_r1_{task}_task_D0/demo.hdf5" for i in range(1)],
        [f"{data_folder_prefix}/chengshu/momagen/{task}_full_vis/r1_{task}_worker_{i}/demo_src_r1_{task}_task_D1/demo.hdf5" for i in range(10)],
        [f"{data_folder_prefix}/chengshu/skillgen/{task}_full/r1_{task}_worker_{i}/demo_src_r1_{task}_skillgen_task_D0/demo.hdf5" for i in range(10)],
    ]
elif task == "dishes_away":
    paths = [
        [f"{data_folder_prefix}/chengshu/momagen/{task}_full/r1_{task}_worker_{i}/demo_src_r1_{task}_task_D0/demo.hdf5" for i in range(1)],
        [f"{data_folder_prefix}/mengdixu/momagen/{task}_full/r1_{task}_worker_{i}/demo_src_r1_{task}_task_D1/demo.hdf5" for i in range(25)],
        # [f"{data_folder_prefix}/chengshu/momagen/{task}_full_vis/r1_{task}_worker_{i}/demo_src_r1_{task}_task_D1/demo.hdf5" for i in range(1)],
        [f"{data_folder_prefix}/chengshu/skillgen/{task}_full/r1_{task}_worker_{i}/demo_src_r1_{task}_skillgen_task_D0/demo.hdf5" for i in range(10)],
    ]

if task == "pick_cup":
    objects = ["coffee_cup_7"]
elif task == "tidy_table":
    objects = ["teacup_601"]
elif task == "dishes_away":
    objects = ["plate_601"]
elif task == "clean_pan":
    objects = ["frying_pan_602"]

objects.extend(["base", "left_eef"])

# objects = ["base"]
# objects = ["teacup_601", "base", "left_eef"]

interval = 200
marker_counter = 7000

for obj_j, obj in enumerate(objects):
    print(f"Plotting diversity for {obj}")
    marker_list_all = []
    for path_j, path_list in enumerate(paths):
        # If D0/D1 wise color
        color = th.tensor(palette[path_j])
        rgba = th.cat((color, th.tensor([1.0])))
        marker_list = []
        pos_list = []
        num_demos_collected = 0
        for path in path_list:
            print(path)
            with h5py.File(path, "r") as f:
                num_demos = len(f["data"].keys())
                for i in range(num_demos):
                    # print(f"Demo {i}")
                    if num_demos_collected == 50:
                        break
                    num_demos_collected += 1
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
                            rgba=rgba,
                        )
                        marker_list.append(marker)
                        pos_list.append(pos)
                        # print(marker_counter)
                        marker_counter += 1

        assert len(marker_list) == len(pos_list)
        assert num_demos_collected == 50
        print(f"Group {path_j}: Plotting {num_demos_collected} demos")
        # print(f"Demo {i}: adding {len(marker_list)} markers")
        og.sim.batch_add_objects(marker_list, [env.scene] * len(marker_list))
        for marker, pos in zip(marker_list, pos_list):
            marker.set_position_orientation(position=pos)
            marker.root_link.visual_meshes["visuals"].material.diffuse_tint = color
        marker_list_all.extend(marker_list)

    og.sim.play()
    for _ in range(10): og.sim.step()

    print(f"Saving image for {obj}: {len(marker_list_all)} markers")
    viewer_camera_img = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3]
    Image.fromarray(viewer_camera_img.cpu().numpy()).save(os.path.join(image_folder, f"diversity_{task}_{obj}.png"))

    # Set marker invisible rather than removing
    for marker in marker_list_all:
        marker.visible = False

    # print(f"Removing markers for {obj}: {len(marker_list_all)} markers")
    # og.sim.batch_remove_objects(marker_list_all)
    # for _ in range(10): og.sim.step()

    og.sim.stop()
    for _ in range(10): og.sim.step()

# +
# visual_markers = [obj for obj in env.scene.objects if isinstance(obj, PrimitiveObject)]
# for marker in visual_markers:
#     marker.visible = False

# +
# f = h5py.File(paths[0], "r")
# for pos in f["data"][f"demo_0"]["datagen_info"]["eef_pose"][::50, :3, 3]:
#     assert False, type(pos)

# +
# f.close()

# +
# If want to remove markers
# og.sim.batch_remove_objects(marker_list)
# for _ in range(10): og.sim.step()

# +
# assert False, (len(marker_list_all))

# +
# og.shutdown()

# +
# import matplotlib.pyplot as plt
# for _ in range(10): og.sim.render()
# print(len(marker_list))
# viewer_camera_img = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3]
# plt.imshow(viewer_camera_img)
# plt.show()
# path = "/home/arpit/test_projects/mimicgen/paper_assets"
# plt.imsave(f'{path}/diversity_fig_tidy_table_D2.jpg', viewer_camera_img.numpy())

# +
# import omnigibson.lazy as lazy
# from pxr import UsdGeom, Usd
# stage = lazy.omni.usd.get_context().get_stage()
# camera_path = og.sim.viewer_camera.prim_path
# camera_prim = stage.GetPrimAtPath(camera_path)
# camera = UsdGeom.Camera(camera_prim)
# # projection = camera.GetProjectionAttr().Get()
# camera.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
# for _ in range(100): og.sim.render()

# og.sim.viewer_camera.horizontal_aperture = 60.0
# og.sim.viewer_camera.vertical_aperture
# og.sim.viewer_camera.focal_length
