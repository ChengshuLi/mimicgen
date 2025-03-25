import h5py
import pickle
import time
import omnigibson as og
import torch as th
from omnigibson.macros import create_module_macros
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.lazy as lazy
from omnigibson.objects.primitive_object import PrimitiveObject

with open("/home/arpit/test_projects/mimicgen/kwargs.pickle", "rb") as f:
    kwargs = pickle.load(f)
    # kwargs["scene"] = {"type": "Scene"}
env = og.Environment(configs=kwargs)

controller_config = {
    "base": {"name": "HolonomicBaseJointController", "motor_type": "position", "command_input_limits": None, "use_impedances": False},
    "trunk": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
    "arm_left": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
    "arm_right": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
    "gripper_left": {"name": "MultiFingerGripperController", "mode": "binary", "command_input_limits": (0.0, 1.0),},
    "gripper_right": {"name": "MultiFingerGripperController", "mode": "binary", "command_input_limits": (0.0, 1.0),},
    "camera": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
}

env.robots[0].reload_controllers(controller_config=controller_config)
env.robots[0]._grasping_mode = "sticky"
robot = env.robots[0]

# with open("/home/arpit/test_projects/mimicgen/scene_0.pickle", "rb") as f:
with open("/home/arpit/test_projects/mimicgen/debug_no_valid_pose.pickle", "rb") as f:
    scene_0 = pickle.load(f)
og.sim.load_state(scene_0, serialized=False)

for _ in range(20): og.sim.step()


f2 = h5py.File("/home/arpit/test_projects/mimicgen/temp_datasets/demo_failed.hdf5", "r")
num_demos = len(f2["data"].keys())
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
    pos = f2["data"][f"demo_{i}"]["datagen_info"]["object_poses"]["teacup"][0][:3,3]
    marker_list[i].set_position_orientation(position=pos)
    base_pos = f2["data"][f"demo_{i}"]["datagen_info"]["base_pose"][-1][:3,3]
    base_marker_list[i].set_position_orientation(position=base_pos)

for _ in range(300): og.sim.step()
breakpoint()