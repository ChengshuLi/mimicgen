import h5py
import json
import numpy as np
import torch as th
import omnigibson as og

from robomimic.utils.file_utils import get_env_metadata_from_dataset
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives


kwargs = {'env': {'action_frequency': 30, 'rendering_frequency': 30, 'physics_frequency': 120, 'device': None, 'automatic_reset': False, 'flatten_action_space': False, 'flatten_obs_space': True, 'initial_pos_z_offset': 0.1, 'external_sensors': [{'sensor_type': 'VisionSensor', 'name': 'external_sensor0', 'relative_prim_path': '/controllable__r1__robot_r1/base_link/external_sensor0', 'modalities': [], 'sensor_kwargs': {'viewport_name': 'Viewport', 'image_height': 1080, 'image_width': 1080}, 'position': [0.0, 0.0, 1.8], 'orientation': [-0.153, 0.153, 0.6903, -0.6903], 'pose_frame': 'parent', 'include_in_obs': False}, {'sensor_type': 'VisionSensor', 'name': 'external_sensor1', 'relative_prim_path': '/controllable__r1__robot_r1/base_link/external_sensor1', 'modalities': [], 'sensor_kwargs': {'viewport_name': 'Viewport', 'image_height': 1080, 'image_width': 1080}, 'position': [-0.2, -0.6, 2.0], 'orientation': [0.4164, -0.1929, -0.3737, 0.806], 'pose_frame': 'parent', 'include_in_obs': False}, {'sensor_type': 'VisionSensor', 'name': 'external_sensor2', 'relative_prim_path': '/controllable__r1__robot_r1/base_link/external_sensor2', 'modalities': [], 'sensor_kwargs': {'viewport_name': 'Viewport', 'image_height': 1080, 'image_width': 1080}, 'position': [-0.2, 0.6, 2.0], 'orientation': [-0.193, 0.4163, 0.8062, -0.3734], 'pose_frame': 'parent', 'include_in_obs': False}]}, 'render': {'viewer_width': 1280, 'viewer_height': 720}, 'scene': {'waypoint_resolution': 0.2, 'num_waypoints': 10, 'trav_map_resolution': 0.1, 'default_erosion_radius': 0.0, 'trav_map_with_objects': True, 'scene_instance': 'Rs_int_task_datagen_pick_0_0_template', 'scene_file': None, 'type': 'InteractiveTraversableScene', 'scene_model': 'Rs_int', 'load_room_types': None, 'load_room_instances': None, 'include_robots': False}, 'robots': [{'type': 'R1', 'name': 'robot_r1', 'action_normalize': False, 'controller_config': {'arm_left': {'name': 'JointController', 'motor_type': 'position', 'pos_kp': 150, 'command_input_limits': None, 'command_output_limits': None, 'use_impedances': False, 'use_delta_commands': False}, 'gripper_left': {'name': 'MultiFingerGripperController', 'mode': 'smooth', 'command_input_limits': 'default', 'command_output_limits': 'default'}, 'arm_right': {'name': 'JointController', 'motor_type': 'position', 'pos_kp': 150, 'command_input_limits': None, 'command_output_limits': None, 'use_impedances': False, 'use_delta_commands': False}, 'gripper_right': {'name': 'MultiFingerGripperController', 'mode': 'smooth', 'command_input_limits': 'default', 'command_output_limits': 'default'}, 'base': {'name': 'HolonomicBaseJointController', 'motor_type': 'velocity', 'vel_kp': 150, 'command_input_limits': [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], 'command_output_limits': [[-0.75, -0.75, -1.0], [0.75, 0.75, 1.0]], 'use_impedances': False}, 'trunk': {'name': 'JointController', 'motor_type': 'position', 'pos_kp': 150, 'command_input_limits': None, 'command_output_limits': None, 'use_impedances': False, 'use_delta_commands': False}}, 'self_collisions': False, 'obs_modalities': [], 'position': [-3.163, -0.26, 0], 'orientation': [0.0, 0.0, 0.0, 1.0], 'grasping_mode': 'assisted', 'sensor_config': {'VisionSensor': {'sensor_kwargs': {'image_height': 1080, 'image_width': 1080}}}, 'reset_joint_pos': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.024999976158142, -1.4500000476837158, -0.4699999988079071, 0.0, 0.5759586691856384, -0.5759586691856384, 2.8274333477020264, 2.8274333477020264, -1.884955644607544, -1.884955644607544, 0.5934119820594788, -0.5934119820594788, 1.2740904092788696, -1.2740904092788696, -1.1344640254974365, 1.1344640254974365, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806]}], 'objects': [], 'task': {'type': 'BehaviorTask', 'activity_name': 'datagen_pick', 'activity_definition_id': 0, 'activity_instance_id': 0, 'predefined_problem': None, 'online_object_sampling': False, 'debug_object_sampling': False, 'highlight_task_relevant_objects': False, 'termination_config': {'max_steps': 50000}, 'reward_config': {'r_potential': 1.0}}, 'wrapper': {'type': None}, 'init_curobo': True}

RESOLUTION = (256, 256)

# Explicity add the depth_linear and rgb modalities
kwargs["robots"][0]["obs_modalities"].append("depth_linear")
kwargs["robots"][0]["obs_modalities"].append("depth")
kwargs["robots"][0]["obs_modalities"].append("rgb")
kwargs["robots"][0]["obs_modalities"].append("seg_instance")

# Setting the camera height and width here because setting it later causes issues
kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_height"] = RESOLUTION[0]
kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_width"] = RESOLUTION[1]
kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["horizontal_aperture"] = 40.0

kwargs["robots"][0]["reset_joint_pos"] = [
        0.0000,
        0.0000,
        0.000,
        0.000,
        0.000,
        -0.0000, # 6 virtual base joint 
        0.5,
        -1.0,
        -0.8,
        -0.0000, # 4 torso joints
        -0.000,
        0.000,
        1.8944,
        1.8945,
        -0.9848,
        -0.9849,
        1.5612,
        1.5621,
        0.9097,
        0.9096,
        -1.5544,
        -1.5545,
        0.0500,
        0.0500,
        0.0500,
        0.0500,
    ]

# Always spawn robot at the origin with no rotation (this is to be compatible with curobo)
kwargs["robots"][0]["position"] = [0.0, 0.0, 0.0]
kwargs["robots"][0]["orientation"] = [0.0, 0.0, 0.0, 1.0]

env = og.Environment(configs=kwargs)
for _ in range(100): og.sim.step()

# f3 = h5py.File("/home/arpit/Downloads/demo.hdf5", "r")
f3 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/r1_pick_cup_no_nav/demo_src_r1_pick_cup_task_D0/demo.hdf5", "r")
# depth = np.array(f3["data"]["demo_0"]["obs"]["robot_r1::robot_r1:eyes:Camera:0::depth_linear"])
# intr = env.robots[0].sensors["robot_r1:eyes:Camera:0"].intrinsic_matrix
# from omnigibson.utils.vision_utils import visualize_pcd
# visualize_pcd(None, depth[0], intr.numpy())

primitive = StarterSemanticActionPrimitives(env,env.robots[0])
coffee_cup = env.scene.object_registry("name", "coffee_cup_7")

for k in f3["data"].keys():
    og.sim.load_state(th.tensor(f3["data"][k]["states"][0]), serialized=True)
    action = primitive._empty_action()
    env.step(action)
    for _ in range(50): og.sim.step()
    pose = coffee_cup.get_position_orientation()
    breakpoint()
