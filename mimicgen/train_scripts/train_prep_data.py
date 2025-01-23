import json
import numpy as np
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
import h5py, argparse, pdb
from robomimic.scripts.split_train_val import split_train_val_from_hdf5
import matplotlib.pyplot as plt
import open3d as o3d
import omnigibson.utils.transform_utils as T
import torch as th
import fpsample
import time
from multiprocessing import Pool
from functools import partial
import sys
import time

# TODO: need to change the offset and fixed normalization range accordingly
PCD_FIXED_OFFSET = np.array([ 2.885, -0.026,  2.533])
PCD_FIXED_OFFSET = np.array([ 1.438, 0.041, 1.509])
PCD_FIXED_NORMALIZATION_RANGE = np.array([0.9, 0.9, 0.9])

# external sensor intrinsic matrix
# K = np.array([
#     [519.2078,   0.0000, 320.0000],
#     [  0.0000, 560.5954, 180.0000],
#     [  0.0000,   0.0000,   1.0000]
#     ]) # 360 x 640
K = np.array([
    [259.6039,   0.0000, 160.0000],
    [  0.0000, 280.2977,  90.0000],
    [  0.0000,   0.0000,   1.0000]
    ]) # 180 x 320


# camera pose
# option 1: position far away
# CAMERA_POSITION = th.tensor([ 1.7492, -0.0424,  1.5371])
# CAMERA_QUAT= th.tensor([0.3379, 0.3417, 0.6236, 0.6166])

# option 2: position far away
CAMERA_POSITION = th.tensor([ 1.0304, -0.0309,  1.0272])
CAMERA_QUAT= th.tensor([0.2690, 0.2659, 0.6509, 0.6583])

PCD_MAX_DEPTH = 2

# print with 3 decimal points
np.set_printoptions(precision=3)

# josiah's data format
"""
 - data
    - demo_0
        - obs
            - robot0::proprio (length, prop_dim)
            - combined::point_cloud' (length, 2048, 4)
        - next_obs
            - robot0::proprio
            - combined::point_cloud'
        - actions (length, action_dim)
        - rewards (length,)
        - dones (length,)
    - demo_1
 - mask

        example rewards
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

       example dones:
       array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True])



"""

# mimicgen generated dataset format
# the original element are in the format of actions, states, obs, datagen_info, src_demo_inds, src_demo_labels
"""
 - data
    - demo_0
        - actions (length, action_dim)
        - states (length, states_dim)
        - obs
            - 'robot_fjtzyj::robot_fjtzyj:eyes:Camera:0::rgb' (length, 128, 128, 4)
            - 'robot_fjtzyj::robot_fjtzyj:eyes:Camera:0::depth' (length, 128, 128)
            - 'task::low_dim' (length, 58)
            - 'external::external_sensor0::rgb' (length, 128, 128, 4)
            - 'external::external_sensor0::depth' (length, 128, 128)
        - datagen_info
            - 'eef_pose' (950, 8, 4)
            - 'object_poses' 
                - coffee_cup (950, 4, 4)
                - paper_cup (950, 8, 4)
                - breakfast_table (950, 8, 4)
            - 'subtask_term_signals'
                - 'grasp_right' (length,)
                - 'ungrasp_right'
                - 'grasp_left'
                - 'ungrasp_left'
            - 'gripper_action' (length, 2)
        - src_demo_inds (num of generated subtasks)
        - src_demo_labels (length, 1)
        - mp_end_steps (length, 2)
        - subtask_lengths (length,)
    - demo_1
"""

"""debugging code"""
def debugging_grasp_obs(demo_data):
    """
    In action_left_gripper, action_right_gripper
      -1 means close the gripper, 1 means open the gripper, TODO: need to double check, no 0 shows up

    In gripper_left_qpos, gripper_right_qpos
      of dim 2, the second dim does not change along the whole trajectory 
      the first dim get smaller when the gripper is closing
    """
    print("")
    print('grasp_left, action_left_gripper, gripper_left_qpos')
    grasp_left = np.array(demo_data['obs']['grasp_left'])
    action_left = demo_data["actions"][:,-9][:,None] 
    gripper_left_qpos = np.array(demo_data['obs']['gripper_left_qpos'])
    gripper_left_info = np.concatenate((grasp_left, action_left), axis=1)
    gripper_left_info = np.concatenate((gripper_left_info, gripper_left_qpos), axis=1)

    # similarly for the right gripper
    grasp_right = np.array(demo_data['obs']['grasp_right'], dtype=np.float32)
    action_right = demo_data["actions"][:,-1][:,None]
    gripper_right_qpos = np.array(demo_data['obs']['gripper_right_qpos'])
    gripper_right_info = np.concatenate((grasp_right, action_right), axis=1)
    gripper_right_info = np.concatenate((gripper_right_info, gripper_right_qpos), axis=1)

    print('gripper_left_info', gripper_left_info[:200, 0])
    print('gripper_right_info', gripper_right_info[:200, 0])


def debugging_camera_imgs(demo_data):
    """
    debugging the camera images
    """
    import matplotlib.pyplot as plt
    # r_rgb_img = demo_data['obs']['robot_fjtzyj::robot_fjtzyj:eyes:Camera:0::rgb']
    # r_depth_img = demo_data['obs']['robot_fjtzyj::robot_fjtzyj:eyes:Camera:0::depth']

    # e_rgb_img = demo_data['obs']['external::external_sensor0::rgb']
    # e_depth_img = demo_data['obs']['external::external_sensor0::depth']

    # print(r_rgb_img.shape, r_depth_img.shape)
    # print(e_rgb_img.shape, e_depth_img.shape)

    # # show four images in two rows

    # fig, axs = plt.subplots(2, 2)
    # axs
    # axs[0, 0].imshow(r_rgb_img[0])
    # # add title
    # axs[0, 0].set_title('robot eyes camera rgb')
    # axs[0, 1].imshow(r_depth_img[0])
    # axs[0, 1].set_title('robot eyes camera depth')
    # axs[1, 0].imshow(e_rgb_img[0])
    # axs[1, 0].set_title('external sensor rgb')
    # axs[1, 1].imshow(e_depth_img[0])
    # axs[1, 1].set_title('external sensor depth')

    render_rgb_img = demo_data['obs']['render::rgb']
    plt.imshow(render_rgb_img[0])

    plt.show()
    import time
    time.sleep(0.1)
    plt.close()


# pcd sanity check
def pcd_sanity_check(pc):
    # visualize with open3D

    print('enter sanity check')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3)) 
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])
    print('number points', pc.shape[0])


"""code blocks"""


def clipping_block_v0(pc):
    # sensor facing the robot, the sensor is far away
    min_z_range = np.min(pc[:, 2]) + 0.43
    mask = pc[:, 2] < min_z_range
    pc = pc[~mask]

    max_z_range = np.max(pc[:, 2]) - 0.6
    mask = pc[:, 2] > max_z_range
    pc = pc[~mask]

    min_x_range = np.min(pc[:, 0]) + 0.3
    mask = pc[:, 0] < min_x_range
    pc = pc[~mask]

    max_x_range = np.max(pc[:, 0]) - 0.1
    mask = pc[:, 0] > max_x_range
    pc = pc[~mask]

    min_y_range = np.min(pc[:, 1]) + 0.3
    mask = pc[:, 1] < min_y_range
    pc = pc[~mask]

    max_y_range = np.max(pc[:, 1]) - 0.3
    mask = pc[:, 1] > max_y_range
    pc = pc[~mask]
        # pc[mask] = 0
    return pc


def clipping_block_v1(pc):
    # sensor is a bit closer
    min_x_range = np.min(pc[:, 0]) + 0.95
    mask = pc[:, 0] < min_x_range
    pc = pc[~mask]

    max_x_range = np.max(pc[:, 0]) - 0.0
    mask = pc[:, 0] > max_x_range
    pc = pc[~mask]

    min_z_range = np.min(pc[:, 2]) + 0.0
    mask = pc[:, 2] < min_z_range
    pc = pc[~mask]

    max_z_range = np.max(pc[:, 2]) - 0.0
    mask = pc[:, 2] > max_z_range
    pc = pc[~mask]

    min_y_range = np.min(pc[:, 1]) + 0.25
    mask = pc[:, 1] < min_y_range
    pc = pc[~mask]

    max_y_range = np.max(pc[:, 1]) - 0.2
    mask = pc[:, 1] > max_y_range
    pc = pc[~mask]
        # pc[mask] = 0
    return pc


def compute_point_cloud_from_depth(
        depth, 
        K, 
        cam_to_img_tf=None,
        world_to_cam_tf=None, 
        pcd_step_vis=False, 
        max_depth=3, 
        sample_type='fps',
        num_points_to_sample=1024,
        clip_scene=True):
    
    # K - 3x3 cam intrinsics matrix
    # tfs - 4x4 homogeneous global pose tf for cam
    # Camera points in -z, so rotate by 180 deg so it points correctly in +z -- this means
    # omni cam_to_img_tf is T.pose2mat(([0, 0, 0], T.euler2quat([np.pi, 0, 0])))
    # max_depth - max depth to consider for point cloud
    # pcd_step_vis - whether to visualize the point cloud at each step for debugging
    # fps - whether to do farthest point sampling
    # random_sample - whether to randomly sample points
    # num_points_to_sample - number of points to sample
    # clip_scene - whether to clip the scene

    h, w = depth.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij", sparse=False)
    assert depth.min() >= 0
    u = x
    v = y
    uv = np.dstack((u, v, np.ones_like(u))) # (img_width, img_height, 3)

    # filter depth
    mask = depth > max_depth
    depth[mask] = 0

    Kinv = np.linalg.inv(K)

    pc = depth.reshape(-1, 1) * (uv.reshape(-1, 3) @ Kinv.T)
    pc = pc.reshape(h, w, 3)

    # If no tfs, use identity matrix
    cam_to_img_tf = np.eye(4) if cam_to_img_tf is None else cam_to_img_tf
    world_to_cam_tf = np.eye(4) if world_to_cam_tf is None else world_to_cam_tf

    pc = np.concatenate([pc.reshape(-1, 3), np.ones((h * w, 1))], axis=-1)  # shape (H*W, 4)

    # Convert using camera transform
    # Create (H * W, 4) vector from pc
    pc = (pc @ cam_to_img_tf.T @ world_to_cam_tf.T)[:, :3].reshape(h, w, 3)

    
    # rotate_start_time = time.time()
    # rotate a point cloud
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
    pc = pc @ R.T
    pc = pc.reshape(-1, 3)
    # print('In compute pcd: new rotate time', time.time() - rotate_start_time)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc_new.reshape(-1, 3)) 
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, axis])
    # import pdb; pdb.set_trace()



    # vis_start_time = time.time()
    # # visualize with open3D
    # pcd = o3d.geometry.PointCloud()
    # print('In compute pcd: init pcd time', time.time() - vis_start_time)
    # pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3)) 
    # print('In compute pcd: init pcd time + assign points', time.time() - vis_start_time)
    # # flip the point clound to align the x, y axis
    # mesh_time = time.time()
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # print('In compute pcd: init mesh time', time.time() - mesh_time)
    # R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
    # print('In compute pcd: get rotation matrix', time.time() - mesh_time)
    # pcd.rotate(R, center=(0, 0, 0))
    # print('In compute pcd: pcd rotate time', time.time() - mesh_time)
    # print('In compute pcd: construct o3d and rotate', time.time() - vis_start_time)

    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, axis])
    # print('In compute pcd: visualize time', time.time() - vis_start_time)

    # pc = np.asarray(pcd.points)
    # import pdb; pdb.set_trace()

    if pcd_step_vis:
        print("")
        print('number points before clipping', pc.shape[0])

    if clip_scene:
        # TODO: this part is not very robust right now, sometimes the point cloud can be none
        # TODO: need to implement an early stop mechanism

        # the clip mechanism when the sensor facing the table, and the sensor is far away
        # filter out the floor
        # pc = clipping_block_v0(pc)
        # corp_time = time.time()
        pc = clipping_block_v1(pc)
        # print('In compute pcd: corp time:', time.time() - corp_time)

    if pcd_step_vis:
        print('number points after clipping', pc.shape[0])

    # transform the center of the point cloud to the origin
    # fixed_offset = -np.mean(pc, axis=0)
    # translate_start_time = time.time()
    pc += PCD_FIXED_OFFSET
    # pcd.translate(PCD_FIXED_OFFSET)
    # print('In compute pcd: translate time', time.time() - translate_start_time)

    # # normalize the point cloud
    pc = pc / PCD_FIXED_NORMALIZATION_RANGE

    # farthest point sampling
    pcd_downsample_start_time = time.time()
    if sample_type == 'fps':
        # print('pc shape', pc.shape)
        # fps_samples_idx = fpsample.fps_sampling(pc, num_points_to_sample)
        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, num_points_to_sample, h=5)
        pc = pc[kdline_fps_samples_idx]
        # print('In compute pcd: fps time used', time.time() - pcd_downsample_start_time)
        # pcd.points = o3d.utility.Vector3dVector(pc)
        if pcd_step_vis:
            print('after fps, number points', pc.shape[0])
        

    elif sample_type=='random':
        # down sample input pointcloud
        if len(pc) > num_points_to_sample:
            indices = np.random.choice(len(pc), num_points_to_sample, replace=False)
            pc = pc[indices]
            # pcd.points = o3d.utility.Vector3dVector(pc)
            if pcd_step_vis:
                print('after random sample, number points', pc.shape[0])
        # print('In compute pcd: random time used', time.time() - pcd_downsample_start_time)
        # print('after random sample, number points', pc.shape[0])

    if pcd_step_vis:
        print("")

    if pcd_step_vis:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3)) 
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, axis])
        import pdb; pdb.set_trace()
        
    # # get points from the point cloud
    # pc = np.asarray(pcd.points)

    return pc


def process_pointcloud_per_demo_parallel(obs, vis_sign=False, sample_type="fps"):
    """
    get point cloud from depth information
    """

    print('start processing point cloud ... ')

    cur_time = time.time()
    # third view key
    rgbd = obs['external::external_sensor0::rgb']
    depth = obs['external::external_sensor0::depth_linear']
    print('depth shape', depth.shape)
    print('rgb shape', rgbd.shape)

    # option 1: transformation matrix 
    world_to_cam_tf = np.array([
        [-1.126e-02, -5.381e-01,  8.428e-01,  1.749e+00],
       [ 9.999e-01, -6.099e-03,  9.470e-03, -4.240e-02],
       [ 4.449e-05,  8.429e-01,  5.381e-01,  1.537e+00],
       [ 0.000e+00,  0.000e+00,  0.000e+00,  1.000e+00]])
    
    # option 2: transformation matrix 
    world_to_cam_tf = T.pose2mat((CAMERA_POSITION, CAMERA_QUAT)).numpy()

    # # for quick debugging
    # depth = depth[500:510, :, :]
    # print('depth shape', depth.shape)
    with Pool(processes=4) as pool:
        pcd_demo = pool.map(
            partial(
                compute_point_cloud_from_depth,
                K=K,
                cam_to_img_tf=None,
                world_to_cam_tf=world_to_cam_tf,
                pcd_step_vis=False,
                max_depth=PCD_MAX_DEPTH,
                sample_type=sample_type,
                num_points_to_sample=NUM_POINTS_TO_SAMPLE,
                clip_scene=True
                ),
                depth
                )
    pcd_demo = np.array(pcd_demo)

    print('finished processing point cloud, pcd shape: ', pcd_demo.shape)
    print('time used:', time.time() - cur_time)   
    print("")

    return pcd_demo


def process_pointcloud_per_demo(obs, vis_sign=True, sample_type="fps"):
    """
    get point cloud from depth information
    """

    print('start processing point cloud ... ')

    cur_time = time.time()
    # third view key
    rgbd = obs['external::external_sensor0::rgb']
    depth = obs['external::external_sensor0::depth_linear']
    print('depth shape', depth.shape)
    print('rgb shape', rgbd.shape)

    # option 1: transformation matrix 
    world_to_cam_tf = np.array([
        [-1.126e-02, -5.381e-01,  8.428e-01,  1.749e+00],
       [ 9.999e-01, -6.099e-03,  9.470e-03, -4.240e-02],
       [ 4.449e-05,  8.429e-01,  5.381e-01,  1.537e+00],
       [ 0.000e+00,  0.000e+00,  0.000e+00,  1.000e+00]])
    
    # option 2: transformation matrix 
    world_to_cam_tf = T.pose2mat((CAMERA_POSITION, CAMERA_QUAT)).numpy()

    if vis_sign:
        # start processing and visualizing the point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd_vis = o3d.geometry.PointCloud()
        firstfirst = True
    
    # without parallel processing 
    pcd_demo = []
    step = 0
    depth_debug = depth[:100]
    for depth_img in depth_debug:
        step += 1
        
        pcd = compute_point_cloud_from_depth(
            depth_img, K, 
            cam_to_img_tf=None, 
            world_to_cam_tf=world_to_cam_tf, 
            pcd_step_vis=False, 
            max_depth=PCD_MAX_DEPTH,
            sample_type='random',
            num_points_to_sample=NUM_POINTS_TO_SAMPLE,
            clip_scene=True
            )
        pcd_demo.append(pcd)

        if vis_sign:
            print('step', step, 'number of points', pcd.shape[0])
            pcd_vis.points = o3d.utility.Vector3dVector(pcd)
            
            if firstfirst:
                vis.add_geometry(pcd_vis)
                firstfirst = False
            else:
                vis.update_geometry(pcd_vis)
            vis.poll_events()
            vis.update_renderer()
    
    if vis_sign:
        vis.destroy_window()
    
    pcd_demo = np.array(pcd_demo)

    print('finished processing point cloud')
    print('time used:', time.time() - cur_time)   
    print('pcd_demo_shape', pcd_demo.shape)
    print("")
    # breakpoint()
    # import pdb; pdb.set_trace()
    return pcd_demo


def parse_obs(obs, obs_type):
    """
    all the obs keys:
     - 'joint_qpos', 'joint_qpos_sin', 'joint_qpos_cos', 'joint_qvel', 'joint_qeffort', 'robot_pos', 'robot_ori_cos', 'robot_ori_sin', 'robot_2d_ori', 'robot_2d_ori_cos', 'robot_2d_ori_sin', 'robot_lin_vel', 'robot_ang_vel', 
     - 'camera_qpos', 'camera_qpos_sin', 'camera_qpos_cos', 'camera_qvel', 
     - 'arm_left_qpos', 'arm_left_qpos_sin', 'arm_left_qpos_cos', 'arm_left_qvel', 'eef_left_pos', 'eef_left_quat', 'grasp_left', 'gripper_left_qpos', 'gripper_left_qvel', 
     - 'arm_right_qpos', 'arm_right_qpos_sin', 'arm_right_qpos_cos', 'arm_right_qvel', 'eef_right_pos', 'eef_right_quat', 'grasp_right', 'gripper_right_qpos', 'gripper_right_qvel', 
     - 'trunk_qpos', 'trunk_qvel', 
     - 'base_qpos', 'base_qpos_sin', 'base_qpos_cos', 'base_qvel', 
     - 'robot_fjtzyj::robot_fjtzyj:eyes:Camera:0::rgb', 'robot_fjtzyj::robot_fjtzyj:eyes:Camera:0::depth', 
     - 'task::low_dim', 
     - 'external::external_sensor0::rgb', 'external::external_sensor0::depth',
     - 'object::dixie_cup', 'object::coffee_cup', 'object::floor', 'object::breakfast_table', 

    """

    if obs_type == "low_dim":
        robot_prop_dict = [
            'joint_qpos', 
            'eef_left_pos', 'eef_left_quat',
            'eef_right_pos', 'eef_right_quat', 
            'gripper_left_qpos', 'gripper_right_qpos',
            ] # robot proprioceptive states
        # get object keys
        obj_key_list = []
        for obj_key in obs.keys():
            if 'object::' in obj_key and "floor" not in obj_key and "table" not in obj_key:
                obj_key_list.append(obj_key)
        obs_key_list = robot_prop_dict + obj_key_list
    elif obs_type == "point_cloud":
        robot_prop_dict = [
            'joint_qpos', 
            'eef_left_pos', 'eef_left_quat',
            'eef_right_pos', 'eef_right_quat', 
            'gripper_left_qpos', 'gripper_right_qpos',
            ] # robot proprioceptive states
        # get object keys
        obj_key_list = []
        for obj_key in obs.keys():
            if 'object::' in obj_key and "floor" not in obj_key and "table" not in obj_key:
                obj_key_list.append(obj_key)
        obs_key_list = robot_prop_dict + obj_key_list + ['combined::point_cloud']
    elif obs_type == "rgb":
        obs_key_list = [
            'robot_fjtzyj::robot_fjtzyj:eyes:Camera:0::rgb',
            'external::external_sensor0::rgb'
        ]
    else:
        raise ValueError("Invalid obs_type")
    
    return obs_key_list


def process_robomimic_dataset(file_path, obs_type, vis_sign=False, sample_type="fps"):
    # the original element are in the format of actions, states, obs, datagen_info, src_demo_inds, src_demo_labels, mp_end_steps, subtask_lengths
    # for each demostration， get the obs, next_obs, actions, rewards, dones
    dataset_dict = {}
    # Open the file
    with h5py.File(file_path, "r") as hdf:
        # Access a group or dataset
        group = hdf["data"]
        # process data for each demo
        for demo_key in group.keys():
            demo_data = group[demo_key]
            print("")
            print('Start processing', demo_key) 

            obs_dict = {}
            next_obs_dict = {}
            num_steps = demo_data['actions'].shape[0] - 1
            # get actions
            actions = demo_data["actions"][:-1] # actions already in range [-1, 1]

            # get rewards and dones
            # assume the data are expert demonstrations and only the last step is the success step
            rewards = np.zeros(num_steps)
            rewards[-1] = 1
            dones = np.zeros(num_steps)
            dones[-1] = 1

            if vis_sign:
                # for pcd debugging, visualized the first 100 steps in each episode
                pcd_demo = process_pointcloud_per_demo(demo_data["obs"], vis_sign=vis_sign, sample_type=sample_type)
                continue

            # get observations
            pcd_demo = process_pointcloud_per_demo_parallel(demo_data["obs"], vis_sign=vis_sign, sample_type=sample_type) # get point cloud from depth images
            obs_key_list = parse_obs(demo_data["obs"], obs_type)
            print(demo_key, 'observation keys', obs_key_list)

            for obs_key in obs_key_list:

                # debugging_grasp_obs(demo_data) # debugging the grasp states
                # debugging_camera_imgs(demo_data) # debugging the camera images

                if obs_key == 'combined::point_cloud':
                    obs_dict[obs_key] = pcd_demo[:-1]
                    next_obs_dict[obs_key] = pcd_demo[1:]
                else:
                    obs_dict[obs_key] = demo_data['obs'][obs_key][:-1]
                    next_obs_dict[obs_key] = demo_data['obs'][obs_key][1:]

                assert obs_dict[obs_key].shape[0] == next_obs_dict[obs_key].shape[0] == num_steps

            demo_dict = {
                "obs": obs_dict,
                "next_obs": next_obs_dict,
                "actions": actions,
                "rewards": rewards,
                "dones": dones
            }

            dataset_dict[demo_key] = demo_dict
    return dataset_dict


def get_demo_subtask_dict(file_path):
    demo_subtask_dict = {}
    with h5py.File(file_path, "r") as hdf:
        group = hdf["data"]
        for demo_key in group.keys():
            demo_data = group[demo_key]
            demo_subtask_dict[demo_key] = {
                "mp_end_steps": np.array(demo_data["mp_end_steps"]), # (num_subtasks, 2)
                "subtask_lengths": np.array(demo_data["subtask_lengths"]) # (num_subtasks,)
            }
    return demo_subtask_dict


def process_subtask_dataset(file_path, output_path):
    """
    segment the trajectory based on subtasks
    in demostration， get the obs, next_obs, actions, rewards, dones
    """

    # get subtasks step information in each demo
    demo_subtask_dict = get_demo_subtask_dict(file_path)

    # retrieve the subtask segmentation
    # - demo_0
    #     - demo_0_subtask_0
    #     - demo_0_subtask_1
    dataset_dict = {}
    with h5py.File(output_path, "r") as hdf:
        # Access a group or dataset
        group = hdf["data"]
        for demo_key in group.keys():
            print("")
            print('Processing', demo_key)
            demo_data = group[demo_key]
            demo_data_subtask = {}
            num_subtasks = demo_subtask_dict[demo_key]["subtask_lengths"].shape[0]
            # for each subtask
            for i in range(num_subtasks):
                print('Processing subtask', i)
                demo_subtask_key = demo_key + "_subtask_" + str(i)
                demo_data_subtask[demo_subtask_key] = {}
                # segment out the trajectory based on the subtask steps
                # TODO: there is a problem for situations where only one arm has motion planner segmentation and the other arm does not have the motion planner segmentation
                subtask_start_step = np.sum(demo_subtask_dict[demo_key]["subtask_lengths"][:i])
                subtask_end_step = subtask_start_step + demo_subtask_dict[demo_key]["subtask_lengths"][i]

                demo_data_subtask[demo_subtask_key]["obs"] = {}
                for obs_key in demo_data["obs"].keys():
                    demo_data_subtask[demo_subtask_key]["obs"][obs_key] = demo_data["obs"][obs_key][subtask_start_step:subtask_end_step]
                
                demo_data_subtask[demo_subtask_key]["next_obs"] = {}
                for next_obs_key in demo_data["next_obs"].keys():
                    demo_data_subtask[demo_subtask_key]["next_obs"][next_obs_key] = demo_data["next_obs"][next_obs_key][subtask_start_step:subtask_end_step]
                
                demo_data_subtask[demo_subtask_key]["actions"] = demo_data["actions"][subtask_start_step:subtask_end_step]
                demo_data_subtask[demo_subtask_key]["rewards"] = demo_data["rewards"][subtask_start_step:subtask_end_step]
                demo_data_subtask[demo_subtask_key]["dones"] = demo_data["dones"][subtask_start_step:subtask_end_step]

                print('Subtask length', demo_data_subtask[demo_subtask_key]["actions"].shape[0])

            dataset_dict[demo_key] = demo_data_subtask

    # asset the number of subtasks in each demo are the same
    cur_num_subtasks = 0
    for demo_key in dataset_dict.keys():
        num_subtasks = len(dataset_dict[demo_key].keys())
        if cur_num_subtasks == 0:
            cur_num_subtasks = num_subtasks
        assert cur_num_subtasks == num_subtasks
    
    print("")
    print("Number of subtasks in each demo are the same", cur_num_subtasks)
    print("")

    # reorganize the dataset and save the data based on subtasks
    # - subtask_0
    #   - demo_0
    #   - demo_1 ...
    # - subtask_1
    new_dataset_dict = {}
    for i in range(cur_num_subtasks):
        subtask_key = "subtask_" + str(i)
        new_dataset_dict[subtask_key] = {}
        for demo_key in dataset_dict.keys():
            new_dataset_dict[subtask_key][demo_key] = dataset_dict[demo_key][demo_key + "_subtask_" + str(i)]

    return new_dataset_dict, cur_num_subtasks


def write_to_hdf5(dict, input_path, output_path):
    with h5py.File(output_path, "w") as f:
        data_group = f.create_group("data")
        for key in dict.keys():
            demo_group = data_group.create_group(key)
            demo = dict[key]
            for key in demo.keys():
                if key in ["obs", "next_obs"]:
                    obs_group = demo_group.create_group(key)
                    obs = demo[key]
                    for key in obs.keys():
                        obs_group.create_dataset(key, data=obs[key])
                else:
                    demo_group.create_dataset(key, data=demo[key])
            demo_group.attrs["total"] = demo_group["actions"].shape[0]
            demo_group.attrs["num_samples"] = demo_group["actions"].shape[0]
    # copy the mask group from input_path to output_path
    with h5py.File(input_path, "r") as in_f:
        mask_group = in_f["mask"]
        with h5py.File(output_path, "r+") as out_f:
            out_f.copy(mask_group, "mask")
    print('Finished writing the data to', output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # add file path argument
    parser.add_argument("--file_path", 
                        type=str, 
                        default="/home/mengdi/dataset/demo_failed.hdf5", 
                        help="hdf5 file path")
    # add output file path argument
    parser.add_argument("--output_path", 
                        type=str, 
                        default="/home/mengdi/dataset/robomimic_dataset.hdf5", 
                        help="output hdf5 file path")
    # add observation type
    parser.add_argument("--obs_type", 
                        type=str, 
                        default="point_cloud", 
                        help="observation key type", 
                        choices=["low_dim", "rgb", "depth", "point_cloud"])
    # trian val split ratio
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.1,
        help="validation ratio, in (0, 1)"
    )

    # pcd number of samples
    parser.add_argument(
        "--num_pcd_samples",
        type=int,
        default=1024,
        help="number of samples after processing pcd"
    )

    parser.add_argument(
        "--vis_sign",
        action="store_true",
        help="whether visualize the pcd when processing"
    )

    parser.add_argument(
        "--fps",
        action="store_true",
        help="use farthest point sampling to sample the point cloud"
    )

    parser.add_argument(
        "--random",
        action="store_true",
        help="use random sampling to sample the point cloud"
    )


    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode: only save the first 50 steps of each demo"
    )

    args = parser.parse_args()

    file_path = args.file_path
    output_path = args.output_path

    global NUM_POINTS_TO_SAMPLE
    NUM_POINTS_TO_SAMPLE = args.num_pcd_samples


    print("")
    print('Start processing the dataset', file_path, '....')
    print("")
    
    # first split the train and val data
    split_train_val_from_hdf5(file_path, val_ratio=args.split_ratio)
    
    assert args.fps != args.random, "Only one of fps and random can be True"
    if args.fps: sample_type = "fps"
    if args.random: sample_type = "random"

    # change to robomimic dataset format
    robomimic_dataset = process_robomimic_dataset(
        file_path=file_path,
        obs_type=args.obs_type,
        vis_sign=args.vis_sign,
        sample_type=sample_type
    )
    if args.vis_sign:
        sys.exit()

    output_path = output_path.replace(".hdf5", "_{}_{}.hdf5".format(sample_type, args.num_pcd_samples))

    if args.debug:
        output_path = output_path.replace(".hdf5", "_debug.hdf5")
    

    print("")
    print('Writing the data to', output_path, '....')
    
    # write to hdf5
    write_to_hdf5(robomimic_dataset, file_path, output_path)

    print("")
    print('Finished writing the data.')

    print("")
    print('Start processing subtasks', output_path, '....')

    # start processing subtasks
    subtask_data_dict, num_subtasks = process_subtask_dataset(file_path, output_path)

    for i in range(num_subtasks):
        subtask_key = "subtask_" + str(i)
        subtask_output_path = output_path.replace(".hdf5", "_subtask_{}.hdf5".format(subtask_key))
        write_to_hdf5(subtask_data_dict[subtask_key], file_path, subtask_output_path)