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
import copy

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
def pcd_vis(pc):
    # visualize with open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3)) 
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])
    print('number points', pc.shape[0])


def color_pcd_vis(color_pcd):
    # visualize with open3D
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(color_pcd[:, :3])
    pcd.points = o3d.utility.Vector3dVector(color_pcd[:,3:]) 
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])
    print('number points', color_pcd.shape[0])


"""code blocks"""

def clipping_block(pc, pcd_offset, clip_bbox_size):
    # clip based on a bbx around the mean of the point cloud

    x_index = 0
    y_index = 1
    z_index = 2

    clip_box_size = {
        'x': clip_bbox_size[0],
        'y': clip_bbox_size[1],
        'z': clip_bbox_size[2],
    }

    update_mean = False
    if update_mean:
        pcd_mean = {
            'x': np.mean(pc[:, x_index]),
            'y': np.mean(pc[:, y_index]),
            'z': np.mean(pc[:, z_index]),
        }
    else:
        pcd_mean = {
            'x': pcd_offset[0],
            'y': pcd_offset[1],
            'z': pcd_offset[2],
        }

    mask = np.ones(pc.shape[0], dtype=bool)

    min_x_range = pcd_mean['x'] - clip_box_size['x']/2
    max_x_range = pcd_mean['x'] + clip_box_size['x']/2
    mask_x = (pc[:, x_index] > min_x_range) * (pc[:, x_index] < max_x_range)

    min_y_range = pcd_mean['y'] - clip_box_size['y']/2
    max_y_range = pcd_mean['y'] + clip_box_size['y']/2
    mask_y = (pc[:, y_index] > min_y_range) * (pc[:, y_index] < max_y_range)

    min_z_range = pcd_mean['z'] - clip_box_size['z']/2
    max_z_range = pcd_mean['z'] + clip_box_size['z']/2
    mask_z = (pc[:, z_index] > min_z_range) * (pc[:, z_index] < max_z_range)

    mask = mask_x * mask_z * mask_y

    pc_clip = pc[mask]

    return pc_clip, mask


def compute_point_cloud_from_rgbd(
        rgbd,
        K, 
        pcd_offset,
        pcd_norm_range,
        clip_bbox_size,
        cam_to_img_tf=None,
        world_to_cam_tf=None, 
        pcd_step_vis=False, 
        max_depth=3, 
        sample_type='fps',
        num_points_to_sample=1024,
        clip_scene=True,
        with_color=True
        ):
    
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

    depth = rgbd[:, :, -1]

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

    # rotate a point cloud
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
    pc = pc @ R.T
    
    # print('color pc shape', pc.shape) 
    color_img = rgbd[:, :, :3].reshape(-1, 3)  # shape (H*W, 3)
    pc = pc.reshape(-1, 3)

    if pcd_step_vis:
        print("")
        print('number points before clipping', pc.shape[0])

    if clip_scene:
        # clip the scene
        pc, mask_clip = clipping_block(
            copy.deepcopy(pc),
            pcd_offset,
            clip_bbox_size
            )
        if pcd_step_vis:
            print('number points after clipping', pc.shape[0])

    # transform 
    pc -= pcd_offset
    pc = pc / pcd_norm_range # # normalize the point cloud

    # get the clipped color
    color_img = color_img[mask_clip]
    color_img = color_img / 255.0 # noramlize the color

    # downsample the pcd
    pcd_downsample_start_time = time.time()
    if sample_type == 'fps':
        # farthest point sampling
        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, num_points_to_sample, h=5)
        pc = pc[kdline_fps_samples_idx]
        color_img = color_img[kdline_fps_samples_idx]
        if pcd_step_vis:
            print('after fps, number points', pc.shape[0])
    elif sample_type=='random':
        # random sample input pointcloud
        if len(pc) > num_points_to_sample:
            indices = np.random.choice(len(pc), num_points_to_sample, replace=False)
            pc = pc[indices]
            color_img = color_img[indices]
            if pcd_step_vis:
                print('after random sample, number points', pc.shape[0])

    if pcd_step_vis:
        print("")

    color_pcd = np.concatenate([color_img, pc], axis=-1)

    if pcd_step_vis:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3)) 
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, axis])
        import pdb; pdb.set_trace()
        
    # # get points from the point cloud
    # pc = np.asarray(pcd.points)
    if with_color:
        assert color_pcd.shape[1] == 6
        return color_pcd
    else:
        assert pc.shape[1] == 3
        return pc


def process_pointcloud_per_demo_parallel(rgbd, sample_type="fps", with_color=True, sensor_info=None):
    """
    get point cloud from depth information
    """

    print('start processing point cloud ... ')

    cur_time = time.time()
    print('rgb shape', rgbd.shape)

    with Pool(processes=4) as pool:
        # TODO: need to double check and verify
        pcd_demo = pool.map(
            partial(
                compute_point_cloud_from_rgbd,
                K=sensor_info['K'], 
                pcd_offset=sensor_info['pcd_offset'],
                pcd_norm_range=sensor_info['pcd_norm_range'],
                clip_bbox_size=sensor_info['clip_bbox_size'],
                cam_to_img_tf=None,
                world_to_cam_tf=sensor_info["world_to_cam_tf"],
                pcd_step_vis=False,
                max_depth=sensor_info['sensor_max_depth'],
                sample_type=sample_type,
                num_points_to_sample=sensor_info['number_points_to_sample'],
                clip_scene=True,
                with_color=with_color
                ),
                rgbd
                )
    pcd_demo = np.array(pcd_demo)

    print('finished processing point cloud, pcd shape: ', pcd_demo.shape)
    print('time used:', time.time() - cur_time)   
    print("")

    return pcd_demo


def process_pointcloud_per_demo(rgbd, vis_sign=True, sample_type="fps", with_color=True, sensor_info=None):
    """
    get point cloud from depth information
    """

    print('start processing point cloud ... ')

    cur_time = time.time()
    print('rgbd shape', rgbd.shape)
    
    if vis_sign:
        # start processing and visualizing the point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd_vis = o3d.geometry.PointCloud()
        firstfirst = True
    
    # without parallel processing 
    pcd_demo = []
    step = 0
    
    for i, rgbd_step in enumerate(rgbd):
        step += 1
        pcd = compute_point_cloud_from_rgbd(
            rgbd=rgbd_step, 
            K=sensor_info['K'], 
            pcd_offset=sensor_info['pcd_offset'],
            pcd_norm_range=sensor_info['pcd_norm_range'],
            clip_bbox_size=sensor_info['clip_bbox_size'],
            cam_to_img_tf=None, 
            world_to_cam_tf=sensor_info["world_to_cam_tf"], 
            pcd_step_vis=False, 
            max_depth=sensor_info['sensor_max_depth'],
            sample_type='fps',
            num_points_to_sample=sensor_info['number_points_to_sample'],
            clip_scene=True,
            with_color=with_color
            )
        pcd_demo.append(pcd)

        if vis_sign:
            # print('step', step, 'number of points', pcd.shape[0])
            if with_color:
                pcd_vis.colors = o3d.utility.Vector3dVector(pcd[:, :3])
                pcd_vis.points = o3d.utility.Vector3dVector(pcd[:, 3:]) 
            else:
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


def process_prop_per_demo(obs):
    """
    process the prop states for each demo
    base_qvel, trunk_qpos, arm_left_qpos, arm_right_qpos, left_gripper_width, right_gripper_width
    """

    # base_qpos = obs['base_qpos'] # steps, 3
    base_qvel = np.array(obs['base_qvel']) # steps, 3
    trunk_qpos = np.array(obs['trunk_qpos']) # steps, 4
    arm_left_qpos = np.array(obs['arm_left_qpos']) # steps, 6
    arm_right_qpos = np.array(obs['arm_right_qpos']) # steps, 6
    gripper_left_qpos = np.array(obs['gripper_left_qpos'])
    left_gripper_width = np.array(gripper_left_qpos).sum(axis=1)[:,None] # steps, 1
    gripper_right_qpos = np.array(obs['gripper_right_qpos'])
    right_gripper_width = np.array(gripper_right_qpos).sum(axis=1)[:,None] # steps, 1
    prop_state = np.concatenate((base_qvel, trunk_qpos, arm_left_qpos, arm_right_qpos, left_gripper_width, right_gripper_width), axis=1) # steps, 21

    return prop_state


def process_bimanual_state_per_demo(obs):
    """
    process the bimanual states for each demo
    base_qvel, trunk_qpos, arm_left_qpos, arm_right_qpos, left_gripper_width, right_gripper_width
    """
    arm_left_qpos = np.array(obs['arm_left_qpos']) # steps, 6
    arm_right_qpos = np.array(obs['arm_right_qpos']) # steps, 6
    gripper_left_qpos = np.array(obs['gripper_left_qpos'])
    left_gripper_width = np.array(gripper_left_qpos).sum(axis=1)[:,None] # steps, 1
    gripper_right_qpos = np.array(obs['gripper_right_qpos'])
    right_gripper_width = np.array(gripper_right_qpos).sum(axis=1)[:,None] # steps, 1
    bimanual_state = np.concatenate((arm_left_qpos, arm_right_qpos, left_gripper_width, right_gripper_width), axis=1) # steps, 14

    return bimanual_state


def process_eef_per_demo(obs):
    """
    process the eef states for each demo
    """
    eef_left_pos = np.array(obs['eef_left_pos']) # steps, 3
    eef_right_pos = np.array(obs['eef_right_pos']) # steps, 3
    eef_left_quat = np.array(obs['eef_left_quat']) # steps, 4
    eef_right_quat = np.array(obs['eef_right_quat']) # steps, 4
    eef_state = np.concatenate((eef_left_pos, eef_right_pos, eef_left_quat, eef_right_quat), axis=1) # steps, 14
    return eef_state


def parse_obs(obs, obs_type, with_color):
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

    # get object keys
    obj_key_list = []
    for obj_key in obs.keys():
        if 'object::' in obj_key and "floor" not in obj_key and "table" not in obj_key:
            obj_key_list.append(obj_key)

    # prepare obs keys based on obs_types
    if obs_type == "low_dim":
        other_keys = [
            'joint_qpos', 
            'prop_state',
            'base_qpos',
            'eef_left_pos', 'eef_left_quat',
            'eef_right_pos', 'eef_right_quat', 
            ] 
        obs_key_list = other_keys + obj_key_list
    
    elif obs_type == "point_cloud":
        other_keys = [
            'joint_qpos', 
            'prop_state',
            'prop_eef_state',
            'prop_eef_basepose',
            ] 
        if with_color:
            other_keys.append('combined::color_point_cloud')
        else:
            other_keys.append('combined::point_cloud')

        obs_key_list = other_keys + obj_key_list

    elif obs_type == "rgb":
        other_keys = [
            'joint_qpos', 
            'prop_state',
            'prop_eef_state',
            'prop_eef_basepose',
            'rgb',
            'depth'
        ]

        obs_key_list = other_keys + obj_key_list

    else:
        raise ValueError("Invalid obs_type")
    
    return obs_key_list


def process_robomimic_dataset(file_path, obs_type, sample_type="fps", with_color=True, vis_sign=False):
    # the original element are in the format of actions, states, obs, datagen_info, src_demo_inds, src_demo_labels, mp_end_steps, subtask_lengths, sensor_info

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
            actions = demo_data["actions"][:-1] # actions already in range [-1, 1]

            # get rewards and dones
            # assume the data are expert demonstrations and only the last step is the success step
            rewards = np.zeros(num_steps)
            rewards[-1] = 1
            dones = np.zeros(num_steps)
            dones[-1] = 1

            # get rgbd information
            if 'external::external_sensor0::rgb' in demo_data["obs"].keys():
                rgb = np.array(demo_data["obs"]['external::external_sensor0::rgb'])[:, :, :, :3]
                depth = np.array(demo_data["obs"]['external::external_sensor0::depth_linear'])
            elif 'external::viewer::rgb' in demo_data["obs"].keys():
                rgb = np.array(demo_data["obs"]['external::viewer::rgb'])[:, :, :, :3]
                depth = np.array(demo_data["obs"]['external::viewer::depth_linear'])[:, :, :, None]
            rgbd = np.concatenate([rgb, depth], axis=-1) # (traj_length, with, height, 4)

            # get point cloud information
            obs_key_list = parse_obs(demo_data["obs"], obs_type, with_color=with_color)
            print(demo_key, 'observation keys', obs_key_list)

            # process point cloud when necessary
            if 'combined::point_cloud' in obs_key_list or 'combined::color_point_cloud' in obs_key_list:
                sensor_info_hdf5 = demo_data['sensor_info']
                sensor_info = {}
                for key in sensor_info_hdf5.keys():
                    sensor_info[key] = np.array(sensor_info_hdf5[key]) # convert to numpy
                if vis_sign:
                    # for pcd debugging, visualized the first 100 steps in each episode
                    pcd_demo = process_pointcloud_per_demo(
                        rgbd, 
                        vis_sign=vis_sign, 
                        sample_type=sample_type,
                        with_color=with_color,
                        sensor_info=sensor_info
                        )
                else:
                    # get observations
                    pcd_demo = process_pointcloud_per_demo_parallel(
                        rgbd, 
                        sample_type=sample_type,
                        with_color=with_color,
                        sensor_info=sensor_info
                        ) # get point cloud from rgbd images
            
            # process the observations
            for obs_key in obs_key_list:
                if "point_cloud" in obs_key:
                    obs_dict[obs_key] = pcd_demo[:-1]
                    next_obs_dict[obs_key] = pcd_demo[1:]
                elif 'rgb' in obs_key:
                    obs_dict[obs_key] = rgb[:-1]
                    next_obs_dict[obs_key] = rgb[1:]
                elif 'depth' in obs_key:
                    obs_dict[obs_key] = depth[:-1]
                    next_obs_dict[obs_key] = depth[1:]
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
    parser.add_argument(
        "--with_color",
        action="store_true",
        help="debug mode: only save the first 50 steps of each demo"
    )
    parser.add_argument(
        "--process_subtasks",
        action="store_true",
        help="process the subtaks"
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
    
    sample_type='default'
    if args.obs_type == "point_cloud":
        assert args.fps != args.random, "Only one of fps and random can be True"
        if args.fps: sample_type = "fps"
        if args.random: sample_type = "random"

    # change to robomimic dataset format
    robomimic_dataset = process_robomimic_dataset(
        file_path=file_path,
        obs_type=args.obs_type,
        sample_type=sample_type,
        with_color=args.with_color,
        vis_sign=args.vis_sign
    )
    # if args.vis_sign:
    #     sys.exit()
    
    # change the processed file name accordingly
    if args.obs_type == "point_cloud":
        output_path = output_path.replace(".hdf5", "_{}_{}.hdf5".format(sample_type, args.num_pcd_samples))
        if args.with_color:
            output_path = output_path.replace(".hdf5", "_color.hdf5")
    if args.debug:
        output_path = output_path.replace(".hdf5", "_debug.hdf5")
    if args.obs_type == "rgb":
        output_path = output_path.replace(".hdf5", "_rgb.hdf5")
    

    print("")
    print('Writing the data to', output_path, '....')
    # write to hdf5
    write_to_hdf5(robomimic_dataset, file_path, output_path)
    print("")
    print('Finished writing the data.')

    if args.process_subtasks:
        print("")
        print('Start processing subtasks', output_path, '....')

        # start processing subtasks
        subtask_data_dict, num_subtasks = process_subtask_dataset(file_path, output_path)

        for i in range(num_subtasks):
            subtask_key = "subtask_" + str(i)
            subtask_output_path = output_path.replace(".hdf5", "_subtask_{}.hdf5".format(subtask_key))
            write_to_hdf5(subtask_data_dict[subtask_key], file_path, subtask_output_path)