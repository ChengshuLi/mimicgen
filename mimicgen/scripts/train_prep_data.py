import json
import numpy as np
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
import h5py, argparse, pdb
from robomimic.scripts.split_train_val import split_train_val_from_hdf5

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
    elif obs_type == "rgb":
        obs_key_list = [
            'robot_fjtzyj::robot_fjtzyj:eyes:Camera:0::rgb',
            'external::external_sensor0::rgb'
        ]
    elif obs_type == "depth":
        obs_key_list = [
            'robot_fjtzyj::robot_fjtzyj:eyes:Camera:0::depth',
            'external::external_sensor0::depth'
        ]
    elif obs_type == "point_cloud":
        obs_key_list = [
            'combined::point_cloud'
        ]
    else:
        raise ValueError("Invalid obs_type")
    
    return obs_key_list

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

def process_robomimic_dataset(file_path, obs_type, corp_demo_for_debug=False):
    # the original element are in the format of actions, states, obs, datagen_info, src_demo_inds, src_demo_labels, mp_end_steps, subtask_lengths
    # for each demostration， get the obs, next_obs, actions, rewards, dones
    dataset_dict = {}
    # Open the file
    with h5py.File(file_path, "r") as hdf:
        # Access a group or dataset
        group = hdf["data"]
        for demo_key in group.keys():
            demo_data = group[demo_key]
            obs_dict = {}
            next_obs_dict = {}
            obs_key_list = parse_obs(demo_data["obs"], obs_type)
            
            print(demo_key, 'observation keys', obs_key_list)
            for obs_key in obs_key_list:

                # debugging_grasp_obs(demo_data) # debugging the grasp states
                # debugging_camera_imgs(demo_data) # debugging the camera images

                obs_dict[obs_key] = demo_data['obs'][obs_key][:-1]
                next_obs_dict[obs_key] = demo_data['obs'][obs_key][1:]
                num_steps = demo_data['obs'][obs_key].shape[0] - 1
                assert obs_dict[obs_key].shape[0] == next_obs_dict[obs_key].shape[0] == num_steps
            actions = demo_data["actions"][:-1]

            # assume the data are expert demonstrations and only the last step is the success step
            rewards = np.zeros(num_steps)
            rewards[-1] = 1
            dones = np.zeros(num_steps)
            dones[-1] = 1

            demo_dict = {
                "obs": obs_dict,
                "next_obs": next_obs_dict,
                "actions": actions,
                "rewards": rewards,
                "dones": dones
            }

            if corp_demo_for_debug:
                total_length = 50
                for obs_key in obs_dict.keys():
                    demo_dict["obs"][obs_key] = demo_dict["obs"][obs_key][:total_length]
                    demo_dict["next_obs"][obs_key] = demo_dict["next_obs"][obs_key][:total_length]
                demo_dict["actions"] = demo_dict["actions"][:total_length]
                demo_dict["rewards"] = demo_dict["rewards"][:total_length]
                demo_dict["dones"] = demo_dict["dones"][:total_length]

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
    parser.add_argument("--obs_type", type=str, default="low_dim", help="observation key type", choices=["low_dim", "rgb", "depth", "point_cloud"])
    # trian val split ratio
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.1,
        help="validation ratio, in (0, 1)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode: only save the first 50 steps of each demo"
    )

    args = parser.parse_args()

    file_path = args.file_path
    output_path = args.output_path

    print("")
    print('Start processing the dataset', file_path, '....')
    print("")
    
    # first split the train and val data
    split_train_val_from_hdf5(file_path, val_ratio=args.split_ratio)
    
    # change to robomimic dataset format
    robomimic_dataset = process_robomimic_dataset(
        file_path=file_path,
        obs_type=args.obs_type,
        corp_demo_for_debug=args.debug
    )

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

    # # Open the file
    # with h5py.File(output_path, "r") as hdf:
    #     # List the groups
    #     print(list(hdf.keys()))  # Prints top-level groups
    #     pdb.set_trace()

    #     # Access a group or dataset
    #     group = hdf["data"]
    #     print(list(group.keys()))  # Prints contents of the group
    #     for demo_key in group.keys():
    #         demo_data = group[demo_key]
    #         print(list(demo_data.keys()))
    #         # print(demo_data.shape, demo_data.dtype)  # Dataset details

    # pdb.set_trace()