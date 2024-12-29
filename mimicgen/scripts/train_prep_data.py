import json
import numpy as np
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
import h5py, argparse, pdb
from robomimic.scripts.split_train_val import split_train_val_from_hdf5

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
    - demo_1
"""

def process_robomimic_dataset(file_path, obs_key_list):
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
            for obs_key in obs_key_list:
                assert obs_key in demo_data['obs'].keys()
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
            dataset_dict[demo_key] = demo_dict
    
    return dataset_dict

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
    parser.add_argument("--obs_key_type", type=str, default="low_dim", help="observation key type", choices=["low_dim", "rgb", "depth", "point_cloud"])
    # trian val split ratio
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.1,
        help="validation ratio, in (0, 1)"
    )

    args = parser.parse_args()

    file_path = args.file_path
    output_path = args.output_path
    if args.obs_key_type == "low_dim":
        obs_key_list = ["task::low_dim"]
    else:
        raise NotImplementedError
    
    # first split the train and val data
    split_train_val_from_hdf5(file_path, val_ratio=args.split_ratio)
    
    # change to robomimic dataset format
    robomimic_dataset = process_robomimic_dataset(
        file_path=file_path,
        obs_key_list=obs_key_list
    )
    
    # write to hdf5
    write_to_hdf5(robomimic_dataset, file_path, output_path)

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