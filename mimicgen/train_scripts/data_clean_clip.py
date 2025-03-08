import h5py
import numpy as np
from mimicgen.train_scripts.train_prep_data import write_to_hdf5
# load hdf5 and only keep the first 1 key



def clip_dataset(data_path, end_key_dict):
    dataset_dict = {}
    with h5py.File(data_path, "r") as hdf:
        # Access a group or dataset
        group = hdf["data"]
        for demo_key in group.keys():
            if demo_key != 'demo_0':
                continue
            
            print("")
            print('Processing', demo_key)
            demo_data = group[demo_key]
            demo_data_clip = {}

            clip_start_step = 0
            clip_end_step = end_key_dict[demo_key]

            demo_data_clip["obs"] = {}
            for obs_key in demo_data["obs"].keys():
                demo_data_clip["obs"][obs_key] = demo_data["obs"][obs_key][clip_start_step:clip_end_step]
            
            demo_data_clip["next_obs"] = {}
            for next_obs_key in demo_data["next_obs"].keys():
                demo_data_clip["next_obs"][next_obs_key] = demo_data["next_obs"][next_obs_key][clip_start_step:clip_end_step]
            
            demo_data_clip["actions"] = demo_data["actions"][clip_start_step:clip_end_step]
            demo_data_clip["rewards"] = demo_data["rewards"][clip_start_step:clip_end_step]
            demo_data_clip["dones"] = demo_data["dones"][clip_start_step:clip_end_step]

            print('Subtask length', demo_data_clip["actions"].shape[0])

            dataset_dict[demo_key] = demo_data_clip

    return dataset_dict



# step 3: increase the observation horizon when training, see what can be the maximum with a100 80G memory 

# step 0: evaluate the transformer policy 

if __name__ == "__main__":
    # file_name = '/home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_r1_cup/robomimic_dataset_D0_fps_4096_color.hdf5'
    # out_path = '/home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_r1_cup/robomimic_dataset_D0_fps_4096_color_clipped.hdf5'
    file_name = '/home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_r1_cup/robomimic_dataset_D0_fps_4096.hdf5'
    out_path = '/home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_r1_cup/robomimic_dataset_D0_fps_4096_clipped.hdf5'

    file_name = '/home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_r1_cup/robomimic_dataset_D0_rgb.hdf5'
    out_path = '/home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_r1_cup/robomimic_dataset_D0_rgb_clipped.hdf5'

    # clip the data to remove the final drifting part
    # TODO: need to first replay the data to get the starting point of the drifting part
    end_key_dict = {
        "demo_0": 1565,
        "demo_1": 1665,
    }
    
    data_dict = clip_dataset(file_name, end_key_dict)
    write_to_hdf5(data_dict, file_name, out_path)

