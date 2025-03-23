import mimicgen.utils.file_utils as MG_FileUtils
import os
import h5py

import matplotlib.pyplot as plt

# merge the hdf5 files for D1 randomization

def merge_hdf5_files(
        saved_dataset_folder_path, 
        new_dataset_folder_path=None
        ):
    new_dataset_path = os.path.join(saved_dataset_folder_path, "demo_500.hdf5")
    print("Merging all hdf5 files in folder: {}".format(saved_dataset_folder_path))
    print("New dataset path: {}".format(new_dataset_path))
    MG_FileUtils.merge_all_hdf5(
            folder=saved_dataset_folder_path,
            new_hdf5_path=new_dataset_path,
            delete_folder=False,
        )

# plot the initial object statistics
def plot_init_states(file_path, save_path, save_name):

    init_states = {}
    # Open the file
    with h5py.File(file_path, "r") as hdf:
        # Access a group or dataset
        group = hdf["data"]
        # process data for each demo
        for demo_key in group.keys():
            demo_data = group[demo_key]
            init_states[demo_key] = {}
            init_states[demo_key]["coffee_cup"] = demo_data['obs']['object::coffee_cup'][0]
            init_states[demo_key]["dixie_cup"] = demo_data['obs']['object::dixie_cup'][0]
        
        mask = hdf["mask"]
        train_mask = mask["train"]
        val_mask = mask["valid"]
        

        # start plotting
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f'Initial object states, train: {len(train_mask)}, val: {len(val_mask)}')
        for key in init_states.keys():
            val_1 = init_states[key]["coffee_cup"][:2]
            print('val_1:', val_1)
            axs[0].scatter(init_states[key]["coffee_cup"][0], init_states[key]["coffee_cup"][1], marker='o', color='b')
            axs[1].scatter(init_states[key]["dixie_cup"][0], init_states[key]["dixie_cup"][1], marker='*', color='b')
        axs[0].set_title('Coffee cup x,y')
        axs[1].set_title('Diexie cup x,y')

        # plot validation with different color
        for val_demo_key in val_mask:
            val_demo_key = val_demo_key.decode("utf-8")
            print('val_demo_key:', val_demo_key)
            axs[0].plot(init_states[val_demo_key]["coffee_cup"][0], init_states[val_demo_key]["coffee_cup"][1], marker='o', color='r')
            axs[1].plot(init_states[val_demo_key]["dixie_cup"][0], init_states[val_demo_key]["dixie_cup"][1],marker='*', color='r')
        # axs[0, 0].plot(init_states[val_mask[0]]["coffee_cup"][:2], marker='o', color='r')
        # axs[0, 1].plot(init_states[val_mask[0]]["dixie_cup"][:2], marker='o', color='r')
        plt.savefig(os.path.join(save_path, f'init_states_{save_name}.png'))
        plt.show()


def load_init_states(data_name=None):
    # file_path = f'/home/mengdi/dataset/test_tiago_cup/demo_{data_name}.hdf5'
    # file_path = f'/home/mengdi/b1k_datagen/mimicgen/datasets/source_og/test_taigo_cup/demo_{data_name}.hdf5'
    file_path = "/home/arpit/test_projects/mimicgen/temp_datasets/demo_failed.hdf5"

    init_states = {}
    # Open the file
    with h5py.File(file_path, "r") as hdf:
        # Access a group or dataset
        group = hdf["data"]
        # process data for each demo
        for demo_key in group.keys():
            demo_data = group[demo_key]
            init_states[demo_key] = demo_data['states'][0]
    
    return init_states


def merge_robomimic_datasets(file_list):
    # TODO: not working now, need to fix

    file_name = file_list[0]

    new_hdf5_path = "home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_tiago_cup/robomimic_dataset_D1_merged_fps_2048.hdf5"

    # write demos in order to new file
    f_new = h5py.File(new_hdf5_path, "w")
    f_new_grp = f_new.create_group("data")

    for i, source_hdf5_path in enumerate(file_list):
        with h5py.File(source_hdf5_path, "r") as f:
            pass
            
    return None


if __name__ == "__main__":
    
    # # merge the hdf5 files of raw generated data
    # saved_dataset_folder_path = '/home/mengdi/dataset/test_tiago_cup/tmp/'
    # merge_hdf5_files(saved_dataset_folder_path)

    # plot the statistics of the initial object states
    data_name = 'D1_500'
    data_path = '/home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_tiago_cup/'
    file_path = f'{data_path}robomimic_dataset_{data_name}_fps_2048.hdf5'
    plot_init_states(file_path, data_path, data_name)

    # try with load with initial states
    # load_init_states()

    # merge the procecced robomimic datasets 

    file_list = [
        '/home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_tiago_cup/robomimic_dataset_D1_10_fps_2048.hdf5',
        '/home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_tiago_cup/robomimic_dataset_D1_64_fps_2048.hdf5',
        '/home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_tiago_cup/robomimic_dataset_D1_500_fps_2048.hdf5',
    ]

    # merge_robomimic_datasets(file_list)