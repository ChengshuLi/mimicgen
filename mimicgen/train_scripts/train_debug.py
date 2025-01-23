import mimicgen.utils.file_utils as MG_FileUtils
import os
import h5py

import matplotlib.pyplot as plt

# merge the hdf5 files for D1 randomization

def merge_hdf5_files(folder_path, new_dataset_path):
    new_dataset_folder_path = '/tmp/core_datasets_og/test_tiago_cup/demo_src_test_tiago_cup_task_D1/'
    tmp_dataset_folder_path = os.path.join(new_dataset_folder_path, "tmp")
    new_dataset_path = os.path.join(new_dataset_folder_path, "demo.hdf5")
    print("Merging all hdf5 files in folder: {}".format(tmp_dataset_folder_path))
    print("New dataset path: {}".format(new_dataset_path))
    MG_FileUtils.merge_all_hdf5(
            folder=tmp_dataset_folder_path,
            new_hdf5_path=new_dataset_path,
            delete_folder=False,
        )


# merge_hdf5_files()

# plot the initial object statistics
def plot_init_states():
    data_name = 'D1_10'
    data_path = '/home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_tiago_cup/'
    file_path = f'{data_path}robomimic_dataset_{data_name}_fps_2048.hdf5'
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
        fig.suptitle('Initial object states')
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
        plt.savefig(os.path.join(data_path, f'init_states_{data_name}.png'))
        plt.show()


# plot_init_states()


def load_init_states(data_name='D1_10'):
    file_path = f'/home/mengdi/dataset/test_tiago_cup/demo_{data_name}.hdf5'
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

# load_init_states()


