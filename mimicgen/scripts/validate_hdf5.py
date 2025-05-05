import h5py
import os

task = "dishes_away"
dataset_folder = f"{task}_wo_joint_limit"

for i in range(10):
    tmp_folder = f"/cvgl2/u/chengshu/mimicgen/datasets/generated_data_mimicgen_format/{dataset_folder}/r1_{task}_worker_{i}/demo_src_r1_{task}_task_D0/tmp"
    tmp_failed_folder = f"/cvgl2/u/chengshu/mimicgen/datasets/generated_data_mimicgen_format/{dataset_folder}/r1_{task}_worker_{i}/demo_src_r1_{task}_task_D0/tmp_failed"
    if os.path.exists(tmp_folder):
        hdf5_files = os.listdir(tmp_folder)
        # print(f"tmp folder {tmp_folder} has {len(hdf5_files)} hdf5 files")
        for hdf5_file in hdf5_files:
            hdf5_path = os.path.join(tmp_folder, hdf5_file)
            try:
                with h5py.File(hdf5_path, "r") as f:
                    if "data" not in f:
                        print("invalid hdf5 file", hdf5_path)
            except:
                print("invalid hdf5 file", hdf5_path)
    if os.path.exists(tmp_failed_folder):
        hdf5_files = os.listdir(tmp_failed_folder)
        # print(f"tmp failed folder {tmp_failed_folder} has {len(hdf5_files)} hdf5 files")
        for hdf5_file in hdf5_files:
            hdf5_path = os.path.join(tmp_failed_folder, hdf5_file)
            try:
                with h5py.File(hdf5_path, "r") as f:
                    if "data" not in f:
                        print("invalid hdf5 file", hdf5_path)
            except:
                print("invalid hdf5 file", hdf5_path)

