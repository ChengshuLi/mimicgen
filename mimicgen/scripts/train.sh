# change to robomimic dataset
python train_prep_data.py --file_path /home/mengdi/dataset/demo_failed.hdf5 --output_path /home/mengdi/dataset/robomimic_dataset.hdf5 --split_ratio 0.5

# start training
python train_mimicgen.py --config train_config_cup.json --mg_config /tmp/core_configs_og/demo_src_test_tiago_cup_task_D0.json

# some dependencies are missing, need to install then
# open3d
# pygpg
# fpsample
# diffusers