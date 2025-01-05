# change to robomimic dataset
python mimicgen/scripts/train_prep_data.py --file_path /home/mengdi/dataset/demo.hdf5 --output_path /home/mengdi/dataset/robomimic_dataset.hdf5 --split_ratio 0.5

python mimicgen/scripts/train_prep_data.py --file_path /home/mengdi/dataset/demo.hdf5 --output_path /home/mengdi/dataset/robomimic_dataset.hdf5 --split_ratio 0.5 --debug

# start training
python mimicgen/scripts/train_mimicgen.py --config mimicgen/scripts/train_config_cup.json --mg_config /tmp/core_configs_og/demo_src_test_tiago_cup_task_D0.json

# some dependencies are missing, need to install then
# open3d
# pygpg
# fpsample
# diffusers